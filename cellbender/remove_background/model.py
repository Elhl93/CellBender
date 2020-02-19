"""Definition of the model and the inference setup, with helper functions."""

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
import pyro.poutine as poutine

from cellbender.remove_background.vae import encoder as encoder_module
import cellbender.remove_background.consts as consts
from cellbender.remove_background.distributions.NegativeBinomialPoissonConv \
    import NegativeBinomialPoissonConv as NBPC
from cellbender.remove_background.distributions.NegativeBinomialPoissonConvApprox \
    import NegativeBinomialPoissonConvApprox as NBPCapprox
from cellbender.remove_background.distributions.NullDist import NullDist
from cellbender.remove_background.distributions.PoissonImportanceMarginalizedGamma \
    import PoissonImportanceMarginalizedGamma
from cellbender.remove_background.exceptions import NanException

from typing import Union
from numbers import Number


class RemoveBackgroundPyroModel(nn.Module):
    """Class that contains the model and guide used for variational inference.

    Args:
        model_type: Which model is being used, one of ['simple', 'ambient',
            'swapping', 'full'].
        encoder: An instance of an encoder object.  Can be a CompositeEncoder.
        decoder: An instance of a decoder object.
        dataset_obj: Dataset object which contains relevant priors.
        use_cuda: Will use GPU if True.

    Attributes:
        All the above, plus
        device: Either 'cpu' or 'cuda' depending on value of use_cuda.

    """

    def __init__(self,
                 model_type: str,
                 encoder: Union[nn.Module, encoder_module.CompositeEncoder],
                 decoder: nn.Module,
                 dataset_obj: 'SingleCellRNACountsDataset',
                 rho_alpha_prior: float = 1.5,
                 rho_beta_prior: float = 20,
                 use_cuda: bool = False):
        super(RemoveBackgroundPyroModel, self).__init__()

        self.model_type = model_type
        self.include_empties = True
        if self.model_type == "simple":
            self.include_empties = False
        self.include_rho = False
        if (self.model_type == "full") or (self.model_type == "swapping"):
            self.include_rho = True

        self.n_genes = dataset_obj.analyzed_gene_inds.size
        self.z_dim = decoder.input_dim
        self.encoder = encoder
        self.decoder = decoder
        self.loss = {'train': {'epoch': [], 'elbo': []},
                     'test': {'epoch': [], 'elbo': []},
                     'params': {}}  # TODO

        # Determine whether we are working on a GPU.
        if use_cuda:
            # Calling cuda() here will put all the parameters of
            # the encoder and decoder networks into GPU memory.
            self.cuda()
            try:
                for key, value in self.encoder.items():
                    value.cuda()
            except KeyError:
                pass
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.use_cuda = use_cuda

        # Priors
        assert dataset_obj.priors['d_std'] > 0, \
            f"Issue with prior: d_std is {dataset_obj.priors['d_std']}, " \
            f"but should be > 0."
        assert dataset_obj.priors['cell_counts'] > 0, \
            f"Issue with prior: cell_counts is " \
            f"{dataset_obj.priors['cell_counts']}, but should be > 0."

        self.d_cell_loc_prior = (np.log1p(dataset_obj.priors['cell_counts'],
                                          dtype=np.float32).item()
                                 * torch.ones(torch.Size([])).to(self.device))
        # TODO:
        self.d_cell_scale_prior = (torch.tensor(0.2).to(self.device))
                                   #np.array(dataset_obj.priors['d_std'],
                                            # dtype=np.float32).item()
                                   # * torch.ones(torch.Size([])).to(self.device))
        self.z_loc_prior = torch.zeros(torch.Size([self.z_dim])).to(self.device)
        self.z_scale_prior = torch.ones(torch.Size([self.z_dim]))\
            .to(self.device)
        self.epsilon_prior = torch.tensor(1000.).to(self.device)
        self.alpha0_loc_prior = (torch.tensor(consts.ALPHA0_PRIOR_LOC)
                                 .float().to(self.device))
        self.alpha0_scale_prior = (torch.tensor(consts.ALPHA0_PRIOR_SCALE)
                                   .float().to(self.device))
        self.flat_alpha = (torch.ones(self.n_genes)
                           / self.n_genes).float().to(self.device)

        if self.model_type != "simple":

            assert dataset_obj.priors['empty_counts'] > 0, \
                f"Issue with prior: empty_counts should be > 0, but is " \
                f"{dataset_obj.priors['empty_counts']}"
            chi_ambient_sum = np.round(dataset_obj.priors['chi_ambient']
                                       .sum().item(),
                                       decimals=4).item()
            assert chi_ambient_sum == 1., f"Issue with prior: chi_ambient " \
                                          f"should sum to 1, but it sums to " \
                                          f"{chi_ambient_sum}"
            chi_bar_sum = np.round(dataset_obj.priors['chi_bar'].sum().item(),
                                   decimals=4)
            assert chi_bar_sum == 1., f"Issue with prior: chi_bar should " \
                f"sum to 1, but is {chi_bar_sum}"

            self.d_empty_loc_prior = (np.log1p(dataset_obj
                                               .priors['empty_counts'],
                                               dtype=np.float32).item()
                                      * torch.ones(torch.Size([]))
                                      .to(self.device))

            self.d_empty_scale_prior = (np.array(dataset_obj.priors['d_std'],
                                                 dtype=np.float32).item()
                                        * torch.ones(torch.Size([])).to(self.device))

            self.p_logit_prior = (dataset_obj.priors['cell_logit']
                                  * torch.ones(torch.Size([])).to(self.device))

            self.chi_ambient_init = dataset_obj.priors['chi_ambient']\
                .to(self.device)
            self.avg_gene_expression = dataset_obj.priors['chi_bar'] \
                .to(self.device)

            self.empty_UMI_threshold = (torch.tensor(dataset_obj.empty_UMI_threshold)
                                        .float().to(self.device))

        else:

            self.avg_gene_expression = None

        self.rho_alpha_prior = (rho_alpha_prior
                                * torch.ones(torch.Size([])).to(self.device))
        self.rho_beta_prior = (rho_beta_prior
                               * torch.ones(torch.Size([])).to(self.device))

    def calculate_lambda(self,
                         epsilon: torch.Tensor,
                         chi_ambient: torch.Tensor,
                         d_empty: torch.Tensor,
                         y: Union[torch.Tensor, None] = None,
                         d_cell: Union[torch.Tensor, None] = None,
                         rho: Union[torch.Tensor, None] = None,
                         chi_bar: Union[torch.Tensor, None] = None):
        """Calculate noise rate based on the model."""

        if self.model_type == "simple" or self.model_type == "ambient":
            lam = epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1) * chi_ambient

        elif self.model_type == "swapping":
            lam = (rho.unsqueeze(-1) * y.unsqueeze(-1)
                   * epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1)
                   + d_empty.unsqueeze(-1)) * chi_bar

        elif self.model_type == "full":
            lam = ((1. - rho.unsqueeze(-1))
                   * epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1) * chi_ambient
                   + rho.unsqueeze(-1) *
                   (y.unsqueeze(-1) * d_cell.unsqueeze(-1)
                    + d_empty.unsqueeze(-1)) * chi_bar)
        else:
            raise NotImplementedError(f"model_type was set to {self.model_type}, "
                                      f"which is not implemented.")

        return lam

    def calculate_mu(self,
                     epsilon: torch.Tensor,
                     d_cell: torch.Tensor,
                     chi: torch.Tensor,
                     y: Union[torch.Tensor, None] = None,
                     rho: Union[torch.Tensor, None] = None):
        """Calculate mean expression based on the model."""

        if self.model_type == 'simple':
            mu = epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1) * chi

        elif self.model_type == 'ambient':
            mu = (y.unsqueeze(-1) * epsilon.unsqueeze(-1)
                  * d_cell.unsqueeze(-1) * chi)

        elif self.model_type == 'swapping' or self.model_type == 'full':
            mu = ((1. - rho.unsqueeze(-1))
                  * y.unsqueeze(-1) * epsilon.unsqueeze(-1)
                  * d_cell.unsqueeze(-1) * chi)

        else:
            raise NotImplementedError(f"model_type was set to {self.model_type}, "
                                      f"which is not implemented.")

        return mu

    def model(self, x: torch.Tensor):
        """Data likelihood model.

            Args:
                x: Mini-batch of data. Barcodes are rows, genes are columns.

        """

        # Register the decoder with pyro.
        pyro.module("decoder", self.decoder)

        # Register the hyperparameter for ambient gene expression.
        if self.include_empties:
            chi_ambient = pyro.param("chi_ambient",
                                     self.chi_ambient_init *
                                     torch.ones(torch.Size([])).to(self.device),
                                     constraint=constraints.simplex)
        else:
            chi_ambient = None

        # # TODO: added this for rho as hyperparameter
        # if self.include_rho:
        #     rho_alpha = pyro.param("rho_alpha",
        #                            0.02 *
        #                            torch.ones(torch.Size([])).to(self.device),
        #                            constraint=constraints.unit_interval)

        # Happens in parallel for each data point (cell barcode) independently:
        with pyro.plate("data", x.size(0),
                        use_cuda=self.use_cuda, device=self.device):

            # Sample z from prior.
            z = pyro.sample("z",
                            dist.Normal(loc=self.z_loc_prior,
                                        scale=self.z_scale_prior)
                            .expand_by([x.size(0)]).to_event(1))

            # Decode the latent code z to get fractional gene expression, chi.
            chi = self.decoder.forward(z)

            # Sample alpha0 based on a prior.
            alpha0 = pyro.sample("alpha0",
                                 dist.LogNormal(loc=self.alpha0_loc_prior,
                                                scale=self.alpha0_scale_prior)
                                 .expand_by([x.size(0)]))

            # Sample d_cell based on priors.
            d_cell = pyro.sample("d_cell",
                                 dist.LogNormal(loc=self.d_cell_loc_prior,
                                                scale=self.d_cell_scale_prior)
                                 .expand_by([x.size(0)]))

            # TODO: changed rho from a latent to a hyperparameter
            # Sample swapping fraction rho.
            if self.include_rho:
                rho = pyro.sample("rho", dist.Beta(self.rho_alpha_prior,
                                                   self.rho_beta_prior)
                                  .expand_by([x.size(0)]))
                # rho = torch.clamp(rho_alpha, min=1e-5, max=0.9)
            else:
                rho = None

            # Sample epsilon based on priors.
            epsilon = torch.ones_like(alpha0).clone()  # TODO
            # epsilon = pyro.sample("epsilon",
            #                       dist.Gamma(concentration=self.epsilon_prior,
            #                                  rate=self.epsilon_prior)
            #                       .expand_by([x.size(0)]))

            # If modelling empty droplets:
            if self.include_empties:

                # Sample d_empty based on priors.
                d_empty = pyro.sample("d_empty",
                                      dist.LogNormal(loc=self.d_empty_loc_prior,
                                                     scale=self.d_empty_scale_prior)
                                      .expand_by([x.size(0)]))

                # Sample y, the presence of a real cell, based on p_logit_prior.
                y = pyro.sample("y",
                                dist.Bernoulli(logits=self.p_logit_prior - 100.)  # TODO
                                .expand_by([x.size(0)]))

            else:
                d_empty = None
                y = None

            # Calculate the mean gene expression counts (for each barcode).
            mu_cell = self.calculate_mu(epsilon=epsilon,
                                        d_cell=d_cell,
                                        chi=chi,
                                        y=y,
                                        rho=rho)

            if self.include_empties:

                # Calculate the background rate parameter (for each barcode).
                lam = self.calculate_lambda(epsilon=epsilon,
                                            chi_ambient=chi_ambient,
                                            d_empty=d_empty,
                                            y=y,
                                            d_cell=d_cell,
                                            rho=rho,
                                            chi_bar=self.avg_gene_expression)
            else:
                lam = torch.zeros([self.n_genes]).to(self.device)

            alpha = alpha0.unsqueeze(-1) * chi

            USE_NBPC = False

            if USE_NBPC:

                # Sample gene expression from our Negative Binomial Poisson
                # Convolution distribution, and compare with observed data.
                c = pyro.sample("obs", NBPC(mu=mu_cell + 1e-10,
                                            alpha=alpha + 1e-10,
                                            lam=lam + 1e-10,
                                            max_poisson=50).to_event(1),
                                obs=x.reshape(-1, self.n_genes))

            else:

                # Use a negative binomial approximation as the observation model.
                c = pyro.sample("obs", NBPCapprox(mu=mu_cell + 1e-10,
                                                  alpha=alpha + 1e-10,  # Avoid NaNs
                                                  lam=lam + 1e-10).to_event(1),
                                obs=x.reshape(-1, self.n_genes))

                # # Compare model with data using approximate marginalization over Poisson rate.
                # c = pyro.sample("obs",
                #                 PoissonImportanceMarginalizedGamma(mu=mu_cell + 1e-10,
                #                                                    alpha=alpha + 1e-10,
                #                                                    lam=lam + 1e-10,
                #                                                    n_samples=10).to_event(1),
                #                 obs=x.reshape(-1, self.n_genes))

            # Additionally use some high-count droplets for cell prob regularization.
            # surely_cell_mask = (torch.where(counts >= self.d_cell_loc_prior.exp(),
            #                                 torch.ones_like(counts),
            #                                 torch.zeros_like(counts))
            #                     .bool().to(self.device))
            # print(f'sure cells = {surely_cell_mask.sum()}')

            # with poutine.mask(mask=surely_cell_mask):
            #
            #     with poutine.scale(scale=0.1):
            #
            #         pyro.sample("obs_cell_y",
            #                     dist.Normal(loc=p_logit_posterior,
            #                                 scale=1.),
            #                     obs=torch.ones_like(y) * 5.)

            if self.include_empties:

                # Put a prior on p_y_logit to maintain balance.
                pyro.sample("p_logit_reg", dist.Normal(loc=self.p_logit_prior,
                                                       scale=2. * torch.ones([1]).to(self.device)))

                # Additionally use the surely empty droplets for regularization,
                # since we know these droplets by their UMI counts.
                counts = x.sum(dim=-1, keepdim=False)
                surely_empty_mask = ((counts < self.empty_UMI_threshold)
                                     .bool().to(self.device))

                with poutine.mask(mask=surely_empty_mask):

                    # TODO: try this
                    with poutine.scale(scale=0.01):

                        if self.include_rho:
                            r = rho.detach()
                        else:
                            r = None

                        # Semi-supervision of ambient expression.
                        lam = self.calculate_lambda(epsilon=epsilon.detach(),
                                                    chi_ambient=chi_ambient,
                                                    d_empty=d_empty,
                                                    y=torch.zeros_like(d_empty),
                                                    d_cell=d_cell.detach(),
                                                    rho=r,
                                                    chi_bar=self.avg_gene_expression)
                        pyro.sample("obs_empty",
                                    dist.Poisson(rate=lam + 1e-10).to_event(1),
                                    obs=x.reshape(-1, self.n_genes))

                    # Semi-supervision of cell probabilities.

                    # print(f'{(p_logit_posterior > 0).sum()} p_logit_posterior > 0, {(p_logit_posterior < 0).sum()} p_logit_posterior < 0')
                    # print(f'p_logit_posterior[> 0].median() is {p_logit_posterior[p_logit_posterior > 0].median()}')
                    # print(f'p_logit_posterior[< 0].median() is {p_logit_posterior[p_logit_posterior < 0].median()}')
                    # print(f'p_logit_posterior[mask].median() is {p_logit_posterior[surely_empty_mask].median()}')

                    with poutine.scale(scale=1.0):

                        p_logit_posterior = pyro.sample("p_passback",
                                                        NullDist(torch.zeros(1)
                                                                 .to(self.device))
                                                        .expand_by([x.size(0)]))

                        # pyro.sample("obs_empty_y",
                        #             dist.Bernoulli(logits=p_logit_posterior),
                        #             obs=torch.zeros_like(y))
                        pyro.sample("obs_empty_y",
                                    dist.Normal(loc=p_logit_posterior,
                                                scale=1.),
                                    obs=torch.ones_like(y) * -5.)  # TODO: try -10

        return {'mu': mu_cell, 'lam': lam, 'alpha': alpha, 'counts': c}

    @config_enumerate(default='parallel')
    def guide(self, x: torch.Tensor):
        """Variational posterior.

            Args:
                x: Mini-batch of data. Barcodes are rows, genes are columns.

        """

        nan_check = False

        if nan_check:
            for param in pyro.get_param_store().keys():
                if torch.isnan(pyro.param(param).sum()):
                    raise NanException(param)

        # Register the encoder(s) with pyro.
        for name, module in self.encoder.items():
            pyro.module("encoder_" + name, module)

        # Initialize variational parameters for d_cell.
        d_cell_scale = pyro.param("d_cell_scale",
                                  self.d_cell_scale_prior *
                                  torch.ones(torch.Size([])).to(self.device),
                                  constraint=constraints.positive)

        # # Initialize variational parameters for alpha0.
        alpha0_scale = pyro.param("alpha0_scale",
                                  torch.tensor(consts.ALPHA0_PRIOR_SCALE).to(self.device),
                                  constraint=constraints.positive)

        if self.include_empties:

            # Initialize variational parameters for d_empty.
            d_empty_loc = pyro.param("d_empty_loc",
                                     self.d_empty_loc_prior *
                                     torch.ones(torch.Size([])).to(self.device),
                                     constraint=constraints.positive)
            d_empty_scale = pyro.param("d_empty_scale",
                                       self.d_empty_scale_prior *
                                       torch.ones(torch.Size([]))
                                       .to(self.device),
                                       constraint=constraints.positive)

            # Register the hyperparameter for ambient gene expression.
            chi_ambient = pyro.param("chi_ambient",
                                     self.chi_ambient_init *
                                     torch.ones(torch.Size([])).to(self.device),
                                     constraint=constraints.simplex)

        # TODO: rho as a hyperparameter
        # Initialize variational parameters for rho.
        if self.include_rho:
            rho_alpha = pyro.param("rho_alpha",
                                   self.rho_alpha_prior *
                                   torch.ones(torch.Size([])).to(self.device),
                                   constraint=constraints.interval(1., 1000.))  # Prevent NaNs
            rho_beta = pyro.param("rho_beta",
                                  self.rho_beta_prior *
                                  torch.ones(torch.Size([])).to(self.device),
                                  constraint=constraints.interval(1., 1000.))

        # Happens in parallel for each data point (cell barcode) independently:
        with pyro.plate("data", x.size(0),
                        use_cuda=self.use_cuda, device=self.device):

            # TODO: changed rho from a latent to a hyperparameter
            # Sample swapping fraction rho.
            if self.include_rho:
                # print(f'rho_alpha ise {rho_alpha}; rho_beta is {rho_beta}')
                rho = pyro.sample("rho", dist.Beta(rho_alpha,
                                                   rho_beta).expand_by([x.size(0)]))

            # Encode the latent variables from the input gene expression counts.
            if self.include_empties:

                # Sample d_empty, which doesn't depend on y.
                d_empty = pyro.sample("d_empty",
                                      dist.LogNormal(loc=d_empty_loc,
                                                     scale=d_empty_scale)
                                      .expand_by([x.size(0)]))

                enc = self.encoder.forward(x=x, chi_ambient=chi_ambient)

            else:
                enc = self.encoder.forward(x=x, chi_ambient=None)

            # Code specific to models with empty droplets.
            if self.include_empties:

                # Regularize based on wanting a balanced p_y_logit.
                pyro.sample("p_logit_reg", dist.Normal(loc=enc['p_y'], scale=2.))

                # Pass back the inferred p_y to the model.
                pyro.sample("p_passback", NullDist(enc['p_y'].detach()))  # TODO

                # Sample the Bernoulli y from encoded p(y).
                y = pyro.sample("y", dist.Bernoulli(logits=enc['p_y']))

                # Mask out empty droplets.
                with poutine.mask(mask=y.bool()):

                    # Sample latent code z for the barcodes containing cells.
                    pyro.sample("z",
                                dist.Normal(loc=enc['z']['loc'],
                                            scale=enc['z']['scale'])
                                .to_event(1))

                    # Sample alpha0 for the barcodes containing cells.
                    pyro.sample("alpha0",
                                dist.LogNormal(loc=enc['alpha0'],
                                               scale=alpha0_scale))

                    # epsilon = pyro.sample("epsilon",
                    #                       dist.Gamma(enc['epsilon'] * self.epsilon_prior,
                    #                                  self.epsilon_prior))  # TODO

                # Gate d_cell_loc so empty droplets do not give big gradients.
                prob = enc['p_y'].sigmoid()  # Logits to probability
                d_cell_loc_gated = (prob * enc['d_loc'] + (1 - prob)
                                    * self.d_cell_loc_prior)

                # Sample d based on the encoding.
                pyro.sample("d_cell", dist.LogNormal(loc=d_cell_loc_gated,
                                                     scale=d_cell_scale))

            else:

                # Sample d based on the encoding.
                pyro.sample("d_cell", dist.LogNormal(loc=enc['d_loc'],
                                                     scale=d_cell_scale))

                # Sample latent code z for each cell.
                pyro.sample("z",
                            dist.Normal(loc=enc['z']['loc'],
                                        scale=enc['z']['scale'])
                            .to_event(1))

                # Sample alpha0 for the barcodes containing cells.
                pyro.sample("alpha0",
                            dist.LogNormal(loc=enc['alpha0'],
                                           scale=alpha0_scale))

    def store_vars(self, x, params=None):
        """Temp method to store params for inspection.
        Keep them in loss, so they get stored.
        """

        if params is None:
            return

        if not self.include_empties:
            return

        # trace the guide
        trace = poutine.trace(self.guide).get_trace(x)

        cells_only = True
        if cells_only:
            cells = trace.nodes['y']['value'] == 1.

        for p in params:
            if p == 'lam':
                if p + ':mean' not in self.loss['params'].keys():
                    self.loss['params'][p + ':mean'] = []
                    self.loss['params'][p + ':std'] = []
                rho = trace.nodes['rho']['value'] if self.model_type != 'ambient' else None
                lam = self.calculate_lambda(epsilon=torch.tensor(1.).to(self.device),
                                            chi_ambient=pyro.param("chi_ambient"),
                                            d_empty=trace.nodes['d_empty']['value'],
                                            y=trace.nodes['y']['value'],
                                            d_cell=trace.nodes['d_cell']['value'],
                                            rho=rho,
                                            chi_bar=self.avg_gene_expression)
                val_mean = lam.mean().detach().cpu().numpy().item()
                val_std = lam.std().detach().cpu().numpy().item()
                self.loss['params'][p + ':mean'].append(val_mean)
                self.loss['params'][p + ':std'].append(val_std)
            elif p != 'y':
                if p + ':mean' not in self.loss['params'].keys():
                    self.loss['params'][p + ':mean'] = []
                    self.loss['params'][p + ':std'] = []
                vals = trace.nodes[p]['value']
                if cells_only:
                    vals = vals[cells]
                val_mean = vals.mean().detach().cpu().numpy().item()
                val_std = vals.std().detach().cpu().numpy().item()
                self.loss['params'][p + ':mean'].append(val_mean)
                self.loss['params'][p + ':std'].append(val_std)
            else:
                if 'y:sum' not in self.loss['params'].keys():
                    self.loss['params']['y:sum'] = []
                self.loss['params'][p + ':sum'].append(trace.nodes[p]['value'].sum().detach().cpu().numpy().item())


def get_ambient_expression() -> Union[np.ndarray, None]:
    """Get ambient RNA expression for 'empty' droplets.

    Return:
        chi_ambient: The ambient gene expression profile, as a normalized
            vector that sums to one.

    Note:
        Inference must have been performed on a model with a 'chi_ambient'
        hyperparameter prior to making this call.

    """

    chi_ambient = None

    if 'chi_ambient' in pyro.get_param_store().keys():
        chi_ambient = to_ndarray(pyro.param('chi_ambient')).squeeze()

    return chi_ambient


def get_rho() -> Union[np.ndarray, None]:
    """Get ambient RNA expression for 'empty' droplets.

    Return:
        chi_ambient: The ambient gene expression profile, as a normalized
            vector that sums to one.

    Note:
        Inference must have been performed on a model with a 'chi_ambient'
        hyperparameter prior to making this call.

    """

    rho = None

    if 'rho_alpha' in pyro.get_param_store().keys() \
            and 'rho_beta' in pyro.get_param_store().keys():
        rho = np.array([to_ndarray(pyro.param('rho_alpha')).item(),
                        to_ndarray(pyro.param('rho_beta')).item()])

    return rho


def to_ndarray(x: Union[Number, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a numeric value or array to a numpy array on cpu."""

    if type(x) is np.ndarray:
        return x

    elif type(x) is torch.Tensor:
        return x.detach().cpu().numpy()

    elif type(x) is Number:
        return np.array(x)

    else:
        raise TypeError(f'to_ndarray() received input of type {type(x)}')
