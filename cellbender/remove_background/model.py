"""Definition of the model and the inference setup, with helper functions."""

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
import pyro.poutine as poutine

from cellbender.remove_background.vae import encoder as encoder_module
import cellbender.remove_background.consts as consts
from cellbender.remove_background.distributions.NegativeBinomialPoissonSumSparse \
    import NegativeBinomialPoissonSumSparse as NBPSS
from cellbender.remove_background.distributions.NullDist import NullDist

from typing import Union, Tuple
from numbers import Number
import logging


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
                 use_cuda: bool = False):
        super(RemoveBackgroundPyroModel, self).__init__()

        self.model_type = model_type
        self.include_empties = True
        if self.model_type == "simple":
            self.include_empties = False

        self.n_genes = dataset_obj.analyzed_gene_inds.size
        self.z_dim = decoder.input_dim
        self.encoder = encoder
        self.decoder = decoder
        self.loss = {'train': {'epoch': [], 'elbo': []},
                     'test': {'epoch': [], 'elbo': []}}

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
        self.d_cell_scale_prior = (np.array(dataset_obj.priors['d_std'],
                                            dtype=np.float32).item()
                                   * torch.ones(torch.Size([])).to(self.device))
        self.z_loc_prior = torch.zeros(torch.Size([self.z_dim])).to(self.device)
        self.z_scale_prior = torch.ones(torch.Size([self.z_dim]))\
            .to(self.device)
        self.epsilon_prior = torch.tensor(1000.).to(self.device)
        self.alpha0_loc_prior = (torch.tensor(consts.ALPHA0_PRIOR_LOC)
                                 .float().to(self.device))
        self.alpha0_scale_prior = (torch.tensor(consts.ALPHA0_PRIOR_SCALE)
                                   .float().to(self.device))

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

            self.d_empty_loc_prior = (np.log1p(dataset_obj
                                               .priors['empty_counts'],
                                               dtype=np.float32).item()
                                      * torch.ones(torch.Size([]))
                                      .to(self.device))
            self.d_empty_scale_prior = (np.array(dataset_obj.priors['d_std'],
                                                 dtype=np.float32).item()
                                        * torch.ones(torch.Size([]))
                                        .to(self.device))

            self.p_logit_prior = (dataset_obj.priors['cell_logit']
                                  * torch.ones(torch.Size([])).to(self.device))

            self.chi_ambient_init = dataset_obj.priors['chi_ambient']\
                .to(self.device)

            self.empty_UMI_threshold = (torch.tensor(dataset_obj.empty_UMI_threshold)
                                        .float().to(self.device))

        else:

            self.avg_gene_expression = None

    def _calculate_mu(self,
                      chi: torch.Tensor,
                      d_cell: torch.Tensor,
                      epsilon: torch.Tensor,
                      y: Union[torch.Tensor, None] = None):
        """Implement a calculation of mean expression based on the model."""

        if self.model_type == "simple":
            """The model is that a latent variable z is drawn from a z_dim 
            dimensional normal distribution.  This latent z is put through the 
            decoder to generate a full vector of fractional gene expression, 
            chi.  Counts are then drawn from a negative binomial distribution 
            with mean d * chi.  d is drawn from a LogNormal distribution with 
            the specified prior.  Phi is the overdispersion of this negative 
            binomial, and is drawn from a Gamma distribution with the specified 
            prior.

            """
            mu = epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1) * chi

        elif self.model_type == "ambient":
            """There is a global hyperparameter called chi_ambient.  This 
            parameter
            is the learned fractional gene expression vector for ambient RNA.
            The model is that a latent variable z is drawn from a z_dim 
            dimensional normal distribution.  This latent z is put through the 
            decoder to generate a full vector of fractional gene expression, 
            chi.  Counts are then drawn from a negative binomial distribution 
            with mean d_cell * chi + d_ambient * chi_ambient.  d_cell is drawn 
            from a LogNormal distribution with the specified prior, as is 
            d_ambient.  Phi is the overdispersion of this negative binomial, 
            and is drawn from a Gamma distribution with the specified prior.

            """
            mu = (y.unsqueeze(-1) * epsilon.unsqueeze(-1)
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

        # Happens in parallel for each data point (cell barcode) independently:
        with pyro.plate("data", x.size(0),
                        use_cuda=self.use_cuda, device=self.device):

            # Sample z from prior.
            z = pyro.sample("z",
                            dist.Normal(self.z_loc_prior,
                                        self.z_scale_prior)
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

            # Sample epsilon based on priors.
            epsilon = torch.ones_like(alpha0).clone()
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
                                dist.Bernoulli(logits=self.p_logit_prior)
                                .expand_by([x.size(0)]))
            else:
                d_empty = None
                y = None

            # Calculate the mean gene expression counts (for each barcode).
            mu_cell = self._calculate_mu(chi=chi,
                                         d_cell=d_cell,
                                         epsilon=epsilon,
                                         y=y)

            lam = (epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1)
                   * chi_ambient)

            # print(f'alpha range is [{alpha0.min():.0f}, '
            #       f'{alpha0.mean():.0f}, {alpha0.max():.0f}]')

            # Sample gene expression from our Negative Binomial Poisson Sum
            # distribution, and compare with observed data.
            c = pyro.sample("obs",
                            NBPSS(mu=mu_cell + 1e-10,
                                  alpha=alpha0.unsqueeze(-1) * chi,
                                  lam=lam + 1e-10,
                                  max_poisson=100).to_event(1),
                            obs=x.reshape(-1, self.n_genes))

            # Additionally use the surely empty droplets for regularization,
            # since we know these droplets by their UMI counts.
            if self.include_empties:

                counts = x.sum(dim=-1, keepdim=False)
                surely_empty_mask = ((counts < self.empty_UMI_threshold)
                                     .type(torch.BoolTensor).to(self.device))

                with poutine.mask(mask=surely_empty_mask):

                    # # Semi-supervision of ambient expression.
                    # pyro.sample("obs_empty",
                    #             dist.Poisson(rate=lam + 1e-10).to_event(1),
                    #             obs=x.reshape(-1, self.n_genes))

                    # Semi-supervision of cell probabilities.
                    p_logit_posterior = pyro.sample("p_passback",
                                                    NullDist(torch.zeros(1)
                                                             .to(self.device))
                                                    .expand_by([x.size(0)]))
                    pyro.sample("obs_empty_y",
                                dist.Bernoulli(logits=p_logit_posterior),
                                obs=torch.zeros_like(y))

    @config_enumerate(default='parallel')
    def guide(self, x: torch.Tensor):
        """Variational posterior.

            Args:
                x: Mini-batch of data. Barcodes are rows, genes are columns.

        """

        # Register the encoder(s) with pyro.
        for name, module in self.encoder.items():
            pyro.module("encoder_" + name, module)

        # Initialize variational parameters for d_cell.
        d_cell_scale = pyro.param("d_cell_scale",
                                  self.d_cell_scale_prior *
                                  torch.ones(torch.Size([])).to(self.device),
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

        # Happens in parallel for each data point (cell barcode) independently:
        with pyro.plate("data", x.size(0),
                        use_cuda=self.use_cuda, device=self.device):

            # TODO: epsilon inference

            # Encode the latent variables from the input gene expression counts.
            if self.include_empties:
                enc = self.encoder.forward(x=x, chi_ambient=chi_ambient)

            else:
                enc = self.encoder.forward(x=x, chi_ambient=None)

            # Code specific to models with empty droplets.
            if self.include_empties:

                # Sample d_empty, which doesn't depend on y.
                pyro.sample("d_empty",
                            dist.LogNormal(loc=d_empty_loc,
                                           scale=d_empty_scale)
                            .expand_by([x.size(0)]))

                # Pass back the inferred p_y to the model.
                pyro.sample("p_passback", NullDist(enc['p_y']))

                # Sample the Bernoulli y from encoded p(y).
                y = pyro.sample("y", dist.Bernoulli(logits=enc['p_y']))
                cell_mask = y.type(torch.BoolTensor).to(self.device)

                # Mask out empty droplets.
                with poutine.mask(mask=cell_mask):

                    # Sample latent code z for the barcodes containing cells.
                    pyro.sample("z",
                                dist.Normal(loc=enc['z']['loc'],
                                            scale=enc['z']['scale'])
                                .to_event(1))

                    # Sample alpha0 for the barcodes containing cells.
                    pyro.sample("alpha0",
                                dist.LogNormal(loc=enc['alpha0']['loc'],
                                               scale=0.1))
                    # pyro.sample("alpha0",
                    #             dist.LogNormal(loc=enc['alpha0']['loc'],
                    #                            scale=enc['alpha0']['scale']))

                # Gate d_cell_loc so empty droplets do not give big gradients.
                prob = enc['p_y'].sigmoid()  # Logits to probability
                d_cell_loc_gated = (prob * enc['d_loc'] + (1 - prob)
                                    * self.d_cell_loc_prior)

                # Sample d based on the encoding.
                pyro.sample("d_cell", dist.LogNormal(loc=d_cell_loc_gated,
                                                     scale=d_cell_scale))

            else:

                # Sample d based on the encoding.
                pyro.sample("d_cell", dist.LogNormal(log=enc['d_loc'],
                                                     scale=d_cell_scale))

                # Sample latent code z for each cell.
                pyro.sample("z",
                            dist.Normal(loc=enc['z']['loc'],
                                        scale=enc['z']['scale'])
                            .to_event(1))

                # Sample alpha0 for the barcodes containing cells.
                pyro.sample("alpha0",
                            dist.LogNormal(loc=enc['alpha0']['loc'],
                                           scale=enc['alpha0']['scale'])
                            .to_event(1))


def get_encodings(model: RemoveBackgroundPyroModel,
                  dataset_obj: 'SingleCellRNACountsDataset',
                  cells_only: bool = True) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """Get inferred quantities from a trained model.

    Run a dataset through the model's trained encoder and return the inferred
    quantities.

    Args:
        model: A trained cellbender.model.VariationalInferenceModel, which will
            be used to generate the encodings from data.
        dataset_obj: The dataset to be encoded.
        cells_only: If True, only returns the encodings of barcodes that are
            determined to contain cells.

    Returns:
        z: Latent variable embedding of gene expression in a low-dimensional
            space.
        d: Latent variable scale factor for the number of UMI counts coming
            from each real cell.  Not in log space, but actual size.  This is
            not just the encoded d, but the mean of the LogNormal distribution,
            which is exp(mean + sigma^2 / 2).
        p: Latent variable denoting probability that each barcode contains a
            real cell.

    """

    logging.info("Encoding data according to model.")

    # Get the count matrix with genes trimmed.
    if cells_only:
        dataset = dataset_obj.get_count_matrix()
    else:
        dataset = dataset_obj.get_count_matrix_all_barcodes()

    # Initialize numpy arrays as placeholders.
    z = np.zeros((dataset.shape[0], model.z_dim))
    d = np.zeros((dataset.shape[0]))
    p = np.zeros((dataset.shape[0]))

    # Get chi ambient, if it was part of the model.
    chi_ambient = get_ambient_expression_from_pyro_param_store()
    if chi_ambient is not None:
        chi_ambient = torch.Tensor(chi_ambient).to(device=model.device)

    # Send dataset through the learned encoder in chunks.
    s = 200
    for i in np.arange(0, dataset.shape[0], s):

        # Put chunk of data into a torch.Tensor.
        x = torch.Tensor(np.array(
            dataset[i:min(dataset.shape[0], i + s), :].todense(),
            dtype=int).squeeze()).to(device=model.device)

        # Send data chunk through encoder.
        enc = model.encoder.forward(x=x, chi_ambient=chi_ambient)

        # Get d_cell_scale from fit model.
        d_sig = to_ndarray(pyro.get_param_store().get_param('d_cell_scale'))

        # Put the resulting encodings into the appropriate numpy arrays.
        z[i:min(dataset.shape[0], i + s), :] = to_ndarray(enc['z']['loc'])
        d[i:min(dataset.shape[0], i + s)] = (np.exp(to_ndarray(enc['d_loc']))
                                             + d_sig.item()**2 / 2)
        try:  # p is not always available: it depends which model was used.
            p[i:min(dataset.shape[0], i + s)] = to_ndarray(enc['p_y'].sigmoid())
        except KeyError:
            p = None  # Simple model gets None for p.

    return z, d, p


def generate_maximum_a_posteriori_count_matrix(
        z: np.ndarray,
        d: np.ndarray,
        p: Union[np.ndarray, None],
        model: RemoveBackgroundPyroModel,
        dataset_obj: 'SingleCellRNACountsDataset',
        cells_only: bool = True,
        chunk_size: int = 200) -> sp.csc.csc_matrix:
    """Make a point estimate of ambient-background-subtracted UMI count matrix.

    Sample counts by maximizing the model posterior based on learned latent
    variables.  The output matrix is in sparse form.

    Args:
        z: Latent variable embedding of gene expression in a low-dimensional
            space.
        d: Latent variable scale factor for the number of UMI counts coming
            from each real cell.
        p: Latent variable denoting probability that each barcode contains a
            real cell.
        model: Model with latent variables already inferred.
        dataset_obj: Input dataset.
        cells_only: If True, only returns the encodings of barcodes that are
            determined to contain cells.
        chunk_size: Size of mini-batch of data to send through encoder at once.

    Returns:
        inferred_count_matrix: Matrix of the same dimensions as the input
            matrix, but where the UMI counts have had ambient-background
            subtracted.

    Note:
        This currently uses the MAP estimate of draws from a Poisson (or a
        negative binomial with zero overdispersion).

    """

    # If simple model was used, then p = None.  Here set it to 1.
    if p is None:
        p = np.ones_like(d)

    # Get the count matrix with genes trimmed.
    if cells_only:
        count_matrix = dataset_obj.get_count_matrix()
    else:
        count_matrix = dataset_obj.get_count_matrix_all_barcodes()

    logging.info("Getting ambient-background-subtracted UMI count matrix.")

    # Ensure there are no nans in p (there shouldn't be).
    p_no_nans = p
    p_no_nans[np.isnan(p)] = 0  # Just make sure there are no nans.

    # Trim everything down to the barcodes we are interested in (just cells?).
    if cells_only:
        d = d[p_no_nans > consts.CELL_PROB_CUTOFF]
        z = z[p_no_nans > consts.CELL_PROB_CUTOFF, :]
        barcode_inds = \
            dataset_obj.analyzed_barcode_inds[p_no_nans
                                              > consts.CELL_PROB_CUTOFF]
    else:
        # Set cell size factors equal to zero where cell probability < 0.5.
        d[p_no_nans < consts.CELL_PROB_CUTOFF] = 0.
        z[p_no_nans < consts.CELL_PROB_CUTOFF, :] = 0.
        barcode_inds = np.arange(0, count_matrix.shape[0])  # All barcodes

    # Get the gene expression vectors by sending latent z through the decoder.
    # Send dataset through the learned encoder in chunks.
    barcodes = []
    genes = []
    counts = []
    s = chunk_size
    for i in np.arange(0, barcode_inds.size, s):

        # TODO: for 117000 cells, this routine overflows (~15GB) memory

        last_ind_this_chunk = min(count_matrix.shape[0], i+s)

        # Decode gene expression for a chunk of barcodes.
        decoded = model.decoder(torch.Tensor(
            z[i:last_ind_this_chunk]).to(device=model.device))
        chi = to_ndarray(decoded)

        # Estimate counts for the chunk of barcodes as d * chi.
        chunk_dense_counts = \
            np.maximum(0,
                       np.expand_dims(d[i:last_ind_this_chunk], axis=1) * chi)

        # Turn the floating point count estimates into integers.
        decimal_values, _ = np.modf(chunk_dense_counts)  # Stuff after decimal.
        roundoff_counts = np.random.binomial(1, p=decimal_values)  # Bernoulli.
        chunk_dense_counts = np.floor(chunk_dense_counts).astype(dtype=int)
        chunk_dense_counts += roundoff_counts

        # Find all the nonzero counts in this dense matrix chunk.
        nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed = \
            np.nonzero(chunk_dense_counts)
        nonzero_counts = \
            chunk_dense_counts[nonzero_barcode_inds_this_chunk,
                               nonzero_genes_trimmed].flatten(order='C')

        # Get the original gene index from gene index in the trimmed dataset.
        nonzero_genes = dataset_obj.analyzed_gene_inds[nonzero_genes_trimmed]

        # Get the actual barcode values.
        nonzero_barcode_inds = nonzero_barcode_inds_this_chunk + i
        nonzero_barcodes = barcode_inds[nonzero_barcode_inds]

        # Append these to their lists.
        barcodes.extend(nonzero_barcodes.astype(dtype=np.uint32))
        genes.extend(nonzero_genes.astype(dtype=np.uint16))
        counts.extend(nonzero_counts.astype(dtype=np.uint32))

    # Convert the lists to numpy arrays.
    counts = np.array(counts, dtype=np.uint32)
    barcodes = np.array(barcodes, dtype=np.uint32)
    genes = np.array(genes, dtype=np.uint16)

    # Put the counts into a sparse csc_matrix.
    inferred_count_matrix = sp.csc_matrix((counts, (barcodes, genes)),
                                          shape=dataset_obj.data['matrix']
                                          .shape)

    return inferred_count_matrix


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

    try:
        # Get fit hyperparameter for ambient gene expression from model.
        chi_ambient = to_ndarray(pyro.param("chi_ambient")).squeeze()
    except KeyError:
        pass

    return chi_ambient


def to_ndarray(x: Union[Number, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a numeric value or array to a numpy array on cpu."""

    if type(x) is Number:
        return np.array(x)

    elif type(x) is np.ndarray:
        return x

    elif type(x) is torch.Tensor:
        return x.detach().cpu().numpy()
