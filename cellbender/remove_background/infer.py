# Posterior inference.

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import torch
from torch.distributions import constraints
import numpy as np
import scipy.sparse as sp
from scipy.stats import mode as scipy_mode

from typing import Tuple, List, Dict
from abc import ABC, abstractmethod


class Posterior(ABC):
    """Base class Posterior handles posterior count inference.

    Args:
        dataset_obj: Dataset object.
        vi_model: VariationalInferenceModel
        counts_dtype: Data type of posterior count matrix.  Can be one of
            [np.uint32, np.float]
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32

    Properties:
        mean: Posterior count mean, as a sparse matrix.

    """

    def __init__(self,
                 dataset_obj,  # Dataset
                 vi_model,  # Trained variational inference model
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: float = 0.5):
        self.dataset_obj = dataset_obj
        self.vi_model = vi_model
        self.analyzed_gene_inds = dataset_obj.analyzed_gene_inds
        self.count_matrix_shape = dataset_obj.data['matrix'].shape
        self.barcode_inds = np.arange(0, self.count_matrix_shape[0])
        self.dtype = counts_dtype
        self.float_threshold = float_threshold
        self._mean = None
        super(Posterior, self).__init__()

    @abstractmethod
    def _get_mean(self):
        """Obtain mean posterior counts and store in self._mean"""
        pass

    @torch.no_grad()
    def _get_encodings(self):
        """Send dataset through a trained encoder."""

        assert "chi_ambient" in pyro.get_param_store().keys(), \
            "Attempting to encode the data before 'chi_ambient' has been " \
            "inferred.  Run inference first."

        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.vi_model.use_cuda,
                                                      analyzed_bcs_only=True,
                                                      batch_size=500,
                                                      shuffle=False)

        self._encodings = {'z': [], 'd': [], 'p': [], 'alpha0': []}

        for data in data_loader:
            # Get latent encodings. (z, d, p)
            enc = self.vi_model.encoder.forward(x=data,
                                                chi_ambient=pyro.param("chi_ambient"))

            self._encodings['z'].append(enc['z']['loc'].detach().cpu().numpy())
            self._encodings['d'].append(enc['d_loc']
                                        .detach().exp().cpu().numpy())
            self._encodings['p'].append(enc['p_y']
                                        .detach().sigmoid().cpu().numpy())
            self._encodings['alpha0'].append(enc['alpha0']['loc']
                                             .detach().exp().cpu().numpy())

        # Concatenate lists.
        for key, value_list in self._encodings.items():
            self._encodings[key] = np.concatenate(tuple(value_list), axis=0)

    @property
    def mean(self) -> sp.csc_matrix:
        if self._mean is None:
            self._get_mean()
        return self._mean

    @property
    def encodings(self) -> Dict[str, np.ndarray]:
        if self._encodings is None:
            self._get_encodings()
        return self._encodings

    @property
    def variance(self):
        raise NotImplemented("Posterior count variance not implemented.")

    def dense_to_sparse(self,
                        chunk_dense_counts: np.ndarray) -> Tuple[List, List, List]:
        """Distill a batch of dense counts into sparse format.
        Barcode numbering is relative to the tensor passed in.
        """

        if self.dtype == np.uint32:

            # Turn the floating point count estimates into integers.
            decimal_values, _ = np.modf(chunk_dense_counts)  # Stuff after decimal.
            roundoff_counts = np.random.binomial(1, p=decimal_values)  # Bernoulli.
            chunk_dense_counts = np.floor(chunk_dense_counts).astype(dtype=int)
            chunk_dense_counts += roundoff_counts

        elif self.dtype == np.float32:

            # Truncate counts at a threshold value.
            chunk_dense_counts = (chunk_dense_counts *
                                  (chunk_dense_counts > self.float_threshold))

        else:
            raise NotImplementedError(f"Count matrix dtype {self.dtype} is not "
                                      f"supported.  Choose from [np.uint32, "
                                      f"np.float32]")

        # Find all the nonzero counts in this dense matrix chunk.
        nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed = \
            np.nonzero(chunk_dense_counts)
        nonzero_counts = \
            chunk_dense_counts[nonzero_barcode_inds_this_chunk,
                               nonzero_genes_trimmed].flatten(order='C')

        # Get the original gene index from gene index in the trimmed dataset.
        nonzero_genes = self.analyzed_gene_inds[nonzero_genes_trimmed]

        return nonzero_barcode_inds_this_chunk, nonzero_genes, nonzero_counts


class ImputedPosterior(Posterior):
    """Posterior count inference using imputation to infer cell mean (d * chi).

    Args:
        dataset_obj: Dataset object.
        vi_model: VariationalInferenceModel
        guide: Variational posterior pyro guide function, optional.  Only
            specify if the required guide function is not vi_model.guide.
        encoder: Encoder that provides encodings of data.
        counts_dtype: Data type of posterior count matrix.  Can be one of
            [np.uint32, np.float]
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32

    Properties:
        mean: Posterior count mean, as a sparse matrix.
        encodings: Encoded latent variables, one per barcode in the dataset.

    """

    def __init__(self,
                 dataset_obj,  # Dataset
                 vi_model,  # Trained variational inference model
                 guide=None,
                 encoder=None,  #: Union[CompositeEncoder, None] = None,
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: float = 0.5):
        self.vi_model = vi_model
        self.use_cuda = vi_model.use_cuda
        self.guide = guide if guide is not None else vi_model.encoder
        self.encoder = encoder if encoder is not None else vi_model.encoder
        self._encodings = None
        self._mean = None
        super(ImputedPosterior, self).__init__(dataset_obj=dataset_obj,
                                               vi_model=vi_model,
                                               counts_dtype=counts_dtype,
                                               float_threshold=float_threshold)

    @torch.no_grad()
    def _get_mean(self):
        """Send dataset through a guide that returns mean posterior counts.

        Keep track of only what is necessary to distill a sparse count matrix.

        """

        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=False,
                                                      batch_size=500,
                                                      shuffle=False)

        barcodes = []
        genes = []
        counts = []
        ind = 0

        for data in data_loader:

            # Get return values from guide.
            dense_counts_torch = self.guide(data, observe=False)
            dense_counts = dense_counts_torch.detach().cpu().numpy()
            bcs_i_chunk, genes_i, counts_i = self.dense_to_sparse(dense_counts)

            # Translate chunk barcode inds to overall inds.
            bcs_i = self.barcode_inds[bcs_i_chunk + ind]

            # Add sparse matrix values to lists.
            barcodes.append(bcs_i)
            genes.append(genes_i)
            counts.append(counts_i)

            # Increment barcode index counter.
            ind += data.shape[0]  # Same as data_loader.batch_size

        # Convert the lists to numpy arrays.
        counts = np.array(np.concatenate(tuple(counts)), dtype=self.dtype)
        barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=self.dtype)
        genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint16)

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)


class ProbPosterior(Posterior):
    """Posterior count inference using a noise count probability distribution.

    Args:
        dataset_obj: Dataset object.
        vi_model: VariationalInferenceModel
        guide: Variational posterior pyro guide function, optional.  Only
            specify if the required guide function is not vi_model.guide.
        encoder: Encoder that provides encodings of data.
        counts_dtype: Data type of posterior count matrix.  Can be one of
            [np.uint32, np.float]
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32

    Properties:
        mean: Posterior count mean, as a sparse matrix.
        encodings: Encoded latent variables, one per barcode in the dataset.

    """

    def __init__(self,
                 dataset_obj,  # Dataset
                 vi_model,  # Trained variational inference model
                 guide=None,
                 encoder=None,  #: Union[CompositeEncoder, None] = None,
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: float = 0.5):
        self.vi_model = vi_model
        self.use_cuda = vi_model.use_cuda
        self.guide = guide if guide is not None else vi_model.encoder
        self.encoder = encoder if encoder is not None else vi_model.encoder
        self._encodings = None
        self._mean = None
        super(ProbPosterior, self).__init__(dataset_obj=dataset_obj,
                                            vi_model=vi_model,
                                            counts_dtype=counts_dtype,
                                            float_threshold=float_threshold)

    @torch.no_grad()
    def _get_mean(self):
        """Send dataset through a guide that returns mean posterior counts.

        Keep track of only what is necessary to distill a sparse count matrix.

        """

        analyzed_bcs_only = True

        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=
                                                      analyzed_bcs_only,
                                                      batch_size=100,
                                                      shuffle=False)

        chi_ambient = pyro.param("chi_ambient")

        barcodes = []
        genes = []
        counts = []
        ind = 0

        for data in data_loader:

            # Compute an estimate of the true counts.
            dense_counts = self._compute_true_counts(data,
                                                     chi_ambient,
                                                     use_map=False,
                                                     n_samples=21)
            bcs_i_chunk, genes_i, counts_i = self.dense_to_sparse(dense_counts)

            # Translate chunk barcode inds to overall inds.
            if analyzed_bcs_only:
                bcs_i = self.dataset_obj.analyzed_barcode_inds[bcs_i_chunk + ind]
            else:
                bcs_i = self.barcode_inds[bcs_i_chunk + ind]

            # Add sparse matrix values to lists.
            barcodes.append(bcs_i)
            genes.append(genes_i)
            counts.append(counts_i)

            # Increment barcode index counter.
            ind += data.shape[0]  # Same as data_loader.batch_size

        # Convert the lists to numpy arrays.
        counts = np.array(np.concatenate(tuple(counts)), dtype=self.dtype)
        barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=self.dtype)
        genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint16)

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)

    @torch.no_grad()
    def _compute_true_counts(self,
                             data: torch.Tensor,
                             chi_ambient: torch.Tensor,
                             use_map: bool = True,
                             n_samples: int = 1) -> np.ndarray:
        """Compute the true de-noised count matrix for this minibatch.

        Can use either a MAP estimate of lambda and mu, or can use a sampling
        approach.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            chi_ambient: Point estimate of inferred ambient gene expression.
            use_map: True to use a MAP estimate of lambda and mu.
            n_samples: If not using a MAP estimate, this specifies the number
                of samples to use in calculating the posterior mean.

        Returns:
            dense_counts: Dense matrix of true de-noised counts.

        """

        if use_map:

            # Calculate MAP estimates of mu and lambda.
            mu_map, lambda_map, alpha_map = \
                self._param_map_estimates(data, chi_ambient)

            # Compute the de-noised count matrix given the MAP estimates.
            dense_counts_torch = self._true_counts_from_params(data,
                                                               mu_map,
                                                               lambda_map,
                                                               alpha_map)

            dense_counts = dense_counts_torch.detach().cpu().numpy()

        else:

            assert n_samples > 0, f"Posterior mean estimate needs to be derived " \
                                  f"from at least one sample: here {n_samples} " \
                                  f"samples are called for."

            dense_counts_torch = torch.zeros((data.shape[0],
                                              data.shape[1],
                                              n_samples), dtype=torch.float32)

            for i in range(n_samples):

                # Sample from mu and lambda.
                mu_sample, lambda_sample, alpha_sample = \
                    self._param_sample(data, chi_ambient)

                # Compute the de-noised count matrix given the MAP estimates.
                dense_counts_torch[..., i] = \
                    self._true_counts_from_params(data,
                                                  mu_sample,
                                                  lambda_sample,
                                                  alpha_sample)

            # Take the mode of the posterior true count distribution.
            dense_counts = dense_counts_torch.detach().cpu().numpy()
            dense_counts = scipy_mode(dense_counts, axis=2)[0].squeeze()

        return dense_counts

    @torch.no_grad()
    def _param_map_estimates(self,
                             data: torch.Tensor,
                             chi_ambient: torch.Tensor) -> Tuple[torch.Tensor,
                                                                 torch.Tensor,
                                                                 torch.Tensor]:
        """Calculate MAP estimates of mu, the mean of the true count matrix, and
        lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            chi_ambient: Point estimate of inferred ambient gene expression.

        Returns:
            mu_map: Dense tensor of Negative Binomial means for true counts.
            lambda_map: Dense tensor of Poisson rate params for noise counts.
            alpha_map: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.

        """

        # Encode latents.
        enc = self.vi_model.encoder.forward(x=data,
                                            chi_ambient=chi_ambient)
        z_map = enc['z']['loc']

        chi_map = self.vi_model.decoder.forward(z_map)
        alpha0_map = enc['alpha0']['loc']
        alpha_map = chi_map * alpha0_map.unsqueeze(-1)

        y = (enc['p_y'] > 0).float()
        d_empty = dist.LogNormal(loc=pyro.param('d_empty_loc'),
                                 scale=pyro.param('d_empty_scale')).mean
        d_cell = dist.LogNormal(loc=enc['d_loc'],
                                scale=pyro.param('d_cell_scale')).mean

        epsilon = torch.ones_like(y).clone()  # TODO

        # Calculate MAP estimates of mu and lambda.
        mu_map = (epsilon.unsqueeze(-1) * y.unsqueeze(-1)
                  * d_cell.unsqueeze(-1) * chi_map) + 1e-10
        lambda_map = (epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1)
                      * chi_ambient) + 1e-10

        return mu_map, lambda_map, alpha_map

    @torch.no_grad()
    def _param_sample(self,
                      data: torch.Tensor,
                      chi_ambient: torch.Tensor) -> Tuple[torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor]:
        """Calculate a single sample estimate of mu, the mean of the true count
        matrix, and lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            chi_ambient: Point estimate of inferred ambient gene expression.

        Returns:
            mu_sample: Dense tensor sample of Negative Binomial mean for true
                counts.
            lambda_sample: Dense tensor sample of Poisson rate params for noise
                counts.
            alpha_sample: Dense tensor sample of Dirichlet concentration params
                that inform the overdispersion of the Negative Binomial.

        """

        # Encode latents.
        enc = self.vi_model.encoder.forward(x=data,
                                            chi_ambient=chi_ambient)
        z = dist.Normal(loc=enc['z']['loc'], scale=enc['z']['scale']).sample()

        chi_sample = self.vi_model.decoder.forward(z)
        alpha0_sample = dist.LogNormal(loc=enc['alpha0']['loc'],
                                       scale=0.1).sample()
        alpha_sample = chi_sample * alpha0_sample.unsqueeze(-1)

        y = dist.Bernoulli(logits=enc['p_y']).sample()
        d_empty = dist.LogNormal(loc=pyro.param('d_empty_loc'),
                                 scale=pyro.param('d_empty_scale')).sample()
        d_cell = dist.LogNormal(loc=enc['d_loc'],
                                scale=pyro.param('d_cell_scale')).sample()
        # epsilon = dist.Gamma(enc['epsilon_loc'] * self.vi_model.epsilon_param,
        #                      self.vi_model.epsilon_param).sample()
        epsilon = torch.ones_like(y).clone()  # TODO

        # Calculate MAP estimates of mu and lambda.
        mu_sample = (epsilon.unsqueeze(-1) * y.unsqueeze(-1)
                     * d_cell.unsqueeze(-1) * chi_sample) + 1e-10
        lambda_sample = (epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1)
                         * chi_ambient) + 1e-10

        return mu_sample, lambda_sample, alpha_sample

    @torch.no_grad()
    def _true_counts_from_params(self,
                                 data: torch.Tensor,
                                 mu_est: torch.Tensor,
                                 lambda_est: torch.Tensor,
                                 alpha_est: torch.Tensor) -> torch.Tensor:
        """Calculate a single sample estimate of mu, the mean of the true count
        matrix, and lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.

        Returns:
            dense_counts_torch: Dense matrix of true de-noised counts.

        """

        # Estimate the max number of noise counts for any entry in the batch.
        max_noise_counts = int(5. * lambda_est.max())
        max_noise_counts = min(max_noise_counts, 100)  # Memory limitations

        # Construct a big tensor of possible noise counts per cell per gene,
        # shape (batch_cells, n_genes, max_noise_counts)
        noise_count_tensor = torch.arange(start=0,
                                          end=max_noise_counts + 1) \
                                 .expand([data.shape[0], data.shape[1], -1]) \
                                 .float().to(device=self.vi_model.device) + 1e-10

        # Compute probabilities of each number of noise counts.
        prob_tensor = (dist.Poisson(lambda_est.unsqueeze(-1))
                       .log_prob(noise_count_tensor)
                       + dist.NegativeBinomial(total_count=alpha_est.unsqueeze(-1),
                                               logits=(mu_est / alpha_est)
                                               .log().unsqueeze(-1))
                       .log_prob(data.unsqueeze(-1) - noise_count_tensor))

        # Find the most probable number of noise counts per cell per gene.
        noise_count_map = torch.argmax(prob_tensor,
                                       dim=-1,
                                       keepdim=False).float()

        # Compute the number of true counts.
        dense_counts_torch = torch.clamp(data - noise_count_map, min=0.)

        return dense_counts_torch
