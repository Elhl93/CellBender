import torch
import torch.nn as nn
import pyro

import cellbender.remove_background.consts as consts

from typing import Dict, List, Union
import warnings


class CompositeEncoder(dict):
    """A composite of several encoders to be run together on the same input.

    This represents an encoder that is a composite of several
    completely separate encoders.  The separate encoders are passed
    in as a dict, where keys are encoder names and values are encoder
    instances.  The output is another dict with the same keys, where values
    are the output tensors created by calling .forward(x) on those encoder
    instances.

    Attributes:
        module_dict: A dictionary of encoder modules.

    """

    def __init__(self, module_dict):
        super(CompositeEncoder, self).__init__(module_dict)
        self.module_dict = module_dict

    def forward(self, **kwargs) \
            -> Dict[str, torch.Tensor]:
        # For each module in the dict of the composite encoder, call forward().
        out = dict()
        for key, value in self.module_dict.items():
            out[key] = value.forward(**kwargs)

            # TODO: tidy up
            if key == 'other':
                for subkey, value in out[key].items():
                    out[subkey] = value
                del out[key]

        return out


class EncodeZ(nn.Module):
    """Encoder module transforms gene expression into latent representation.

    The number of input units is the total number of genes and the number of
    output units is the dimension of the latent space.  This encoder transforms
    a point in high-dimensional gene expression space to a point in a
    low-dimensional latent space, via a learned transformation involving
    fully-connected layers.  The output is two vectors: one vector of latent
    variable means, and one vector of latent variable standard deviations.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of dimensions of the latent space in which gene
            expression will be embedded.
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        loc_out: torch.nn.Linear fully-connected output layer for the location
            of each point in latent space.
        sig_out: torch.nn.Linear fully-connected output layer for the standard
            deviation of each point in latent space.  Must result in positive
            values.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].

    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 input_transform: str = None):
        super(EncodeZ, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = input_transform

        # Set up the linear transformations used in fully-connected layers.
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.loc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.sig_out = nn.Linear(hidden_dims[-1], output_dim)

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Define the forward computation to go from gene expression to latent
        # representation.

        # Transform input.
        x = x.reshape(-1, self.input_dim)
        x = transform_input(x, self.transform)

        # Compute the hidden layers.
        hidden = self.softplus(self.linears[0](x))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        # Compute the outputs: loc is any real number, scale must be positive.
        loc = self.loc_out(hidden)
        scale = torch.exp(self.sig_out(hidden))

        return {'loc': loc.squeeze(), 'scale': scale.squeeze()}


class EncodeAlpha0(nn.Module):
    """Encoder module transforms gene expression into latent Dirichlet precision.

    Input dimension is the total number of genes and the output dimension is
    one.  This encoder transforms a point in high-dimensional gene expression
    space to a single number, alpha0: the precision of a Dirichlet
    distribution, where  the Dirichlet concentration parameters are
    chi * alpha0.  The output is two numbers: one the mean, and one the
    standard deviation, both in log space, both positive.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.

    Attributes:
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        loc_out: torch.nn.Linear fully-connected output layer for the location
            of each point in latent space.
        sig_out: torch.nn.Linear fully-connected output layer for the standard
            deviation of each point in latent space.  Must result in positive
            values.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].

    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super(EncodeAlpha0, self).__init__()
        self.input_dim = input_dim

        # Set up the linear transformations used in fully-connected layers.
        self.linears = nn.ModuleList([nn.Linear(input_dim + 1, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.loc_out = nn.Linear(hidden_dims[-1], 1)
        self.sig_out = nn.Linear(hidden_dims[-1], 1)

        # Initialization of outputs.
        self.initial_loc = None

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

    def forward(self,
                x: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Define the forward computation to go from gene expression to latent
        # representation.

        # Shape the input.
        x = x.reshape(-1, self.input_dim)

        # Calculate the number of nonzero genes.
        nnz = (x > 0).sum(dim=-1, keepdim=True).float()

        # Compute the hidden layers.
        hidden = self.softplus(self.linears[0](torch.cat((nnz, x), dim=-1)))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        # Compute the outputs: loc is any real number, scale must be positive.
        loc = self.softplus(self.loc_out(hidden))
        scale = torch.exp(self.sig_out(hidden))

        # Initialize outputs to a given value.
        if self.initial_loc is None:
            self.initial_loc = loc.mean().item()
        loc = loc - self.initial_loc + consts.ALPHA0_PRIOR_LOC

        return {'loc': loc.squeeze(), 'scale': scale.squeeze()}


class EncodeD(nn.Module):
    """Encoder module that transforms gene expression into latent cell size.

    The number of input units is the total number of genes and the number of
    output units is the dimension of the latent space.  This encoder transforms
    a point in high-dimensional gene expression space to a latent cell size
    estimate, via a learned transformation involving fully-connected
    layers.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of dimensions of the size estimate (1).
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        param: The log of the number of counts where the transition from
            cells to empty droplets is expected to occur.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        output: torch.nn.Linear fully-connected output layer for the size
            of each input barcode.
        input_dim: Size of input gene expression.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].
        This encoder acts as if each barcode input contains a cell.  In reality,
        barcodes that do not contain a cell will not propagate gradients back
        to this encoder, due to the design of the rest of the model.

    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 input_transform: str = None, log_count_crossover: float = 7.):
        super(EncodeD, self).__init__()
        self.input_dim = input_dim
        self.transform = input_transform
        self.param = log_count_crossover

        # Set up the linear transformations used in fully-connected layers.
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Define the forward computation to go from gene expression to cell
        # probabilities.

        # Transform input and calculate log total UMI counts per barcode.
        x = x.reshape(-1, self.input_dim)
        log_sum = x.sum(dim=-1, keepdim=True).log1p()
        x = transform_input(x, self.transform)

        # Compute the hidden layers and the output.
        hidden = self.softplus(self.linears[0](x))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        return self.softplus(self.output(hidden)
                             + self.softplus(log_sum - self.param)
                             + self.param).squeeze()


class EncodeNonEmptyDropletLogitProb(nn.Module):
    """Encoder module that transforms gene expression into cell probabilities.

    The number of input units is the total number of genes and the number of
    output units is the dimension of the latent space.  This encoder transforms
    a point in high-dimensional gene expression space to a latent probability
    that a given barcode contains a real cell, via a learned transformation
    involving fully-connected layers.  This encoder uses both the gene
    expression counts as well as an estimate of the ambient RNA profile in
    order to output a cell probability.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of dimensions of the probability estimate (1).
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        param: The log of the number of counts where the transition from
            cells to empty droplets is expected to occur.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        output: torch.nn.Linear fully-connected output layer for the size
            of each input barcode.
        input_dim: Size of input gene expression.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].
        The output is in the form of a logit, so can be any real number.  The
        transformation from logit to probability is a sigmoid.

    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 log_count_crossover: float, input_transform: str = None):
        super(EncodeNonEmptyDropletLogitProb, self).__init__()
        self.input_dim = input_dim
        self.transform = input_transform
        self.param = log_count_crossover
        self.INITIAL_WEIGHT_FOR_LOG_COUNTS = 2.
        self.OUTPUT_SCALE = 1.
        self.log_count_crossover = log_count_crossover

        # Set up the linear transformations used in fully-connected layers.
        # Adjust initialization conditions to start with a reasonable output.
        self.linears = nn.ModuleList([nn.Linear(1 + 2*input_dim,
                                                hidden_dims[0])])
        with torch.no_grad():
            self.linears[-1].weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            # Initialize p so that it starts out based (almost) on UMI counts.
            with torch.no_grad():
                self.linears[-1].weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        # Initialize p to be a sigmoid function of UMI counts.
        with torch.no_grad():
            self.output.weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
            # self.output.bias.data.copy_(torch.tensor([self.INITIAL_OUTPUT_BIAS_FOR_LOG_COUNTS]))

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

        # Set up the initial bias.
        self.offset = None

    def forward(self,
                x: torch.Tensor,
                chi_ambient: torch.Tensor,
                **kwargs) -> torch.Tensor:
        # Define the forward computation to go from gene expression to cell
        # probabilities.  The log of the total UMI counts is concatenated with
        # the input gene expression and the estimate of the difference between
        # the ambient RNA profile and this barcode's gene expression to form
        # an augmented input.

        # Transform input and calculate log total UMI counts per barcode.
        x = x.reshape(-1, self.input_dim)
        log_sum = x.sum(dim=-1, keepdim=True).log1p()
        x = transform_input(x, self.transform)

        # Form a new input by concatenation.
        # Compute the hidden layers and the output.
        hidden = self.softplus(self.linears[0](torch.cat((log_sum,
                                                          x,
                                                          x - chi_ambient),
                                                         dim=-1)))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        out = self.output(hidden).squeeze(-1)

        if self.offset is None:

            # Expected number of empties.
            expected_empties = (log_sum < self.log_count_crossover).sum()
            sort_out = torch.argsort(out, descending=False)
            # self.offset = out[sort_out][expected_empties + 1]  # works
            # self.offset = (out[sort_out][:expected_empties].median().item()
            #                + out[sort_out][expected_empties:].median().item()) / 2  # works

            cells = (log_sum > self.log_count_crossover).squeeze()
            cell_median = out[cells].median().item()
            empty_median = out[~cells].median().item()
            self.offset = empty_median + (cell_median - empty_median) * 3 / 4

            print(f'Seems there are {expected_empties} empties out of {x.shape[0]}')
            print(f'The offset is {self.offset}')

        return (out - self.offset) * self.OUTPUT_SCALE


class EncodeNonZLatents(nn.Module):
    """Encoder module that transforms data into all latents except z.

    The number of input units is the total number of genes plus four
    hand-crafted features, and the number of output units is 5: these being
    latents logit_p, d, epsilon, alpha0, and alpha1.  This encoder transforms
    a point in high-dimensional gene expression space into latents.  This
    encoder uses both the gene expression counts as well as an estimate of the
    ambient RNA profile in order to compute latents.

    Args:
        n_genes: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        output: torch.nn.Linear fully-connected output layer for the size
            of each input barcode.
        n_genes: Size of input gene expression.

    Returns:
        output: Dict containing -
            logit_p: Logit probability that droplet contains a cell
            d: Cell size scale factor
            epsilon: Value near one that represents droplet RT efficiency
            alpha0: Dirichlet concentration parameter sum

    Notes:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].
        The output is in the form of a dict.  Ouput for cell probability is a
        logit, so can be any real number.  The transformation from logit to
        probability is a sigmoid.
        Several heuristics are used to try to encourage a good initialization.

    """

    def __init__(self,
                 n_genes: int,
                 hidden_dims: List[int],
                 log_count_crossover: float,  # prior on log counts of smallest cell
                 input_transform: str = None):
        super(EncodeNonZLatents, self).__init__()
        self.n_genes = n_genes
        self.transform = input_transform
        self.output_dim = 4

        # Values related to logit cell probability
        self.INITIAL_WEIGHT_FOR_LOG_COUNTS = 2.
        self.P_OUTPUT_SCALE = 1.
        self.log_count_crossover = log_count_crossover

        # Values related to epsilon
        self.EPS_OUTPUT_SCALE = 0.05

        # Set up the linear transformations used in fully-connected layers.
        # Adjust initialization conditions to start with a reasonable output.
        self.linears = nn.ModuleList([nn.Linear(3 + self.n_genes,
                                                hidden_dims[0])])
        with torch.no_grad():
            self.linears[-1].weight[0][0] = 1.  # self.INITIAL_WEIGHT_FOR_LOG_COUNTS  # TODO
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            # Initialize p so that it starts out based (almost) on UMI counts.
            with torch.no_grad():
                self.linears[-1].weight[0][0] = 1.  # self.INITIAL_WEIGHT_FOR_LOG_COUNTS  # TODO
        self.output = nn.Linear(hidden_dims[-1], self.output_dim)
        # Initialize p to be a sigmoid function of UMI counts.
        with torch.no_grad():
            self.output.weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
            # Prevent a negative weight from starting something inverted.
            self.output.weight[1][0] = torch.abs(self.output.weight[1][0])
            self.output.weight[2][0] = torch.abs(self.output.weight[2][0])
            self.output.weight[3][0] = torch.abs(self.output.weight[3][0])

        # Set up the non-linear activations.
        self.nonlin = nn.Softplus()
        self.softplus = nn.Softplus()

        # Set up the initial biases.
        self.offset = None

        # Set up the initial scaling for values of x.
        # self.x_scaling = 1.  # None  # TODO
        self.x_scaling = None

        # Set up initial values for overlap normalization.
        self.overlap_mean = None
        self.overlap_std = None

    def _poisson_log_prob(self, lam, value):
        return (lam.log() * value) - lam - (value + 1).lgamma()

    def forward(self,
                x: torch.Tensor,
                chi_ambient: torch.Tensor,
                **kwargs) -> torch.Tensor:
        # Define the forward computation to go from gene expression to cell
        # probabilities.  The log of the total UMI counts is concatenated with
        # the input gene expression and the estimate of the difference between
        # the ambient RNA profile and this barcode's gene expression to form
        # an augmented input.

        x = x.reshape(-1, self.n_genes)

        # Calculate log total UMI counts per barcode.
        counts = x.sum(dim=-1, keepdim=True)
        log_sum = counts.log1p()

        # Calculate the log of the number of nonzero genes.
        log_nnz = (x > 0).sum(dim=-1, keepdim=True).float().log1p()

        # Calculate a similarity between expression and ambient.
        # overlap = (x * chi_ambient).sum(dim=-1, keepdim=True) / (x * x).sum(dim=-1, keepdim=False)
        overlap = self._poisson_log_prob(lam=counts * chi_ambient.detach().unsqueeze(0),
                                         value=x).sum(dim=-1, keepdim=True)  # TODO: this overlap is good 2020/02/16
        if self.overlap_mean is None:
            self.overlap_mean = (overlap.max() + overlap.min()) / 2
            self.overlap_std = overlap.max() - overlap.min()
        overlap = (overlap - self.overlap_mean) / self.overlap_std * 5

        # counts = counts.squeeze()
        # print(f'overlap[counts > 1000].mean() is {overlap[counts > 1000].mean()}')
        # print(f'overlap[counts < 1000].mean() is {overlap[counts < 1000].mean()}')

        # Apply transformation to data.
        x = transform_input(x, self.transform)

        # TODO: is this helpful?
        if self.x_scaling is None:
            n_std_est = 10
            num = int(self.n_genes * 0.4)
            std_estimates = torch.zeros([n_std_est])
            for i in range(n_std_est):
                idx = torch.randperm(x.nelement())
                std_estimates[i] = x.view(-1)[idx][:num].std().item()
            robust_std = std_estimates.median().item()
            self.x_scaling = 1. / robust_std / 100.  # Get values on a level field
            print(f'std is {x.std().item()} and robust_std is {robust_std}')
            print(f'scaled x has mean {(x * self.x_scaling).mean()} and std {(x * self.x_scaling).std()}')

        # print(f'(x * self.x_scaling).mean() is {(x * self.x_scaling).mean()}')
        # print(f'(x * self.x_scaling).std() is {(x * self.x_scaling).std()}')

        # Calculate a similarity between expression and ambient.
        # overlap = (x * chi_ambient).sum(dim=-1, keepdim=True) * self.n_genes / 50.

        # print(f'overlap.mean() is {overlap.mean()}')

        # Form a new input by concatenation.
        # Compute the hidden layers and the output.
        x_in = torch.cat((log_sum,
                          log_nnz,
                          overlap,
                          x * self.x_scaling),
                         dim=-1)

        hidden = self.nonlin(self.linears[0](x_in))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.nonlin(self.linears[i](hidden))

        out = self.output(hidden).squeeze(-1)

        if self.offset is None:

            self.offset = dict()

            # Heuristic for initialization of logit_cell_probability.
            cells = (log_sum > self.log_count_crossover).squeeze()
            cell_median = out[cells, 0].median().item()
            empty_median = out[~cells, 0].median().item()
            self.offset['logit_p'] = empty_median + (cell_median - empty_median) * 9. / 10  # * 3. / 4

            # Heuristic for initialization of d.
            self.offset['d'] = out[cells, 1].median().item()

            # Heuristic for initialization of epsilon.
            self.offset['epsilon'] = out[cells, 2].mean().item()

            # Heuristic for initialization of alpha.
            self.offset['alpha0'] = out[cells, 3].mean().item()

            # print(f"cell log_sum.mean() is {log_sum[cells].mean()}")
            # print(f"~cell log_sum.mean() is {log_sum[~cells].mean()}")
            # print(f"cell log_nnz.mean() is {log_nnz[cells].mean()}")
            # print(f"~cell log_nnz.mean() is {log_nnz[~cells].mean()}")
            # print(f"cell overlap.mean() is {overlap[cells].mean()}")
            # print(f"~cell overlap.mean() is {overlap[~cells].mean()}")
            # print(f"x.mean() is {x.mean()}")
            # print(f"x.std() is {x.std()}")

        p_y_logit = ((out[:, 0] - self.offset['logit_p'])
                     * self.P_OUTPUT_SCALE).squeeze()

        # Feed z back in to the last layer of p_y_logit.  # TODO: try?
        # TODO: try clipping outputs to safe ranges (to prevent nans / overflow)

        return {'p_y': p_y_logit,
                'd_loc': self.softplus(out[:, 1] - self.offset['d']
                                       + self.softplus(log_sum.squeeze()
                                                       - self.log_count_crossover)
                                       + self.log_count_crossover).squeeze(),
                'epsilon': ((out[:, 2] - self.offset['epsilon']).squeeze()
                            * self.EPS_OUTPUT_SCALE + 1.),
                'alpha0': self.softplus((out[:, 3] - self.offset['alpha0']) + 7.).squeeze()}
        # NOTE: if alpha0 initialization is too small, there is not enough difference
        # between NB and Poisson, and y reverts to zero.


def transform_input(x: torch.Tensor, transform: str) -> Union[torch.Tensor,
                                                              None]:
    """Transform input to encoder.

    Args:
        x: Input torch.Tensor
        transform: Specifies which transformation to perform.  Must be one of
            ['log', 'normalize'].

    Returns:
        Transformed input as a torch.Tensor of the same type and shape as x.

    """

    if transform is None:
        return x

    elif transform == 'log':
        x = x.log1p()
        return x

    elif transform == 'normalize':
        x = x / x.sum(dim=-1, keepdim=True)
        return x

    else:
        warnings.warn("Specified an input transform that is not "
                      "supported.  Choose from 'log' or 'normalize'.")
        return None
