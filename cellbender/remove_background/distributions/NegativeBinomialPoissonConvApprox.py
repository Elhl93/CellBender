import pyro.distributions as dist

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import \
    broadcast_all, probs_to_logits, lazy_property, logits_to_probs

from numbers import Number


class TorchNegativeBinomialPoissonConvApprox(Distribution):
    r"""
    Creates a distribution for a random variable Z = X + Y such that:

        X ~ NegativeBinomial(:attr:`mu`, :attr:`alpha`)
        Y ~ Poisson(:attr:`lam`)

    where :attr:`mu` is the mean of X, `alpha` is the inverse of the overdispersion of X, and
    :attr:`lam` is the rate of Y.

    Args:
        mu (Number, Tensor): mean of the negative binomial variable
        alpha (Number, Tensor): inverse overdispersion of the negative binomial variable
        lam (Number, Tensor): rate of the Poisson variable
    """
    arg_constraints = {'mu': constraints.positive,
                       'alpha': constraints.positive,
                       'lam': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, mu, alpha, lam, validate_args=None):
        self.mu, self.alpha, self.lam = broadcast_all(mu, alpha, lam)
        if isinstance(mu, Number) and isinstance(alpha, Number) \
                and isinstance(lam, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(TorchNegativeBinomialPoissonConvApprox,
              self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TorchNegativeBinomialPoissonConvApprox,
                                         _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.lam = self.lam.expand(batch_shape)

        super(TorchNegativeBinomialPoissonConvApprox,
              new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @staticmethod
    def _poisson_log_prob(lam, value):
        return (lam.log() * value) - lam - (value + 1).lgamma()

    @staticmethod
    def _neg_binom_log_prob(mu, alpha, value):
        return ((value + alpha).lgamma() - (value + 1).lgamma() - alpha.lgamma()
                + alpha * (alpha.log() - (alpha + mu).log())
                + value * (mu.log() - (alpha + mu).log()))

    @staticmethod
    def _poisson_log_prob_zero(lam):
        return - lam

    @staticmethod
    def _neg_binom_log_prob_zero(mu, alpha):
        return alpha * (alpha.log() - (alpha + mu).log())

    @staticmethod
    def _poisson_log_prob_one(lam):
        return lam.log() - lam

    @staticmethod
    def _neg_binom_log_prob_one(mu, alpha):
        # log_gamma(2.) = 0
        # log_gamma(z + 1) - log_gamma(z) = ln(z)
        return (alpha + 1) * (alpha.log() - (alpha + mu).log()) + mu.log()

    @staticmethod
    def _poisson_log_prob_two(lam):
        return 2. * lam.log() - lam - 0.69314718  # ln(2)

    @staticmethod
    def _neg_binom_log_prob_two(mu, alpha):
        # log_gamma(3.) = 0.6931 = ln(2.)
        return ((2. + alpha).lgamma() - 0.69314718 - alpha.lgamma()
                + alpha * (alpha.log() - (alpha + mu).log())
                + 2. * (mu.log() - (alpha + mu).log()))

    def log_prob(self, value):
        """Empirically, it seems that a moment-matched negative binomial is a
        sufficient approximation except for when value = 1 or 2.
        In those cases, we can do a brute-force computation, and the result is
        quite close to the full brute-force computation.
        """
        if self._validate_args:
            self._validate_sample(value)
        mu, alpha, lam, value = broadcast_all(self.mu, self.alpha, self.lam, value)

        # moment-matched negative binomial approximation
        mean_hat = mu + lam
        alpha_hat = mean_hat.pow(2) * alpha * mu.pow(-2)
        nb_approx_log_prob = self._neg_binom_log_prob(mu=mean_hat,
                                                      alpha=alpha_hat,
                                                      value=value)

        # # (brute-force) analytical solution for counts = 1
        # one_indices = (value == 1)
        # poisson_log_prob_zero = self._poisson_log_prob_zero(lam[one_indices])
        # neg_binom_log_prob_zero = self._neg_binom_log_prob_zero(mu[one_indices],
        #                                                         alpha[one_indices])
        # poisson_log_prob_one = self._poisson_log_prob_one(lam[one_indices])
        # neg_binom_log_prob_one = self._neg_binom_log_prob_one(mu[one_indices],
        #                                                       alpha[one_indices])
        # total_log_prob_one = torch.logsumexp(
        #     torch.cat(((poisson_log_prob_zero + neg_binom_log_prob_one).unsqueeze(-1),
        #                (poisson_log_prob_one + neg_binom_log_prob_zero).unsqueeze(-1)), dim=-1),
        #     dim=-1, keepdim=False)
        #
        # # (brute-force) analytical solution for counts = 2
        # two_indices = (value == 2)
        # poisson_log_prob_zero = self._poisson_log_prob_zero(lam[two_indices])
        # neg_binom_log_prob_zero = self._neg_binom_log_prob_zero(mu[two_indices],
        #                                                         alpha[two_indices])
        # poisson_log_prob_one = self._poisson_log_prob_one(lam[two_indices])
        # neg_binom_log_prob_one = self._neg_binom_log_prob_one(mu[two_indices],
        #                                                       alpha[two_indices])
        # poisson_log_prob_two = self._poisson_log_prob_two(lam[two_indices])
        # neg_binom_log_prob_two = self._neg_binom_log_prob_two(mu[two_indices],
        #                                                       alpha[two_indices])
        # total_log_prob_two = torch.logsumexp(
        #     torch.cat(((poisson_log_prob_zero + neg_binom_log_prob_two).unsqueeze(-1),
        #                (poisson_log_prob_one + neg_binom_log_prob_one).unsqueeze(-1),
        #                (poisson_log_prob_two + neg_binom_log_prob_zero).unsqueeze(-1)), dim=-1),
        #     dim=-1, keepdim=False)
        #
        # # replace brute-forced values
        # log_prob = nb_approx_log_prob
        # log_prob[one_indices] = total_log_prob_one
        # log_prob[two_indices] = total_log_prob_two

        # poisson for small mu
        empty_indices = (mu < 1e-5)
        poisson_log_prob = self._poisson_log_prob(lam=lam[empty_indices],
                                                  value=value[empty_indices])

        # replace
        log_prob = nb_approx_log_prob
        log_prob[empty_indices] = poisson_log_prob

        return log_prob


# We wrap the Torch distribution inside a Pyro distribution.
# This is as simple as inheriting
# distributions.torch_distribution.TorchDistributionMixin.
# It adds the required extra attributes.
class NegativeBinomialPoissonConvApprox(TorchNegativeBinomialPoissonConvApprox,
                                        dist.torch_distribution
                                        .TorchDistributionMixin):
    pass
