"""Interface for implementations."""

import abc
from dataclasses import dataclass

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Information(abc.ABC):
    """Interface for information operators."""

    def __init__(self, f, /, *, ode_order):
        self.f = f
        self.ode_order = ode_order

    def tree_flatten(self):
        children = ()
        aux = self.f, self.ode_order
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        f, ode_order = aux
        return cls(f, ode_order=ode_order)

    @abc.abstractmethod
    def begin_correction(self, x, /, *, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, *, extrapolated, cache):
        raise NotImplementedError

    @abc.abstractmethod
    def evidence_sqrtm(self, *, observed):
        raise NotImplementedError


@dataclass(frozen=True)
class Implementation(abc.ABC):
    """Implementation interface."""

    @abc.abstractmethod
    def init_corrected(self, *, taylor_coefficients):
        raise NotImplementedError

    @abc.abstractmethod
    def init_output_scale_sqrtm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_error_estimate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, m0, /, *, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, *, linearisation_pt, l0, cache, output_scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(self, *, linearisation_pt, l0, cache, output_scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol(self, *, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def condense_backward_models(self, *, bw_init, bw_state):  # noqa: D102
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_transition(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_noise(self, *, rv_proto):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, *, rv, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_backwards(self, *, init, linop, noise):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_model(self, *, init, linop, noise):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_backwards(self, init, linop, noise, base_samples):
        raise NotImplementedError

    # todo: make the extract_*_from_* functions use this one?
    @abc.abstractmethod
    def extract_mean_from_marginals(self, mean):
        raise NotImplementedError