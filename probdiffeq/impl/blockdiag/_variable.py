import jax
import jax.numpy as jnp

from probdiffeq.impl import _variable
from probdiffeq.impl.blockdiag import _normal


class VariableBackend(_variable.VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + (rv.cholesky @ unit_sample[..., None])[..., 0]

    def to_multivariate_normal(self, rv):
        mean = jnp.reshape(rv.mean.T, (-1,), order="F")
        cov = jax.scipy.linalg.block_diag(*self._cov_dense(rv.cholesky))
        return (mean, cov)

    def _cov_dense(self, cholesky):
        if cholesky.ndim > 2:
            return jax.vmap(self._cov_dense)(cholesky)
        return cholesky @ cholesky.T
