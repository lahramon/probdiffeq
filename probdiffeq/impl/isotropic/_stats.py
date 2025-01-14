import jax
import jax.numpy as jnp

from probdiffeq.impl import _stats
from probdiffeq.impl.isotropic import _normal


class StatsBackend(_stats.StatsBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def mahalanobis_norm_relative(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        residual_white_matrix = jnp.linalg.qr(residual_white.T, mode="r")
        return jnp.reshape(jnp.abs(residual_white_matrix) / jnp.sqrt(rv.mean.size), ())

    def logpdf(self, u, /, rv):
        # if the gain is qoi-to-hidden, the data is a (d,) array.
        # this is problematic for the isotropic model unless we explicitly broadcast.
        if jnp.ndim(u) == 1:
            u = u[None, :]

        def logpdf_scalar(x, r):
            dx = x - r.mean
            w = jax.scipy.linalg.solve_triangular(r.cholesky.T, dx, trans="T")

            maha_term = jnp.dot(w, w)

            diagonal = jnp.diagonal(r.cholesky, axis1=-1, axis2=-2)
            slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))
            logdet_term = 2.0 * slogdet
            return -0.5 * (logdet_term + maha_term + x.size * jnp.log(jnp.pi * 2))

        # Batch in the "mean" dimension and sum the results.
        rv_batch = _normal.Normal(1, None)
        return jnp.sum(jax.vmap(logpdf_scalar, in_axes=(1, rv_batch))(u, rv))

    def mean(self, rv):
        return rv.mean

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return jax.vmap(self.standard_deviation)(rv)
        return jnp.sqrt(jnp.dot(rv.cholesky, rv.cholesky))

    def sample_shape(self, rv):
        return rv.mean.shape
