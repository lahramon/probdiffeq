"""Calibration tools."""

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _calib


def output_scale():
    """Construct (a buffet of) isotropic calibration strategies."""
    mle = _output_scale_mle()
    dynamic = _output_scale_dynamic()
    free = _output_scale_free()

    return _calib.CalibrationBundle(mle=mle, dynamic=dynamic, free=free)


def _output_scale_mle():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def update(s, /):
        return s

    @jax.tree_util.Partial
    def extract(s, /):
        if s.ndim > 0:
            return s[-1] * jnp.ones_like(s)
        return s

    return _calib.Calib(init=init, update=update, extract=extract)


def _output_scale_dynamic():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def update(s, /):  # todo: make correct
        return s

    @jax.tree_util.Partial
    def extract(s):
        return s

    return _calib.Calib(init=init, update=update, extract=extract)


def _output_scale_free():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def update(s, /):
        return s

    @jax.tree_util.Partial
    def extract(s, /):
        return s

    return _calib.Calib(init=init, update=update, extract=extract)