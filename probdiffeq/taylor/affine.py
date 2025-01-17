r"""Taylor-expand the solution of an initial value problem (IVP)."""

from typing import Callable

import jax
import jax.experimental.jet
import jax.experimental.ode


def affine_recursion(vf: Callable, initial_values: tuple[jax.Array, ...], /, num: int):
    """Evaluate the Taylor series of an affine differential equation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.

    """
    if num == 0:
        return initial_values

    fx, jvp_fn = jax.linearize(vf, *initial_values)

    tmp = fx
    fx_evaluations = [tmp := jvp_fn(tmp) for _ in range(num - 1)]
    return [*initial_values, fx, *fx_evaluations]
