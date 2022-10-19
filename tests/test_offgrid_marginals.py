"""Tests for IVP solvers."""
import jax.numpy as jnp
from pytest_cases import parametrize_with_cases

from odefilter import controls, ivpsolve


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver, info_op",
    cases=".recipe_cases",
    prefix="solver_",
    has_tag=("solve", "filter"),
)
def test_offgrid_marginals_filter(vf, u0, t0, t1, p, solver, info_op):
    solution = ivpsolve.solve(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
        info_op=info_op,
        atol=1e-1,
        rtol=1e-1,
    )
    midpoint = (solution[0].t + solution[1].t) / 2
    dense = solver.offgrid_marginals(
        t=midpoint, state=solution[1], state_previous=solution[0]
    )

    assert isinstance(dense, type(solution))

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    close_to_left = solver.offgrid_marginals(
        t=solution[0].t + 1e-4, state=solution[1], state_previous=solution[0]
    )
    close_to_right = solver.offgrid_marginals(
        t=solution[1].t - 1e-4, state=solution[1], state_previous=solution[0]
    )
    assert jnp.allclose(close_to_left.u, solution[0].u, atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(close_to_right.u, solution[0].u, atol=1e-3, rtol=1e-3)

    # Repeat the same but interpolating via *_searchsorted:
    # check we correctly landed in the first interval
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    dense = solver.offgrid_marginals_searchsorted(ts=ts, solution=solution)
    assert jnp.allclose(dense.t, ts)
    assert jnp.allclose(dense.u[0], solution.u[0], atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(dense.u[0], solution.u[1], atol=1e-3, rtol=1e-3)


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver, info_op",
    cases=".recipe_cases",
    prefix="solver_",
    has_tag=("solve", "smoother"),
)
def test_offgrid_marginals_smoother(vf, u0, t0, t1, p, solver, info_op):

    solution = ivpsolve.solve(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
        info_op=info_op,
        atol=1e-1,
        rtol=1e-1,
        control=controls.ClippedIntegral(),
    )
    midpoint = (solution[0].t + solution[1].t) / 2
    dense = solver.offgrid_marginals(
        t=midpoint, state=solution[1], state_previous=solution[0]
    )
    assert isinstance(dense, type(solution))

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    close_to_left = solver.offgrid_marginals(
        t=solution[0].t + 1e-4, state=solution[1], state_previous=solution[0]
    )
    close_to_right = solver.offgrid_marginals(
        t=solution[1].t - 1e-4, state=solution[1], state_previous=solution[0]
    )
    assert jnp.allclose(close_to_left.u, solution[0].u, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(close_to_right.u, solution[1].u, atol=1e-3, rtol=1e-3)

    # Repeat the same but interpolating via *_searchsorted:
    # check we correctly landed in the first interval
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    dense = solver.offgrid_marginals_searchsorted(ts=ts, solution=solution)
    assert jnp.allclose(dense.t, ts)
    assert jnp.allclose(dense.u[0], solution.u[0], atol=1e-3, rtol=1e-3)
    assert jnp.allclose(dense.u[-1], solution.u[-1], atol=1e-3, rtol=1e-3)