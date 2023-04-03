"""Tests for IVP solvers."""
import jax
import jax.numpy as jnp
import pytest
import pytest_cases
import pytest_cases.filters

from probdiffeq import ivpsolve, solution, test_util
from probdiffeq.strategies import filters, smoothers


@pytest_cases.fixture(scope="session", name="solution_native_python_while_loop")
@pytest_cases.parametrize_with_cases("ode_problem", cases="..problem_cases")
def fixture_solution_native_python_while_loop(ode_problem):
    solver = test_util.generate_solver(num_derivatives=1)
    sol = ivpsolve.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1,
        rtol=1e-2,
    )
    return sol, solver


def test_solution_is_iterable(solution_native_python_while_loop):
    sol, _ = solution_native_python_while_loop
    assert isinstance(sol[0], type(sol))
    assert len(sol) == len(sol.t)


def test_getitem_raises_error_for_nonbatched_solutions(
    solution_native_python_while_loop,
):
    """__getitem__ only works for batched solutions."""
    sol, _ = solution_native_python_while_loop
    with pytest.raises(ValueError):
        _ = sol[0][0]
    with pytest.raises(ValueError):
        _ = sol[0, 0]


def test_loop_over_solution_is_possible(solution_native_python_while_loop):
    solution, _ = solution_native_python_while_loop

    i = 0
    for i, sol in zip(range(2 * len(solution)), solution):
        assert isinstance(sol, type(solution))

    assert i == len(solution) - 1


# Maybe this test should be in a different test suite, but it does not really matter...
def test_marginal_nth_derivative_of_solution(solution_native_python_while_loop):
    sol, _ = solution_native_python_while_loop

    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = sol.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == sol.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with pytest.raises(ValueError):
        sol.marginals.marginal_nth_derivative(100)


def test_offgrid_marginals_filter(solution_native_python_while_loop):
    sol, solver = solution_native_python_while_loop
    t0, t1 = sol.t[0], sol.t[-1]

    # todo: this is hacky. But the tests get faster?
    if isinstance(solver.strategy, filters.Filter):
        # Extrapolate from the left: close-to-left boundary must be similar,
        # but close-to-right boundary must not be similar
        u_left, _ = solution.offgrid_marginals(
            t=sol[0].t + 1e-4,
            solution=sol[1],
            solution_previous=sol[0],
            solver=solver,
        )
        u_right, _ = solution.offgrid_marginals(
            t=sol[1].t - 1e-4,
            solution=sol[1],
            solution_previous=sol[0],
            solver=solver,
        )
        assert jnp.allclose(u_left, sol[0].u, atol=1e-3, rtol=1e-3)
        assert not jnp.allclose(u_right, sol[0].u, atol=1e-3, rtol=1e-3)

        # Repeat the same but interpolating via *_searchsorted:
        # check we correctly landed in the first interval
        ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
        u, _ = solution.offgrid_marginals_searchsorted(
            ts=ts, solution=sol, solver=solver
        )
        assert jnp.allclose(u[0], sol.u[0], atol=1e-3, rtol=1e-3)
        assert not jnp.allclose(u[0], sol.u[1], atol=1e-3, rtol=1e-3)


def test_offgrid_marginals_smoother(solution_native_python_while_loop):
    sol, solver = solution_native_python_while_loop
    t0, t1 = sol.t[0], sol.t[-1]

    # todo: this is hacky. But the tests get faster?
    if isinstance(solver.strategy, smoothers.Smoother):
        # Extrapolate from the left: close-to-left boundary must be similar,
        # but close-to-right boundary must not be similar
        u_left, _ = solution.offgrid_marginals(
            t=sol[0].t + 1e-4,
            solution=sol[1],
            solution_previous=sol[0],
            solver=solver,
        )
        u_right, _ = solution.offgrid_marginals(
            t=sol[1].t - 1e-4,
            solution=sol[1],
            solution_previous=sol[0],
            solver=solver,
        )
        assert jnp.allclose(u_left, sol[0].u, atol=1e-3, rtol=1e-3)
        assert jnp.allclose(u_right, sol[1].u, atol=1e-3, rtol=1e-3)

        # Repeat the same but interpolating via *_searchsorted:
        # check we correctly landed in the first interval
        ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
        u, _ = solution.offgrid_marginals_searchsorted(
            ts=ts, solution=sol, solver=solver
        )
        assert jnp.allclose(u[0], sol.u[0], atol=1e-3, rtol=1e-3)
        assert jnp.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)


@pytest_cases.fixture(scope="session", name="solution_save_at")
@pytest_cases.parametrize_with_cases("ode_problem", cases="..problem_cases")
def fixture_solution_save_at(ode_problem):
    solver = test_util.generate_solver(strategy_factory=smoothers.FixedPointSmoother)

    save_at = jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=4)
    sol = ivpsolve.solve_and_save_at(
        ode_problem.vector_field,
        ode_problem.initial_values,
        save_at=save_at,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1,
        rtol=1e-2,
    )
    return sol, solver


@pytest_cases.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_grid_samples(solution_save_at, shape):
    sol, solver = solution_save_at

    key = jax.random.PRNGKey(seed=15)
    u, samples = solution.sample(key, solution=sol, solver=solver, shape=shape)
    assert u.shape == shape + sol.u.shape
    assert samples.shape == shape + sol.marginals.hidden_state.sample_shape

    # Todo: test values of the samples by checking a chi2 statistic
    #  in terms of the joint posterior. But this requires a joint_posterior()
    #  method, which is only future work I guess. So far we use the eye-test
    #  in the notebooks, which looks good.