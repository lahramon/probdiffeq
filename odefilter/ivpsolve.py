"""Solve initial value problems."""


from functools import partial

import jax
import jax.numpy as jnp

from odefilter import _control_flow, taylor

# The high-level checkpoint-style routines


@partial(jax.jit, static_argnums=[0, 5])
def simulate_terminal_values(
    vector_field, initial_values, t0, t1, solver, info_op, parameters=()
):
    """Simulate the terminal values of an initial value problem.

    Thin wrapper around :func:`odefilter_terminal_values`.
    """
    _assert_not_scalar(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=lambda *x: vector_field(t0, *x, *parameters),
        initial_values=initial_values,
        num=solver.strategy.implementation.num_derivatives,
    )

    info_op_curried = info_op(vector_field)
    return odefilter_terminal_values(
        lambda t, *xs: info_op_curried(t, *xs, *parameters),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
    )


@partial(jax.jit, static_argnums=[0])
def odefilter_terminal_values(info, taylor_coefficients, t0, t1, solver):
    """Simulate the terminal values of an ODE with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    state0 = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        info_op=info,
        solver=solver,
    )
    return solver.extract_fn(state=solution)


@partial(jax.jit, static_argnums=[0, 4])
def simulate_checkpoints(
    vector_field, initial_values, ts, solver, info_op, parameters=()
):
    """Solve an IVP and return the solution at checkpoints.

    Thin wrapper around :func:`odefilter_checkpoints`.
    """
    _assert_not_scalar(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=lambda *x: vector_field(ts[0], *x, *parameters),
        initial_values=initial_values,
        num=solver.strategy.implementation.num_derivatives,
    )

    info_op_curried = info_op(vector_field)
    return odefilter_checkpoints(
        lambda t, *xs: info_op_curried(t, *xs, *parameters),
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
    )


@partial(jax.jit, static_argnums=[0])
def odefilter_checkpoints(info, taylor_coefficients, ts, solver):
    """Simulate checkpoints of an ODE solution with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            info_op=info,
            solver=solver,
        )
        return s_next, s_next

    state0 = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=ts[0])

    _, solution = _control_flow.scan_with_init(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=ts[1:],
        reverse=False,
    )
    return solver.extract_fn(state=solution)


# Full solver routines


def solve(vector_field, initial_values, t0, t1, solver, info_op, parameters=()):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.

    !!! warning
        The parameters are essentially static. Why?
        Because we use ``lambda t, y: f(t, y, p)``-style implementations
        and pass this lambda function to lower-level implementations,
        which have static "fun" arguments. Since we cannot jit this function.
        the lower-level stuff must recompile... :(
    """
    _assert_not_scalar(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=lambda *x: vector_field(t0, *x, *parameters),
        initial_values=initial_values,
        num=solver.strategy.implementation.num_derivatives,
    )

    # todo: because of this line, the function recompiles
    #  every single time it is called.
    #  This is because odefilter() marks the info_op as static, and because
    #  info_op() creates a new function every time it is called.
    #  Is it sufficient to make information operators cache output?
    info_op_curried = info_op(vector_field)

    # todo: this lambda function below is newly created at every
    #  call to solve() and therefore we recompile steps
    #  every single time. This is strange.
    return odefilter(
        lambda t, *xs: info_op_curried(t, *xs, *parameters),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
    )


def odefilter(info_op, taylor_coefficients, t0, t1, solver):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    generator = _odefilter_generator(
        info_op, taylor_coefficients=taylor_coefficients, t0=t0, t1=t1, solver=solver
    )
    forward_solution = _control_flow.tree_stack([sol for sol in generator])
    return solver.extract_fn(state=forward_solution)


def _odefilter_generator(info_op, taylor_coefficients, t0, t1, solver):
    """Generate an ODE filter solution iteratively."""
    _assert_not_scalar(taylor_coefficients)
    state = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)
    yield state
    while state.solution.t < t1:
        state = solver.step_fn(state=state, info_op=info_op, t1=t1)
        yield state


# Auxiliary routines


def _assert_not_scalar(x, /):
    """Verify the initial conditions are not scalar.

    There is no clear mechanism for the internals if the IVP is
    scalar. Therefore, we don't allow them for now.

    todo: allow scalar problems.
    """
    is_not_scalar = jax.tree_util.tree_map(lambda x: jnp.ndim(x) > 0, x)
    assert jax.tree_util.tree_all(is_not_scalar)


def _advance_ivp_solution_adaptively(info_op, t1, state0, solver):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.solution.t < t1

    def body_fun(s):
        state = solver.step_fn(state=s, info_op=info_op, t1=t1)
        return state

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
