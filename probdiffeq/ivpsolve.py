"""Routines for estimating solutions of initial value problems."""

import functools
import warnings

import jax
import jax.numpy as jnp

from probdiffeq import _ivpsolve_impl
from probdiffeq.backend import tree_array_util
from probdiffeq.impl import impl
from probdiffeq.solvers import markov, _common

# todo: change the Solution object to a simple
#  named tuple containing (t, full_estimate, u_and_marginals, stats).
#  No need to pre/append the initial condition to the solution anymore,
#  since the user knows it already.


class Solution:
    """Estimated initial value problem solution."""

    def __init__(self, t, u, output_scale, marginals, posterior, num_steps):
        """Construct a solution object."""
        self.t = t
        self.u = u
        self.output_scale = output_scale
        self.marginals = marginals
        self.posterior = posterior
        self.num_steps = num_steps

    def __repr__(self):
        """Evaluate a string-representation of the solution object."""
        return (
            f"{self.__class__.__name__}("
            f"t={self.t},"
            f"u={self.u},"
            f"output_scale={self.output_scale},"
            f"marginals={self.marginals},"
            f"posterior={self.posterior},"
            f"num_steps={self.num_steps},"
            ")"
        )

    def __len__(self):
        """Evaluate the length of a solution."""
        if jnp.ndim(self.t) < 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access a single item of the solution."""
        if jnp.ndim(self.t) < 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)

        if jnp.ndim(self.t) == 1 and item != -1:
            msg = "Access to non-terminal states is not available."
            raise ValueError(msg)

        return jax.tree_util.tree_map(lambda s: s[item, ...], self)

    def __iter__(self):
        """Iterate through the solution."""
        if jnp.ndim(self.t) <= 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)

        for i in range(self.t.shape[0]):
            yield self[i]


def _sol_flatten(sol):
    children = (
        sol.t,
        sol.u,
        sol.marginals,
        sol.posterior,
        sol.output_scale,
        sol.num_steps,
    )
    aux = ()
    return children, aux


def _sol_unflatten(_aux, children):
    t, u, marginals, posterior, output_scale, n = children
    return Solution(
        t=t,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=n,
    )


jax.tree_util.register_pytree_node(Solution, _sol_flatten, _sol_unflatten)


def simulate_terminal_values(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0
) -> Solution:
    """Simulate the terminal values of an initial value problem."""
    save_at = jnp.asarray([t1])
    (_t, solution_save_at), _, num_steps = _ivpsolve_impl.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        t0,
        initial_condition,
        save_at=save_at,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )
    # "squeeze"-type functionality (there is only a single state!)
    squeeze_fun = functools.partial(jnp.squeeze, axis=0)
    solution_save_at = jax.tree_util.tree_map(squeeze_fun, solution_save_at)
    num_steps = jax.tree_util.tree_map(squeeze_fun, num_steps)

    # I think the user expects marginals, so we compute them here
    posterior, output_scale = solution_save_at
    marginals = posterior.init if isinstance(posterior, markov.MarkovSeq) else posterior
    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=t1,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def solve_and_save_at(
    vector_field, initial_condition, save_at, adaptive_solver, dt0
) -> Solution:
    """Solve an initial value problem and return the solution at a pre-determined grid.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    if not adaptive_solver.solver.strategy.is_suitable_for_save_at:
        msg = (
            f"Strategy {adaptive_solver.solver.strategy} should not "
            f"be used in solve_and_save_at. "
        )
        warnings.warn(msg, stacklevel=1)

    (_t, solution_save_at), _, num_steps = _ivpsolve_impl.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        save_at[0],
        initial_condition,
        save_at=save_at[1:],
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )

    # I think the user expects the initial condition to be part of the state
    # (as well as marginals), so we compute those things here
    posterior_t0, *_ = initial_condition
    posterior_save_at, output_scale = solution_save_at
    _tmp = _userfriendly_output(posterior=posterior_save_at, posterior_t0=posterior_t0)
    marginals, posterior = _tmp
    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=save_at,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def solve_and_save_every_step(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0
) -> Solution:
    """Solve an initial value problem and save every step.

    This function uses a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.
    """
    if not adaptive_solver.solver.strategy.is_suitable_for_save_every_step:
        msg = (
            f"Strategy {adaptive_solver.solver.strategy} should not "
            f"be used in solve_and_save_every_step."
        )
        warnings.warn(msg, stacklevel=1)

    (t, solution_every_step), _dt, num_steps = _ivpsolve_impl.solve_and_save_every_step(
        jax.tree_util.Partial(vector_field),
        t0,
        initial_condition,
        t1=t1,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )
    # I think the user expects the initial time-point to be part of the grid
    # (Even though t0 is not computed by this function)
    t = jnp.concatenate((jnp.atleast_1d(t0), t))

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
    posterior, output_scale = solution_every_step
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0)
    marginals, posterior = _tmp

    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=t,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def solve_fixed_grid(vector_field, initial_condition, grid, solver) -> Solution:
    """Solve an initial value problem on a fixed, pre-determined grid."""
    # Compute the solution
    _t, (posterior, output_scale) = _ivpsolve_impl.solve_fixed_grid(
        jax.tree_util.Partial(vector_field), initial_condition, grid=grid, solver=solver
    )

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0)
    marginals, posterior = _tmp

    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=grid,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=jnp.arange(1.0, len(grid)),
    )

def solve_fixed_grid_arr(vector_field:list, initial_condition, grid:list, solver, use_filter=False) -> Solution:
    """Solve an initial value problem on a fixed, pre-determined grid,
    with an array of vector fields and corresponding time grids. 
    Time grids should not have overlapping intervals."""
    # Compute the solution
    initial_condition_markovseq_arr = [initial_condition[0]] * len(vector_field)
    initial_condition_output_scale_arr = [initial_condition[1]] * len(vector_field)
    output_scale_arr = [None] * len(vector_field)
    posterior_arr = [None] * len(vector_field)
    for i, vf in enumerate(vector_field):
        _t, _tmp = _ivpsolve_impl.solve_fixed_grid(
            jax.tree_util.Partial(vf),
            (initial_condition_markovseq_arr[i], initial_condition_output_scale_arr[i]),
            grid=grid[i],
            solver=solver,
        )
        posterior_arr[i], output_scale_arr[i] = _tmp
        if i < len(vector_field)-1:
            # one more prediction step to starting time of next interval
            dt = grid[i+1][0] - grid[i][-1]
            # get starting state
            initial_condition_markovseq_transition = jax.tree_util.tree_map(lambda s: s[-1, ...], posterior_arr[i])
            output_scale_transition = jax.tree_util.tree_map(lambda s: s[-1, ...], output_scale_arr[i])
            
            state = solver.init(grid[i][-1], (initial_condition_markovseq_transition, output_scale_transition))
            error, _observed, state_strategy = solver.strategy.predict_error(
                state.strategy,
                dt=dt,
                vector_field=vector_field[i],
            )
            error_f, _observed_f, state_strategy_f = solver.strategy.predict_error(
                state.strategy,
                dt=dt,
                vector_field=vector_field[i+1],
            )
            output_scale_inf = jnp.array(1e9)
            
            state_strategy_complete = solver.strategy.complete(
                state_strategy,
                output_scale=output_scale_inf
            )

            # one normal solver step looks like this:
            state_strategy = state.strategy
            hidden_begin, extra_begin = solver.strategy.extrapolation.begin(state_strategy.hidden, state_strategy.aux_extra, dt=dt)
            t = state_strategy.t + dt
            error_begin, observed_begin, corr_begin = solver.strategy.correction.estimate_error(
                hidden_begin, state_strategy.aux_corr, vector_field=vector_field[i+1], t=t
            )
            state_begin = _State(t=t, hidden=hidden_begin, aux_extra=extra_begin, aux_corr=corr_begin)
            hidden_compl_1, extra_compl = solver.strategy.extrapolation.complete(
                state_begin.hidden, state_begin.aux_extra, output_scale=output_scale_transition
            )
            hidden_compl_2, corr_compl = solver.strategy.correction.complete(hidden_compl_1, state_begin.aux_corr)
            state_updt = _State(t=state_begin.t, hidden=hidden_compl_2, aux_extra=extra_compl, aux_corr=corr_compl)

            # updated:
            state_strategy = state.strategy
            hidden_begin, extra_begin = solver.strategy.extrapolation.begin(state_strategy.hidden, state_strategy.aux_extra, dt=dt)
            t = state_strategy.t + dt
            error_begin, observed_begin, corr_begin = solver.strategy.correction.estimate_error(
                hidden_begin, state_strategy.aux_corr, vector_field=vector_field[i], t=t
            )
            state_begin = _State(t=t, hidden=hidden_begin, aux_extra=extra_begin, aux_corr=corr_begin)
            hidden_compl_1, extra_compl = solver.strategy.extrapolation.complete(
                state_begin.hidden, state_begin.aux_extra, output_scale=output_scale_transition
            )
            # until here everything should be fine
            hidden_compl_2, corr_compl = solver.strategy.correction.complete(hidden_compl_1, state_begin.aux_corr)
            state_updt = _State(t=state_begin.t, hidden=hidden_compl_2, aux_extra=extra_compl, aux_corr=corr_compl)

            # state_predict = _common.State(strategy=state_strategy, output_scale=state.output_scale)
            state_predict = _common.State(strategy=state_strategy, output_scale=output_scale_inf)
            t_predict, (posterior_predict, output_scale_predict) = solver.extract(state_predict)

            # set infinite noise for vector field prediction (nonsmooth -> could be anything)
            posterior_predict.cholesky = posterior_predict.cholesky.at[-1,-1].set(1e9)
            # condition on vector field at beginning of next interval
            error, observed, corr = solver.strategy.correction.estimate_error(posterior_predict, None, vector_field=vector_field[i+1],t=t_predict)
            # hidden_updt, corr_updt = solver.strategy.correction.complete(state_strategy.hidden, corr)
            # hidden_updt, corr_updt = solver.strategy.correction.complete(state_strategy.hidden, state_strategy.aux_corr)

            # get new initial condition as last value from previous posterior
            initial_condition_markovseq_arr[i+1] = posterior_predict
            initial_condition_output_scale_arr[i+1] = output_scale_predict

    # Stitch together smoothed solution (smoothing marginals computed in userfriendly output)
    posterior_t0, output_scale_t0 = initial_condition # this just extracts the MarkovSeq part (no output scale)
    posterior_t0_with_output_scale = initial_condition

    if not use_filter:
        # TODO: Implement smoothing part
        for i in range(len(vector_field)-1,-1,-1):
            # get MarkovSeq from beginning of current interval

            _tmp = _userfriendly_output(posterior=posterior_arr[i], posterior_t0=posterior_t0)
            marginals, posterior = _tmp
    else:
        # prepend the initial condition to the computed marginals for each subinterval
        for i in range(len(vector_field)):
            posterior_arr[i] = tree_array_util.tree_prepend(initial_condition_markovseq_arr[i], posterior_arr[i])
            output_scale_arr[i] = jnp.hstack((initial_condition_output_scale_arr[i], output_scale_arr[i]))
        # concatenate
        posterior = tree_array_util.tree_concatenate(posterior_arr)
        output_scale = tree_array_util.tree_concatenate(output_scale_arr)
        marginals = posterior

    t = jnp.concatenate((*grid,))
    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=t,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=jnp.arange(1.0, len(t)),
    )

def _userfriendly_output(*, posterior, posterior_t0):
    if isinstance(posterior, markov.MarkovSeq):
        # Compute marginals
        posterior_no_filter_marginals = markov.select_terminal(posterior)
        marginals = markov.marginals(posterior_no_filter_marginals, reverse=True)

        # Prepend the marginal at t1 to the computed marginals
        marginal_t1 = jax.tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        marginals = tree_array_util.tree_append(marginals, marginal_t1)

        # Prepend the marginal at t1 to the inits
        init_t0 = posterior_t0.init
        init = tree_array_util.tree_prepend(init_t0, posterior.init)
        posterior = markov.MarkovSeq(init=init, conditional=posterior.conditional)
    else:
        posterior = tree_array_util.tree_prepend(posterior_t0, posterior)
        marginals = posterior
    return marginals, posterior
