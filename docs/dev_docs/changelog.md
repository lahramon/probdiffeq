# Change log

## v0.2.0

Notable breaking changes:

* `solution_routines.py` has been renamed to `ivpsolve.py`. 
  The contents of both modules are identical.
  This is done to pave the way for boundary value problem solvers
  and to tighten the correspondence between solvers and solution routines.
* `solvers.py` has been renamed to `ivpsolvers.py`. 
  The contents of both modules are identical.
  This is done to pave the way for boundary value problem solvers
  and to tighten the correspondence between solvers and solution routines.
* `cubature.py` has been moved to `probdiffeq.implementations`.
  The contents of both modules are identical.
* `dense_output.py` has been renamed to `solution.py` and from now on also contains
  the `Solution` object (formerly in `solvers.py`). 
  This has been done to pave the way for boundary value problem solvers.
* `solution.negative_marginal_log_likelihood` has been renamed to
  `solution.log_marginal_likelihood` and its output is -1 times the output of the former function.
  The new term is mathematically more accurate, implements less logic and has a shorter name.
  The same applies to `solution.negative_marginal_log_likelihood_terminal_values`, which
  has become `solution.log_marginal_likelihood_terminal_values`.
* `norm_of_whitened_residual_sqrtm()` has been renamed to `mahalanobis_norm(x, /)` and is a function of one argument now.
  This is mathematically more accurate; the function should depend on an input.
* The recipes in implementation.recipes are not class-methods anymore but functions.
  For instance, instead of `recipes.IsoTS0.from_params(**kwargs)` users must call `recipes.ts0_iso(**kwargs)`.
  The advantages of this change are much less code to achieve the same logic, 
  more freedom to change background-implementations without worrying about API, 
  and improved ease of maintenance (no more classmethods, no more custom pytree node registration.)


Notable enhancements:

* Scalar solvers are now part of the public API. While all "other" methods are for IVPs of shape `(d,)`,
  scalar solvers target problems of shape `()` (i.e. if the initial values are floats, not arrays).
* The public API has been defined (see the developer docs). Notably, this document describes changes in which modules necessitate an entry in this changelog.



Notable bug fixes:

* The log-pdf behaviour of Gaussian random variables has been corrected (previously, the returned values were incorrect).
  This means that the behaviour of, e.g., parameter estimation scripts will change slightly.
  A related bugfix in the whitened residuals implies that the DenseTS1 is not exactly equivalent 
  to tornadox.ReferenceEK1 anymore (because the latter still has the same error).


## Prior to v0.2.0

This changelog has been started between v0.1.4 and 0.2.0.