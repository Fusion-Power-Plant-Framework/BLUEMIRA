Performing an Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following example to demonstrate how to set up
and perform an optimisation with some non-linear constraints.

Suppose that we wish to find

.. math::

    \min_{x \in \mathbb{R}^2} \sqrt{x_2} \tag{1}

subject to constraints

.. math::

    x_2 \ge (a_1x_1 + b_1)^3 \tag{2}

.. math::

    x_2 \ge (a_2 x_1 + b_2)^3 \tag{3}

for parameters
:math:`a_1 = 2`, :math:`b_1 = 0`, :math:`a_2 = -1`, :math:`b_2 = 1`.

This problem expects a minimum at
:math:`\boldsymbol{x} = ( \frac{1}{3}, \frac{8}{27} )`.

.. note::

    This example is ripped straight from the
    `NLOpt docs <https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/#example-nonlinearly-constrained-problem>`_.

.. _optimisation-functions-objective_function:

Objective Function
------------------

First we must define our objective function;
the function we wish to minimise.
The function must take a single argument: a 1-D ``numpy`` array,
where each element in the array is an optimisation parameter.
It must return a float.

For our example, our objective function looks like this,
note that we don't actually need to use :math:`x_1` (``x[0]``) here:

.. code-block:: python

    def f_objective(x: np.ndarray) -> float:
        """Objective function for optimisation to find solution to eqn (1)."""
        return np.sqrt(x[1])

If using a gradient-based optimisation algorithm,
it helps to define an analytical gradient.
If you do not give an analytical gradient,
a numerical one will be estimated.

As with the objective, the gradient function must take a 1-D ``numpy`` array,
containing the optimisation parameters as its only argument.
It must return a ``numpy`` array of the same length,
where each index :math:`i` contains the partial derivative
:math:`\frac{\partial f}{\partial x_i}`
for the corresponding optimisation parameter.

.. code-block:: python

    def df_objective(x) -> np.ndarray:
        """Gradient of the objective function."""
        return np.array([0.0, 0.5 / np.sqrt(x[1])])

.. _optimisation-functions-constraints:

Constraints
-----------

A constraint function must take a single argument,
the array of optimisation parameters.
Both vector-valued and scalar constraints are supported,
so the function can return a float, or,
an array of length :math:`m`,
where :math:`m` is the dimensionality of the constraint.

Given a constraint function :math:`f_c`, and optimisation parameters
:math:`\boldsymbol{x}`,
an equality constraint has the form

.. math::

    f_c(\boldsymbol{x}) = 0

and an inequality constraint has the form

.. math::

    f_c(\boldsymbol{x}) \le 0


In our example, we have two inequality constraints.
We'll need to adjust these slightly to match the required form.
Equation 2 becomes

.. math::

    (a_1 x_1 + b_1)^3 - x_2 \le 0

and equation 3 becomes

.. math::

    (a_2 x_1 + b_2)^3 - x_2 \le 0


Notice that these two constraints are similar,
so we can define a single function,
then use lambdas to set ``a`` and ``b`` to get the required values.

.. code-block:: python

    def f_constraint(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Inequality constraint."""
        return np.array([(a * x[0] + b) ** 3 - x[1]])

    f_constraint_1 = lambda x: f_constraint(x, 2, 0)
    f_constraint_2 = lambda x: f_constraint(x, -1, 1)

We can also define the gradient of the constraint.
Note that this isn't strictly necessary, as,
if a gradient-based optimiser is used, a numerical approximation is made.
However, an analytical gradient will be more reliable.

The constraint's gradient function takes the array of optimisation parameters,
and returns an array with shape :math:`m \times n`.

The partial derivatives of the constraint in our example are

.. math::
    :nowrap:

    \begin{gather*}
    \frac{\partial f_c}{\partial x_1} = 3a(a x_1 + b)^2 \\
    \frac{\partial f_c}{\partial x_2} = -1
    \end{gather*}

So our Python function will be

.. code-block:: python

    def df_constraint(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Inequality constraint gradient."""
        return np.array([3 * a * (a * x[0] + b) ** 2, -1.0])


Note that we are using two separate constraints here,
but it can sometimes be more convenient to express multiple constraints
in a single vector-valued one.
In this case that vector-valued constraint, and its gradient,
could look like this

.. code-block:: python

    def vector_constraint(x: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
        return np.array([f_constraint(x, a1, b1), f_constraint(x, a2, b2)])


    def d_vector_constraint(x: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
        return np.vstack([df_constraint(x, a1, b1), df_constraint(x, a2, b2)])

.. note::

    Not all optimisation algorithms support non-linear constraints.
    They can only be used with
    ``SLSQP``, ``COBYLA``, and ``ISRES``.

The Optimise Function
---------------------

Now that we have our objective function and constraints,
we can perform the optimisation.
To do this, we use the
:py:func:`~bluemira.optimisation._optimise.optimise` function.

Within this function, we can select the optimisation algorithm to use,
parameter bounds, stopping conditions, the initial guess
(if not given, the center of the bounds is used), and whether to record the
history of the optimisation parameters
(this is off by default, as it impacts run time performance).

.. code-block:: pycon

    >>> result = optimise(
    ...     f_objective,
    ...     df_objective=df_objective,
    ...     algorithm="SLSQP",
    ...     x0=np.array([1, 1]),
    ...     opt_conditions={"xtol_rel": 1e-10, "max_eval": 1000},
    ...     keep_history=True,
    ...     bounds=(np.array([-np.inf, 0]), np.array([np.inf, np.inf])),
    ...     ineq_constraints=[
    ...         {
    ...             "f_constraint": lambda x: f_constraint(x, 2, 0),
    ...             "df_constraint": lambda x: df_constraint(x, 2, 0),
    ...             "tolerance": np.array([1e-8]),
    ...         },
    ...         {
    ...             "f_constraint": lambda x: f_constraint(x, -1, 1),
    ...             "df_constraint": lambda x: df_constraint(x, -1, 1),
    ...             "tolerance": np.array([1e-8]),
    ...         },
    ...     ],
    ... )
    >>> print(result)
    OptimiserResult(x=array([0.33333528, 0.29629148]), n_evals=18)

The Optimisation Problem Class
------------------------------

As an alternative to the :py:func:`~bluemira.optimisation._optimise.optimise`
function,
it is possible to take a class-based approach to performing an optimisation.
This can have several benefits, including
Liskov Substitution of optimisation problems,
shared state between objective functions and constraints,
and logical grouping of related functionality.

To define an optimisation problem, inherit from
:py:class:`~bluemira.optimisation.problem.OptimisationProblem`
and implement the interface.

You must implement the
:py:meth:`~bluemira.optimisation.problem.OptimisationProblem.objective`
method.

You can optionally override:

* :py:meth:`~bluemira.optimisation.problem.OptimisationProblem.df_objective`
    Must return the gradient of the objective function
    at the given parameterisation.
    If this is not overridden, and a gradient-based algorithm is used,
    a gradient will be numerically estimated.
    See also, :ref:`optimisation-functions-objective_function`.
* :py:meth:`~bluemira.optimisation.problem.OptimisationProblem.eq_constraints`
    Must return a list of
    :py:class:`~bluemira.optimisation.typing.ConstraintT`
    dictionaries, defining equality constraints.
    See also, :ref:`optimisation-functions-constraints`.
* :py:meth:`~bluemira.optimisation.problem.OptimisationProblem.ineq_constraints`
    Must return a list of
    :py:class:`~bluemira.optimisation.typing.ConstraintT`
    dictionaries, defining inequality constraints.
    See also, :ref:`optimisation-functions-constraints`.
* :py:meth:`~bluemira.optimisation.problem.OptimisationProblem.bounds`
    Must return the lower and upper bounds of the optimisation parameters.
    The default is to return :code:`(-np.inf, np.inf)`.

See
:doc:`here <../examples/optimisation/nonlinearly_constrained_problem>`
for an implemented example of an :code:`OptimisationProblem`.

Available Optimisation Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several optimisation algorithms that can be used within Bluemira.
Including gradient and non-gradient based.

- SLSQP
- COBYLA
- SBPLX
- MMA
- BFGS
- DIRECT
- DIRECT_L
- CRS
- ISRES

See the :py:class:`~bluemira.optimisation._algorithm.Algorithm`
enum for a reliably up-to-date list.
