# %%
r"""
Heat Diffusion Process with Inequality (betts_10_57)
====================================================

This is example 10.57 from *John T. Betts, Practical Methods for Optimal
Control Using Nonlinear Programming*, 3rd edition, chapter 10: Test Problems.
It deals with the 'discretization' of a PDE.

More details may be found in section 6.2 of this book.

Notes
-----

- The equations of motion consist of N - 1 differential quations and N - 1
  algebraic equations. The latter are bound to be non - positive.
- As the time appears explicitly in the equations of motion, and currently
  opty does not support this, a function T(t) is defined and a known
  trajectory map is provided to opty.


**States**

- :math:`y_1, .....y_{N-1}` : state variables

**Specifieds**

- :math:`u_0, u_{\ pi}` : control variables

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import time
from opty import Problem
from opty.utils import MathJaxRepr

# sphinx_gallery_thumbnail_number = 2

# %%
# Equations of motion.
t = me.dynamicsymbols._t
T = sm.symbols('T', cls=sm.Function)

N = 20

y = list(me.dynamicsymbols(f'y:{N-1}'))
u0, upi = me.dynamicsymbols('u0 upi')

# Parameters
q1 = 1.e-3
q2 = 1.e-3
a = 0.5
b = 0.2
c = 1.0
delta = sm.pi/N


def g(k, t):
    x = k * sm.pi/N
    return c * (sm.sin(x) * sm.sin(sm.pi*t/5) - a) - b


eom = sm.Matrix([
    -y[0].diff(t) + 1/delta**2 * (y[1] - 2*y[0] + u0),
    *[-y[i].diff(t) + 1/delta**2 * (y[i+1] - 2*y[i] + y[i-1])
      for i in range(1, N-2)],
    -y[N-2].diff(t) + 1/delta**2 * (upi - 2*y[N-2] + y[N-3]),
    *[g(k, T(t)) - y[k] for k in range(N-1)],
])

MathJaxRepr(eom)

# %%
# Set Up and Solve the Optimization Problem
# -----------------------------------------

t0, tf = 0.0, 5.0
num_nodes = 2001
interval_value = (tf - t0) / (num_nodes - 1)

state_symbols = y
specified_symbols = (u0, upi)

times = np.linspace(t0, tf, num=num_nodes)

# %%
# Specify the objective function and form the gradient.


def obj(free):
    value1 = interval_value * ((delta/2 + q1) *
                               np.sum(free[N*num_nodes:(N+1)*num_nodes]**2))
    value2 = interval_value * ((delta/2 + q2) *
                               np.sum(free[(N+1)*num_nodes:
                                           (N+2)*num_nodes]**2))
    value3 = delta * interval_value * np.sum(free[0:N*num_nodes]**2)
    return value1 + value2 + value3


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[N*num_nodes:(N+1)*num_nodes] = \
        (2 * (delta/2 + q1) * interval_value *
         free[N*num_nodes:(N+1)*num_nodes])
    grad[(N+1)*num_nodes:(N+2)*num_nodes] = \
        (2 * (delta/2 + q2) * interval_value *
         free[(N+1)*num_nodes:(N+2)*num_nodes])
    grad[0:N*num_nodes] = (2 * delta * interval_value * free[0:N*num_nodes])
    return grad


# %%
# Specify the instance constraints
instance_constraints = (
    *[y[i].func(t0) for i in range(N-1)],
    u0.func(t0),
    upi.func(t0),
)

# %%
# Inequality bounds on the last 19 eoms.
eom_bounds = {k: (-np.inf, 0.0) for k in range(N-1, 2*N-2)}

# %%
# Create the optimization problem.
prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    instance_constraints=instance_constraints,
    known_trajectory_map={T(t): times},
    eom_bounds=eom_bounds,
)

# %%
# Give some rough estimates for the trajectories.
initial_guess = np.zeros(prob.num_free)

# %%
# Find the optimal solution.
for _ in range(1):
    start = time.time()
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    Jstar = 0.468159793
    print(f"Objective value achieved: {info['obj_val']:.4f}, as per the book "
          f"it is {Jstar:.4f}, so the error is: "
          f"{(info['obj_val'] - Jstar) / (Jstar)*100:.3f} % ")
    print(f'Time taken for the simulation: {time.time() - start:.2f} s')

# %%
# Plot the optimal state and input trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True, show_bounds=True)

# %%
# Plot the objective function.
_ = prob.plot_objective_value()
