 # %%
"""
Kepler' s Equation
==================

This is based on example 10.64 from Betts' book "Practical Methods for Optimal
Control Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

Only the second EOM is from the book, the first EOM ist just a simple ODE,
without any connection to the problem.
The third EOM is just something I made up to have a second algebraic equation.

All I want to show is that opty can handle algebraic equations only, too.
One has to add some dummy differential equation to satisfy its formal needs.


**States**

- :math:`y, E_1, E_2` : state variables

**Specifieds**

- :math:`u` : control variable

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem

# %%
# Equations of Motion
# -------------------
#
t = me.dynamicsymbols._t
T = sm.symbols('T', cls=sm.Function)
y, E1, E2 = me.dynamicsymbols('y E1 E2')
u = me.dynamicsymbols('u')

# %%
# equations of motion

eom = sm.Matrix([
        -y.diff(t) + sm.cos(y) + u**2,
        -E1 + T(t)*sm.sin(E1) + 1.0,
        -E2 - T(t)*sm.cos(E2) - 1.0,
])
sm.pprint(eom)

# %%
# Set up and Solve the Optimization Problem
# -----------------------------------------
num_nodes = 751
t0, tf = 0.0, 0.9
state_symbols = (y, E1, E2)

interval_value = (tf - t0)/(num_nodes - 1)
times = np.linspace(t0, tf, num_nodes)

# Specify the objective function and form the gradient.

# Specify the symbolic instance constraints, as per the example.
instance_constraints = (
    y.func(t0) - 0.0,
    E1.func(t0) - 1.0,
    E2.func(t0) + 1.0,
)

def obj(free):
    return sum([free[i]**2 for i in range(num_nodes)])

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[:num_nodes] = 2*free[:num_nodes]
    return grad

# %%
# Create the optimization problem and set any options.
prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints=instance_constraints,
        known_trajectory_map={T(t): times}
)

prob.add_option('max_iter', 1000)

# Give some rough estimates for the trajectories.
initial_guess = np.ones(prob.num_free)

# Find the optimal solution.
for _ in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.3f}')

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function.
prob.plot_objective_value()
