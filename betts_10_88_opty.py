# %%
"""
Nonconvex Delay
===============

This is example 10.88 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.
(actually it is 10.90, as I use :math:`\\sigma = 5`)

This simulation shows the use of instance constrains of the form:
:math:`x_i(t_0) = x_j(t_f)`.


**States**

- :math:`x_1,....., x_N` : state variables

**Controls**

- :math:`u_1,....., u_N` : control variables

Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their meaning.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function

# %%
# Equations of Motion
# -------------------
t = me.dynamicsymbols._t

N = 20
tf = 0.1
sigma = 5

x = me.dynamicsymbols(f'x:{N}')
u = me.dynamicsymbols(f'u:{N}')

eom = sm.Matrix([
    *[-x[i].diff(t) + 1**2 - u[i] for i in range(sigma)],
    *[-x[i].diff(t) + x[i-sigma]**2 - u[i] for i in range(sigma, N)],
])

sm.pprint(eom)

# %%
# Define and Solve the Optimization Problem
# -----------------------------------------
num_nodes = 501
t0 = 0.0
interval_value = (tf - t0) / (num_nodes - 1)

state_symbols = x
unkonwn_input_trajectories = u

# %%
# Specify the objective function and form the gradient.
objective = sm.Integral(sum([x[i]**2 + u[i]**2 for i in range(N)]), t)

obj, obj_grad = create_objective_function(
    objective,
    state_symbols,
    unkonwn_input_trajectories,
    tuple(),
    num_nodes,
    interval_value
)

# %%

instance_constraints = (
    x[0].func(t0) - 1.0,
    *[ x[i].func(t0) - x[i-1].func(tf) for i in range(1, N)],
)

limit_value = np.inf
bounds = {
   x[i]: (0.7, limit_value) for i in range(1, N)
}

# %%
# Iterate
# -------

prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
        bounds=bounds,
    )

prob.add_option('max_iter', 1000)

initial_guess = np.ones(prob.num_free) * 0.1
# Find the optimal solution.
for i in range(2):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book '+
        f'it is {2.79685770}, so the error is: '
        f'{(info['obj_val'] - 2.79685770)/2.79685770*100:.3f} % ')
    print('\n')


# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function.
prob.plot_objective_value()


# %%
# Are the inequality constraints satisfied?
min_x = np.min(solution[1: N*num_nodes-1])
if min_x >= 0.7:
    print(f"Minimal value of the x\u1D62 is: {min_x:.12f} >= 0.7, so satisfied")
else:
    print(f"Minimal value of the x\u1D62 is: {min_x:.12f} < 0.7, so not satisfied")

