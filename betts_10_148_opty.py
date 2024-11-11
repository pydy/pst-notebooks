# %%
"""
Zermolo's Problem
=================

This is example 10.148 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.
The goal is to minimize the final time :math:`t_f` to reach the point (0 / 0)

**States**

- :math:`x, y` : state variables

**Controls**

- :math:`\theta` : control variable

Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their meaning.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem

# %%
# Equations of motion.
t = me.dynamicsymbols._t

x, y = me.dynamicsymbols('x, y')
theta = me.dynamicsymbols('theta')
h = sm.symbols('h')

#Parameters
V = 1.0
c = -1.0

eom = sm.Matrix([
    -x.diff(t) + V*sm.cos(theta) + c*y,
    -y.diff(t) + V*sm.sin(theta),
])
sm.pprint(eom)

# %%
# Define and solve the optimization problem.
num_nodes = 1001
t0 = 0*h
tf = (num_nodes - 1) * h
interval_value = h

state_symbols = (x, y)
unkonwn_input_trajectories = (theta, )

# Specify the objective function and form the gradient.
def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

instance_constraints = (
    x.func(t0) - 3.5,
    y.func(t0) + 1.8,
    x.func(tf),
    y.func(tf),
)

bounds = {
        h: (0.0, 0.5),
}

# Create the optimization problem and set any options.
prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
        bounds=bounds,
)
print(prob.num_free)
# prob.add_option('nlp_scaling_method', 'gradient-based')
prob.add_option('max_iter', 5000)

# Give some rough estimates for the trajectories.
initial_guess = np.ones(prob.num_free) * 0.1

# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {(num_nodes-1)*info['obj_val']:.4f}, as per the book ' +
        f'it is {5.26493205}, so the error is: '
        f'{((num_nodes-1)*info['obj_val'] - 5.26493205)/5.26493205*100:.3f} % ')

# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()
