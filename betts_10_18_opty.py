  # %%
"""
Brachistochrone
===============

This is example 10.18 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.
It is described in more detail in section 4.16.2 of the book.

A classic example of the calculus of variations is to find the brachistochrone,
defined as that smooth curve joining two points A and B
(not underneath one another) along which a particle will slide from A to B
under gravity in the fastest possible time. (Of course, here already assumed
that the shape of the curve is known up to a parameter)

**States**

- :math:`x, y, v` : state variables

**Controls**

- :math:`u` : control variables

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt
import time

# %%
# Equations of motion.
# --------------------
start = time.time()
t = me.dynamicsymbols._t

x, y, v = me.dynamicsymbols('x y v')
u = me.dynamicsymbols('u')

h_fast = sm.symbols('h_fast')

eom = sm.Matrix([
    -x.diff() + v*sm.cos(u),
    -y.diff() + v*sm.sin(u),
    -v.diff() + 32.185 * sm.sin(u),
])
sm.pprint(eom)

# %%
# Define and solve the optimization problem.
num_nodes = 1001
iterations = 2000

interval_value = h_fast
t0, tf = 0*h_fast, (num_nodes-1)*h_fast

state_symbols = (x, y, v)
unkonwn_input_trajectories = (u, )

# Specify the objective function and form the gradient.
# Her the duration is to be minimized
def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

instance_constraints = (
        x.func(t0),
        y.func(t0),
        v.func(t0),

        x.func(tf) - 1.0,
)

bounds = {
        h_fast: (0.0, 1.0/(num_nodes-1)),
        x: (0.0, 10.0),
        y: (0.0, 10.0),
        v: (0.0, 10.0),
        u: (0, np.pi/2),
}

prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
        bounds=bounds,
)

# %%
# Solve the optimization problem.
# Give some rough estimates for the trajectories.
initial_guess = np.ones(prob.num_free)

# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Duration is: {solution[-1]*(num_nodes-1):.4f}, ' +
          f'as per the book it is {0.312480130}, so the deviation is: ' +
        f'{(solution[-1]*(num_nodes-1) - 0.312480130)/0.312480130*100 :.3e} %')
#    print(f'p(tf) = {solution[num_nodes-1]:.4f}' +
#          f'as per the book it is {7571.6712}, so the deviation is: ' +
#        f'{(solution[num_nodes-1] - 7571.6712)/7571.6712*100 :.3e} %')

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_title('Brachistochrone')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.plot(solution[0: num_nodes], -solution[num_nodes: 2*num_nodes])
print(f'Time taken: {time.time() - start:.2f} seconds')

