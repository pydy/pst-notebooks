 # %%
"""
Underwater Vehicle
==================

This is example 10.6 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

**States**

- $y_1, ..., y_{10]}$ : state variables

**Specifieds**

- $u_1, ..., u_4$ : control variables

Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their meaning.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import time

# %%
# Equations of motion.
t = me.dynamicsymbols._t
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = me.dynamicsymbols('y1 y2 y3 y4 y5 y6 y7 y8 y9 y10')
u1, u2, u3, u4 = me.dynamicsymbols('u1 u2 u3 u4')

uy1 = y1.diff(t)
uy2 = y2.diff(t)
uy3 = y3.diff(t)
uy4 = y4.diff(t)
uy5 = y5.diff(t)
uy6 = y6.diff(t)
uy7 = y7.diff(t)
uy8 = y8.diff(t)
uy9 = y9.diff(t)
uy10 = y10.diff(t)

cx = 0.5
rx = 0.1
ux = 2.0
cz = 0.1
uz = 0.1

E = sm.exp(-((y1 - cx)/rx)**2)
Rx = -ux*E*(y1-cx)*((y3 - cz)/cz)**2
Rz = -uz*E*((y3-cz)/cz)**2

eom = sm.Matrix([-uy1 + y7*sm.cos(y6)*sm.cos(y5) + Rx,
                -uy2 + y7* sm.sin(y6) * sm.cos(y5),
                -uy3 - y7*sm.sin(y5) + Rz,
                -uy4 + y8 +y9*sm.sin(y4)*sm.tan(y5) + y10*sm.cos(y4)*sm.tan(y5),
                -uy5 + y9*sm.cos(y4) - y10*sm.sin(y4),
                -uy6 + y9*sm.sin(y4)/sm.cos(y5) + y10*sm.cos(y4)/sm.cos(y5),
                -uy7 + u1,
                -uy8 + u2,
                -uy9 + u3,
                -uy10 + u4,
])
sm.pprint(eom)

# %%

def solve_optimization(nodes, tf, iterations):
    t0, tf = 0.0, tf
    num_nodes = nodes
    interval_value = (tf - t0)/(num_nodes - 1)

    state_symbols = (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10)
    specified_symbols = (u1, u2, u3, u4)

    # Specify the objective function and form the gradient.
    start = time.time()
    obj_func = sm.Integral(u1**2 + u2**2 + u3**2 + u4**2, t)
    sm.pprint(obj_func)
    obj, obj_grad = create_objective_function(obj_func,
                                          state_symbols,
                                          specified_symbols,
                                          tuple(),
                                          num_nodes,
                                          node_time_interval=interval_value)

    # Specify the symbolic instance constraints, as per the example
    instance_constraints = (
        y1.func(t0),
        y2.func(t0),
        y3.func(t0) - 0.2,
        y4.func(t0) - np.pi/2,
        y5.func(t0) - 0.1,
        y6.func(t0) + np.pi/4,
        y7.func(t0) - 1.0,
        y8.func(t0),
        y9.func(t0) - 0.5,
        y10.func(t0) - 0.1,

        y1.func(tf) - 1.0,
        y2.func(tf) - 0.5,
        y3.func(tf),
        y4.func(tf) - np.pi/2,
        y5.func(tf),
        y6.func(tf), 
        y7.func(tf),
        y8.func(tf),
        y9.func(tf),
        y10.func(tf),
    )

    bounds = {u1: (-15.0, 15.0),
            u2: (-15.0, 15.0),
            u3: (-15.0, 15.0),
            u4: (-15.0, 15.0),
            y4: (np.pi/2 - 0.02, np.pi/2 + 0.02),
    }

    # Create the optimization problem and set any options.
    prob = Problem(obj,
                   obj_grad,
                   eom,
                   state_symbols,
                   num_nodes,
                   interval_value,
                   instance_constraints=instance_constraints,
                   bounds=bounds,
    )

    #prob.add_option('nlp_scaling_method', 'gradient-based')

    # Give some rough estimates for the trajectories.
    initial_guess = np.zeros(prob.num_free)

    # Find the optimal solution.
    for i in range(iterations):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
        print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
            f'it is {236.528}, so the error is: '
            f'{(info['obj_val'] - 236.528)/236.528*100:.3f} % ')
        print(f'Time taken for the simulation: {time.time() - start:.2f} s')

    # Plot the optimal state and input trajectories.
    prob.plot_trajectories(solution)

    # Plot the constraint violations.
    prob.plot_constraint_violations(solution)

    # Plot the objective function as a function of optimizer iteration.
    prob.plot_objective_value()

# %%
# As per the example tf = 1.0
tf = 1.0
num_nodes = 101
iterations = 1
solve_optimization(num_nodes, tf, iterations)
