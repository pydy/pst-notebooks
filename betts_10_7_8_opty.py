 # %%
"""
Hypersensitive Control
======================

This is example 10.7 from Betts' Test Problems.
Note: Below is example 10.8, which is mathematically EXACTLY the same, but the
formulationis different. this has a big impact on the optimization.

**States**

- y : state variable
- uy: its speed

**Specifieds**

- u : control variable

Note: the state variable uy is only needed because opt needs minimum two
differential equations in the equations of motion.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function, parse_free
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# %%
# Equations of motion.
t = me.dynamicsymbols._t
y, uy, u = me.dynamicsymbols('y uy, u')

eom = sm.Matrix([ uy - y.diff(t), -uy - y**3 + u])
sm.pprint(eom)

# %%

def solve_optimization(nodes, tf):
    t0, tf = 0.0, tf
    num_nodes = nodes
    interval_value = (tf - t0)/(num_nodes - 1)
    #times = np.linspace(t0, tf, num=num_nodes)


    # Provide some reasonably realistic values for the constants.

    state_symbols = (y, uy)
    specified_symbols = (u,)


    # Specify the objective function and form the gradient.
    start = time.time()
    obj_func = sm.Integral(y**2 + u**2, t)
    sm.pprint(obj_func)
    obj, obj_grad = create_objective_function(obj_func,
                                          state_symbols,
                                          specified_symbols,
                                          tuple(),
                                          num_nodes,
                                          node_time_interval=interval_value)


    # Specify the symbolic instance constraints, as per the example
    instance_constraints = (
    #    y.func(t0),
    #    y.func(tf),
        y.func(t0) - 1,
        y.func(tf) - 1.5,
    )


    # Create the optimization problem and set any options.
    prob = Problem(obj, obj_grad, eom, state_symbols,
               num_nodes, interval_value,
               instance_constraints=instance_constraints,
    )

    prob.add_option('nlp_scaling_method', 'gradient-based')

    # Give some rough estimates for the x and y trajectories.
    initial_guess = np.zeros(prob.num_free)
    prob.plot_trajectories(initial_guess)

    # Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
        f'it is {6.7241}, so the error is: '
        f'{(info['obj_val'] - 6.7241)/6.7241*100:.3f} % ')
    print(f'Time taken for the simulation: {time.time() - start:.2f} s')

    # Plot the optimal state and input trajectories.
    prob.plot_trajectories(solution)

    # Plot the constraint violations.
    #fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    prob.plot_constraint_violations(solution)

    # Plot the objective function as a function of optimizer iteration.
    prob.plot_objective_value()

# %%
# As per the example tf = 10000
tf = 10000
num_nodes = 5001
solve_optimization(num_nodes, tf)

# %%
# As per the plot of the solution y(t) it seems, that most of the time y(t) = 0,
# only at the very beginning and the very end it is different from 0.
# So, it may make sense to use a smaller  tf
tf = 10
num_nodes = 5001
solve_optimization(num_nodes, tf)


# %%
"""
Hypersensitive Control
======================

This is example 10.8 from Betts' Test Problems.
Mathematically it is EXACTLY the same as the previous example, only
the formulation is different.
The comparison to example 10.7 above shows, that different formulations of the
same problem, may have drastic influences of the optimization.

**States**

- y, z : state variables
- uy, uz: their speeds

**Specifieds**

- u : control variable
"""

# %%
# Equations of motion.
t = me.dynamicsymbols._t
y, z, uy, uz, u = me.dynamicsymbols('y z uy uz u')

eom = sm.Matrix([uy - y.diff(t), uz - z.diff(t), uy + y**3 - u, uz - y**2 - u**2])
sm.pprint(eom)

# %%

def solve_optimization(nodes, tf):
    t0, tf = 0.0, tf
    num_nodes = nodes
    interval_value = (tf - t0)/(num_nodes - 1)
    #times = np.linspace(t0, tf, num=num_nodes)


    # Provide some reasonably realistic values for the constants.

    state_symbols = (y, z, uy, uz)
    specified_symbols = (u,)


    # Specify the objective function and form the gradient.
    start = time.time()
    obj_func = sm.Integral(uz, t)
    sm.pprint(obj_func)
    obj, obj_grad = create_objective_function(obj_func,
                                          state_symbols,
                                          specified_symbols,
                                          tuple(),
                                          num_nodes,
                                          node_time_interval=interval_value)


    # Specify the symbolic instance constraints, as per the example
    instance_constraints = (
    #    uy.func(t0),
        y.func(t0) - 1,
        y.func(tf) - 1.5,
    )


    # Create the optimization problem and set any options.
    prob = Problem(obj, obj_grad, eom, state_symbols,
               num_nodes, interval_value,
               instance_constraints=instance_constraints,
    )

    prob.add_option('nlp_scaling_method', 'gradient-based')

    # Give some rough estimates for the x and y trajectories.
    initial_guess = np.zeros(prob.num_free)
    prob.plot_trajectories(initial_guess)

    # Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
        f'it is {6.7241}, so the error is: '
        f'{(info['obj_val'] - 6.7241)/6.7241*100:.3f} % ')

    print(f'Time taken for the simulation: {time.time() - start:.2f} s')

    # Plot the optimal state and input trajectories.
    prob.plot_trajectories(solution)

    # Plot the constraint violations.
    #fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    prob.plot_constraint_violations(solution)

    # Plot the objective function as a function of optimizer iteration.
    prob.plot_objective_value()

# %%
# As we konw from above, that tf = 10 gives better results, we use this value
# here only.
tf = 10
num_nodes = 5001
solve_optimization(num_nodes, tf)