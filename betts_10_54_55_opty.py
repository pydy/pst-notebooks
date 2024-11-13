  # %%
"""
Maximum Range of a Hang Glider
==============================

This is example 10.54 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

**States**

- :math:`x, y, v_x, v_y` : state variables

**Controls**

- :math:`C_L` : control variable


Note:
While these equations of motion may look harmless, they are not, as is
stated in section 8.5 of the a.m. book - and is evident here, too. While I do
get results close to those in the book, opty never converges, no matter what
I tried.

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

x, y, vx, vy = me.dynamicsymbols('x, y, vx, vy')
CL = me.dynamicsymbols('CL')
h = sm.symbols('h')

# Parameters
uM = 2.5
R = 100.0
C0 = 0.034
k = 0.069662
vbx = 13.227567500
m = 100.0
S = 14.0
rho = 1.13
g = 9.80665
vby = -1.2875005200

CD = C0 + k*CL**2
X = (x/R - 2.5)**2
Vy = vy - uM*(1 - X)*sm.exp(-X)
vr = sm.sqrt(vx**2 + Vy**2)
D = 0.5*CD*rho*S*vr**2
L = 0.5*CL*rho*S*vr**2

sinnu = Vy/vr
cosnu = vx/vr

# equations of motion.
eom = sm.Matrix([
    -x.diff(t) + vx,
    -y.diff(t) + vy,
    -vx.diff(t) - (D*cosnu + L*sinnu)/m,
    -vy.diff(t) + (-D*sinnu + L*cosnu)/m - g,
])
sm.pprint(eom)

# %%
# Define and solve the optimization problem.
num_nodes = 1501
iterations = 3000

t0, tf = 0*h, (num_nodes - 1) * h
interval_value = h

state_symbols = (x, y, vx, vy)
unkonwn_input_trajectories = (CL, )

# Specify the objective function and form the gradient.
def obj(free):
    return -free[num_nodes-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[num_nodes-1] = -1.0
    return grad

instance_constraints = (
    x.func(t0),
    y.func(t0) - 1000.0,
    vx.func(t0) - vbx,
    vy.func(t0) - vby,
    y.func(tf) - 900.0,
    vx.func(tf) - vbx,
    vy.func(tf) - vby,
)

bounds = {
        h: (0.0, 0.5),
        CL: (0, 1.4),
}

# %%
# Create the optimization problem and set any options.
prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
        bounds=bounds,
#        integration_method='midpoint',
    )

#prob.add_option('nlp_scaling_method', 'gradient-based')
prob.add_option('max_iter', iterations)

# Give some rough estimates for the trajectories.
initial_guess = np.ones(prob.num_free) * 0.1
s1 = list(np.linspace(0.0, 1250, num_nodes))
s2 = list(np.linspace(1000, 900, num_nodes))
s3 = list(np.linspace(vbx, vbx, num_nodes))
s4 = list(np.linspace(vby, vby, num_nodes))
s5 = list(np.linspace(0.0, 1.0, num_nodes))
initial_guess = np.array(s1 + s2 + s3 + s4 + s5 + [100/num_nodes])

# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)

    print(info['status_msg'])
    print(f'Objective value achieved: {-info['obj_val']:.4f}, ' +
          f'as per the book it is {1248.03103}, so the error is: ' +
        f'{(-info['obj_val'] - 1248.03103)/1248.03103 :.3e} ')

    print(f'optimum time is {(num_nodes-1)*solution[-1]:.3f}' +
          f' as per the book it is 98.436940, so the error is: ' +
          f'{((num_nodes-1)*solution[-1] - 98.436940)/98.436940 :.3e} ')

solution1 = solution

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

print(f'Time taken: {time.time() - start:.2f} seconds')

# =============================================================================
# =============================================================================
# %%
# Example 10.55 from Betts' book "Practical Methods for Optimal Control Using
# Nonlinear Programming", 3rd edition, Chapter 10: Test Problems.
# This is equivalent to problem 10.54, but formulated differently.
#
# **states**
#
# - :math:`x, y, v_x, v_y, t_f` : state variables
#
# **controls**
#
# - :math:`C_L` : control variable
#
# Equations of motion
# -------------------
start = time.time()
t = me.dynamicsymbols._t

tf = me.dynamicsymbols('tf')

eom = sm.Matrix([
    -x.diff(t) + tf*vx,
    -y.diff(t) + tf*vy,
    -vx.diff(t) - tf*(D*cosnu + L*sinnu)/m,
    -vy.diff(t) + tf*(-D*sinnu + L*cosnu)/m - tf*g,
    -tf.diff(t),
])
sm.pprint(eom)

# %%
# Define and solve the optimization problem.

state_symbols = (x, y, vx, vy, tf)
t0, tend = 0.0, 1.0
interval_value = tend / (num_nodes - 1)

# Specify the objective function and form the gradient.
def obj(free):
    return -free[num_nodes - 1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[num_nodes-1] = -1.0
    return grad

instance_constraints = (
    x.func(t0),
    y.func(t0) - 1000.0,
    vx.func(t0) - vbx,
    vy.func(t0) - vby,
    y.func(tend) - 900.0,
    vx.func(tend) - vbx,
    vy.func(tend) - vby,
)

bounds = {
        tf: (0.0, 120),
        CL: (0, 1.4),
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

#prob.add_option('nlp_scaling_method', 'gradient-based')
prob.add_option('max_iter', iterations)

# I use the results from above as initial guess.
s1 = list(solution1[:4*num_nodes])
s2= [solution1[-1]*(num_nodes-1)  for _ in range(num_nodes)]
s3 = list(solution1[-num_nodes-1:-1])
initial_guess = np.array(s1 + s2 + s3)


# %%
# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {-info['obj_val']:.4f}, ' +
          f'as per the book it is {1248.03103}, so the error is: ' +
        f'{(-info['obj_val'] - 1248.03103)/1248.03103:.3e}')

    print(f'Final time: {solution[5*num_nodes-1]}' +
          f' as per the book it is 98.436940, so the error is: ' +
          f'{(solution[5*num_nodes-1] - 98.436940)/98.436940 :.3e} '
    )

# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

print(f'Time taken: {time.time() - start:.2f} seconds')

# %%
# Compare the two solutions.
difference = np.empty(4*num_nodes)
for i in range(4*num_nodes):
    difference[i] = solution1[i] - solution[i]

max_x = np.max(np.abs(solution1[0:num_nodes]))
max_y = np.max(np.abs(solution1[num_nodes:2*num_nodes]))
max_vx = np.max(np.abs(solution1[2*num_nodes:3*num_nodes]))
max_vy = np.max(np.abs(solution1[3*num_nodes:4*num_nodes]))
max_list = [max_x, max_y, max_vx, max_vy]

fig, ax = plt.subplots(4, 1, figsize=(8, 6), constrained_layout=True)
names = ['x', 'y', 'v_x', 'v_y']
times = np.linspace(t0, (num_nodes-1)*solution1[-1], num_nodes)
for i in range(4):
    ax[i].plot(times, difference[i*num_nodes:(i+1)*num_nodes]/max_list[i])
    ax[i].set_ylabel(f'difference in {names[i]}')
ax[0].set_title('Relative difference in the state variables between the two solutions')
ax[-1].set_xlabel('time');
