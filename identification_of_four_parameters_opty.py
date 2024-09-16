 # %%
"""

Parameter Identification with Non-Contiguous Measurements.
==========================================================

for parameter estimation it is common to collect measurements of a system's
trajectories for distinct experiments. for example, if you are identifying the
parameters of a mass-spring-damper system, you will exite the system with
different initial conditions multiple times. The date cannot simply be stacked
and the identification run because the measurement data would be discontinuous
between trials.
A work around in opty is to create a set of differential equations with unique
state variables for each measurement trial that all share the same constant
parameters. You can then identify the parameters from all measurement trials
simultaneously by passing the uncoupled differential equations to opty.

For exaple:
Four measurements of the location of a simple system consisting of a mass
connected to a fixed point by a spring and a damper, with a
force = :math:`F_1 \sin(\omega \cdot t)` acting on the mass. The movement is in a
horizontal direction. :math:`c, F_1, k, \omega` are to be identified.


**State Variables**

- :math:`x_i`: position of the mass of the i - th measurement trial [m]
- :math:`u_i`: speed of the mass of the i - th measurement trial [m/s]

**Parameters**

- :math:`m`: mass for both systems system [kg]
- :math:`c`: damping coefficient for both systems [Ns/m]
- :math:`k`: spring constant for both systems [N/m]
- :math:`l_0`: natural length of the spring [m]
- :math:`F_1`: amplitude of the force [N]
- :math:`\omega`: frequency of the force [rad/s]

"""
# %%
# Set up the equations of motion and integrate them to get the measurements.
#
import sympy as sm
import numpy as np
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from opty import Problem

number_of_measurements = 15
t0, tf = 0, 10
num_nodes = 400
times = np.linspace(t0, tf, num_nodes)
t_span = (t0, tf)
np.random.seed(1234)

x = me.dynamicsymbols(f'x:{number_of_measurements}')
u = me.dynamicsymbols(f'u:{number_of_measurements}')
fx = me.dynamicsymbols('f_x')
m, c, k, l0 = sm.symbols('m, c, k, l0')
F1, omega = sm.symbols('F1, omega')
t = me.dynamicsymbols._t
T = sm.symbols('T', cls=sm.Function)

eom1 = sm.Matrix([x[i].diff(t) - u[i] for i in range(number_of_measurements)])
eom2 = sm.Matrix([m*u[i].diff(t) + c*u[i] + k*(x[i] - l0) - F1*sm.sin(omega*T(t))
        for i in range(number_of_measurements)])
eom = eom1.col_join(eom2)
# %%
# Print the equations of motion.
#
sm.pprint(eom)
# %%
# Equations of motion for the solve_ivp integration.
#
rhs1 = np.array([u[i] for i in range(number_of_measurements)])
rhs2 = np.array([1/m * (-c*u[i] - k*(x[i] - l0) + F1*sm.sin(omega*t))
    for i in range(number_of_measurements)])
rhs = np.concatenate((rhs1, rhs2))

qL = x + u
pL = [m, c, k, l0, F1, omega, t]

rhs_lam = sm.lambdify(qL + pL, rhs)

def gradient(t, x, args):
    args[-1] = t
    return rhs_lam(*x, *args).reshape(2*number_of_measurements)


x0 = np.array([3 + i  for i in range(number_of_measurements)] + [0 for i in range(number_of_measurements)])
pL_vals = [1.0, 0.25, 2.0, 1.0, 6.0, 3.0, t0]

resultat1 = solve_ivp(gradient, t_span, x0, t_eval = times, args=(pL_vals,))
resultat = resultat1.y.T

# %%
# Create the measurements: simply add noise to the results of the integration.
measurements = []

for i in range(number_of_measurements):
    measurements.append(resultat[:, i] + np.random.randn(resultat.shape[0]) *
        1.0/(i+1) + np.random.randn(1)*1.0/(i+1))

# %%
# Set up the Estimation Problem.
# --------------------------------

# If some measurement is considered more reliable, its weight w can be increased.
#
# objective = :math:`\int_{t_0}^{t_f} \left[ \sum_{i=1}^{i = text{number_of_measurements}} (w_i (x_i - x_i^m)^2 \right], dt`
#
state_symbols = x + u

interval_value = (tf - t0) / (num_nodes - 1)

par_map = {m: pL_vals[0],
        l0: pL_vals[3],
}

w = [1 + i for i in range(number_of_measurements)]

def obj(free):
    return interval_value * np.sum([w[i] * np.sum(
        (free[i*num_nodes:(i+1)*num_nodes] - measurements[i])**2)
        for i in range(number_of_measurements)]
)

def obj_grad(free):
    grad = np.zeros_like(free)
    for i in range(number_of_measurements):
        grad[i*num_nodes: (i+1)*num_nodes] = 2*w[i]*interval_value*(
            free[i*num_nodes:(i+1)*num_nodes] - measurements[i]
        )
    return grad

bounds = {
    c: (0, 1),
    k: (1, 3),
    F1: (5, 10),
    omega: (2, 7),
}

problem = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    known_trajectory_map={T(t): np.linspace(t0, tf, num_nodes)},
    bounds=bounds,
    time_symbol=me.dynamicsymbols._t,
)
# This gives the unknown parameters, and their sequence in the solution vector.
print('unknown parameters are:', problem.collocator.unknown_parameters)
# %%
# Initial guess.
#
list1 = [list(measurements[i])for i in range(number_of_measurements)]
list1 = list(np.array(list1).flat)
initial_guess = np.array(list1
    + list(np.zeros(number_of_measurements*num_nodes))
#    + list(np.zeros(num_nodes))
    + [0, 0, 0, 0])

# %%
# Solve the Optimization Problem.
#
solution, info = problem.solve(initial_guess)
print(info['status_msg'])
problem.plot_objective_value()
# %%
problem.plot_constraint_violations(solution)
# %%
# Results obtained.
#------------------
#
print(f'Estimate of dampening constant is {solution[-3]:.2f} ')
print(f'Estimate of spring constant is    {solution[-2]:.2f} ')
print(f'Estimate of force is              {solution[-4]:.2f} ')
print(f'Estimate of the frequency is      {solution[-1]:.2f} ')

# %%
# Plot the measurements.
fig, ax = plt.subplots(number_of_measurements, 1, figsize=(8,
            2*number_of_measurements), sharex=True)
for i in range(number_of_measurements):
    ax[i].plot(times, measurements[i])
    ax[i].set_ylabel(f'Measurement {i+1} [m]')
ax[0].set_title('Measurements')
ax[-1].set_xlabel('Time [sec]');

plt.show()