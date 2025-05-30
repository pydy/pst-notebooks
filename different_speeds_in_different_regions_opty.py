# %%

r"""
Different Speeds in Different Regions
=====================================

Objectives
----------

- Show how to use inequalities on the equations of motion.
- Show how to use differentiable functions to approximate their non-
  differentiable counterparts.
- Show that values of the parameter map may be changed in a loop without having
  to build ``Problem`` again.

Introduction
------------

A particle should move from the origin to the final point in minimum time.
The maximum speed is limited to different values in different regions.
This is accomplished by using a smooth approximation of a 'step' function in
connection with inequality constraints on the equations of motion.

Notes
-----

The example shows how a local minimum may be converged to which is not even
close to the global minimum. Depending on the initial guess, the path will
partially cross the disc, while it is faster to go around it.


**States**

- :math:`x, y` : position of the particle
- :math:`u_x, u_y` : velocity of the particle


**Controls**

- :math:`f_x, f_y` : force applied to the particle

**Parameters**

- :math:`m` : mass of the particle [kg]
- :math:`x_0, y_0` : center of the circle [m]
- :math:`\textrm{radius}` : radius of the circle [m]
- :math:`\mu_1, \mu_2` : maximum speed in the regions [m/s]
- :math:`\textrm{steepness}` : steepness of the step function

"""

import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib as mp
from IPython.display import HTML
mp.rcParams['animation.embed_limit'] = 2**128

# %%
# Define the smooth hump function.


def smooth_step(x, a, k):
    """gives approx 1.0 for a < x, 0.0 otherwise.
    The larger k, the steeper the transition"""
    return 0.5 * (sm.tanh(k * (x - a)) + 1.0)


# %%
# Set Up the Equations of Motion, Kane's Method
# ---------------------------------------------
N = me.ReferenceFrame('N')
O, P = sm.symbols('O P', cls=me.Point)
P.set_vel(N, 0)
t = me.dynamicsymbols._t

x, y, ux, uy = me.dynamicsymbols('x y u_x u_y')
fx, fy = me.dynamicsymbols('f_x f_y')

m, x0, y0, radius = sm.symbols('m, x_0, y_0, radius')
mu1, mu2 = sm.symbols('mu_1 mu_2')
steepness = sm.symbols('steepness')
friction = sm.symbols('friction')

P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)
bodies = [me.Particle('P', P, m)]

forces = [(P, fx * N.x + fy * N.y - ux * friction * N.x - uy * friction * N.y)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

KM = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
# %%
# Add the eoms to bound the speeds in the different regions.
ort = (x - x0)**2 + (y - y0)**2
radius1 = radius**2
step1 = smooth_step(ort, radius1, steepness)  # outside the circle if = 1.0
speed_mag = sm.sqrt(ux**2 + uy**2)

eom1 = sm.Matrix([
    speed_mag * step1,
    speed_mag * (1 - step1),
    speed_mag,
])
eom = eom.col_join(eom1)

# %%
# Bound the magnitude of the driving force, but not its direction.
eom = eom.col_join(sm.Matrix([sm.sqrt(fx**2 + fy**2)]))
print((f'The equations of motion have {sm.count_ops(eom)} operations and have'
      f' shape {eom.shape}'))
eom
# %%
# Set Up the Optimization Problem
# -------------------------------
h = sm.symbols('h')
interval = h
num_nodes = 501
t0, tf = 0.0, h * (num_nodes - 1)

state_symbols = [x, y, ux, uy]

par_map = {
    m: 1.0,
    x0: 5.0,
    y0: 5.0,
    radius: 4.5,
    mu1: 5.0,
    mu2: 1.5,
    steepness: 60.0,
    friction: 1.0,
}

# %%
# Plot the step function.
a, xt, k, xx = sm.symbols('a xt k, xx')
eval_step = sm.lambdify((xt, a, k), smooth_step(xt, a, k))
x_vals = np.linspace(-0.2, 0.2, 100)
k = par_map[steepness]
epsilon = 1.e-2
loesung = fsolve(lambda xx: eval_step(xx, 0.0, k) - epsilon, 0.0)

fig, ax = plt.subplots(figsize=(6.5, 1.5), layout='constrained')
ax.plot(x_vals, eval_step(x_vals, 0.0, k))
ax.axvline(0.0, color='black', lw=0.5)
ax.axhline(0.0, color='black', lw=0.5)
ax.axhline(1.0, color='black', lw=0.5)
ax.axvline(loesung[0], color='red', lw=0.5)
ax.axvline(-loesung[0], color='red', lw=0.5)
_ = ax.set_title((f"Smooth step function with steepness = {k} \n The red "
                  f"lines are the points where \n the function is closer than "
                  f"{epsilon:.1e} to 0.0 or 1.0"))

# %%
# Build ``Problem``.


def obj(free):
    """minimize the variable time interval."""
    return free[-1]


def obj_grad(free):
    """Gradient of the objective function."""
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


instance_constraints = (
    x.func(t0) - 0.0,
    y.func(t0) - 0.0,
    x.func(tf) - 10.0,
    y.func(tf) - 10.0,
)

bounds = {
    h: (0.0, 1.0),
    x: (0.0, 10.0),
    y: (0.0, 10.0),
    ux: (0.0, np.inf),
    uy: (0.0, np.inf),
}

limit = 400
eom_bounds = {
    4: (0.0, par_map[mu1]),
    5: (0.0, par_map[mu2]),
    6: (0.0, max(par_map[mu1], par_map[mu2])),
    7: (0.0, limit),
}

prob = Problem(
    obj,
    obj_grad,
    eom, state_symbols,
    num_nodes,
    interval,
    instance_constraints=instance_constraints,
    known_parameter_map=par_map,
    bounds=bounds,
    eom_bounds=eom_bounds,
    time_symbol=t,
)

prob.add_option('max_iter', 35000)

# %%
# Use existing solution if available, else solve the problem.
fname = (f'different_speeds_in_different_regions_opty{num_nodes}_nodes_'
         f'solution.csv')

# Solve the problem. Pick a reasonable initial guess.
initial_guess = np.ones(prob.num_free) * 0.5
x_values = np.linspace(0, 10, num_nodes)
y_values = np.linspace(0, 10, num_nodes)
initial_guess[:num_nodes] = x_values
initial_guess[num_nodes:2*num_nodes] = y_values

if os.path.exists(fname):
    initial_guess = np.loadtxt(fname)
    for _ in range(3):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
        print(f'Iterations needed: {len(prob.obj_value)}')
else:
    it0 = 0
    for i in range(6):
        # Values of par_map may be changed in a loop without having to build
        # Problem again.
        par_map[steepness] = 10 + 10 * i

        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
        it1 = len(prob.obj_value)
        print('Iterations needed', it1 - it0)
        it0 = it1
    _ = prob.plot_objective_value()
    print(f'Total iterations needed were: {len(prob.obj_value)}')

# %%
# Plot trajectories.
_ = prob.plot_trajectories(solution, show_bounds=True)
np.savetxt(fname, solution, fmt='%.12f')
# %%
# PLot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# Animate the Motion
# ------------------
fps = 15

state_vals, input_vals, _, h_val = prob.parse_free(solution)
tf = h_val * (num_nodes - 1)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

# create additional point for the speed vector
arrow_head = sm.symbols('arrow_head', cls=me.Point)
arrow_head.set_pos(P, ux * N.x + uy * N.y)

coordinates = P.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(arrow_head.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
coords_lam = sm.lambdify((*state_symbols, fx, fy, *pL), coordinates,
                         cse=True)


def init():
    # needed to give the picture the right size.
    xmin, xmax = -1.0, 11.
    ymin, ymax = -1.0, 11.

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid()

    ax.axvline(xmin, color='black', lw=1)
    ax.scatter(0.0, 0.0, color='red', s=20)
    ax.scatter(10.0, 10.0, color='green', s=20)

    circle = plt.Circle((par_map[x0], par_map[y0]), par_map[radius],
                        color='grey')
    ax.add_patch(circle)
    # Initialize the block
    line1 = ax.scatter([], [], color='blue', s=100)
    line2, = ax.plot([], [], color='red', lw=0.5)
    pfeil = ax.quiver([], [], [], [], color='green', scale=25,
                      width=0.004, headwidth=8)

    return fig, ax, line1, line2, pfeil


fig, ax, line1, line2, pfeil = init()


def update(t):
    message = (f'running time {t:.2f} sec \n The speed is the green arrow \n '
               f'Speed in the disk is {par_map[mu2]} m/s, speed outside the '
               f'disk is {par_map[mu1]} m/s')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pL_vals)

    koords_x = []
    koords_y = []
    for t1 in np.linspace(t0, tf, int(fps * (tf - t0))):
        if t1 <= t:
            coords = coords_lam(*state_sol(t1), *input_sol(t1), *pL_vals)
            koords_x.append(coords[0, 0])
            koords_y.append(coords[1, 0])
    line2.set_data(koords_x, koords_y)
    line1.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(coords[0, 1] - coords[0, 0], coords[1, 1] - coords[1, 0])


frames = np.linspace(t0, tf, int(fps * (tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
display(HTML(animation.to_jshtml()))
plt.close(fig)
