# %%
r"""
Pendulum with Explicit Times
============================

Objectives
----------

- Show how to handle an explicit occurance of time in the equations of motion,
  which opty presently cannot handle directly.
- Show how in a special case the current opty limitation about instance
  constraints not allowing ranges may be overcome.


Introduction
------------

The task is to get a pendulum which is hanging straight down to swing an come
rest within a specified sector. A torque of the form
:math:`\textrm{driving}_{\textrm{torque}} = t \cdot F`
is applied to the pendulum. As opty presently does not support explicit time
in the equations of motion, one has to introduce an additional state
variable, :math:`T` and set :math:`\dfrac{dT}{dt} = 1` in the equations of
motion. Setting an instance constraint :math:`T(t_0) = 0` will ensure that
:math:`T` is equal to the time at any point in time.

In this special case, it is clear, that the pendulum will come to rest at one
of the edges of the sector. So,

- a state variable :math:`\textrm{window}` is introduced which is equal to the
  minimum of the distance of :math:`q` to the upper and lower edge of the
  sector respectively.
- The instance constraint :math:`\textrm{window}(t_f) = 0` is set, which means
  that the pendulum will come to rest at one of the edges of the sector.


**States**

- :math:`q` - angle of the pendulum
- :math:`u` - angular velocity of the pendulum
- :math:`T` - time
- :math:`\textrm{window}` - auxiliary state, see above

**Known Parameters**

- :math:`m_p` - mass of the pendulum [kg]
- :math:`l_e` - length of the pendulum [kg]
- :math:`i_{ZZ}` - moment of inertia of the pendulum [kg m^2]
- :math:`g` - gravitational acceleration [m/s^2]
- :math:`\nu` - damping coefficient [kg m^2/s]
- :math:`\textrm{lower}` - lower edge of the sector [rad]
- :math:`\textrm{upper}` - upper edge of the sector [rad]

**Specifieds**

- :math:`F` - driving force [N]

**unknown Parameters**

- :math:`h` - time step [s]


"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem

from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib as mp
mp.rcParams['animation.embed_limit'] = 2**128

# %%
# Set Up the Equations of Motion
# -------------------------------


def min_diff(a, b, epsilon):
    """
    Returns the minimum difference between two reals a and b.
    """
    return 0.5 * (a + b - sm.sqrt((a - b)**2 + epsilon))


N, A = sm.symbols('N A', cls=me.ReferenceFrame)
O, P = sm.symbols('O P', cls=me.Point)
O.set_vel(N, 0)
t = me.dynamicsymbols._t

h = sm.symbols('h')

q, u = me.dynamicsymbols('q u')
T, F = me.dynamicsymbols('T, F')
window = me.dynamicsymbols('window')
upper, lower = sm.symbols('upper lower')

mp, g, le, iZZ, nu = sm.symbols('m_p, g, le, i_ZZ, nu')

A.orient_axis(N, q, N.z)
A.set_ang_vel(N, u * N.z)

P.set_pos(O, -le * A.y)
P.v2pt_theory(O, N, A)

inert = me.inertia(A, 0, 0, iZZ)
pendulum = me.RigidBody('pendulum', P, A, mp, (inert, P))
bodies = [pendulum]

driving_torque = T * F
forces = [(A, driving_torque * N.z - u * nu * A.z), (P, -mp * g * N.y)]

kd = sm.Matrix([u - q.diff(t)])

KM = me.KanesMethod(N,
                    q_ind=[q],
                    u_ind=[u],
                    kd_eqs=kd)

fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(sm.Matrix([T.diff(t) - 1]))

eom = eom.col_join(sm.Matrix([window - min_diff((q - lower)**2,
                                                (q - upper)**2, 1.e-4)]))
eom
# %%
# Set Up the Problem and Solve it
# -------------------------------

state_symbols = [q, u, T, window]

num_nodes = 301
t0, tf = 0.0, h * (num_nodes - 1)
interval_value = h

par_map = {}
par_map[mp] = 1.0
par_map[g] = 9.81
par_map[le] = 1.0
par_map[iZZ] = 1.0
par_map[nu] = 0.1
par_map[lower] = 7/8*np.pi
par_map[upper] = -6/8*np.pi

bounds = {
    h: (0.0, 0.5),
    F: (-0.5, 0.5),
}

instance_constraints = (
    q.func(t0) - 0.0,
    u.func(t0) + 0.0,
    T.func(t0) - 0.0,
    window.func(tf) - 0.0,
    u.func(tf) - 0.0,
)


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds,
    backend='numpy'
    )

initial_guess = np.ones(prob.num_free) * 0.1
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
_ = prob.plot_objective_value()

# %%
# Plot the violations of the constraints.
_ = prob.plot_constraint_violations(solution, subplots=True)
# %%
# Plot the trajectories.
fig, axes = plt.subplots(6, 1, figsize=(6.5, 6.5), layout='constrained',
                         sharex=True)
prob.plot_trajectories(solution, show_bounds=True, axes=axes)
sol, input, constant_values, _ = prob.parse_free(solution)
driving_torque_lam = sm.lambdify((*state_symbols, F), driving_torque,
                                 cse=True)

driving_torque_values = []
for i in range(num_nodes):
    driving_torque_values.append(driving_torque_lam(*sol[:, i], input[i]))
times = prob.time_vector(solution)
axes[-1].plot(times, driving_torque_values)
axes[-1].set_ylabel('Driving Torque')
axes[-1].set_xlabel('Time [s]')
_ = axes[-1].set_title('Actual Driving Torque')


# %%
# Animation
# ---------
fps = 20

state_vals, input_vals, constant_values, h_vals = prob.parse_free(solution)
t_arr = prob.time_vector(solution)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

act_torque = [state_vals[2, i] * input_vals[i] for i in range(num_nodes)]
torque_sol = CubicSpline(t_arr, act_torque)

pL, pL_vals = zip(*par_map.items())
coordinates = P.pos_from(O).to_matrix(N)
coords_lam = sm.lambdify((*state_symbols, F, *pL), coordinates,
                         cse=True)


# sphinx_gallery_thumbnail_number = 3
def init():
    xmin, xmax = -par_map[le] - 0.5, par_map[le] + 0.5
    ymin, ymax = -par_map[le] - 0.5, par_map[le] + 0.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid()
    ax.scatter(0.0, 0.0, color='black', marker='o', s=100)  # origin O

    # draw the sector where the pendulum should come to a rest
    XX = np.linspace(0.0, xmax, 200)
    YY = XX * np.tan(par_map[lower] + np.pi/2)
    #ax.plot(XX, YY, color='red', lw=0.25)
    YY1 = -XX * np.tan(par_map[upper] + np.pi/2.0)
    #ax.plot(-XX, YY, color='red', lw=0.25)
    oben = np.linspace(ymax, ymax, 200)
    ax.fill_between(XX, YY, oben, color='red', alpha=0.2, interpolate=True)
    ax.fill_between(-XX, YY1, oben, color='red', alpha=0.2, interpolate=True)

    line1 = ax.scatter([], [], color='blue', marker='o', s=100)   # point P
    line2, = ax.plot([], [], color='magenta', lw=1)  # connecting line
    arrow = ax.quiver([], [], [], [], color='green', scale=15,
                      width=0.004, headwidth=8)  # torque arrow
    return fig, ax, line1, line2, arrow


# Function to update the plot for each animation frame
fig, ax, line1, line2, arrow = init()


def update(t):
    message = ((f'Running time {t:.2f} sec \n The green arrow corresponds to '
                f'the driving torque \n The red area is the sector where '
                f'the pendulum should come to a rest'))
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t),
                        *pL_vals)

    line1.set_offsets([coords[0, 0], coords[1, 0]])
    line2.set_data([0.0, coords[0, 0]], [0.0, coords[1, 0]])
    arrow.set_offsets([0.0, 0.0])
    arrow.set_UVC(torque_sol(t), 0.0)
    return line1, line2, arrow


tf = h_vals * (num_nodes - 1)
frames = np.linspace(t0, tf, int(fps * (tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)

display(HTML(animation.to_jshtml()))
plt.close(fig)
