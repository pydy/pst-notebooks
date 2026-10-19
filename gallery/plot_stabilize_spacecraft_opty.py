# %%
r"""
Stabilize Spacecraft
====================

Description
-----------

A spacecraft rotates in outer space where no gravity is present.
The spacecraft is modeled as a thin walled box of side lengths
:math:`l_1`, :math:`l_2`, and :math:`l_3`.
A massless linkage shaped as a ``T`` is rotably attached to the the center of
mass of the spacecraft. Control torques :math:`t_x, t_y, t_z` are applied to
it. A mass modeled as a ball of mass :math:`m_m` and radius :math:`r_m` can
move frictionlessly along the'horizontal part' of the linkage and a
control force :math:`f_z` is applied to it.

The goal is to orient the spacecraft in such a way, that the body fixed frame
of the spacecraft (A1) aligns with the inertial frame (N) in their z direction.
and that the rotation of the spacecraft around its x and y axes is minimized.

Notes
-----

- The idea is from https://arc.aiaa.org/doi/abs/10.2514/3.55956?journalCode=jgc
  which was written for another purpose. The linkage design is changed.
- As there are no external forces or torques acting on the system, its
  angular momentum should be constant. The solution so obtained was far
  from keeping it constant. When adding equations forcing it to be constant,
  it became constant within numerical errors.
  It seems unclear, whether such an enforcing helps or hurts the solution.
- Solving the system as an ODE, with the controls replaced by simple functions
  only kept the angular momentum approximately constant when very small
  tolerances ``atol, rtol`` were used with ``solve_ivp``. (This refers to
  a previous version using Euler angles. I did not try this with
  quaternions.)
- No statements can be made about the kinetic energy of the system as
  confirmed by Carlos Roithmayr (private communication).
- The objective function is to minimize the 'total movement around the
  x and y axes of the spacecraft', while it reaches the final position.
- While 1.5.0 is the latest released version of opty, the developmental
  version must be used to run this code.
- When I changed to quaternions for Euler angles, execution time dropped to
  one hour from about five hours before.
- Increasing ``fps`` in the animation program will make the animation smooother
  at the expense of increased storage.

**States**

- :math:`q_0, q_1, q_2, q_3` : quaternion of the spacecraft's body
  fixed frame (A1) relative to the inertial frame (N).
- :math:`u_0, u_1, u_2, u_3` : velocities
- :math:`q_{00}, q_4, q_5, q_6` : quaternion of the linkage's body fixed frame
  (A2) relative to the spacecraft's body fixed frame.
- :math:`u_{00}, u_4, u_5, u_6` : velocities.
- :math:`r, u_r` : Position and velocity of the mass along the
  horizontal part (A2.z) of the linkage.
- :math:`x_1, x_2, x_3` : Position of the spacecraft's center of mass in the
  inertial frame.
- :math:`ux_1, ux_2, ux_3` : Velocity of the spacecraft's center of mass in the
  inertial frame.

**Controls**

- :math:`t_x, t_y, t_z` : Torques acting on the :math:`A2_x, A2_y, A2_z` axes
  of the linkage.
- :math:`f_z` : Force acting on the mass along the :math:`A2_z` axis of
  the linkage.

**Parameters**

- :math:`m` : Mass of the spacecraft.
- :math:`m_m` : Mass of the moving mass.
- :math:`r_m` : Radius of the moving mass.
- :math:`b, c` : Geometric parameters of the linkage.
- :math:`l_1, l_2, l_3` : side lengths of the spacecraft.
"""

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from opty.direct_collocation import Problem

# %%
# Set up Kane's Equations of Motion
# ---------------------------------
#
# Define the reference frames and points.
# Define the time symbols t.
# Set the speed of the reference point O to zero.
N, A1, A2 = sm.symbols('N A1 A2', cls=me.ReferenceFrame)
O, Dmc, P1, P2, P3 = sm.symbols('O Dmc P1 P2 P3', cls=me.Point)
t = me.dynamicsymbols._t
O.set_vel(N, 0)

# %%
# Coordinates and speeds of Dmc.
x1, x2, x3, ux1, ux2, ux3 = me.dynamicsymbols('x1 x2 x3 ux1 ux2 ux3')

# %%
# Rotation and angular velocity of the spacecraft and the linkage.
q0, q1, q2, q3, q00, q4, q5, q6 = me.dynamicsymbols('q0 q1 q2 q3 q00 q4 q5 q6')
u0, u1, u2, u3, u00, u4, u5, u6 = me.dynamicsymbols('u0 u1 u2 u3 u00 u4 u5 u6')

# %%
# Control inputs for opty.
tx, ty, tz, fz = me.dynamicsymbols('tx ty tz fz')

# %%
# Position and speed of the moving mass.
r, ur = me.dynamicsymbols('r ur')

# %%
# Parameters of the system.
m, mm, rm = sm.symbols('m mm rm')
b, c, l1, l2, l3 = sm.symbols('b c l1 l2 l3')


# %%
# Body fixed frame of the spacecraft.
A1.orient_quaternion(N, (q0, q1, q2, q3))

# %%
# Body fixed frame of the linkage.
#
# Also used as body fixed frame of the
# moving mass. (As the mass is rotationally symmetric, any frame would do.)
A2.orient_quaternion(A1, (q00, q4, q5, q6))

# %%
# Mass center of the spacecraft.
Dmc.set_pos(O, x1 * N.x + x2 * N.y + x3 * N.z)
Dmc.set_vel(N, ux1 * N.x + ux2 * N.y + ux3 * N.z)

# %%
# Position of the linkage where the upright part meets the horizontal part.
P2.set_pos(Dmc, b * A2.x)
P2.v2pt_theory(Dmc, N, A2)
print('angular velocity of A2 in A1:')
A2.ang_vel_in(A1)

# %%
# Position of the mass moving on the horizontal part of the linkage.
P3.set_pos(P2, r * A2.z)
P3.set_vel(A2, ur * A2.z)


# %%
# Inertia of a box with thin walls and side lengths :math:`l_1, l_2, l_3`
# (from the internet).
iXX = m / (12*(l1*l2 + l2*l3 + l1*l3)) * (
    l2**3 * (l1 + l3) + l3**3*(l1 + l2) + 3*l1*l2*l3*(l2 + l3))

iYY = m / (12*(l1*l2 + l2*l3 + l1*l3)) * (
    l1**3 * (l2 + l3) + l3**3*(l1 + l2) + 3*l1*l2*l3*(l1 + l3))

iZZ = m / (12*(l1*l2 + l2*l3 + l1*l3)) * (
    l1**3 * (l2 + l3) + l2**3*(l1 + l3) + 3*l1*l2*l3*(l1 + l2))

I_spacecraft = me.inertia(A1, iXX, iYY, iZZ)
spacecraft = me.RigidBody('spacecraft', Dmc, A1, m, (I_spacecraft, Dmc))

# %%
# Inertia of a solid sphere (for the moving mass).
iXXp = 2 / 5 * mm * rm**2
iYYp = iXXp
iZZp = iXXp
I_masse = me.inertia(A2, iXXp, iYYp, iZZp)
masse = me.RigidBody('masse', P3,  A2, mm, (I_masse, P3))
bodies = [spacecraft, masse]

# %%
# Forces and torques acting on the linkage and the mass,
# and the reaction forces acting on the spacecraft.
FL = [
    (A2, tx * A2.x + ty * A2.y + tz * A2.z),
    (A1, -tx * A2.x - ty * A2.y - tz * A2.z),
    (P3, fz * A2.z),
    (Dmc, -fz * A2.z),
    (A2, P3.pos_from(Dmc).cross(fz * A2.z)),
    (A1, -P3.pos_from(Dmc).cross(fz * A2.z))
]


# %%
# Differential kinematic equations for the system.
kd = sm.Matrix([
    q0.diff(t) - u0,
    q1.diff(t) - u1,
    q2.diff(t) - u2,
    q3.diff(t) - u3,
    q00.diff(t) - u00,
    q4.diff(t) - u4,
    q5.diff(t) - u5,
    q6.diff(t) - u6,
    r.diff(t) - ur,
    x1.diff(t) - ux1,
    x2.diff(t) - ux2,
    x3.diff(t) - ux3,
])

# %%
# Configuration and speed constraints for the quaternions.
config_constr = sm.Matrix([
    q0**2 + q1**2 + q2**2 + q3**2 - 1,
    q00**2 + q4**2 + q5**2 + q6**2 - 1
])

# %%
# Speed constraints are obtained by differentiating the configuration
# constraints with respect to time. Needed for Kane's method.
speed_constr = config_constr.diff(t)


# %%
# Finish setting up Kane's EOMs.
q_ind = [q1, q2, q3, q4, q5, q6, r, x1, x2, x3]
q_dep = [q0, q00]
u_ind = [u1, u2, u3, u4, u5, u6, ur, ux1, ux2, ux3]
u_dep = [u0, u00]

kane = me.KanesMethod(
    N,
    q_ind,
    u_ind,
    q_dependent=q_dep,
    u_dependent=u_dep,
    kd_eqs=kd,
    configuration_constraints=config_constr,
    velocity_constraints=speed_constr,
)
fr, frstar = kane.kanes_equations(bodies, FL)

eom = kd.col_join(fr + frstar)

# %%
# Force the angular momentum to be constant, equal to its initial value.
H01, H02, H03 = sm.symbols('H01 H02 H03')
H_total = sum([b.angular_momentum(O, N) for b in bodies])

eom_H = sm.Matrix([H_total.dot(N.x) - H01,
                   H_total.dot(N.y) - H02,
                   H_total.dot(N.z) - H03])

eom = eom.col_join(eom_H)

# %%
# Create a dictionary replacing the time derivatives of the generalized
# coordinates with their expressions from the kinematic differential equations.
A_kin, b_kin = sm.linear_eq_to_matrix(kd, [i.diff(t) for i in q_ind + q_dep])
loesung = A_kin.LUsolve(b_kin)
kin_dict = {i.diff(t): loesung[j] for j, i in enumerate(q_ind + q_dep)}

# %%
# Force the x, y components of the angular velocity of A1 to be zero at
# :math:`t_f`.
A1_ang_x, A1_ang_y = me.dynamicsymbols('A1_ang_x A1_ang_y')

eom_ang = sm.Matrix([A1.ang_vel_in(N).dot(N.x) - A1_ang_x,
                     A1.ang_vel_in(N).dot(N.y) - A1_ang_y])
eom = eom.col_join(eom_ang)

# %%
# Force the ang. accelerations to be zero at :math:`t_f`.
A1_ang_x_dot, A1_ang_y_dot = me.dynamicsymbols('A1_ang_x_dot A1_ang_y_dot')

A1_ang_vel = me.msubs(A1.ang_vel_in(N), kin_dict)
A1_ang_acc = A1_ang_vel.diff(t, N)
eom_ang_acc = sm.Matrix([A1_ang_acc.dot(N.x) - A1_ang_x_dot,
                         A1_ang_acc.dot(N.y) - A1_ang_y_dot])
eom = eom.col_join(eom_ang_acc)

# %%
# Add the quaternion normalization constraints to the EOMs.
eom = eom.col_join(config_constr)

# %%
# Print information about the EOMs.
print(f"The eoms have {sm.count_ops(eom):,} operations"
      f" {sm.count_ops(sm.cse(eom))} operations after cse")
print('Shape of the EOMs:', eom.shape)


# %%
# Compile some functions for later use.
qLL = q_ind + q_dep + u_ind + u_dep
pLL = [m, mm, rm, b, c, l1, l2, l3]

# %%
# Define the total angular momentum and its components.
H_total = sum([me.msubs(b.angular_momentum(O, N), kin_dict) for b in bodies])
moment = [H_total.dot(uv) for uv in N]
moment_lam = sm.lambdify(qLL + pLL, moment, cse=True)

# %%
# Define the kinetic energy of the system, but keep in mind the notes above.
kin_energy = sum([me.msubs(body.kinetic_energy(N), kin_dict)
                  for body in bodies])
kin_lam = sm.lambdify(qLL + pLL, kin_energy, cse=True)

# %%
# If initial rotations, speeds are give as angles und convert then to
# quaternions and quaternion rates.


def euler123_quaternions(a, ad):
    # a  = [a1, a2, a3]   Euler angles in body-fixed 1-2-3 sequence
    # ad = [a1d, a2d, a3d] d/dt of Euler angles
    # Returns q = [q0..q3] and qd = dq/dt.
    # From the internet. Formulae likely in textbooks on quaternions.

    a1, a2, a3 = a
    a1d, a2d, a3d = ad
    c1, s1 = np.cos(a1/2), np.sin(a1/2)
    c2, s2 = np.cos(a2/2), np.sin(a2/2)
    c3, s3 = np.cos(a3/2), np.sin(a3/2)

    q0 = c1*c2*c3 - s1*s2*s3
    q1 = s1*c2*c3 + c1*s2*s3
    q2 = c1*s2*c3 - s1*c2*s3
    q3 = c1*c2*s3 + s1*s2*c3

    qd0 = -0.5*(q1*a1d + q2*a2d + q3*a3d)
    qd1 = 0.5*(q0*a1d + (c1*c2*s3 - s1*s2*c3)*a2d + q2*a3d)
    qd2 = 0.5*(-q3*a1d + (c1*c2*c3 + s1*s2*s3)*a2d - q1*a3d)
    qd3 = 0.5*(q2*a1d + (s1*c2*c3 - c1*s2*s3)*a2d + q0*a3d)

    return np.array([q0, q1, q2, q3]), np.array([qd0, qd1, qd2, qd3])


# %%
# Set up for opty
# ---------------
state_symbols = q_ind + q_dep + u_ind + u_dep
num_nodes = 500
t0, tf = 0.0, 10.0

interval_value = (tf - t0) / (num_nodes - 1)

# %%
# Set the parameters.
par_map = {}
par_map[m] = 1.0
par_map[mm] = 1.0
par_map[rm] = 0.1
par_map[b] = 0.75
par_map[c] = 0.5
par_map[l1] = 2.0
par_map[l2] = 2.0
par_map[l3] = 2.0

# %%
# Needed for lambdified functions later.
pL_vals = [par_map[i] for i in par_map.keys()]


# %%
# Define the objective function:
#
# The x and y components of the angular velocity of A1 w.r.t.
# the inertial frame N
# should be minimized. so the objective function is the sum of the squares of
# the x and y components of the angular velocity of A1 w.r.t. N. and
# the gradient function is the jacobian of the objective function
# with respect to the state symbols.
#
A1_ang = me.msubs(A1.ang_vel_in(N), kin_dict)
omega = me.msubs(sm.Matrix([A1_ang.dot(N.x)**2 +
                            A1_ang.dot(N.y)**2]), kin_dict)
omega_jac = omega.jacobian(state_symbols)
omega_lam = sm.lambdify(state_symbols + pLL, omega, cse=True)
omega_jac_lam = sm.lambdify(state_symbols + pLL, omega_jac, cse=True)
test = np.random.rand(len(state_symbols))


def obj(free):
    summe = 0
    params = np.empty(len(state_symbols))
    for i in range(num_nodes):
        params = np.array([free[i + j*num_nodes]
                           for j in range(len(state_symbols))])
        summe += omega_lam(*params, *pL_vals).squeeze()
    return summe * interval_value


def obj_grad(free):
    grad = np.zeros_like(free)
    for i in range(num_nodes):
        params = np.array([free[i + j*num_nodes]
                           for j in range(len(state_symbols))])
        omega_jac_vall = omega_jac_lam(*params, *pL_vals).squeeze()
        for j in range(len(state_symbols)):
            grad[i + j*num_nodes] = omega_jac_vall[j].squeeze()
    return grad * interval_value


# %%
# Limit of the control torques and force.
grenze = 25.0

# %%
# This limits the velocity and acceleration for the last portion of the nodes
# of the angular velocity of A1 w.r.t. N and of its acceleration.
lim_low_vel = np.full(num_nodes, -np.inf)
lim_up_vel = np.full(num_nodes, np.inf)
lim_low_acc = np.full(num_nodes, -np.inf)
lim_up_acc = np.full(num_nodes, np.inf)

lim_vel = 5.0
lim_acc = 30.0
los = int(num_nodes * 0.9)
delta = num_nodes - los

lim_low_vel[los:] = np.array([(lim_vel / delta) * (i - los) - lim_vel
                              for i in range(los, num_nodes)])
lim_up_vel[los:] = np.array([(-lim_vel / delta) * (i - los) + lim_vel
                             for i in range(los, num_nodes)])
lim_low_acc[los:] = np.array([(lim_acc / delta) * (i - los) - lim_acc
                              for i in range(los, num_nodes)])
lim_up_acc[los:] = np.array([(-lim_acc / delta) * (i - los) + lim_acc
                             for i in range(los, num_nodes)])
# %%
# Define the bounds.
bounds = {
    tx: (-grenze, grenze),
    ty: (-grenze, grenze),
    tz: (-grenze, grenze),
    fz: (-grenze, grenze),
    r: (-0.5 * min(par_map[l1], par_map[l2], par_map[l3]),
        0.5 * min(par_map[l1], par_map[l2], par_map[l3])),
    A1_ang_x: (lim_low_vel, lim_up_vel),
    A1_ang_y: (lim_low_vel, lim_up_vel),
    A1_ang_x_dot: (lim_low_acc, lim_up_acc),
    A1_ang_y_dot: (lim_low_acc, lim_up_acc),
 }

# %%
# Set the initial values for the states.
#
# Initial rotations and speeds of A1 w.r.t. N
alpha = 1.0
beta = 1.0
gamma = 1.0

alpha_dot = 2.0
beta_dot = 2.0
gamma_dot = 2.0

q01, q01d = euler123_quaternions([alpha, beta, gamma],
                                       [alpha_dot, beta_dot, gamma_dot])

# %%
# Initial rotations and speeds of A2 w.r.t. A1
alpha = 0.0
beta = 0.0
gamma = 0.0

alpha_dot = 0.0
beta_dot = 0.0
gamma_dot = 0.0

q001, q001d = euler123_quaternions([alpha, beta, gamma],
                                         [alpha_dot, beta_dot, gamma_dot])

# %%
# Set the initial values for the states.
q01, q11, q21, q31 = q01
q001, q41, q51, q61 = q001
r1 = par_map[c]
x11 = 0.0
x21 = 0.0
x31 = 0.0
u01, u11, u21, u31 = q01d
u001, u41, u51, u61 = q001d
ur1 = 0.0
ux11 = 0.0
ux21 = 0.0
ux31 = 0.0

instance_constraints = [
    q0.func(t0) - q01,
    q1.func(t0) - q11,
    q2.func(t0) - q21,
    q3.func(t0) - q31,
    q00.func(t0) - q001,
    q4.func(t0) - q41,
    q5.func(t0) - q51,
    q6.func(t0) - q61,
    r.func(t0) - r1,
    x1.func(t0) - x11,
    x2.func(t0) - x21,
    x3.func(t0) - x31,
    u0.func(t0) - u01,
    u1.func(t0) - u11,
    u2.func(t0) - u21,
    u3.func(t0) - u31,
    u00.func(t0) - u001,
    u4.func(t0) - u41,
    u5.func(t0) - u51,
    u6.func(t0) - u61,
    ur.func(t0) - ur1,
    ux1.func(t0) - ux11,
    ux2.func(t0) - ux21,
    ux3.func(t0) - ux31,
    q1.func(tf) - 0 % (2*np.pi),
    q2.func(tf) - 0 % (2*np.pi),
]

# %%
# Get values of the initial angular momentum components. The angular momentum
# must be equal to the initial values.
qL_vals = [q11, q21, q31, q41, q51, q61, r1, x11, x21, x31] + [q01, q001]
uL_vals = [u11, u21, u31, u41, u51, u61, ur1, ux11, ux21, ux31] + [u01, u001]

H_fix1, H_fix2, h_fix3 = moment_lam(*qL_vals, *uL_vals, *pL_vals)
par_map.update({H01: H_fix1, H02: H_fix2, H03: h_fix3})

# %%
# Set up a Problem instance.
prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
)

print('length of free vector: ', prob.num_free)

# %%
# Solve the problem.
initial_guess = np.ones(prob.num_free)
prob.add_option('max_iter', 15000)

start_time = time.time()
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
end_time = time.time()
minutes, seconds = divmod(end_time - start_time, 60)
print(f"Time needed to solve the problem: {int(minutes)} "
      f"min {seconds:.2f} sec")

# %%
# Plot objective value.
_ = prob.plot_objective_value()

# %%
# Plot the jacobian of the problem.
_ = prob.plot_jacobian_sparsity()

# %%
# Plot the components of the angular velocity.
ang_vel = me.msubs(A1.ang_vel_in(N), kin_dict)
ang_vel_components = [ang_vel.dot(uv) for uv in N]
ang_vel_lam = sm.lambdify(qLL + pLL, ang_vel_components, cse=True)
names = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
resultat, *_ = prob.parse_free(solution)
times = prob.time_vector(solution)
fig, ax = plt.subplots(figsize=(8, 3))
for i, name in enumerate(names):
    ax.plot(times, [ang_vel_lam(*resultat[:, j], *pL_vals)[i]
                    for j in range(resultat.shape[1])], label=names[i])
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity [rad/s]')
ax.axhline(0.0, color='k', lw=0.5, ls='--')
ax.set_title('Angular Velocity Components')
_ = ax.legend()
# %%
# Plot the state trajectories.
axes = prob.plot_trajectories(solution, show_bounds=True)
for i in (0, 1, 2):
    axes[i].axhline(0, color='k', linestyle='--')

# %%
# Plot the violations of the constraints.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# How well is the angular momentum kept?
resultat, *_, = prob.parse_free(solution)
pLL = [m, mm, rm, b, c, l1, l2, l3]
qLL = q_ind + q_dep + u_ind + u_dep
H_total = sum([me.msubs(b.angular_momentum(O, N), kin_dict) for b in bodies])
moment = [H_total.dot(uv) for uv in N]
moment_lam = sm.lambdify(qLL + pLL, moment, cse=True)

pL_vals = [par_map[i] for i in pLL]
H_norm = np.linalg.norm(moment_lam(*resultat, *pL_vals)[:][2:], axis=0)
H_max = np.max(H_norm)
H_min = np.min(H_norm)
delta_H = (H_max - H_min) / H_max
print("Max deviaiton in angular momentum norm from being constant is: "
      f"{delta_H:.3e}")
times = prob.time_vector(solution)

fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True, layout='constrained')
for i in range(3):
    ax[0].plot(times[2:], moment_lam(*resultat, *pL_vals)[i][2:],
               label=f'H_{i+1}')
ax[0].set_title('Components of the angular momerntum')
ax[-1].set_xlabel('Time')
ax[0].legend()

ax[1].plot(times, kin_lam(*resultat, *pL_vals))
_ = ax[1].set_title('Kinetic energy')


# %%
# Revert back to Euler angles XYZ for the plot.
# Needed for the rotation of the spacecraft.


def quat_to_euler_xyz(q):
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q

    s2 = 2.0 * (q1*q3 + q0*q2)
    # -1 <= s2 <= 1 needed because of arcsin.
    s2 = np.clip(s2, -1.0, 1.0)

    a2 = np.arcsin(s2)
    a1 = np.arctan2(-2.0*(q2*q3 - q0*q1), 1.0 - 2.0*(q1**2 + q2**2))
    a3 = np.arctan2(-2.0*(q1*q2 - q0*q3), 1.0 - 2.0*(q2**2 + q3**2))

    return [a1, a2, a3]

# %%
# Animation
# ---------
# The motion of the center of mass is suppressed to better see the rotations.
# The rotating box was written by AI. I just adapted it to my needs.
#
v_ang = me.msubs(A1.ang_vel_in(N), kin_dict)
v_ang = [v_ang.dot(uv) for uv in N]
v_ang_lam = sm.lambdify(qLL + pLL, v_ang, cse=True)
# Start the animation program.
fps = 7

# Interpolation of the solution
t_arr = np.asarray(prob.time_vector(solution))
y_arr = resultat
state_sol = interp1d(
    t_arr,
    y_arr,
    kind="cubic",
    axis=1
)

coordinates = Dmc.pos_from(O).to_matrix(N)
for point in (P2, P3):
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))
coordinates_lam = sm.lambdify(qLL + pLL, coordinates, cse=True)

l11 = par_map[l1]
l21 = par_map[l2]
l31 = par_map[l3]

T_END = tf  # animation duration in seconds
ALPHA = 0.25  # face transparency (0 = invisible, 1 = opaque)


def euler_angles(t):
    q11 = state_sol(t)[0]
    q21 = state_sol(t)[1]
    q31 = state_sol(t)[2]
    return q11, q21, q31


# Kinematics


def R1(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def R2(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R3(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_NA(q1, q2, q3):
    """Rotation matrix from A to N (body-fixed 1-2-3 sequence)."""
    return R1(q1) @ R2(q2) @ R3(q3)


# Corner points in A relative to M (rows = points)
h = np.array([l11, l21, l31]) / 2.0
corners_A = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
# index = 4*ix + 2*iy + iz  with ix, iy, iz in {0 (negative), 1 (positive)}

# Faces as index lists into the corners
# (pairs of opposite faces share a colour)
faces_idx = [
    [0, 1, 3, 2], [4, 5, 7, 6],   # -e1 / +e1
    [0, 1, 5, 4], [2, 3, 7, 6],   # -e2 / +e2
    [0, 2, 6, 4], [1, 3, 7, 5],   # -e3 / +e3
]
face_colors = ["tab:red"] * 2 + ["tab:green"] * 2 + ["tab:blue"] * 2

# Edges (for a clean wireframe)
edges_idx = [(0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def pose(t):
    """Return corners in N, centre M in N, and rotation matrix R_NA."""
    R = rotation_matrix_NA(*euler_angles(t))
    M = (0.0, 0.0, 0.0)
    corners_N = M + corners_A @ R.T
    return corners_N, M, R


times = np.concatenate((np.arange(0.0, tf, 1.0 / fps), [tf]))

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("$N_1$ [m]")
ax.set_ylabel("$N_2$ [m]")
ax.set_zlabel("$N_3$ [m]")

r_max = 0.5 * np.linalg.norm([l11, l21, l31])
lo = -r_max
hi = +r_max
ctr = 0.5 * (lo + hi)
half = 0.5 * (hi - lo)
ax.set_xlim(-half, half)
ax.set_ylim(-half, half)
ax.set_zlim(-half, half)
ax.set_box_aspect((1, 1, 1))

# Translucent faces
c0, _, _ = pose(0.0)
poly = Poly3DCollection([c0[f] for f in faces_idx],
                        facecolors=face_colors, edgecolors="k",
                        linewidths=1.0, alpha=ALPHA)
ax.add_collection3d(poly)

# Body-fixed axes of A (drawn from M)
axis_len = 1.0 * max(l11, l21, l31)
axis_colors = ["tab:red", "tab:green", "tab:blue"]
axis_lines = [ax.plot([], [], [], color=col, lw=0.5)[0] for col in axis_colors]
axis_labels = [ax.text(0, 0, 0, f"$A1_{i+1}$", color=axis_colors[i])
               for i in range(3)]

# Inertial frame N at origin
for i, (col, lab) in enumerate(zip("rgb", ["N1", "N2", "N3"])):
    d = np.zeros(3)
    d[i] = half * 0.4
    ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=col, lw=1, ls="--",
            alpha=0.6)

# Centre of mass
com_dot, = ax.plot([], [], [], "ko", ms=5)
trail, = ax.plot([], [], [], "k-", lw=1, alpha=0.5)
title = ax.set_title("")

# Create P2, P3
pt2, = ax.plot([], [], [], 'o', color='blue', markersize=5)
pt3, = ax.plot([], [], [], 'o', color='red', markersize=15)

# Create the lines connecting Dmc to P2 and PO2 to P3
line1, = ax.plot([], [], [], '-', color='red', lw=1.5)
line2, = ax.plot([], [], [], '-', color='red', lw=1.5)

# create the arrow showing angular velocity
arrow_head = np.array(
    [v_ang_lam(*state_sol(0), *pL_vals)[0] -
     coordinates_lam(*state_sol(0), *pL_vals)[0, 0],
     v_ang_lam(*state_sol(0), *pL_vals)[1] -
     coordinates_lam(*state_sol(0), *pL_vals)[1, 0],
     v_ang_lam(*state_sol(0), *pL_vals)[2] -
     coordinates_lam(*state_sol(0), *pL_vals)[2, 0]])

arrow = ax.quiver(*[0.0, 0.0, 0.0], *arrow_head,
                  arrow_length_ratio=0.15,
                  color='green', linewidth=2)


def update(k):
    global arrow
    t = times[k]
    corners_N, M, R = pose(t)

    poly.set_verts([corners_N[f] for f in faces_idx])

    for i, line in enumerate(axis_lines):
        tip = M + axis_len * R[:, i]
        line.set_data_3d([M[0], tip[0]], [M[1], tip[1]], [M[2], tip[2]])
        axis_labels[i].set_position_3d(tip)

    q1, q2, q3 = euler_angles(t)
    title.set_text(f"Running time = {t:4.2f} sec \n "
                   " Angles of rotation of the spacecraft = "
                   f"({np.degrees(q1):.0f}°,"
                   f" {np.degrees(q2):.0f}°, {np.degrees(q3):.0f}°) \n"
                   "The red ball ist the mass moved around to stabilize "
                   "the spacecraft. \n The green arrow gives the"
                   " angular velocity of the spacecraft.")

    # Update the positions of P2, P3, and the connecting lines
    coords = coordinates_lam(*state_sol(t), *pL_vals)
    p2x = coords[0, 1] - coords[0, 0]
    p2y = coords[1, 1] - coords[1, 0]
    p2z = coords[2, 1] - coords[2, 0]
    pt2.set_data_3d([p2x], [p2y], [p2z])

    p3x = coords[0, 2] - coords[0, 0]
    p3y = coords[1, 2] - coords[1, 0]
    p3z = coords[2, 2] - coords[2, 0]
    pt3.set_data_3d([p3x], [p3y], [p3z])

    line1.set_data_3d([p2x, p3x], [p2y, p3y], [p2z, p3z])
    line2.set_data_3d([p2x, 0.0], [p2y, 0.0], [p2z, 0.0])

    # As quiver cannot be updated easily in 3D remove the old arrow and
    # create a new one
    arrow.remove()
    arrow_head = (np.array([v_ang_lam(*state_sol(t), *pL_vals)[0],
                           v_ang_lam(*state_sol(t), *pL_vals)[1],
                           v_ang_lam(*state_sol(t), *pL_vals)[2]]))

    arrow = ax.quiver(*[0.0, 0.0, 0.0], *arrow_head,
                      arrow_length_ratio=0.15,
                      color='green', linewidth=2)

    return poly, com_dot, trail, *axis_lines, pt2, pt3, line1, line2, arrow


anim = FuncAnimation(fig, update, frames=len(times),
                     interval=1000 / fps, blit=False)

plt.show()
