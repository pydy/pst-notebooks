# %%
r"""
Back-Up Tractor with Two Trailers
=================================

Description
-----------

A truck with two trailers has to drive backwards on a wavy street as fast as
possible. The truck has front steering and rear wheel drive.
Truck and trailer bodies are modeled as thin rods of length :math:`l_i`
and mass :math:`m_i`, and the wheels are modeled as point masses of mass
:math:`m_w`. The truck is moved by a force :math:`F` on the rear axle
and a steering torque :math:`Torq` on the front axle.

There is no speed allowed in the direction of the axles.

Notes
-----

- This would likely be hard for even a very experienced human driver.
- Here, starting with a simpler (less wavy) street helped.
- This simulation shows, that ``opty`` can solve pretty complicated
  problems, if one has the patience to wait for a solution.

**States**

- :math:`q_{10}, q_{20}, q_{30}` : angles of the truck and trailers.
  :math:`q_{10}` w.r.t. the inertial frame, :math:`q_{20}`
  w.r.t. the truck, and :math:`q_{30}` w.r.t. the first trailer.
- :math:`q_{11}, q_{21}, q_{31}` : steering angles of the axles, w.r.t.
  the respective bodies.
- :math:`x, y` : position of the truck's center of mass in the
  inertial frame.
- :math:`u_{10}, u_{20}, u_{30}` : angular velocities of the truck and
  trailers.
- :math:`u_{11}, u_{21}, u_{31}` : angular velocities of the steering
  axles.
- :math:`u_x, u_y` : velocities of the truck's center of mass in the
  inertial frame.

**Control Inputs**

- :math:`F` : force on the rear axle of the truck.
- :math:`Torq` : steering torque on the front axle of the truck.

**Parameters**

- :math:`m_1, m_2, m_3` : masses of the truck and trailers.
- :math:`m_w` : mass of the wheels.
- :math:`l_1, l_2, l_3` : lengths of the truck and trailers.
- :math:`l_d` : length of the hitch.
- :math:`l_{ax}` : half the length of the axles.
- :math:`g` : acceleration due to gravity.
- :math:`reibung` : coefficient of friction.

**Symbols Used in the Code**

- :math:`N` : inertial frame.
- :math:`A_{i0}` : body-fixed frames, where :math:`i` is the body.
- :math:`A_{i1}` : steering axle frames, where :math:`i` is the body.
- :math:`O` : origin of the inertial frame.
- :math:`Dmc_i` : mass center of body :math:`i`.
- :math:`Pi_{fl}, Pi_{fr}, Pi_{bl}, Pi_{br}` : front left, front right,
  back left, and back right attachment points for the wheels of body :math:`i`.

"""

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import FancyArrowPatch
from opty import Problem
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


# %%
# Kanes Equation of Motion
# ------------------------
#
# Geometry.

start_time = time.time()
N, A10, A11, A20, A21, A30, A31 = sm.symbols(
    'N A10 A11 A20 A21 A30 A31', cls=me.ReferenceFrame)
O = me.Point('O')
O.set_vel(N, 0)
t = me.dynamicsymbols._t

Dmc1, P1f, P1b, Dmc2, P2f, P2b, Dmc3, P3f, P3b = sm.symbols(
    'Dmc1 P1f P1b Dmc2 P2f P2b Dmc3 P3f P3b', cls=me.Point)

P1fl, P1fr, P1bl, P1br = sm.symbols(
    'P1fl P1fr P1bl P1br', cls=me.Point)
P2fl, P2fr, P2bl, P2br = sm.symbols(
    'P2fl P2fr P2bl P2br', cls=me.Point)
P3fl, P3fr, P3bl, P3br = sm.symbols(
    'P3fl P3fr P3bl P3br', cls=me.Point)

q10, q11, q20, q21, q30, q31 = me.dynamicsymbols('q10 q11 q20 q21 q30 q31')
u10, u11, u20, u21, u30, u31 = me.dynamicsymbols('u10 u11 u20 u21 u30 u31')
x, y, ux, uy = me.dynamicsymbols('x y ux uy')
F, Torq = me.dynamicsymbols('F Torq')

# %%
# Parameters.
m1, m2, m3, mw, l1, l2, l3, ld, lax, g, reibung = sm.symbols(
    'm1 m2 m3 mw l1 l2 l3 ld lax g reibung')

# %%
# Orient the body fixed frames and set their angular velocities.
A10.orient_axis(N, q10, N.z)
A10.set_ang_vel(N, u10 * N.z)
A11.orient_axis(A10, q11, N.z)
A11.set_ang_vel(A10, u11 * N.z)
A20.orient_axis(A10, q20, N.z)
A20.set_ang_vel(A10, u20 * N.z)
A21.orient_axis(A20, q21, N.z)
A21.set_ang_vel(A20, u21 * N.z)
A30.orient_axis(A20, q30, N.z)
A30.set_ang_vel(A20, u30 * N.z)
A31.orient_axis(A30, q31, N.z)
_ = A31.set_ang_vel(A30, u31 * N.z)

# %%
# Define the various points and set their velocities.
#
# The truck.
Dmc1.set_pos(O, x * N.x + y * N.y)
Dmc1.set_vel(N, ux * N.x + uy * N.y)
P1f.set_pos(Dmc1, l1/2 * A10.x)
P1f.v2pt_theory(Dmc1, N, A10)
P1b.set_pos(Dmc1, -l1/2 * A10.x)
P1b.v2pt_theory(Dmc1, N, A10)
P1fl.set_pos(P1f, lax * A11.y)
P1fl.v2pt_theory(P1f, N, A11)
P1fr.set_pos(P1f, -lax * A11.y)
P1fr.v2pt_theory(P1f, N, A11)
P1bl.set_pos(P1b, lax * A10.y)
P1bl.v2pt_theory(P1b, N, A10)
P1br.set_pos(P1b, -lax * A10.y)
_ = P1br.v2pt_theory(P1b, N, A10)

# %%
# Trailer 1.
P2f.set_pos(P1b, -ld * A21.x)
P2f.v2pt_theory(P1b, N, A21)
Dmc2.set_pos(P2f, -l2/2 * A20.x)
Dmc2.v2pt_theory(P2f, N, A20)
P2b.set_pos(Dmc2, -l2/2 * A20.x)
P2b.v2pt_theory(Dmc2, N, A20)
P2fl.set_pos(P2f, lax * A21.y)
P2fl.v2pt_theory(P2f, N, A21)
P2fr.set_pos(P2f, -lax * A21.y)
P2fr.v2pt_theory(P2f, N, A21)
P2bl.set_pos(P2b, lax * A20.y)
P2bl.v2pt_theory(P2b, N, A20)
P2br.set_pos(P2b, -lax * A20.y)
_ = P2br.v2pt_theory(P2b, N, A20)

# %%
# Trailer 2.
P3f.set_pos(P2b, -ld * A31.x)
P3f.v2pt_theory(P2b, N, A31)
Dmc3.set_pos(P3f, -l3/2 * A30.x)
Dmc3.v2pt_theory(P3f, N, A30)
P3b.set_pos(Dmc3, -l3/2 * A30.x)
P3b.v2pt_theory(Dmc3, N, A30)
P3fl.set_pos(P3f, lax * A31.y)
P3fl.v2pt_theory(P3f, N, A31)
P3fr.set_pos(P3f, -lax * A31.y)
P3fr.v2pt_theory(P3f, N, A31)
P3bl.set_pos(P3b, lax * A30.y)
P3bl.v2pt_theory(P3b, N, A30)
P3br.set_pos(P3b, -lax * A30.y)
_ = P3br.v2pt_theory(P3b, N, A30)

# %%
# Define the bodies.

IZZtruck = 1/12 * m1 * l1**2
IZZtrailer1 = 1/12 * m2 * l2**2
IZZtrailer2 = 1/12 * m3 * l3**2
I_truck = me.inertia(A10, 0, 0, IZZtruck)
I_trailer1 = me.inertia(A20, 0, 0, IZZtrailer1)
I_trailer2 = me.inertia(A30, 0, 0, IZZtrailer2)

truck = me.RigidBody('Truck', Dmc1, A10, m1, (I_truck, Dmc1))
trailer1 = me.RigidBody('Trailer1', Dmc2, A20, m2, (I_trailer1, Dmc2))
trailer2 = me.RigidBody('Trailer2', Dmc3, A30, m3, (I_trailer2, Dmc3))

P1fla = me.Particle('P1fla', P1fl, mw)
P1fra = me.Particle('P1fra', P1fr, mw)
P1bla = me.Particle('P1bla', P1bl, mw)
P1bra = me.Particle('P1bra', P1br, mw)
P2fla = me.Particle('P2fla', P2fl, mw)
P2fra = me.Particle('P2fra', P2fr, mw)
P2bla = me.Particle('P2bla', P2bl, mw)
P2bra = me.Particle('P2bra', P2br, mw)
P3fla = me.Particle('P3fla', P3fl, mw)
P3fra = me.Particle('P3fra', P3fr, mw)
P3bla = me.Particle('P3bla', P3bl, mw)
P3bra = me.Particle('P3bra', P3br, mw)

bodies = [truck, trailer1, trailer2, P1fla, P1fra, P1bla, P1bra, P2fla,
          P2fra, P2bla, P2bra, P3fla, P3fra, P3bla, P3bra]

# %%
# Define the forces and torques.
FL = [(P1b, F * A10.x), (A11, Torq * N.z),
      (Dmc1, -reibung * m1 * g * Dmc1.vel(N)),
      (Dmc2, -reibung * m2 * g * Dmc2.vel(N)),
      (Dmc3, -reibung * m3 * g * Dmc3.vel(N))]

# %%
# Kinematic differential equations.
kd = sm.Matrix([
    q10.diff(t) - u10,
    q11.diff(t) - u11,
    q20.diff(t) - u20,
    q21.diff(t) - u21,
    q30.diff(t) - u30,
    q31.diff(t) - u31,
    ux - x.diff(t),
    uy - y.diff(t),
])

# %%
# Speed constraints: No speed in direction of the axles.
speed_constr = sm.Matrix([
    P1f.vel(N).dot(A11.y),
    P1b.vel(N).dot(A10.y),
    P2f.vel(N).dot(A21.y),
    P2b.vel(N).dot(A20.y),
    P3f.vel(N).dot(A31.y),
    P3b.vel(N).dot(A30.y)
])

# %%
# Form Kane's equations of motion.

q_ind = [q10, q11, q20, q21, q30, q31, x, y]
u_ind = [u11, ux]
u_dep = [u10, u20, u21, u30, u31, uy]

kanes = me.KanesMethod(
    N,
    q_ind,
    u_ind,
    kd_eqs=kd,
    u_dependent=u_dep,
    velocity_constraints=speed_constr,
)
fr, frstar = kanes.kanes_equations(bodies, FL)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(speed_constr)

# %%
# Add additional constraints.
#
# Form the street.
ast, bst, cst, aux = sm.symbols('ast bst cst aux')


def smooth_step(x, a, steepness=10):
    # returns 1 for x << a and 0 for x >> a, with a
    # smooth transition in between.
    return 0.5 * (1 - sm.tanh(steepness * (x - a)))


def street_top(aux):
    return (ast * (1 - smooth_step(aux, 0.0)) * sm.sin(2 * sm.pi / bst * aux) +
            cst * (0.5 + ast/cst/2 * smooth_step(aux, 0.0)))


def street_bottom(aux):
    return (ast * (1 - smooth_step(aux, 0.0)) * sm.sin(2 * sm.pi / bst * aux) -
            cst * (0.5 + ast/cst/2 * smooth_step(aux, 0.0)))


# %%
# Wheels and mass centers to remain in the street.
punkte = [P1fl, P1fr, P1bl, P1br,
          P2fl, P2fr, P2bl, P2br,
          P3fl, P3fr, P3bl, P3br,
          Dmc1, Dmc2, Dmc3]

alg_eqs = sm.Matrix([
    *[punkt.pos_from(O).dot(N.y) -
      street_bottom(punkt.pos_from(O).dot(N.x)) for punkt in punkte],
    *[street_top(punkt.pos_from(O).dot(N.x)) -
      punkt.pos_from(O).dot(N.y) for punkt in punkte],
])

eom = eom.col_join(alg_eqs)

# %%
# Needed for final condition and to ensure that it moves backwards in the
# positive x direction.

x3b, y3b = me.dynamicsymbols('x3b y3b')
ende = sm.Matrix([
    x3b - P3b.pos_from(O).dot(N.x),
    y3b - P3b.pos_from(O).dot(N.y),
    x3b - x,
])
eom = eom.col_join(ende)
print(f"the eoms contain {sm.count_ops(eom):,} operations, "
      f"{sm.count_ops(sm.cse(eom)):,} operations after cse and have "
      f"shape {eom.shape}")

# %%
# Set Up the Optimization
# -----------------------

state_symbols = q_ind + u_ind + u_dep

num_nodes = 801
t0, tf = 0.0, 8.0
interval_value = (tf - t0) / (num_nodes - 1)

# %%
# Set the parameters.
par_map = {}
par_map[ast] = 0.0
par_map[bst] = 15.0
par_map[cst] = 10.0
par_map[m1] = 1.0
par_map[m2] = 1.0
par_map[m3] = 1.0
par_map[mw] = 0.25
par_map[l1] = 1.0
par_map[l2] = 2.0
par_map[l3] = 3.0
par_map[ld] = 0.75
par_map[lax] = 0.75
par_map[g] = 9.81
par_map[reibung] = 0.0

# %%
# Objective function: Minimimize the energy needed.


def obj(free):
    return np.sum(free[16*num_nodes:18*num_nodes]**2) * interval_value


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[16*num_nodes:18*num_nodes] = \
        2 * free[16*num_nodes:18*num_nodes] * interval_value
    return grad


# %%
# Set the instance constraints.
instance_constraints = [
    q10.func(t0) - np.pi,
    q11.func(t0) - 0.0,
    q20.func(t0) - 0.0,
    q21.func(t0) - 0.0,
    q30.func(t0) - 0.0,
    q31.func(t0) - 0.0,
    x.func(t0) + 15.0,
    y.func(t0) - 0.0,
    u10.func(t0) - 0.0,
    u11.func(t0) - 0.0,
    u20.func(t0) - 0.0,
    u21.func(t0) - 0.0,
    u30.func(t0) - 0.0,
    u31.func(t0) - 0.0,
    ux.func(t0) - 0.0,
    uy.func(t0) - 0.0,
    x3b.func(t0) - 0.0,

    x3b.func(tf) - 30.0,
    ux.func(tf) - 0.0,
    uy.func(tf) - 0.0,
    u10.func(tf) - 0.0,
    u11.func(tf) - 0.0,
    u20.func(tf) - 0.0,
    u21.func(tf) - 0.0,
    u30.func(tf) - 0.0,
    u31.func(tf) - 0.0,
]

# %%
# While travelling the angle between two adjacent vehicles must be larger than
# :math:`\frac{\pi}{2}`.
# The front axles of the trailers must be less than
# :math:`\frac{\pi}{ 3}` w.r.t. the trailers, and the steering axle must be
# less than :math:`\frac{\pi}{4}` w.r.t. the truck.
# (Arbitrary choices).
#
# Bound the force and the steering torque.

limit = 75.0
limitT = 10.0
bounds = {
    F: (-limit, limit),
    Torq: (-limitT, limitT),
    q20: (-np.pi/2, np.pi/2),
    q30: (-np.pi/2, np.pi/2),
    q11: (-np.pi/4, np.pi/4),
    q21: (-np.pi/3, np.pi/3),
    q31: (-np.pi/3, np.pi/3),
}

# %%
# The truck and trailers must stay on the road.
limit1, limit2 = 0.0, 15.0
eom_bounds = {eq: (limit1, limit2) for eq in range(16, 46)} | {48: (3.0, 10.0)}


# %%
# Set Up the Problem and Solve It
# -------------------------------

for i in range(3):
    anfang = time.time()
    if i == 1:
        par_map[ast] = 6.70
    if i == 2:
        par_map[ast] = 8.5
        limit = 75.0
        limitT = 35.0
        bounds[F] = (-limit, limit)
        bounds[Torq] = (-limitT, limitT)

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
        eom_bounds=eom_bounds,
        time_symbol=t,
    )

    if i == 0:
        np.random.seed(0)
        initial_guess_1 = np.random.rand(prob.num_free)
        initial_guess_1 = np.ones(prob.num_free)
        initial_guess_1[6*num_nodes: 7*num_nodes] = np.linspace(-15.0, 30.0,
                                                                num_nodes)  # x
        initial_guess_1[7*num_nodes: 8*num_nodes] = 0.0  # y
        initial_guess = initial_guess_1

    else:
        initial_guess = solution
    prob.add_option('max_iter', 5000)
    solution, info = prob.solve(initial_guess)
    # Save the various initial guesses for plotting
    if i == 0:
        initial_guess_2 = solution
    if i == 1:
        initial_guess_3 = solution
    print(info['status_msg'])
    ende = time.time()
    print(f"Time needed for iteration {i + 1}: {ende - anfang:.2f} seconds \n")

# %%
# Plot the trajectories.

_ = prob.plot_trajectories(solution, show_bounds=True)

# %%
# Plot the constraint violations.

_ = prob.plot_constraint_violations(solution, subplots=True, show_bounds=True)

# %%
# Plot the objective function.

_ = prob.plot_objective_value()

# %%
# Animation
# ---------

print("Total time needed for the simulation: "
      f"{(time.time() - start_time) / 60:.2f} minutes")
fps = 15

strasse_top_lam = sm.lambdify((aux, ast, bst, cst), street_top(aux), cse=True)
strasse_bottom_lam = sm.lambdify((aux, ast, bst, cst), street_bottom(aux),
                                 cse=True)

resultat, inputs, *_, h_vals = prob.parse_free(solution)
resultat = resultat.T
inputs = inputs.T
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = interp1d(t_arr, resultat, kind='cubic', axis=0)
input_sol = interp1d(t_arr, inputs, kind='cubic', axis=0)

pL = [key for key in par_map.keys()]
pL_vals = [par_map[key] for key in par_map.keys()]
qL = q_ind + u_ind + u_dep + [F, Torq]

# Define the end of the force_vector.
arrow_head = me.Point('arrow_head')
scale = 2.0
arrow_head.set_pos(P1b, F / scale * A10.x)

# Get the coordinates of the points in the inertial frame for plotting.
coords = Dmc1.pos_from(O).to_matrix(N)
for point in (P1f, P1b, P2f, P2b, P3f, P3b,
              P1fl, P1fr, P1bl, P1br,
              P2fl, P2fr, P2bl, P2br,
              P3fl, P3fr, P3bl, P3br, arrow_head, Dmc1):
    coords = coords.row_join(point.pos_from(O).to_matrix(N))
coords_lam = sm.lambdify(qL + pL, coords, cse=True)

fig, ax = plt.subplots(figsize=(9, 9), layout='constrained')

arrow = FancyArrowPatch([0.0, 0.0], [0.0, 0.0],
                        arrowstyle='-|>',     # nicer arrow head
                        mutation_scale=20,    # makes head bigger
                        linewidth=1,
                        color='green')
ax.add_patch(arrow)

# Plot the points
P1f_p = ax.scatter([], [], color='blue', s=5)
P1b_p = ax.scatter([], [], color='blue', s=5)
P2f_p = ax.scatter([], [], color='orange', s=5)
P2b_p = ax.scatter([], [], color='orange', s=5)
P3f_p = ax.scatter([], [], color='red', s=5)
P3b_p = ax.scatter([], [], color='red', s=5)

P1fl_p = ax.scatter([], [], color='blue', s=5)
P1fr_p = ax.scatter([], [], color='blue', s=5)
P1bl_p = ax.scatter([], [], color='blue', s=5)
P1br_p = ax.scatter([], [], color='blue', s=5)
P2fl_p = ax.scatter([], [], color='orange', s=5)
P2fr_p = ax.scatter([], [], color='orange', s=5)
P2bl_p = ax.scatter([], [], color='orange', s=5)
P2br_p = ax.scatter([], [], color='orange', s=5)
P3fl_p = ax.scatter([], [], color='red', s=5)
P3fr_p = ax.scatter([], [], color='red', s=5)
P3bl_p = ax.scatter([], [], color='red', s=5)
P3br_p = ax.scatter([], [], color='red', s=5)

# lines for axes
fax1, = ax.plot([], [], color='blue')
bax1, = ax.plot([], [], color='blue')
fax2, = ax.plot([], [], color='orange')
bax2, = ax.plot([], [], color='orange')
fax3, = ax.plot([], [], color='red')
bax3, = ax.plot([], [], color='red')

# Lines for the truck and trailers
truck_p, = ax.plot([], [], color='blue', lw=1)
trailer1_p, = ax.plot([], [], color='orange', lw=1)
trailer2_p, = ax.plot([], [], color='red', lw=1)

# Tow bars
towbar1_p, = ax.plot([], [], color='black')
towbar2_p, = ax.plot([], [], color='black')

# Show the actual path
Dmc1_p, = ax.plot([], [], color='gray', linestyle='-', lw=0.5)

# Size of the plot.
margin = (par_map[l3] + par_map[l2] + 2.0 * par_map[ld] + par_map[l1] / 2 +
          par_map[ast] / 2)
x_max = np.max(resultat[:, 6]) + margin
x_min = np.min(resultat[:, 6]) - margin / 4.0
y_max = np.max(resultat[:, 7]) + margin
y_min = np.min(resultat[:, 7]) - margin
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', fontsize=15)
ax.set_ylabel('y (m)', fontsize=15)

# Plot the street and fill it.
XX = np.linspace(x_min, x_max, 100)
YY = strasse_top_lam(XX, par_map[ast], par_map[bst], par_map[cst])
ZZ = strasse_bottom_lam(XX, par_map[ast], par_map[bst], par_map[cst]) - 0.5
ax.plot(XX, YY, color='black')
ax.plot(XX, ZZ, color='black')
ax.fill_between(XX, YY, y2=y_max, color='black', alpha=0.5)
ax.fill_between(XX, ZZ, y2=y_min, color='black', alpha=0.5)

ax.vlines(-15.0, strasse_top_lam(-15, par_map[ast], par_map[bst],
                                 par_map[cst]),
          strasse_bottom_lam(-15, par_map[ast], par_map[bst], par_map[cst]),
          color='red', lw=0.75, linestyle='--')
ax.vlines(30.0, strasse_top_lam(30, par_map[ast], par_map[bst],
                                par_map[cst]),
          strasse_bottom_lam(30, par_map[ast], par_map[bst], par_map[cst]),
          color='green', lw=0.75, linestyle='--')

# Show the initial guess for the path of Dmc1 for the first loop.
ax.plot(initial_guess_1[6*num_nodes:7*num_nodes],
        initial_guess_1[7*num_nodes:8*num_nodes],
        color='gray', linestyle='--', lw=0.5)
# Show the initial guess for the path of Dmc1 for the second loop.
ax.plot(initial_guess_2[6*num_nodes:7*num_nodes],
        initial_guess_2[7*num_nodes:8*num_nodes],
        color='blue', linestyle='--', lw=0.5)

ax.plot(initial_guess_3[6*num_nodes:7*num_nodes],
        initial_guess_3[7*num_nodes:8*num_nodes],
        color='green', linestyle='--', lw=0.5)


def update(t):
    ax.set_title(f"Running time: {t:.2f} s \n "
                 "Control force is the green arrow with magnitude: "
                 f"{np.abs(input_sol(t))[0]:.2f} N \n "
                 f"The tractor is blue, the first trailer is orange "
                 "and the second trailer is red."
                 f"The tow bars are black \n"
                 "Grey dotted line is the initial guess for the first loop, "
                 f"\n blue dotted line is the initial guess for the second "
                 "loop, green dotted for the third loop, "
                 f"\n The grey solid line is the actual path taken by "
                 "the mass center of the truck."
                 )
    coords_vals = coords_lam(*state_sol(t)[0:], *input_sol(t)[0:2],
                             *pL_vals)

    # Update the various objects
    arrow.set_positions(np.array([coords_vals[0, 2], coords_vals[1, 2]]),
                        np.array([coords_vals[0, 19], coords_vals[1, 19]]))

    P1f_p.set_offsets((coords_vals[0, 1], coords_vals[1, 1]))
    P1b_p.set_offsets((coords_vals[0, 2], coords_vals[1, 2]))
    P2f_p.set_offsets((coords_vals[0, 3], coords_vals[1, 3]))
    P2b_p.set_offsets((coords_vals[0, 4], coords_vals[1, 4]))
    P3f_p.set_offsets((coords_vals[0, 5], coords_vals[1, 5]))
    P3b_p.set_offsets((coords_vals[0, 6], coords_vals[1, 6]))
    P1fl_p.set_offsets((coords_vals[0, 7], coords_vals[1, 7]))
    P1fr_p.set_offsets((coords_vals[0, 8], coords_vals[1, 8]))
    P1bl_p.set_offsets((coords_vals[0, 9], coords_vals[1, 9]))
    P1br_p.set_offsets((coords_vals[0, 10], coords_vals[1, 10]))
    P2fl_p.set_offsets((coords_vals[0, 11], coords_vals[1, 11]))
    P2fr_p.set_offsets((coords_vals[0, 12], coords_vals[1, 12]))
    P2bl_p.set_offsets((coords_vals[0, 13], coords_vals[1, 13]))
    P2br_p.set_offsets((coords_vals[0, 14], coords_vals[1, 14]))
    P3fl_p.set_offsets((coords_vals[0, 15], coords_vals[1, 15]))
    P3fr_p.set_offsets((coords_vals[0, 16], coords_vals[1, 16]))
    P3bl_p.set_offsets((coords_vals[0, 17], coords_vals[1, 17]))
    P3br_p.set_offsets((coords_vals[0, 18], coords_vals[1, 18]))

    fax1.set_data([coords_vals[0, 7], coords_vals[0, 8]],
                  [coords_vals[1, 7], coords_vals[1, 8]])
    bax1.set_data([coords_vals[0, 9], coords_vals[0, 10]],
                  [coords_vals[1, 9], coords_vals[1, 10]])
    fax2.set_data([coords_vals[0, 11], coords_vals[0, 12]],
                  [coords_vals[1, 11], coords_vals[1, 12]])
    bax2.set_data([coords_vals[0, 13], coords_vals[0, 14]],
                  [coords_vals[1, 13], coords_vals[1, 14]])
    fax3.set_data([coords_vals[0, 15], coords_vals[0, 16]],
                  [coords_vals[1, 15], coords_vals[1, 16]])
    bax3.set_data([coords_vals[0, 17], coords_vals[0, 18]],
                  [coords_vals[1, 17], coords_vals[1, 18]])

    truck_p.set_data([coords_vals[0, 1], coords_vals[0, 2]],
                     [coords_vals[1, 1], coords_vals[1, 2]])
    trailer1_p.set_data([coords_vals[0, 3], coords_vals[0, 4]],
                        [coords_vals[1, 3], coords_vals[1, 4]])
    trailer2_p.set_data([coords_vals[0, 5], coords_vals[0, 6]],
                        [coords_vals[1, 5], coords_vals[1, 6]])

    towbar1_p.set_data([coords_vals[0, 2], coords_vals[0, 3]],
                       [coords_vals[1, 2], coords_vals[1, 3]])
    towbar2_p.set_data([coords_vals[0, 4], coords_vals[0, 5]],
                       [coords_vals[1, 4], coords_vals[1, 5]])

    # Trace Dmc1
    x_list, y_list = [], []
    for zeit in np.arange(t0, t, 1.0/fps):
        coords_vals1 = coords_lam(*state_sol(zeit)[0:],
                                  *input_sol(zeit)[0:2],
                                  *pL_vals)
        x_list.append(coords_vals1[0, 2])
        y_list.append(coords_vals1[1, 2])
    Dmc1_p.set_data(x_list, y_list)

    return (P1f_p, P1b_p, P2f_p, P2b_p, P3f_p, P3b_p,
            P1fl_p, P1fr_p, P1bl_p, P1br_p,
            P2fl_p, P2fr_p, P2bl_p, P2br_p,
            P3fl_p, P3fr_p, P3bl_p, P3br_p,
            fax1, bax1, fax2, bax2, fax3, bax3,
            truck_p, trailer1_p, trailer2_p,
            towbar1_p, towbar2_p, Dmc1_p)


ani = FuncAnimation(fig, update,
                    frames=np.concatenate((np.arange(0, tf, 1.0/fps),
                                           np.array([tf]))),
                    interval=1000/fps, blit=False)

plt.show()
