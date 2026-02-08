# %%
r"""
Ellipse Rolling on Very Uneven Street
=====================================

Objectives
----------

- Show an application of the ``events`` feature of
  ``scipy.integrate.solve_ivp`` to detect events during integration. (Here
  the event is a :math:`2^{\text{nd}}` contact point of the ellipse with
  the street.)

Description
-----------

An ellipse is rolling on an uneven street without slipping or jumping.
The street may be quite wavy so second contact points may appear.

Notes
-----

- The main issues are geometrical: how to find the center of the ellipse given
  the contact point and the slope of the street at this point, and how to
  relate the speed of the contact point to the angular speed of the ellipse.
  This is described in detail in the simulation
  ``Ellipse rolling on uneven street``
- Reaction forces on the contact point :math:`C_P` cannot be calculated in this
  model. Presumably because it is not attached to the ellipse as far as this
  model is concerned.
- Note the coment in the function ``event`` near the statement ``np.isclose``
- The special case of the ellipse running on a horizontal line is solved
  explicitly here:
  https://www.mapleprimes.com/DocumentFiles/210428_post/rolling-ellipse.pdf

**Parameters**

- :math:`N` : inertial frame
- :math:`A` : frame fixed to the ellipse
- :math:`P_0` : point fixed in *N*
- :math:`Dmc` : center of the ellipse
- :math:`C_P` : contact point
- :math:`P_o` : location of the particle fixed to the ellipse

- :math:`q, u` : angle of rotation of the ellipse, its speed
- :math:`x, u_x` : X coordinate of the contact point CP, its speed
- :math:`m_x, m_y, um_x, um_y` : coordinates of the center of the ellipse,
  its speeds

- :math:`m, m_o` : mass of the ellipse, of the particle attached to the ellipse
- :math:`a, b` : semi axes of the ellipse
- :math:`\textrm{amplitude}, \textrm{frequenz}` : parameters for the street.
- :math:`i_{ZZ}`: moment of inertia of the ellipse around the Z axis
- :math:`\alpha, \beta`: determine the location of the particle w.r.t. Dmc
- :math:`aux_x, aux_y, f_x, f_y`: needed for the reaction forces
- :math:`rhs_0....rhs_6`: place holders for :math:`RHS = MM^{-1} \cdot force`
  calculated numerically later. Needed for the reaction forces.

 """

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, root, fsolve
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib import patches

# %%
# Needed to exit a loop.


class Rausspringen(Exception):
    pass


# %%
# Set up the Equations of Motion
# ------------------------------
#
# Frames and points.
N, A = sm.symbols('N, A', cls=me.ReferenceFrame)
P0, Dmc, CP, Po = sm.symbols('P0, Dmc, CP, Po', cls=me.Point)
t = sm.symbols('t')
P0.set_vel(N, 0)


# %%
# Parameters of the system.
m, mo, g, CPx, CPy, a, b, iZZ, alpha, beta, amplitude, frequenz = sm.symbols(
    'm, mo, g, CPx, CPy, a, b, iZZ, alpha, beta, amplitude, frequenz')


# %%
# Center of mass of the ellipse.
mx, my, umx, umy = me.dynamicsymbols('mx, my, umx, umy')


# %%
# Rotation of the ellipse, coordinates of the contact point.
q, x, y, u, ux, uy = me.dynamicsymbols('q, x, y, u, ux, uy')


# %%
# Needed for the reaction forces.
auxx, auxy, fx, fy = me.dynamicsymbols('auxx, auxy, fx, fy')


# %%
# Placeholders for the right-hand sides of the ODEs. Needed for the reaction
# forces.
rhs_list = [sm.symbols('rhs' + str(i)) for i in range(10)]


# %%
# Orientation of the body fixed frame of the ellipse
A.orient_axis(N, q, N.z)
A.set_ang_vel(N, u*N.z)


# %%
# Model the street.
#
# It is a parabola, open to the top, with superimposed sinus waves.
# Then its osculating circle is calculated.

rumpel = 3  # the higher the number the more 'uneven the street'

strasse = sum([amplitude/j * sm.sin(j*frequenz * x)
               for j in range(1, rumpel)])
strassen_form = (amplitude/2. * x)**2
gesamt = strassen_form + strasse
gesamtdx = gesamt.diff(x)
r_max = ((sm.S(1.) + (gesamt.diff(x))**2)**sm.S(3/2)/gesamt.diff(x, 2))


# %%
# Formula to find the center of an ellipse with semi axes a, b, given
# an point P(x / y) on the ellipse and the slope of the ellipse at this point.
# (Long dialogue with chatGPT)
#
#  - x : x - coordinate (in the inertial system N) of the point
#  - y : y - coordinate (in the inertial system N) of the point
#  - q : rotation of the ellipse
#  - :math:`k_0` : slope of the ellipse at P
#
#
# this gives the center :math:`M(m_x / m_y)` of the ellipse as:
#
# - :math:`m_x = x - \dfrac{k0 \cdot (a^2 \cdot \cos^2(q) + b^2 \cdot
#   \sin^2(q)) -
#   (a^2 - b^2) \cdot \sin(q) \cdot \cos(q)}{\sqrt{k0^2 \cdot (a^2 \cdot
#   \cos^2(q) + b^2 \cdot \sin^2(q)) - 2\cdot k0 \cdot (a^2 - b^2) \cdot
#   \sin(q) \cdot \cos(q) + (a^2 \cdot \sin^2(q) + b^2 \cdot \cos^2(q))}}`
#
# - :math:`m_y = y - \dfrac{k0 \cdot(a^2 - b^2) \cdot \sin(q) \cdot \cos(q) -
#   (a^2 \cdot \cos^2(q) + b^2 \cdot \sin^2(q))}{\sqrt{k0^2 \cdot (a^2 \cdot
#   \cos^2(q) + b^2 \cdot \sin^2(q)) - 2\cdot k0 \cdot (a^2 - b^2) \cdot
#   \sin(q) \cdot \cos(q) + (a^2 \cdot \sin^2(q) + b^2 \cdot \cos^2(q))}}`
#
# The slope of the ellipse at the contact point must be equal to the slope of
# the road at the contact point.

k0 = gesamt.diff(x)
denom = sm.sqrt(k0**2 * (a**2 * sm.cos(q)**2 + b**2 * sm.sin(q)**2) -
                2 * k0 * (a**2 - b**2) * sm.sin(q) * sm.cos(q) +
                (a**2 * sm.sin(q)**2 + b**2 * sm.cos(q)**2))

num_x = k0 * (a**2 * sm.cos(q)**2 + b**2 * sm.sin(q)**2) - \
            (a**2 - b**2) * sm.sin(q) * sm.cos(q)

num_y = k0 * (a**2 - b**2) * sm.sin(q) * sm.cos(q) - \
            (a**2 * sm.sin(q)**2 + b**2 * sm.cos(q)**2)

mx_c = x - num_x / denom
my_c = gesamt - num_y / denom


# %%
# For the velocity constraints the speed of the center of the ellipse is
# needed.

umx_c = mx_c.diff(t).subs({x.diff(t): ux, q.diff(t): u})
umy_c = my_c.diff(t).subs({x.diff(t): ux, q.diff(t): u})


# %%
# Relationship of :math:`\dot{x}(t)` to :math:`\dot{q}(t)`:
#
# The goal is to find the speed of the contact point as a function of the
# angular velocity of the ellipse, and of course other parameters.
#
# The formula below was found by (the paid for version of) chatGPT
#
# :math:`\dot {x}(t) = - \dfrac{(1 + \left[\frac{d}{dx}f(x)\right]^2) \cdot a^2
# \cdot b^2}
# {\left(a^2 \cdot(\sin(q) - \frac{d}{dx}f(x) \cdot \cos(q))^2 + b^2
# \cdot(\cos(q) +
# \frac{d}{dx}f(x) \cdot \sin(q))^2 \right)^{\frac{3}{2}} - a^2 \cdot b^2 \cdot
# \frac{d^2}{dx^2}f(x) } \cdot \dot{q}(t)`
#

# %%
fdx = gesamt.diff(x)
fdxdx = fdx.diff(x)

denom = ((a*a*(sm.sin(q) - fdx*sm.cos(q))**2 + b*b*(sm.cos(q) +
                                                    fdx*sm.sin(q))**2)**(3/2) -
         a**2 * b**2 * fdxdx)

num = (1 + gesamt.diff(x)**2) * a**2 * b**2

rhsx = -u * num / denom

# %%
# Contact point position and velocity.
CP.set_pos(P0, x*N.x + y*N.y)
CP.set_vel(N, ux*N.x + uy*N.y)


# %%
# Ellipse center position and velocity.
Dmc.set_pos(P0, mx*N.x + my*N.y)
Dmc.set_vel(N, umx*N.x + umy*N.y + auxx*N.x + auxy*N.y)


# %%
# particle on ellipse position and velocity
Po.set_pos(Dmc, a*alpha*A.x + b*beta*A.y)
_ = Po.v2pt_theory(Dmc, N, A)


# %%
# Kane's Equations.

Inert = me.inertia(A, 0., 0., iZZ)
bodye = me.RigidBody('bodye', Dmc, A, m, (Inert, Dmc))
Poa = me.Particle('Poa', Po, mo)
BODY = [bodye, Poa]

FL = [(Dmc, -m*g*N.y + fx*N.x + fy*N.y), (Po, -mo*g*N.y)]

kd = sm.Matrix([
    u - q.diff(t),
    ux - x.diff(t),
    uy - y.diff(t),
    umx - mx.diff(t),
    umy - my.diff(t),
])

speed_constr = sm.Matrix([
    umx - umx_c,
    umy - umy_c,
    ux - rhsx,
    uy - gesamt.diff(t),
])

q1 = [q, x, y, mx, my]
u_ind = [u]
u_dep = [ux, uy, umx, umy]
aux = [auxx, auxy]

KM = me.KanesMethod(
    N,
    q_ind=q1,
    u_ind=u_ind,
    u_dependent=u_dep,
    kd_eqs=kd,
    velocity_constraints=speed_constr,
    u_auxiliary=aux,
)

fr, frstar = KM.kanes_equations(BODY, FL)
MM = KM.mass_matrix_full

# %%
# As Dmc has both real speeds and auxiliary speeds, the force vector
# contains the reaction forces :math:`f_x, f_y`. As they do no work
# they must be set to zero.
force = KM.forcing_full.subs({fx: 0., fy: 0.})


# %%
# Replace the accelerations appearing in eingepraegt with the corresponding
# place holders for the right-hand sides of the equations of motion.
eingepraegt_dict = {
    sm.Derivative(x, t): rhsx,
    sm.Derivative(u, t): rhs_list[5],
    sm.Derivative(umx, t): rhs_list[8],
    sm.Derivative(umy, t): rhs_list[9]}
eingepraegt1 = (KM.auxiliary_eqs).subs(eingepraegt_dict)


# %%
# Print some information about the symbolic expressions.
print('eingepraegt1 DS', me.find_dynamicsymbols(eingepraegt1))
print('eingepraegt1 free symbols', eingepraegt1.free_symbols)
print(f'eingepraegt1 has {sm.count_ops(eingepraegt1):,} operations', '\n')

print('force DS', me.find_dynamicsymbols(force))
print('force free symbols', force.free_symbols)
print(f'force has {sm.count_ops(force):,} operations', '\n')

print('MM DS', me.find_dynamicsymbols(MM))
print('MM free symbols', MM.free_symbols)
print(f'MM has {sm.count_ops(MM):,} operations', '\n')

# %%
# Define the energy.

# %%
pot_energie = (m * g * me.dot(Dmc.pos_from(P0), N.y) +
               mo * g * me.dot(Po.pos_from(P0), N.y))
kin_energie = sum([koerper.kinetic_energy(N).subs({i: 0. for i in aux})
                   for koerper in BODY])
print(me.find_dynamicsymbols(kin_energie))


# %%
# Compilation.

qL = q1 + u_ind + u_dep
pL = [m, mo, g, a, b, iZZ, alpha, beta, amplitude, frequenz]
F = [fx, fy]

MM_lam = sm.lambdify(qL + pL, MM, cse=True)
force_lam = sm.lambdify(qL + pL, force, cse=True)

gesamt = gesamt.subs({CPx: x})
gesamt_lam = sm.lambdify([x] + pL, gesamt, cse=True)
eingepraegt_lam = sm.lambdify(F + qL + pL + rhs_list, eingepraegt1,
                              cse=True)
pot_lam = sm.lambdify(qL + pL, pot_energie, cse=True)
kin_lam = sm.lambdify(qL + pL, kin_energie, cse=True)

r_max_lam = sm.lambdify([x] + pL, r_max, cse=True)


# %%
# Numerical Integration
# ---------------------
#
# Set the parameters.

m1 = 1.0
mo1 = 1.0
g1 = 9.8
a1 = 1.
b1 = 2.
amplitude1 = 0.5
frequenz1 = 1.5

alpha1 = 0.5
beta1 = 0.5

q11 = 1
u11 = 2.0
x11 = 10.0

intervall = 10.0

iZZ1 = 0.25 * m1 * (a1**2 + b1**2)
pL_vals = [m1, mo1, g1, a1, b1, iZZ1, alpha1, beta1, amplitude1, frequenz1]

punkte = 200
schritte = int(intervall * punkte)

# %%
# Find the initial center of the ellipse.

gesamt_dx = gesamt.diff(x)
gesamt_dx_lam = sm.lambdify([x] + [amplitude, frequenz], gesamt_dx, cse=True)

k0 = gesamt_dx

denom = sm.sqrt(k0**2 * (a**2 * sm.cos(q)**2 + b**2 * sm.sin(q)**2) -
                2 * k0 * (a**2 - b**2) * sm.sin(q) * sm.cos(q) +
                (a**2 * sm.sin(q)**2 + b**2 * sm.cos(q)**2))

num_x = k0 * (a**2 * sm.cos(q)**2 + b**2 * sm.sin(q)**2) - \
            (a**2 - b**2) * sm.sin(q) * sm.cos(q)

num_y = k0 * (a**2 - b**2) * sm.sin(q) * sm.cos(q) - \
            (a**2 * sm.sin(q)**2 + b**2 * sm.cos(q)**2)

mx = x - num_x / denom
my = gesamt - num_y / denom

mx_lam = sm.lambdify([x, q, a, b, amplitude, frequenz], mx, cse=True)
my_lam = sm.lambdify([x, q, a, b, amplitude, frequenz], my, cse=True)
mx1 = mx_lam(x11, q11, a1, b1, amplitude1, frequenz1)
my1 = my_lam(x11, q11, a1, b1, amplitude1, frequenz1)


# %%
# To get correct intial dependent speeds, the speed constraints must be solved.
matrix_A = speed_constr.jacobian((ux, uy, umx, umy))
vector_b = speed_constr.subs({ux: 0., uy: 0., umx: 0., umy: 0.})
loesung = (matrix_A.LUsolve(-vector_b)).subs({q.diff(t): u,
                                              x.diff(t): rhsx})
print('loesung DS', me.find_dynamicsymbols(loesung))
loesung = [loesung[i] for i in range(4)]
loesung_lam = sm.lambdify([x, q, u, a, b, amplitude, frequenz], loesung,
                          cse=True)
initial_speeds = loesung_lam(x11, q11, u11, a1, b1, amplitude1, frequenz1)


# %%
# Needed to calculate :math:`\frac{d}{dt}q(t)` after the impact, so the
# kinetic energy is the same right before and right after the impact.

kinetic_energy = kin_energie.subs({umx: loesung[2], umy: loesung[3]})
print(me.find_dynamicsymbols(kinetic_energy))

kin_lam_before = sm.lambdify([q, x, u, umx, umy] + pL, kin_energie, cse=True)
kin_lam_after = sm.lambdify([u, q, x] + pL, kinetic_energy, cse=True)


def new_u_function(u0, args):
    q_old, x_old, x_new, u_old, umx_old, umy_old, pL = args
    return (kin_lam_after(u0, q_old, x_new, *pL) -
            kin_lam_before(q_old, x_old, u_old, umx_old, umy_old, *pL))


# %%
# This function finds the lower of the two possible y coordinates of the
# ellipse for a given x coordinate of the contact point given rotation q,
# and given center of the ellipse.


def ellipse_y_min(q, x, mx, my, a, b):
    dx = x - mx
    s = np.sin(q)
    c = np.cos(q)

    A = s*s/a**2 + c*c/b**2
    B = 2*(dx*s*c*(1/a**2 - 1/b**2) - my*A)
    C = (dx*dx*(c*c/a**2 + s*s/b**2) - 2*dx*my*s*c*(1/a**2 - 1/b**2) +
         my*my*A - 1)

    D = B*B - 4*A*C
    if D < 0:
        return None  # no intersection

    return (-B - np.sqrt(D)) / (2*A)


# %%
# This give the slope of the ellipse at the contact point.
# Only used to verify the initial conditions.
X = (x - mx) * sm.cos(q) + (gesamt - my) * sm.sin(q)
Y = -(x - mx) * sm.sin(q) + (gesamt - my) * sm.cos(q)

KK1 = X * sm.cos(q) / a**2 - Y * sm.sin(q) / b**2
KK2 = X * sm.sin(q) / a**2 + Y * sm.cos(q) / b**2
KK = -KK1 / KK2
KK_lam = sm.lambdify([q, x, a, b, amplitude, frequenz], KK, cse=True)
gesamt_dx_lam = sm.lambdify([x] + [amplitude, frequenz], gesamt_dx, cse=True)


# %%
# Plot initial position of the ellipse.
gesamt_lam = sm.lambdify([x] + [amplitude, frequenz], gesamt, cse=True)
XX = np.linspace(-15, 15, 200)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(XX, gesamt_lam(XX, amplitude1, frequenz1))
ax.set_aspect('equal')
ax.set_title('Initial position of the ellipse')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
elli = patches.Ellipse((mx1, my1), width=2.*a1, height=2.*b1,
                       angle=np.rad2deg(q11), zorder=1, fill=True,
                       color='red', ec='black')
ax.add_patch(elli)
ax.scatter(mx_lam(x11, q11, a1, b1, amplitude1, frequenz1),
           my_lam(x11, q11, a1, b1, amplitude1, frequenz1), s=50)

y1 = gesamt_lam(x11, amplitude1, frequenz1)
_ = ax.scatter(x11, y1, s=50, color='orange')
print('Slope of ellipse and of street should be the same at the contact '
      'point:')
print(f"initial slope of street:  {gesamt_dx_lam(x11, amplitude1,
      frequenz1):.5f}")
print(f"initial slope of ellipse: {KK_lam(q11, x11, a1, b1, amplitude1,
      frequenz1):.5f}")

y_min = ellipse_y_min(q11, x11, mx1, my1, a1, b1)
if y_min is not None:
    ax.scatter(x11, y_min, s=20, color='black')

# %%
# Numerical Integration
# ---------------------
#
# Ensure that the particle is inside the ellipse.
if alpha1**2/a1**2 + beta1**2/b1**2 >= 1.:
    raise ValueError('Particle is outside the ellipse')


# %%
# max osculating circle of an ellipse

r_max = max(a1**2/b1, b1**2/a1)

# %%
# Find out if a second contact point is possible. If so, ensure at the
# starting position there is only one contact point.


def func2(x, args):
    # just needed to get the arguments matching for minimize
    return np.abs(r_max_lam(x, *args))


x0 = 0.1            # initial guess
minimal = minimize(func2, x0, pL_vals)
if r_max < (x111 := minimal.get('fun')):
    print(f'selected r_max = {r_max} is less than '
          f'radius = {x111:.2f}, hence no 2nd contact point possible.\n')
else:
    print(f'selected r_max {r_max} is larger than '
          f'{x111:.2f}, hence 2nd contact point possible.\n')
    # check whether at starting position there is only one contact point
    diameter = 2 * max(a1, b1)
    for xi in np.linspace(x11 - diameter, x11 + diameter, 200):
        yi = gesamt_lam(xi, amplitude1, frequenz1)
        y_min = ellipse_y_min(q11, xi, mx1, my1, a1, b1)
        if y_min is not None:
            if yi >= y_min + 1e-8:
                if not np.isclose(xi, x11, atol=1e-5):
                    raise ValueError('At the starting position there is more '
                                     'than one contact point!')

# %%
# Define the function to interrupt the integration when the ellipse hits a
# second contact point.


def event(t, y, args):
    global CP2, zaehler
    zaehler += 1
    # Only look in the direction the ellipse is moving
    vorzeichen = np.sign(y[5])
    diameter = 2.0 * np.max([a1, b1])
    # now look for a second contact point
    for xi in np.linspace(y[1], y[1] - vorzeichen * diameter, 200):
        try:
            yi = gesamt_lam(xi, amplitude1, frequenz1)
            y_min = ellipse_y_min(y[0], xi, y[3], y[4], a1, b1)
            if y_min is None or yi < y_min:
                pass
            else:
                # setting atol avoids detecting contact points too close to
                # the existing one. Otherwise it does not converge. Of course
                # the rad must not be too uneven.
                if not np.isclose(xi, y[1], atol=diameter/10.0):
                    # this is the second contact point
                    CP2 = xi
                    raise Rausspringen()
                else:
                    pass
        except Rausspringen:
            return yi - y_min  # event occurred
    if y_min is None:
        return -1.0
    else:
        return yi - y_min  # no event occurred


# %%
# Print initial conditions.

y0 = [q11, x11, gesamt_lam(x11, amplitude1, frequenz1), mx1, my1] + \
     [u11] + initial_speeds
print('initial conditions are:')
for i in range(len(y0)):
    print(f"{str(qL[i])} = {y0[i]:.3f}")


# %%
# if True, this stops the integration if event occurs.
event.terminal = True

# %%
# if True, data related to the occurence of events are printed.
event_info = False

# %%
# Right hand side of the differential equations.


def gradient(t, y, args):
    vals = np.concatenate((y, args))
    sol = np.linalg.solve(MM_lam(*vals), force_lam(*vals))
    return np.array(sol).T[0]


# %%
# Some variables needed to collect data.
runtime = 0.
starttime = 0.
starttime1 = 0.
zaehler = 0
ergebnis = []
count_events = 0
event_times = []
times = np.linspace(0.0, intervall, schritte)
max_step = 0.001

# %%
# Here the 'piecewise' integration starts.
while starttime < intervall:
    resultat1 = solve_ivp(gradient, (starttime, float(intervall)), y0,
                          t_eval=times, args=(pL_vals,),
                          atol=1.e-10,
                          rtol=1.e-10,
                          events=event,
                          method='DOP853',
                          max_step=max_step,
                          )

    resultat = resultat1.y.T
    if event_info:
        print('\n')
        print(resultat1.message)
        print('resultat shape', resultat.shape)

    if resultat1.y_events[0].size > 0.:
        count_events += 1
        # a 2nd contact point has been found, the results when the event
        # occurred are in resultat1.y_events[0][0]
        y0 = resultat1.y_events[0][0]
        kin_before_impact = kin_lam_before(y0[0], y0[1], y0[5], y0[8], y0[9],
                                           *pL_vals)
        if event_info:
            print('generalized coordinates at exit time'
                  f'{resultat1.y_events[0][0]}')

        # Calculate the new u, so the kinetic energy remains constant
        # q_old, x_old, x_new, u_old, umx_old, umy_old, pL = args
        args1 = [y0[0], y0[1], CP2, y0[5], y0[8], y0[9], pL_vals]
        u_new = fsolve(new_u_function, y0[5], args=args1)
        y0[5] = u_new[0]
        if event_info:
            print('Corrected angular speed u to keep kinetic energy '
                  f'constant: {y0[5]:.5f}')

        # Calculate the new dependent speeds
        y0[1] = CP2
        y0[2] = gesamt_lam(y0[1], amplitude1, frequenz1)

        new_speeds = loesung_lam(y0[1], y0[0], y0[5], a1, b1, amplitude1,
                                 frequenz1)
        y0[6] = new_speeds[0]
        y0[7] = new_speeds[1]
        y0[8] = new_speeds[2]
        y0[9] = new_speeds[3]

        kin_after_impact = kin_lam_before(y0[0], y0[1], y0[5], y0[8], y0[9],
                                          *pL_vals)
        if event_info is True:
            print(f'Kinetic energy before event: {kin_before_impact:.5f}, '
                  f'after event: {kin_after_impact:.5f}\n')

        event_times.append(resultat1.t_events[0][0])
        if event_info:
            print('time of event:', resultat1.t_events[0][0])
            print('New contact point CP2 at x = '
                  f'CP {y0[1]:.2f} /  {y0[2]:.2f}')
        starttime = resultat1.t_events[0][0]

        schritte = int((intervall - starttime) * punkte)
        times = np.linspace(starttime, intervall, schritte)

        ergebnis.append(resultat)
    else:
        break

print(resultat1.message)
print('Total number of events (2nd contact points):', count_events)

# %%
# Stack the individual results of the last integration, to get the
# complete results.
if count_events > 0:
    ergebnis.append(resultat)
    resultat = np.vstack(ergebnis)
    print(resultat.shape)

# %%
# Set these values for the subsequent plots below.
schritte = resultat.shape[0]
times = np.linspace(0., intervall, schritte)
print('how often did solve_ivp call event:', zaehler)

# %%
# Plot whatever generalized coordinates are of interest.

fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
bezeichnung = ['q', 'x', 'y', 'mx', 'my', 'u', 'ux', 'uy', 'umx', 'umy']
for i in (1, 2, 3, 4):
    ax.plot(times, resultat[:, i], label=bezeichnung[i])
ax.set_xlabel('time (sec)')
ax.axhline(0, color='gray', lw=0.5, linestyle='--')
ax.set_title(f'Generalized coordinates \n The red lines indicate the '
             f'times when a $2^{{nd}}$ contact point occured.')
_ = ax.legend()

for i in event_times:
    ax.axvline(i, color='red', lw=0.5, linestyle='--')

# %%
# Energy

kin_np = np.empty(schritte)
pot_np = np.empty(schritte)
total_np = np.empty(schritte)

for i in range(schritte):
    kin_np[i] = kin_lam(*[resultat[i, j] for j in range(resultat.shape[1])],
                        *pL_vals)
    pot_np[i] = pot_lam(*[resultat[i, j] for j in range(resultat.shape[1])],
                        *pL_vals)
    total_np[i] = kin_np[i] + pot_np[i]

fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
ax.plot(times, pot_np, label='potential energy')
ax.plot(times, kin_np, label='kinetic energy')
ax.plot(times, total_np, label='total energy')
ax.set_xlabel('time (sec)')
ax.set_ylabel("energy (Nm)")
ax.set_title('Energies of the system')
ax.legend()
total_max = np.max(total_np)
total_min = np.min(total_np)
print('max deviation of total energy from being constant is {:.2e} % of max'
      ' total energy'.format((total_max - total_min)/total_max * 100))


# %%
# Reaction forces.
#
# First :math:`RHS1 = MM^{-1} \cdot force` is solved numerically.
# Then, *eingepraegt_lam(...) = 0.0* for :math:`f_x, f_y` is solved.
# Iterantion sometimes helps to improve the results.

RHS1 = np.zeros((schritte, resultat.shape[1]))
for i in range(schritte):
    RHS1[i, :] = np.linalg.solve(
        MM_lam(*[resultat[i, j] for j in range(resultat.shape[1])], *pL_vals),
        force_lam(*[resultat[i, j]
                    for j in range(resultat.shape[1])],
                  *pL_vals)).reshape(resultat.shape[1])


def func(x11, *args):
    return eingepraegt_lam(*x11, *args).reshape(len(F))


kraft = np.empty((schritte, len(F)))
x0 = tuple([1. for i in range(len(F))])   # initial guess
for i in range(schritte):
    for _ in range(2):
        y00 = [resultat[i, j] for j in range(resultat.shape[1])]
        RHS2 = [RHS1[i, j] for j in range(RHS1.shape[1])]
        args = tuple((y00 + pL_vals + RHS2))
        A = root(func, x0, args=args)
        x0 = tuple(A.x)  # updated initial guess, may improve convergence
    kraft[i] = A.x

fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
ax.plot(times, kraft[:, 0], label='fx')
ax.plot(times, kraft[:, 1], label='fy')
ax.set_title('Reaction forces on Dmc, the center of the ellipse')
ax.set_xlabel('time (sec)')
ax.set_ylabel('force (N)')
_ = ax.legend()

# %%
# Animation
# ---------

fps = 8

t_arr = np.linspace(0.0, intervall, schritte)
state_sol = CubicSpline(t_arr, resultat)

coordinates = Po.pos_from(P0).to_matrix(N)
coords_lam = sm.lambdify(qL + pL, coordinates, cse=True)


def init():
    xmin, xmax = np.min(resultat[:, 1]) - 3., np.max(resultat[:, 1]) + 3.
    ymin, ymax = np.min(resultat[:, 2]) - 3., np.max(resultat[:, 2]) + 3.

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid()

    XX = np.linspace(xmin, xmax,  200)
    ax.plot(XX, gesamt_lam(XX, amplitude1, frequenz1), color='black')

    elli = patches.Ellipse((resultat[0, 3], resultat[0, 4]), width=2.*a1,
                           height=2.*b1, angle=np.rad2deg(resultat[0, 0]),
                           zorder=1, fill=True, color='red', ec='black')
    ax.add_patch(elli)

    line1 = ax.scatter([], [], color='blue', s=30)
    line2 = ax.scatter([], [], color='orange', s=30)
    line3 = ax.scatter([], [], color='black', s=30)
    line4 = ax.axvline(0, color='black', lw=0.75, linestyle='--')
#    line5 = ax.axvline(0, color='blue', lw=0.75, linestyle='--')

    return fig, ax, elli, line1, line2, line3, line4


fig, ax, elli, line1, line2, line3, line4 = init()


def update(t):
    message = ((f'Running time {t:.2f} sec \n The black dot is the '
                'particle'))
    ax.set_title(message, fontsize=12)

    elli.set_center((state_sol(t)[3], state_sol(t)[4]))
    elli.set_angle(np.rad2deg(state_sol(t)[0]))
    coords = coords_lam(*[state_sol(t)[j] for j in range(resultat.shape[1])],
                        *pL_vals).flatten()
    line1.set_offsets((state_sol(t)[3], state_sol(t)[4]))
    line2.set_offsets((state_sol(t)[1], state_sol(t)[2]))
    line3.set_offsets((coords[0], coords[1]))
    line4.set_xdata([state_sol(t)[1], state_sol(t)[1]])
#    line5.set_xdata([state_sol(t)[3], state_sol(t)[3]])
    return elli, line1, line2, line3, line4


frames = np.linspace(0, intervall, int(fps * (intervall)))
animation = FuncAnimation(fig, update, frames=frames, interval=1500/fps)
plt.show()
