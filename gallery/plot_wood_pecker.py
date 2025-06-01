# %%
r"""
Wood Pecker
===========

Objective
---------

- Show how to use sympy.Piecewise(....) to set up equations of motion.


Description
-----------

A ring with rectangular crossection, inner radius :math:`r_i` and outer radius
:math:`r_o` slides on a vertical peg with :math:`r_p < r_i`. It has mass
:math:`m_R` and central moment of inertia around the Z axis of :math:`i_{ZZR}`.
A body, the **woodpecker**, of mass :math:`m_W` and  central moment of inertia
around the Z axis of :math:`i_{ZZW}` is attached to the ring via a torque
spring of strength :math:`k_W`.
The pivoting point P of the woodpecker is at distance :math:`r_o` from the
center of gravity of the ring, :math`Dmc_R`, the center of gravity of the
woodpecker, :math:`Dmc_W` is at a distance :math:`d_W` from P.

The angle of rotation of the ring is :math:`q_R`. if the ring tilts more than
an angle :math:`max_{kipp}` it contacts the peg.
Then a force :math:`F_t = k_t \cdot \Delta(max_{\textrm{kipp}}, q_R)`
effectively stops the ring from tilting much more. A strong friction force,
:math:`F_f = -uy_R \cdot reibung \cdot F_t` effectively stops the further
sliding of the ring.

This link shows a video of what is simulated.
https://github.com/Peter230655/Miscellaneous/blob/main/woodpecker.MOV


Parameters and Variables

- :math:`N`: inertial frame
- :math:`AR`: body fixed frame of the ring
- :math:`AW`: body fixed frame of the woodpecker
- :math:`O`: fixed point (origin of N)
- :math:`P`: point on outer surface of the ring, where the woodpecker is
  attached to the ring
- :math:`Dmc_R, Dmc_W`: centers of gravity of the ring and the woodpecker
- :math:`m_R, m_W, i_{ZZR}, i_{ZZW}`: masses and moments of inertia of ring
  and woodpecker
- :math:`q_R, q_W, u_R, u_W`: angles and angular speeds of AR, AW
- :math:`x_R, y_R, ux_R, uy_R`: location and speed of :math:`Dmc_R`.
  :math:`Dmc_R` can only move in Y direction, so :math:`x_R, ux_R \equiv 0`
- :math:`k_W`: constant of the torque spring connecting the woodpecker to the
  ring
- :math:`k_R`: constant of the spring, with which the peg pushes against the
  ring upon impact.
- :math:`\textrm{reibung}`: coefficient of friction between ring and peg upon
  contact.
- :math:`r_o`: outer radius of the ring
- :math:`d`: distance from P to :math:`Dmc_W`
- :math:`max_{\textrm{kipp}}`: tilting angle at which the ring touches the peg
- :math:`\textrm{korrekt}`: put the 'neutral location' of the torque spring at
  :math:`\textrm{korrekt}` radians w.r.t. AW. Set by trial and error to get the
  woodpecker to wiggle around AW.x

"""
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import animation
from matplotlib import patches
import matplotlib.pyplot as plt

# %%
# Set up the System
# -----------------

N, AR, AW = sm.symbols('N, AR, AW', cls=me.ReferenceFrame)
O, DmcR, DmcW, P = sm.symbols('O, DmcR, DmcW, P', cls=me.Point)

t = me.dynamicsymbols._t

qR, uR, qW, uW = me.dynamicsymbols('qR, uR, qW, uW')
yR, uyR = me.dynamicsymbols('yR, uyR')

mR, mW, g, iZZR, iZZW, ro, d, max_kipp, kR, kW, reibung, korrekt = \
    sm.symbols(('mR, mW, g, iZZR, iZZW, ro, d, max_kipp, kR, kW, reibung,'
                'korrekt'))

O.set_vel(N, 0)

AR.orient_axis(N, qR, N.z)
AR.set_ang_vel(N, uR * N.z)
AW.orient_axis(AR, qW, AR.z)
rot = AW.ang_vel_in(N)
AW.set_ang_vel(AR, uW * AR.z)
rot1 = AW.ang_vel_in(N)

DmcR.set_pos(O, yR*N.y)
DmcR.set_vel(N, uyR*N.y)

P.set_pos(DmcR, ro*AR.x - ro/5.*AR.y)
P.v2pt_theory(DmcR, N, AR)

DmcW.set_pos(P, d*AW.x)
DmcW.v2pt_theory(P, N, AW)

iR = me.inertia(AR, 0., 0., iZZR)
iW = me.inertia(AW, 0., 0., iZZW)
Ring = me.RigidBody('Ring', DmcR, AR, mR, (iR, DmcR))
Woodpecker = me.RigidBody('Woodpecker', DmcW, AW, mW, (iW, DmcW))
BODY = [Ring, Woodpecker]

# %%
# Set up the **forces** acting on the system.

# The torque needs to to effectively stop the ring from penetrating the peg,
# and a force to stop the sliding of the ring.
# They just have to be strong, this is why it should not matter whether they
# are set up mechanically correct.
# I return *eindring*, because I need it for the spring energy below.\
# I return hilfs to plot the forces.
#
# NOTE: sm.Piecewise((..), (..), .., (..)) returns the first conditon which is
# True from left to right. Therefore, the sequence is important.

# %%


def kraefte():
    FG = [(DmcR, -mR*g*N.y), (DmcW, -mW*g*N.y)]         # gravity
    # torque if woodpecker is not 'in line' with the ring
    FT1 = [(AW, -kW * (qW-korrekt) * N.z), (AR, kW * (qW-korrekt) * N.z)]

    # When - max_kipp > qR or qR > max_kipp, the ring touches the peg
    faktor1 = sm.Piecewise((1., -max_kipp > qR), (1., qR >= max_kipp),
                           (0., True))

    # the 'penetration' is qR - max_kipp if qR > max_kipp, and -(qR + max_kipp)
    # if qR < -max_kipp
    eindring = sm.Piecewise((qR - max_kipp, qR > max_kipp),
                            (-qR - max_kipp, qR < -max_kipp), (0., True))

    # This penetration will create a torque on AR. The direction of the torque
    # is needed,
    faktor2 = sm.Piecewise((-1., max_kipp < qR), (1., True))
    kraft = kR * eindring * faktor1 * faktor2
    FT2 = [(AR, kraft * N.z)]

    # friction acting on DmcR to stop the sliding. It is speed dependent to
    # avoid that the ring might be driven upward.
    kraft2 = sm.Abs(kraft)
    FR = [(DmcR, -uyR * reibung * kraft2 * N.y)]
    FL = FG + FT1 + FT2 + FR
    hilfs = [kraft, -uyR * reibung*kraft2]
    return FL, eindring, hilfs


FL, eindring, krafte = kraefte()
# %%
# Set up Kane's equations

q_ind = [qR, qW, yR]
u_ind = [uR, uW, uyR]
kd = [i - j.diff(t) for i, j in zip(u_ind, q_ind)]

KM = me.KanesMethod(N, q_ind=q_ind, u_ind=u_ind, kd_eqs=kd)
fr, frstar = KM.kanes_equations(BODY, FL)
MM = KM.mass_matrix_full
force = KM.forcing_full

print('force DS', me.find_dynamicsymbols(force))
print('force free symbols', force.free_symbols)
print((f'force has {sm.count_ops(force)} operations, '
       f'{sm.count_ops(sm.cse(force))} operations after cse', '\n'))

print('MM DS', me.find_dynamicsymbols(MM))
print('MM free symbols', MM.free_symbols)
print((f'MM has {sm.count_ops(MM)} operations, '
       f'{sm.count_ops(sm.cse(MM))} operations after cse', '\n'))


# %%
# Set up the **energies**
kin_energie = BODY[0].kinetic_energy(N) + BODY[1].kinetic_energy(N)
pot_energie = mR*g*yR + mW*g*me.dot(DmcW.pos_from(O), N.y)
spring_energie = 0.5 * kW * (qW - korrekt)**2 + 0.5 * kR * eindring**2

# %%
# Convert the **sympy functions** into **numpy functions**.
qL = q_ind + u_ind
pL = [mR, mW, g, iZZR, iZZW, ro, d, kR, kW, reibung, max_kipp, korrekt]

combined = sm.lambdify(qL + pL, [MM, force, kin_energie, pot_energie,
                                 spring_energie], cse=True)
energien = sm.lambdify(qL + pL, [kin_energie, pot_energie, spring_energie],
                       cse=True)

krafte_lam = sm.lambdify(qL + pL, krafte, cse=True)

# %%
# Numerical Integration
# ---------------------
#
# Variables / starting coordinates
mR1 = 1.
mW1 = 15.
iZZR1 = 1.
iZZW1 = 10.
ro1 = 1.
d1 = 2.
g1 = 9.81

kR1 = 1.e6
kW1 = 2.e3
reibung1 = 0.75
max_kipp1 = 2.    # in degrees. Will be converted to radians later

korrekt1 = 10.    # in degrees. Just to make the woodpecker to swing more
                  # 'in the center'.

qR1 = 0
qW1 = -30   # in degrees. Will be converted to radians later
yR1 = 0.
uR1 = 0.
uW1 = 0.
uy1 = 0.

intervall = 30.     # intervall in sec to be simulated
schritte = 500     # steps per sec to be returned

pL_vals = [mR1, mW1, g1, iZZR1, iZZW1, ro1, d1, kR1, kW1, reibung1,
           np.deg2rad(max_kipp1), np.deg2rad(korrekt1)]
y0 = [qR1, np.deg2rad(qW1), yR1, uR1, uW1, uy1]


def gradient(t, y, args):
    sol = np.linalg.solve(combined(*y, *args)[0], combined(*y, *args)[1])
    return np.array(sol).T[0]


times = np.linspace(0., intervall, int(schritte*intervall))
t_span = (0., intervall)

resultat1 = solve_ivp(gradient, t_span, y0, t_eval=times, args=(pL_vals,))
resultat = resultat1.y.T
print('Shape of result: ', resultat.shape)
print(resultat1.message)

# %%
# Plot some generalized coordinates

bezeichnung = [str(i) for i in qL]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
for i in (0, 1):
    ax1.plot(times, np.rad2deg(resultat[:, i]), label=bezeichnung[i])
ax1.set_ylabel('angle (degrees)')
ax1.set_title('Generalized coordinates')
ax1.axhline(max_kipp1, linestyle='--', color='black', lw=0.5)
ax1.axhline(-max_kipp1, linestyle='--', color='black', lw=0.5)
ax1.legend()

kraft1 = np.array(krafte_lam(*[resultat[:, i]
                               for i in range(resultat.shape[1])],
                             *pL_vals)[0])
kraft2 = np.array(krafte_lam(*[resultat[:, i]
                               for i in range(resultat.shape[1])],
                             *pL_vals)[1])
ax2.plot(times, kraft1, label='Torque to stop rotation of the ring')
ax2.plot(times, kraft2, label='Friction to slow down the ring')
ax2.set_title('Torque, Force')
ax2.legend()
ax3.plot(times, resultat[:, 2], label='Location of the ring')
ax3.plot(times, resultat[:, 5], label='linear speed of the ring')
ax3.set_xlabel('time (sec)')
ax3.set_title('location and speed of the ring')
ax3.axhline(0., linestyle='--', color='black', lw=0.5)
_ = ax3.legend()

# %%
# Energies of the system.
#
# As expected the total energy drops as there is friction.

kin_np = np.array(energien(*[resultat[:, i]
                             for i in range(resultat.shape[1])], *pL_vals)[0])
pot_np = np.array(energien(*[resultat[:, i]
                             for i in range(resultat.shape[1])], *pL_vals)[1])
spring_np = np.array(energien(*[resultat[:, i]
                                for i in range(resultat.shape[1])],
                              *pL_vals)[2])
total_np = kin_np + pot_np + spring_np

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.plot(times, kin_np, label='kinetic energy')
ax1.plot(times, pot_np, label='potential energy')
ax1.plot(times, spring_np, label='spring energy')
ax1.plot(times, total_np, label='total energy')
ax1.set_ylabel('energy(Nm)')
ax1.set_title('Energies of the system')
ax1.legend()
ax2.plot(times, spring_np, label='spring energy')
ax2.set_title('Spring energy separated out')
ax2.set_xlabel('time (sec)')
ax2.set_ylabel('energy (Nm)')
_ = ax2.legend()

# %%
# Animation
# ---------

# reduce the number of points of time to around zeitpunkte
times2 = []
resultat2 = []
index2 = []

# =======================
zeitpunkte = 500
# =======================

reduction = max(1, int(len(times)/zeitpunkte))

for i in range(len(times)):
    if i % reduction == 0:
        times2.append(times[i])
        resultat2. append(resultat[i])

schritte2 = len(times2)
print(f'animation used {schritte2} points in time')
resultat2 = np.array(resultat2)
times2 = np.array(times2)

# set up the ring, which is modelled as a rectangle of width = 2 * ro,
# and height = width / 5
# attachment point of the weoodpecker on the ring is point P
P_loc = [me.dot(P.pos_from(O), uv) for uv in (N.x, N.y)]
DmcW_loc = [me.dot(DmcW.pos_from(O), uv) for uv in (N.x, N.y)]
# set up the anchor point of the rectangle
P1, P2, P3 = sm.symbols('P1, P2, P3', cls=me.Point)
P1.set_pos(DmcR, -ro * AR.x - ro / 2.5 * AR.y)
P2.set_pos(DmcW, 2. * ro*AW.y)   # head of woodpecker
P3.set_pos(P2, -d * AW.x)  # end of beak
P1_loc = [me.dot(P1.pos_from(O), uv) for uv in (N.x, N.y)]
P2_loc = [me.dot(P2.pos_from(O), uv) for uv in (N.x, N.y)]
P3_loc = [me.dot(P3.pos_from(O), uv) for uv in (N.x, N.y)]

P_loc_lam = sm.lambdify(qL + pL, P_loc, cse=True)
P1_loc_lam = sm.lambdify(qL + pL, P1_loc, cse=True)
P2_loc_lam = sm.lambdify(qL + pL, P2_loc, cse=True)
P3_loc_lam = sm.lambdify(qL + pL, P3_loc, cse=True)


DmcW_loc_lam = sm.lambdify(qL + pL, DmcW_loc, cse=True)

P_x = np.array(P_loc_lam(*[resultat2[:, i] for i in range(resultat2.shape[1])],
                         *pL_vals)[0])
P_y = np.array(P_loc_lam(*[resultat2[:, i] for i in range(resultat2.shape[1])],
                         *pL_vals)[1])

P1_x = np.array(P1_loc_lam(*[resultat2[:, i]
                             for i in range(resultat2.shape[1])], *pL_vals)[0])
P1_y = np.array(P1_loc_lam(*[resultat2[:, i]
                             for i in range(resultat2.shape[1])], *pL_vals)[1])

P2_x = np.array(P2_loc_lam(*[resultat2[:, i]
                             for i in range(resultat2.shape[1])], *pL_vals)[0])
P2_y = np.array(P2_loc_lam(*[resultat2[:, i]
                             for i in range(resultat2.shape[1])], *pL_vals)[1])

P3_x = np.array(P3_loc_lam(*[resultat2[:, i]
                             for i in range(resultat2.shape[1])], *pL_vals)[0])
P3_y = np.array(P3_loc_lam(*[resultat2[:, i]
                             for i in range(resultat2.shape[1])], *pL_vals)[1])


DmcW_x = np.array(DmcW_loc_lam(*[resultat2[:, i]
                                 for i in range(resultat2.shape[1])],
                               *pL_vals)[0])
DmcW_y = np.array(DmcW_loc_lam(*[resultat2[:, i]
                                 for i in range(resultat2.shape[1])],
                               *pL_vals)[1])

DmcR_x = np.array([0. for _ in range(schritte2)])
DmcR_y = resultat2[:, 2]

# needed to give the picture the right size.
xmin = -ro1 - 1
xmax1 = np.max(DmcW_x)
xmax2 = np.max(DmcR_x)
xmax = max(xmax1, xmax2) + 2.

ymin1 = np.min(DmcW_y)
ymin2 = np.min(DmcR_y)
ymin = min(ymin1, ymin2) - 1
ymax1 = np.max(DmcW_y)
ymax2 = np.max(DmcR_y)
ymax = max(ymax1, ymax2) + 2


def animate_pendulum(times):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'aspect': 'equal'})

    ax.axis('on')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axvline(-ro1/2., color='black')
    ax.axvline(ro1/2, color='black')
    ax.axvspan(-ro1/2., ro1/2, color='black', alpha=0.5)

    line1, = ax.plot([], [], color='blue')   # line connecting P to DmcW
    # line connecting DmcW to P2
    line2, = ax.plot([], [], color='green', marker='o', markersize=5)
    # line connecting P2 to P3
    line3, = ax.plot([], [], color='red', marker='o', markersize=5)

    line4, = ax.plot([DmcW_x[0]], [DmcW_y[0]], color='green', marker='o',
                     ms=10)
    line5, = ax.plot([P2_x[0]], [P2_y[0]], color='green', marker='o', ms=25)
    line6, = ax.plot([P2_x[0]], [P2_y[0]], color='yellow', marker='o', ms=5)

    ring = patches.Rectangle((P1_x[0], P1_y[0]), width=2.*ro1, height=2.*ro1/5,
                             angle=np.rad2deg(resultat2[0, 0]),
                             rotation_point='center', color='blue')
    ax.add_patch(ring)

    def animate(i):
        message = (f'running time {times[i]:.2f}')
        ax.set_title(message, fontsize=12)
        ax.set_xlabel('X direction', fontsize=12)
        ax.set_ylabel('Y direction', fontsize=12)
        ring.set_xy((P1_x[i], P1_y[i]))
        ring.set_angle(np.rad2deg(resultat2[i, 0]))

        werte_x = [P_x[i], DmcW_x[i]]
        werte_y = [P_y[i], DmcW_y[i]]
        line1.set_data([werte_x, werte_y])

        werte_x = [DmcW_x[i], P2_x[i]]
        werte_y = [DmcW_y[i], P2_y[i]]
        line2.set_data([werte_x, werte_y])

        werte_x = [P2_x[i], P3_x[i]]
        werte_y = [P2_y[i], P3_y[i]]
        line3.set_data([werte_x, werte_y])

        line4.set_data([[DmcW_x[i]], [DmcW_y[i]]])
        line5.set_data([[P2_x[i]], [P2_y[i]]])
        line6.set_data([[P2_x[i]], [P2_y[i]]])

        return line1, line2, line3, line4, line5, line6

    anim = animation.FuncAnimation(fig, animate, frames=schritte2,
                                   interval=150*np.max(times2) / schritte2,
                                   blit=True)
    return anim


anim = animate_pendulum(times2)
plt.show()
