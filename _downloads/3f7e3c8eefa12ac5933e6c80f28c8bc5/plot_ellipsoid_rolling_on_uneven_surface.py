# %%
r"""
Ellipsoid rolling on an uneven surface
======================================

Objective
---------

- Show how to use the instructions for an simpler example (here: ellipse
  rolling on an uneven line) to solve a more complex problem
  (here: ellipsoid rolling on an uneven surface) and utilize ``sympy``'s
  capabilities to do the calculations.


Description
-----------

An ellipsoid rolls on an uneven surface. The surface is described by a
function of the form z = f(x, y). The function must be smooth enough to
ensure that the ellipsoid has only one point of contact with the surface at
any time.
A particle is attached to the surface of the ellipsoid.

Notes
-----

- These are the instructions for the case on hand:
  https://www.dropbox.com/scl/fi/tfcxldh5zmycwkm7michi/Planar_Ellipse_Rolling_on_a_Curve.pdf?rlkey=t4bn3zxm1k9z5vh72epgazxvy&st=4v8k4d52&dl=0
  Equations mentioned below refer to the equation in this paper by Dr. Carlos
  Rothmayr.
- The total energy of the system is conserved very well.
  The speed of the contact point :math:`\mathbf{v}_{\bar{E}}` is very small,
  but these are necessary conditions only.
- From the animation it is impossible to judge whether the ellipsoid is really
  rolling. It can be made smoother by increasing ``fps`` but at the cost of
  a longer build time.
- This simulation shows the effect ``cse`` may have on the number of
  operations. E.g. in the case of the mass matrix, they drop from 4,000,000 to
  1000, which is a huge difference for the numerical integration.

**States**

- :math:`q_1, q_2, q_3` : Rotation of the ellipsoid
- :math:`u_1, u_2, u_3` : Angular velocity of the ellipsoid
- :math:`x, y` : Position of subsequent points of contact
- :math:`u_x, u_y` : Velocity of subsequent points of contact

**Parameters**

- :math:`a, b, c` : Semi-axes of the ellipsoid
- :math:`p_1, p_2, s_1, s_2, s_3` : Parameters of the surface
- :math:`\alpha, \beta, \gamma` : Determines where the particle is positioned
  on the surface of the ellipsoid
- :math:`m_{el}, m_p` : Mass of the ellipsoid and the particle
- :math:`g` : Gravitational acceleration
- :math:`\bar{E}` : Contact point between the ellipsoid and the surface
- :math:`E^{\star}` : Center of the ellipsoid

"""
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
from opty.utils import MathJaxRepr

# %%
# Set up the uneven surface.

xs, ys, p1, p2, s1, s2, s3 = sm.symbols('xs ys p1 p2 s1 s2 s3')


def surface(xs, ys, p1, p2, s1, s2, s3):
    return p1*xs**2 + p2*ys**2 + s1 * sm.sin(s2 * xs) * sm.sin(s3 * ys) - 5


# %%
# Curvatures of the surface:
# K = Gaussian curvature
# H = mean curvature
# (from the internet)
# This is used to avoid that the ellipsoid may have more than one contact
# point at any time.

f = surface(xs, ys, p1, p2, s1, s2, s3)

K = ((f.diff(xs, xs) * f.diff(ys, ys) - f.diff(xs, ys)**2) /
     (1 + f.diff(xs)**2 + f.diff(ys)**2)**2)

H = ((f.diff(xs, xs) * (1 + f.diff(ys)**2) -
      2 * f.diff(xs, ys) * f.diff(xs) * f.diff(ys) +
      f.diff(ys, ys) * (1 + f.diff(xs)**2)) /
     (2 * (1 + f.diff(xs)**2 + f.diff(ys)**2)**(3/2)))

kappa1 = H + sm.sqrt(H**2 - K)
kappa2 = H - sm.sqrt(H**2 - K)

kappa1_lam = sm.lambdify((xs, ys, p1, p2, s1, s2, s3), kappa1, cse=True)
kappa2_lam = sm.lambdify((xs, ys, p1, p2, s1, s2, s3), kappa2, cse=True)

# %%
# Set up the unit normal vector of the surface.

N = me.ReferenceFrame('N')
O = me.Point('O')
O.set_vel(N, 0)
t = me.dynamicsymbols._t
x, y, z = me.dynamicsymbols('x y z', real=True)
ux, uy, uz = me.dynamicsymbols('ux uy uz', real=True)
f = sm.Function('f')(x, y)
f = surface(x, y, p1, p2, s1, s2, s3)
normal_surface = -f.diff(x)*N.x - f.diff(y)*N.y + N.z
normal_surface = normal_surface / normal_surface.magnitude()

# %%
# Set up the ellipsoid and its normal.
#
# NOTE: The normal points away from the center of the ellipsoid, that means
# the normal of the surface and the normal of the ellipsoid point in 'opposite
# directions'.
# This is the reason, further down :math:`\lambda =
# \bf{-}\sqrt{\textrm{expression}}` is used.

E = me.ReferenceFrame('E')
a, b, c = sm.symbols('a b c')
e1, e2, e3 = sm.symbols('e1 e2 e3')  # points on the surface of the ellipsoid
ellipsoid = e1**2/a**2 + e2**2/b**2 + e3**2/c**2 - 1
normal_ellipsoid = (ellipsoid.diff(e1)*E.x + ellipsoid.diff(e2)*E.y +
                    ellipsoid.diff(e3)*E.z)

# %%
# Print normal vector (not normalized) on the surface of the ellipsoid.
MathJaxRepr(normal_ellipsoid)

# %%
# Rotate the ellipsoid. E is the reference frame of the ellipsoid,
# N is the inertial frame.
q1, q2, q3 = me.dynamicsymbols('q1 q2 q3', real=True)
u1, u2, u3 = me.dynamicsymbols('u1 u2 u3', real=True)
E.orient_body_fixed(N, (q1, q2, q3), 'XYZ')

# The angular velocity of the ellipsoid is needed later for the rolling
# condition.

omega = E.ang_vel_in(N)

# %%
# Print the angular velocity of the ellipsoid expressed in the body fixed
# frame.

MathJaxRepr(omega)

# %%
# The r.h.s. of eq (4) is dot multiplied by :math:`\hat e_i`
# so, :math:`e_{11} = \dfrac{e_1}{a^2},\hspace{2pt} e_{21} =
# \dfrac{e_2}{b^2}, \hspace{2pt}, \hspace{2pt} e_{31}
# = \dfrac{e_3}{c^2}`.
#
# LAMBDA corresponds to :math:`\lambda` in the paper.

LAMBDA = sm.symbols('LAMBDA')
e11 = ((2 * LAMBDA * normal_surface).dot(E.x)).simplify()
e21 = ((2 * LAMBDA * normal_surface).dot(E.y)).simplify()
e31 = ((2 * LAMBDA * normal_surface).dot(E.z)).simplify()

# %%
# Solve for :math:`\lambda`
# :math:`e_{i1}` is eq (5) dot multiplied by :math:`\hat e_i`
# Then the :math:`e_{i1}` are inserted in eq (3) and solved for
# :math:`\lambda^2`

e11 = e11 * a**2
e21 = e21 * b**2
e31 = e31 * c**2
expr = (e11/a)**2 + (e21/b)**2 + (e31/c)**2 - 1
expr = sm.Matrix([expr])
substitution = sm.symbols('substitution')
expr = expr.subs({LAMBDA**2: substitution})

# %%
# Use ULsolve.
# LU.solve is used as much faster than sm.linsolve, see point 4 here:
# https://moorepants.github.io/learn-multibody-dynamics/holonomic-eom.html

Mk = expr.jacobian([substitution])
gk = expr.subs({substitution: 0})
loesung = -Mk.LUsolve(gk)


# %%
# The negative root is taken, cf. the discussion right below eq (16)
lam_loesung = -sm.sqrt(loesung[0])
print(f"lam_loesung has {sm.count_ops(lam_loesung)} operations, ")


# %%
# Get the expressions for :math:`e_i` corresponding to eq (10)
e11 = e11.subs(LAMBDA, lam_loesung)
e21 = e21.subs(LAMBDA, lam_loesung)
e31 = e31.subs(LAMBDA, lam_loesung)
print(f"e11 has {sm.count_ops(e11)} operations, "
      f"{sm.count_ops(sm.cse(e11))} after cse")
print(f"e21 has {sm.count_ops(e21)} operations, "
      f"{sm.count_ops(sm.cse(e21))} after cse")
print(f"e31 has {sm.count_ops(e31)} operations, "
      f"{sm.count_ops(sm.cse(e31))} after cse")

# %%
# Now kinematics is used.
#
# r_Ebar_Estar = :math:`\bar r^{\bar{E}E^{\star}}`, the vector from the
# contact point :math:`\bar E` to the center of the ellipsoid :math:`E^{\star}`

Estar, Ebar = sm.symbols('Estar Ebar', cls=me.Point)
r_Ebar_Estar = -(e11*E.x + e21*E.y + e31*E.z)

# %%
# This corresponds to equation (21)
Estar.set_pos(O, x*N.x + y*N.y + f*N.z + r_Ebar_Estar)
vEstar = Estar.pos_from(O).diff(t, N)
print(f"vEstar has {sm.count_ops(
      sm.Matrix([vEstar.dot(N.x), vEstar.dot(N.y), vEstar.dot(N.z)]))}"
      f" operations, {sm.count_ops(
            sm.cse(
                  sm.Matrix([vEstar.dot(N.x), vEstar.dot(N.y),
                             vEstar.dot(N.z)])))} after cse")

# %%
# This corresponds to equation (22)
vEbar = vEstar + omega.cross(-r_Ebar_Estar)

# %%
# Rolling condition: :math:`\mathbf{v}_{\bar{E}} = 0`
constraint1 = sm.Matrix([[vEbar.dot(N.x)], [vEbar.dot(N.y)]])

# %%
# Solve using LUsolve.
Mk = constraint1.jacobian((x.diff(t), y.diff(t)))
gk = constraint1.subs({x.diff(t): 0, y.diff(t): 0})
loesung = -Mk.LUsolve(gk)

print('loesung DS', me.find_dynamicsymbols(loesung, reference_frame=N))
print(f"loesung contains {sm.count_ops(loesung):,} operations, "
      f"{sm.count_ops(sm.cse(loesung)):,} after cse")
omega

# %%
# Set up Kane's Equations.

print_output = True

alpha, beta, gamma = sm.symbols('alpha beta gamma')
mel, mp, g = sm.symbols('mel mp, g')
punkt1 = me.Point('punkt1')  # particle on the ellipsoid
punkt1.set_pos(Estar, a*alpha*E.x + b*beta*E.y + c*gamma*E.z)
punkt1.v2pt_theory(Estar, N, E)

iXX = 1/5 * mel * (b**2 + c**2)
iYY = 1/5 * mel * (a**2 + c**2)
iZZ = 1/5 * mel * (a**2 + b**2)
inertia_ellipsoid = me.inertia(E, iXX, iYY, iZZ)
ellipsoid = me.RigidBody('ellipsoid', Estar, E, mel,
                         (inertia_ellipsoid, Estar))

punkta = me.Particle('punkta', punkt1, mp)
bodies = [ellipsoid, punkta]

forces = [(punkt1, -mp*g*N.z), (Estar, -mel*g*N.z)]

kd = sm.Matrix([
    ux - x.diff(t),
    uy - y.diff(t),
    u1 - q1.diff(t),
    u2 - q2.diff(t),
    u3 - q3.diff(t)
])

speed_constraints = sm.Matrix([
    ux - loesung[0],
    uy - loesung[1],
])

q_ind = [q1, q2, q3, x, y]
u_ind = [u1, u2, u3]
u_dep = [ux, uy]

kanes = me.KanesMethod(
    N,
    q_ind,
    u_ind,
    kd_eqs=kd,
    u_dependent=u_dep,
    velocity_constraints=speed_constraints)

fr, frstar = kanes.kanes_equations(bodies, forces)

MM = kanes.mass_matrix_full
force = kanes.forcing_full

# %%
# Print some information about the mass matrix and the forcing vector.

if print_output:
    print('mass Matrix dynamic symbols',
          me.find_dynamicsymbols(MM, reference_frame=N))
    print(f"mass matrix contains {sm.count_ops(MM):,} operations, "
          f"{sm.count_ops(sm.cse(MM)[0]):,} after cse, \n")
    print('forcing dynamic symbols',
          me.find_dynamicsymbols(force, reference_frame=N))
    print(f"forcing contains {sm.count_ops(force):,} operations, "
          f"{sm.count_ops(sm.cse(force)[0]):,} after cse")

# %%
# Compilation.

qL = q_ind + u_ind + u_dep
pL = [mel, mp, g, a, b, c, p1, p2, s1, s2, s3, alpha, beta, gamma]
MM_lam = sm.lambdify(qL + pL, MM, cse=True)
force_lam = sm.lambdify(qL + pL, force, cse=True)

# %%
# Parameters.

a1 = 3.0
b1 = 1.5
c1 = 1.5
mel1 = 1.0
mp1 = 1.0
g1 = 9.81
p11 = 0.03
p21 = 0.03
s11 = 0.05
s21 = 1
s31 = 1
alpha1 = 0.3
beta1 = 0.5

# %%
# Initial conditions
q1_0 = 1.0
q2_0 = 2.0
q3_0 = 3.0
x_0 = 10.0
y_0 = 10.0
u1_0 = 1.0
u2_0 = 1.0
u3_0 = -1.0

# %%
# Ensure the particle is on the surface of the ellipsoid.
if (alpha1 < 0 or alpha1 > 1) or (beta1 < 0 or beta1 > 1):
    raise ValueError(r"alpha1 and beta1 must be between 0 and 1")
else:
    gamma1 = np.sqrt(1 - alpha1**2 - beta1**2)

# %%
# Get the initial dependent speeds.
loesung = loesung.subs({q1.diff(t): u1, q2.diff(t): u2, q3.diff(t): u3})
loesung_lam = sm.lambdify([q1, q2, q3, x, y, u1, u2, u3] + pL, loesung,
                          cse=True)
ux_0, uy_0 = loesung_lam(q1_0, q2_0, q3_0, x_0, y_0, u1_0, u2_0, u3_0,
                         mel1, mp1, g1, a1, b1, c1, p11, p21, s11, s21, s31,
                         alpha1, beta1, gamma1)
ux_0 = ux_0[0]
uy_0 = uy_0[0]

# %%
# Minimal osculating circles: Ellipsoid must be able to roll.


def func1(x0, *args):
    xs, ys = x0
    p11, p21, s11, s21, s31 = args
    return 1.0/np.sqrt(kappa1_lam(xs, ys, p11, p21, s11, s21, s31)**2)


def func2(x0, *args):
    xs, ys = x0
    p11, p21, s11, s21, s31 = args
    return 1.0/np.sqrt(kappa2_lam(xs, ys, p11, p21, s11, s21, s31)**2)


args = (p11, p21, s11, s21, s31)
x0 = np.array([0.0, 0.0])
res1 = minimize(func1, x0, args=args, bounds=((None, None), (None, None)))

res2 = minimize(func2, x0, args=args, bounds=((None, None), (None, None)))

ell_osc = max([
    b1**2 / a1, c1**2 / a1,
    a1**2 / b1, c1**2 / b1,
    a1**2 / c1, b1**2 / c1
])

# %%
# Ensure that only one point of contact :math:`\bar E` exists.
if ell_osc >= min(res1.fun, res2.fun):
    raise ValueError("Ellipse too large / surface too uneven, "
                     "rolling may not be possible")

# %%
# Numerical Integration
# ---------------------

interval = 10.0
punkte = 100

schritte = int(interval * punkte)
times = np.linspace(0., interval, schritte)
t_span = (0., interval)

pL_vals = [mel1, mp1, g1, a1, b1, c1, p11, p21, s11, s21, s31,
           alpha1, beta1, gamma1]
y0 = [q1_0, q2_0, q3_0, x_0, y_0, u1_0, u2_0, u3_0, ux_0, uy_0]
y0 = np.array(y0)

# %%
# Initial speed of the contact point :math:`\mathbf{v}_{\bar{E}}`
# for the given initial conditions.
subs_dict = {i.diff(t): u for i, u in zip(q_ind, u_ind + u_dep)}
vel_Ebar = sm.Matrix([vEbar.dot(N.x), vEbar.dot(N.y),
                      vEbar.dot(N.z)]).subs(subs_dict)
vel_Ebar_lam = sm.lambdify(qL + pL, vel_Ebar, cse=True)
v_Ebar_start = vel_Ebar_lam(*y0, *pL_vals)
print('Initial speed of the contact point, should be zero ideally: '
      f'\n {v_Ebar_start}')

vel_Estar = sm.Matrix([vEstar.dot(N.x), vEstar.dot(N.y),
                       vEstar.dot(N.z)]).subs(subs_dict)
vel_Estar_lam = sm.lambdify(qL + pL, vel_Estar, cse=True)
v_Estar_start = vel_Estar_lam(*y0, *pL_vals)
print(f'Initial speed of the center of the ellipsoid: \n {v_Estar_start}')

# %%
# Right hand side of the ODE system. Solve the ODE numerically.


def gradient(t, y, args):
    sol = np.linalg.solve(MM_lam(*y, *args), force_lam(*y, *args))
    return np.array(sol).squeeze()


resultat1 = solve_ivp(gradient, t_span, y0, t_eval=times, args=(pL_vals,),
                      method='DOP853',
                      atol=1.e-10,
                      rtol=1.e-10,
                      )

resultat = resultat1.y.T
print('resultat shape', resultat.shape, '\n')
print(resultat1.message)
print(f"To integrate {interval} sec, {resultat1.nfev} "
      "evaluations of the gradient were needed.")

# %%
# Plot generalized coordinates and speeds.

bezeichnung = ['$q_1$', '$q_2$', '$q_3$', '$x$', '$y$', '$u_1$', '$u_2$',
               '$u_3$', '$u_x$', '$u_y$']
fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True,
                       layout='constrained')
for i in range(5):
    ax[0].plot(times, resultat[:, i], label=bezeichnung[i])
ax[0].legend()
ax[0].set_title('Generalized Coordinates')

for i in range(5, 10):
    ax[1].plot(times, resultat[:, i], label=bezeichnung[i])
ax[1].legend()
ax[1].set_xlabel('Time [s]')
_ = ax[1].set_title('Generalized Speeds')

# %%
# Plot the energies and the speed of the contact point.
#
# Speed of contact point :math:`\bar E` should be zero.

kin_energy = sum([body.kinetic_energy(N) for body in bodies]).subs(subs_dict)
pot_energy = mp * g * punkt1.pos_from(O).dot(N.z) + \
    mel * g * Estar.pos_from(O).dot(N.z)

kin_lam = sm.lambdify(qL + pL, kin_energy, cse=True)
pot_lam = sm.lambdify(qL + pL, pot_energy, cse=True)

kin_np = np.empty(resultat.shape[0])
pot_np = np.empty(resultat.shape[0])
total_np = np.empty(resultat.shape[0])
for i in range(resultat.shape[0]):
    kin_np[i] = kin_lam(*resultat[i], *pL_vals)
    pot_np[i] = pot_lam(*resultat[i], *pL_vals)
total_np = kin_np + pot_np

t_max = np.max(total_np)
t_min = np.min(total_np)
print("max. deviation of total energy for being constant: "
      f"{(t_max - t_min) / t_max * 100:.2e} %")


fig, ax = plt.subplots(2, 1, figsize=(10, 4), layout='constrained')
ax[0].plot(times, kin_np, label='Kinetic Energy')
ax[0].plot(times, pot_np, label='Potential Energy')
ax[0].plot(times, total_np, label='Total Energy')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Energy [J]')
ax[0].set_title('Energy of the System')
ax[0].legend()


vel_Ebar_np = np.empty((resultat.shape[0], 3))
for i in range(resultat.shape[0]):
    vel_Ebar_np[i] = vel_Ebar_lam(*resultat[i], *pL_vals).squeeze()

msg = [r'$V_{\bar{E}_x}$', r'$V_{\bar{E}_y}$', r'$V_{\bar{E}_z}$']
for i in range(3):
    ax[1].plot(times, vel_Ebar_np[:, i], label=msg[i])
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Velocity [m/s]')
ax[1].set_title(r'Velocity of $\bar{E}$')
_ = ax[1].legend()

# %%
# Animate the ellipsoid.
#
# This routine was largely written by chatGPT. I just adapted it to the
# problem at hand.

fps = 5


def ellipsoid_points(a, b, c, n=50):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    u, v = np.meshgrid(u, v)

    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)

    return x, y, z


def rotation_matrix(q1, q2, q3):
    Rz = np.array([
        [np.cos(q1), -np.sin(q1), 0],
        [np.sin(q1),  np.cos(q1), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(q2), 0, np.sin(q2)],
        [0, 1, 0],
        [-np.sin(q2), 0, np.cos(q2)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(q3), -np.sin(q3)],
        [0, np.sin(q3),  np.cos(q3)]
    ])

    return Rz @ Ry @ Rx


X0, Y0, Z0 = ellipsoid_points(a1, b1, c1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

Estar_pos = Estar.pos_from(O).to_matrix(N)
Estar_pos_lam = sm.lambdify(qL + pL, Estar_pos, cse=True)
pos_np = np.empty((resultat.shape[0], 3))
for i in range(resultat.shape[0]):
    pos_np[i] = Estar_pos_lam(*resultat[i], *pL_vals).squeeze()

pos_min = np.min(pos_np, axis=0)
pos_max = np.max(pos_np, axis=0)
pos_min = min(pos_min) - max(a1, b1, c1)
pos_max = max(pos_max) + max(a1, b1, c1)

ax.set_xlim(pos_min, pos_max)
ax.set_ylim(pos_min, pos_max)
ax.set_zlim(pos_min, pos_max)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [m]', fontsize=15)
ax.set_ylabel('Y [m]', fontsize=15)
_ = ax.set_zlabel('Z [m]', fontsize=15)

# Coordinates of Estar and punkt1 for visualization
coordinates = Estar.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(punkt1.pos_from(O).to_matrix(N))
coords_lam = sm.lambdify(qL + pL, coordinates, cse=True)

# Interpolate the results.
t_arr = np.linspace(0.0, interval, schritte)
state_sol = CubicSpline(t_arr, resultat)

# Plot the uneven surface.
surface = ax.plot_surface(X0, Y0, Z0, cmap='inferno', edgecolor='none')
line1 = ax.scatter([], [], [], color='black', s=50)

x_s = np.linspace(pos_min, pos_max, 500)
y_s = np.linspace(pos_min, pos_max, 500)
surface_lam = sm.lambdify((x, y, p1, p2, s1, s2, s3), f, cse=True)

X, Y = np.meshgrid(x_s, y_s)
Z = surface_lam(X, Y, p11, p21, s11, s21, s31)

surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.25)

fig.colorbar(surf, shrink=0.5, aspect=20)


def update(times):
    global surface
    surface.remove()  # remove previous surface
    ax.set_title(f"Running time: {times:.2f} sec")
    coords = coords_lam(*state_sol(times), *pL_vals)

    # Moving center
    E = np.array([
        coords[0, 0],
        coords[1, 0],
        coords[2, 0]
    ])

    # Rotations
    q1 = state_sol(times)[0]  # q1
    q2 = state_sol(times)[1]  # q2
    q3 = state_sol(times)[2]  # q3

    R = rotation_matrix(q1, q2, q3)

    pts = np.vstack((X0.flatten(), Y0.flatten(), Z0.flatten()))
    rotated = R @ pts

    rotated[0] += E[0]
    rotated[1] += E[1]
    rotated[2] += E[2]

    Xr = rotated[0].reshape(X0.shape)
    Yr = rotated[1].reshape(Y0.shape)
    Zr = rotated[2].reshape(Z0.shape)
    surface = ax.plot_surface(Xr, Yr, Zr, cmap='inferno', edgecolor='none')
    line1._offsets3d = ([coords[0, 1]], [coords[1, 1]], [coords[2, 1]])

    return line1, surface


frames = np.arange(0.0, interval, 1 / fps)
ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)

plt.show()
