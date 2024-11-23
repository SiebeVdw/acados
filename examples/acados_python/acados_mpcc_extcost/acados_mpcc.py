import numpy as np
import matplotlib.pyplot as plt
from loader import loader, loader_circle
from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca
from utils import initial_guess

reference_track, current_state, _ = loader()
# reference_track, current_state, _ = loader_circle()

ocp = AcadosOcp()

##############################
# set model
##############################
n_control_points = reference_track.shape[0]
model = export_bicycle_model(n_control_points)
ocp.model = model
ocp.dims.np = 2*n_control_points
ocp.parameter_values = np.zeros((2*n_control_points, ))
# nonlinear circular constraints
ocp.dims.nh = 1
ocp.constraints.uh = np.array([0.0])
ocp.constraints.lh = np.array([-10])

nx = model.x.rows()
nu = model.u.rows()
N = 20
dt = 0.1
Tf = N * dt

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

# set cost type
ocp.cost.cost_type = 'EXTERNAL'


##############################
#set constraints
##############################
# set simple input constraints
velocity_limit = 5.0
wheelradius = 0.1
steering_limit = 0.5
ocp.constraints.idxbu = np.array([0, 1, 2])
ocp.constraints.lbu = np.array([-velocity_limit/wheelradius, -steering_limit, 0.001])
ocp.constraints.ubu = np.array([velocity_limit/wheelradius, steering_limit, 1.2]) # 10 times higher then Zander's because ACADOS multiplies by dt=0.1
# set simple state constraints
ocp.constraints.idxbx = np.array([3,4,5])
ocp.constraints.lbx = np.array([-np.pi/4, 0.0, 1e-8])
ocp.constraints.ubx = np.array([np.pi/4, 5.0, 1.0])
ocp.constraints.idxbx_e = np.array([3,4,5])
ocp.constraints.lbx_e = np.array([-np.pi/4, 0.0, 1e-8])
ocp.constraints.ubx_e = np.array([np.pi/4, 5.0, 1.0])
# set initial condition
ocp.constraints.x0 = current_state


##############################
# set solver
##############################
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
# ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
# ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
# ocp.solver_options.nlp_solver_type = 'SQP_RTI'
ocp.solver_options.print_level = 1
# set max iterations
# ocp.solver_options.qp_solver_iter_max = 2000

ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
# set the params
for i in range(N+1):
    ocp_solver.set(i, "p", reference_track[:,:2].flatten())

print(f"Initial cost: {ocp_solver.get_cost()}")

# initial guess
x_guess, u_guess = initial_guess(reference_track[:,:2], N, dt)
for i in range(N+1):
    ocp_solver.set(i, "x", x_guess[i, :])
for i in range(N):
    ocp_solver.set(i, "u", u_guess[i, :])



##############################
# solve and interpret
##############################
status = ocp_solver.solve()
print(f"total solve time: {ocp_solver.get_stats('time_tot')*1e3} ms")

# print states and inputs
X = np.zeros(N+1)
Y = np.zeros(N+1)
THETA = np.zeros(N+1)
TAU = np.zeros(N+1)
print("\n States:")
for i in range(N+1):
    x = ocp_solver.get(i,"x")
    X[i] = x[0]
    Y[i] = x[1]
    THETA[i] = x[2]
    TAU[i] = x[5]
    print(f"x: {x[0]}, y: {x[1]}, theta: {x[2]*180/np.pi}°, delta: {x[3]*180/np.pi}°, v: {x[4]}, tau: {x[5]}")
U_a = np.zeros(N)
U_s = np.zeros(N)
U_z = np.zeros(N)
print("\n Inputs:")
for i in range(N):
    u = ocp_solver.get(i,"u")
    U_a[i] = u[0]
    U_s[i] = u[1]
    U_z[i] = u[2]
    print(f'alpha: {U_a[i]}, phi: {U_s[i]}, zeta: {U_z[i]}')

# calculate theta from X and Y
for i in range(N):
    print(f"calculated theta = {ca.atan2(Y[i+1]-Y[i],X[i+1]-X[i])*180/np.pi}, solution theta = {THETA[i]*180/np.pi}")



# spline
degree = 2
kk = np.linspace(0, 1, n_control_points - degree + 1)
knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
c_points = ca.MX.sym("c_points", 2, n_control_points)
tau = ca.MX.sym("tau")
spline = ca.bspline(tau, reference_track[:,:2].T, knots, [degree], 2, {})
spline_function = ca.Function("spline", [tau], [spline])
spline_derivative = ca.jacobian(spline, tau)
spline_derivative_function = ca.Function("spline_derivative", [tau], [spline_derivative])

xpos = ca.MX.sym("xpos")
ypos = ca.MX.sym("ypos")
phi =   ca.atan2((spline_derivative_function(tau)[1] + 1e-6), spline_derivative_function(tau)[0] + 1e-6)
ec  =   ca.sin(phi) * (xpos - spline_function(tau)[0]) - ca.cos(phi) * (ypos - spline_function(tau)[1])
el  = - ca.cos(phi) * (xpos - spline_function(tau)[0]) - ca.sin(phi) * (ypos - spline_function(tau)[1])

ec_function = ca.Function("ec", [xpos, ypos, tau], [ec])
el_function = ca.Function("el", [xpos, ypos, tau], [el])


# print the cost and its contibutions
ql = 1e3   # longitudinal cost
qc = 1e2  # lateral cost
ra = 1e-2  # acceleration cost
rs = 1e-2  # steering cost
rz = 1e1   # zeta cost

print(f"\n cost: {ocp_solver.get_cost()}")
longitudinal_cost, lateral_cost, acceleration_cost, steering_cost, zeta_cost = 0.0, 0.0, 0.0, 0.0, 0.0
for i in range(N):
    ec_val = ec_function(X[i], Y[i], TAU[i]).full().flatten()
    el_val = el_function(X[i], Y[i], TAU[i]).full().flatten()
    longitudinal_cost += ql*el_val**2*dt
    lateral_cost += qc*ec_val**2*dt
    acceleration_cost += ra*U_a[i]**2*dt
    steering_cost += rs*U_s[i]**2*dt
    zeta_cost -= rz*U_z[i]*dt
# no final point cost included yet!
# longitudinal_cost += ql*el_function(TAU[N]).full().flatten()**2
# lateral_cost += qc*ec_function(TAU[N]).full().flatten()**2
print(f"longitudinal cost: {longitudinal_cost}")
print(f"lateral cost: {lateral_cost}")
print(f"acceleration cost: {acceleration_cost}")
print(f"steering cost: {steering_cost}")
print(f"zeta cost: {zeta_cost}")
print(f'sum of costs: {longitudinal_cost + lateral_cost + acceleration_cost + steering_cost + zeta_cost}')



# plot coordinates, colors from light to dark
plt.figure()
plt.scatter(X, Y, c=range(N+1), cmap='viridis', label="Vehicle States")
plt.scatter(reference_track[:,0], reference_track[:,1], c=range(N+1), cmap='plasma', marker='x', label="Reference Track")
plt.plot(x_guess[:,0], x_guess[:,1], 'o', color='red', label="Initial Guess")

# plot dotted lines connecting the states (X,Y) with the spline points spline(tau)
for i in range(N+1):
    # Spline point at current tau
    spline_xy = spline_function(TAU[i]).full().flatten()
    
    # Draw line connecting state to spline
    plt.plot([X[i], spline_xy[0]], [Y[i], spline_xy[1]], 'k--', linewidth=0.5)

    # Compute errors
    ec_val = ec_function(X[i], Y[i], TAU[i]).full().flatten()
    el_val = el_function(X[i], Y[i], TAU[i]).full().flatten()

    # Compute spline tangent direction
    spline_der = spline_derivative_function(TAU[i]).full().flatten()
    tangent_dir = spline_der / (np.linalg.norm(spline_der) + 1e-6)  # Normalize tangent

    # Lateral error vector
    lateral_vector = np.array([-tangent_dir[1], tangent_dir[0]]) * ec_val

    # Longitudinal error vector
    longitudinal_vector = tangent_dir * el_val

    # Add arrows for longitudinal and lateral errors
    plt.arrow(X[i], Y[i], lateral_vector[0], lateral_vector[1], color='blue', head_width=0.05, length_includes_head=True, label="Lateral Error" if i == 0 else "")
    plt.arrow(X[i], Y[i], longitudinal_vector[0], longitudinal_vector[1], color='green', head_width=0.05, length_includes_head=True, label="Longitudinal Error" if i == 0 else "")

plt.axis('equal')
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Visualization of Longitudinal and Lateral Errors")
plt.show()


# # plot the spline for some tau values
# tau_values = np.linspace(0,1,1000)
# spline_values = np.zeros((1000, 2))
# for i, tau_val in enumerate(tau_values):
#     spline_values[i] = spline_function(tau_val, reference_track[:,:2].T).full().flatten()
# plt.figure()
# plt.plot(spline_values[:,0], spline_values[:,1], )
# plt.plot(reference_track[:,0], reference_track[:,1], 'x')
# plt.show()


