import numpy as np
import matplotlib.pyplot as plt
from loader import loader, loader_circle
from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca
from utils import initial_guess
from pathlib import Path
import yaml


x_vars = ['x', 'y', 'theta', 'delta', 'v', 'tau']
u_vars = ['alpha', 'phi', 'zeta']
with open(Path(__file__).parent / "parameters.yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# reference_track, current_state, _ = loader()
ref_track = np.load(Path(__file__).parent / "maps" / "fssim_fsg.npy")
# shift the reference track 
# ref_track = np.roll(ref_track, -9, axis=0)


reference_track = ref_track[:20]
start_angle = np.arctan2(reference_track[1,1] - reference_track[0,1], reference_track[1,0] - reference_track[0,0])
current_state = np.array([reference_track[0,0], reference_track[0,1], start_angle, 0.0, 0.0, 0.00001])
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
steering_limit = 5.0
ocp.constraints.idxbu = np.array([0, 1, 2])
ocp.constraints.lbu = np.array([params['alpha_min'], params['phi_min'], params['zeta_min']])
ocp.constraints.ubu = np.array([params['alpha_max'], params['phi_max'], params['zeta_max']]) # 10 times higher then Zander's because ACADOS multiplies by dt=0.1
# set simple state constraints
ocp.constraints.idxbx = np.array([3,4,5])
ocp.constraints.lbx = np.array([params['delta_min']*np.pi/180, params['v_min'], params['tau_min']])
ocp.constraints.ubx = np.array([params['delta_max']*np.pi/180, params['v_max'], params['tau_max']])
ocp.constraints.idxbx_e = np.array([3,4,5])
ocp.constraints.lbx_e = np.array([params['delta_min']*np.pi/180, params['v_min'], params['tau_min']])
ocp.constraints.ubx_e = np.array([params['delta_max']*np.pi/180, params['v_max'], params['tau_max']])
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

# initial guess Zander
x_guess, u_guess = initial_guess(reference_track[:,:2], N, dt)
for i in range(N+1):
    ocp_solver.set(i, "x", x_guess[i, :])
ocp_solver.set(N, "x", x_guess[N, :])
for i in range(N):
    ocp_solver.set(i, "u", u_guess[i, :])

# initial guess previous solution shifted
# path = Path(__file__).parent / "results" / "solution.npz"
# data = np.load(path)
# x_guess = data['x']
# u_guess = data['u']
# for i in range(N):
#     ocp_solver.set(i, "x", x_guess[i+1, :])
# ocp_solver.set(N, "x", x_guess[N, :])
# for i in range(N-1):
#     ocp_solver.set(i, "u", u_guess[i+1, :])
# ocp_solver.set(N-1, "u", u_guess[N-1, :])
# ocp.constraints.x0 = x_guess[0, :]



##############################
# solve and save
##############################
# solve multiple times the same problem
# for _ in range(30):
#     status = ocp_solver.solve()
#     x_sol = np.array([ocp_solver.get(i,"x") for i in range(N+1)])
#     u_sol = np.array([ocp_solver.get(i,"u") for i in range(N)])
#     for i in range(N+1):
#         ocp_solver.set(i, "x", x_sol[i, :])
#     for i in range(N):
#         ocp_solver.set(i, "u", u_sol[i, :])
#     ocp.constraints.x0 = x_sol[0, :]



status = ocp_solver.solve()
print(f"total solve time: {ocp_solver.get_stats('time_tot')*1e3} ms")

# get x and u solutions
x_sol = np.array([ocp_solver.get(i,"x") for i in range(N+1)])
u_sol = np.array([ocp_solver.get(i,"u") for i in range(N)])
# save solutions as .npz file
path = Path(__file__).parent / "results" / "solution"
np.savez(path, x=x_sol, u=u_sol)





##############################
# results + interpretation
##############################

# calculate total cost and its contributions
ql, qc, ra, rs, rz = params['ql'], params['qc'], params['ra'], params['rs'], params['rz'], 
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
# errors
xpos = ca.MX.sym("xpos")
ypos = ca.MX.sym("ypos")
phi =   ca.atan2((spline_derivative_function(tau)[1] + 1e-6), spline_derivative_function(tau)[0] + 1e-6)
ec  =   ca.sin(phi) * (xpos - spline_function(tau)[0]) - ca.cos(phi) * (ypos - spline_function(tau)[1])
el  = - ca.cos(phi) * (xpos - spline_function(tau)[0]) - ca.sin(phi) * (ypos - spline_function(tau)[1])
ec_function = ca.Function("ec", [xpos, ypos, tau], [ec])
el_function = ca.Function("el", [xpos, ypos, tau], [el])
# calculate cost
print(f"\n cost: {ocp_solver.get_cost()}")
longitudinal_cost, lateral_cost, acceleration_cost, steering_cost, zeta_cost = 0.0, 0.0, 0.0, 0.0, 0.0
for i in range(N):
    ec_val = ec_function(x_sol[i, x_vars.index('x')], x_sol[i, x_vars.index('y')], x_sol[i, x_vars.index('tau')]).full().flatten()
    el_val = el_function(x_sol[i, x_vars.index('x')], x_sol[i, x_vars.index('y')], x_sol[i, x_vars.index('tau')]).full().flatten()
    longitudinal_cost += ql*el_val**2*dt
    lateral_cost += qc*ec_val**2*dt
    acceleration_cost += ra*u_sol[i, u_vars.index('alpha')]**2*dt
    steering_cost += rs*u_sol[i, u_vars.index('phi')]**2*dt
    zeta_cost -= rz*u_sol[i, u_vars.index('zeta')]*dt
print(f"longitudinal cost: {longitudinal_cost}")
print(f"lateral cost: {lateral_cost}")
print(f"acceleration cost: {acceleration_cost}")
print(f"steering cost: {steering_cost}")
print(f"zeta cost: {zeta_cost}")
print(f'sum of costs: {longitudinal_cost + lateral_cost + acceleration_cost + steering_cost + zeta_cost}')

# plot inputs and state timeseries
fig, ax = plt.subplots(2,4)
# inputs
t = np.linspace(0,Tf,N)
ax[0,0].plot(t, u_sol[:,u_vars.index('alpha')])
ax[0,0].step(t, u_sol[:,u_vars.index('alpha')], where='post')
ax[0,0].plot(t, np.ones(N)*params['alpha_min'], 'r--')
ax[0,0].plot(t, np.ones(N)*params['alpha_max'], 'r--')
ax[0,0].set_ylabel('m/s²')
ax[0,0].set_title('alpha')
ax[0,1].plot(t, u_sol[:,u_vars.index('phi')])
ax[0,1].step(t, u_sol[:,u_vars.index('phi')], where='post')
ax[0,1].plot(t, np.ones(N)*params['phi_min'], 'r--')
ax[0,1].plot(t, np.ones(N)*params['phi_max'], 'r--')
ax[0,1].set_ylabel('m/s')
ax[0,1].set_title('phi')
ax[0,2].plot(t, u_sol[:,u_vars.index('zeta')])
ax[0,2].step(t, u_sol[:,u_vars.index('zeta')], where='post')
ax[0,2].plot(t, np.ones(N)*params['zeta_min'], 'r--')
ax[0,2].plot(t, np.ones(N)*params['zeta_max'], 'r--')

ax[0,2].set_title('zeta')
# states
t = np.linspace(0,Tf,N+1)
ax[1,0].plot(t, x_sol[:,x_vars.index('theta')])
ax[1,0].set_title('theta')
ax[1,0].set_ylabel('rad')
ax[1,1].plot(t, x_sol[:,x_vars.index('delta')])
# ax[1,1].plot(t, np.ones(N+1)*params['delta_min'], 'r--')
# ax[1,1].plot(t, np.ones(N+1)*params['delta_max'], 'r--')
ax[1,1].set_title('delta')
ax[1,1].set_ylabel('rad')
ax[1,2].plot(t, x_sol[:,x_vars.index('v')])
ax[1,2].set_title('velocity')
ax[1,2].plot(t, np.ones(N+1)*params['v_min'], 'r--')
ax[1,2].plot(t, np.ones(N+1)*params['v_max'], 'r--')
ax[1,2].set_ylabel('m/s')
ax[1,3].plot(t, x_sol[:,x_vars.index('tau')])
ax[1,3].plot(t, np.ones(N+1)*params['tau_min'], 'r--')
ax[1,3].plot(t, np.ones(N+1)*params['tau_max'], 'r--')
ax[1,3].set_title('tau')
ax[1,3].set_ylabel('%')

# plot xy plot with extra features
plt.figure()
# plt.scatter(x_sol[:,x_vars.index('x')], x_sol[:,x_vars.index('y')], c=range(N+1), cmap='viridis', label="Vehicle States")
plt.plot(x_sol[:,x_vars.index('x')], x_sol[:,x_vars.index('y')], 'o-', label="Vehicle States")
plt.scatter(ref_track[:,0], ref_track[:,1], c=range(ref_track.shape[0]), cmap='plasma', marker='x', label="Reference Track")
plt.plot(x_guess[:,0], x_guess[:,1], 'o', color='red', label="Initial Guess")

# plot dotted lines connecting the states (X,Y) with the spline points spline(tau)
for i in range(N+1):
    # Spline point at current tau
    spline_xy = spline_function(x_sol[i,x_vars.index('tau')]).full().flatten()
    
    # Draw line connecting state to spline
    plt.plot([x_sol[i,x_vars.index('x')], spline_xy[0]], [x_sol[i,x_vars.index('y')], spline_xy[1]], 'k--', linewidth=0.5)

    # Compute errors
    ec_val = ec_function(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], x_sol[i,x_vars.index('tau')]).full().flatten()
    el_val = el_function(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], x_sol[i,x_vars.index('tau')]).full().flatten()

    # Compute spline tangent direction
    spline_der = spline_derivative_function(x_sol[i,x_vars.index('tau')]).full().flatten()
    tangent_dir = spline_der / (np.linalg.norm(spline_der) + 1e-6)  # Normalize tangent

    # Lateral error vector
    lateral_vector = np.array([-tangent_dir[1], tangent_dir[0]]) * ec_val

    # Longitudinal error vector
    longitudinal_vector = tangent_dir * el_val

    # Add arrows for longitudinal and lateral errors
    plt.arrow(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], lateral_vector[0], lateral_vector[1], color='blue', head_width=0.05, length_includes_head=True, label="Lateral Error" if i == 0 else "")
    plt.arrow(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], longitudinal_vector[0], longitudinal_vector[1], color='green', head_width=0.05, length_includes_head=True, label="Longitudinal Error" if i == 0 else "")

plt.axis('equal')
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Visualization of Longitudinal and Lateral Errors")

# show all plots
plt.show()


# # plot the spline for some tau values
# tau_values = np.linspace(0,1,1000)
# spline_values = np.zeros((1000, 2))
# for i, tau_val in enumerate(tau_values):
#     spline_values[i] = spline_function(tau_val).full().flatten()
# plt.figure()
# plt.plot(spline_values[:,0], spline_values[:,1], )
# plt.scatter(reference_track[:,0], reference_track[:,1], c=range(reference_track.shape[0]), cmap='plasma', marker='x')
# plt.show()
