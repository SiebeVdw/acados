import numpy as np
import matplotlib.pyplot as plt
from loader import loader
from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca

reference_track, current_state, u_prev = loader()

ocp = AcadosOcp()

##############################
# set model
##############################
model = export_bicycle_model()
control_points = ca.MX.sym('control_points', 2*reference_track.shape[0], 1)
model.p = control_points
ocp.model = model
ocp.dims.np = 2*reference_track.shape[0]
ocp.parameter_values = np.zeros((2*reference_track.shape[0], ))

nx = model.x.rows()
nu = model.u.rows()
N = 20
dt = 0.1
Tf = N * dt

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

##############################
# set cost
##############################
ql = 1e1   # longitudinal cost
qc = 1e-7  # lateral cost
ra = 1e-5  # acceleration cost
rs = 1e-1  # steering cost
rz = 2e2   # zeta cost
Qn = np.diag([ql, qc])
R = np.diag([ra, rs, rz])
W = np.block([[Qn, np.zeros((2, nu))], [np.zeros((nu, 2)), R]])

# spline through the control points
degree = 2
kk = np.linspace(0, 1, reference_track.shape[0] - degree + 1)
knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
tau = ca.MX.sym("tau")
c_points = ca.MX.sym("c_points", 2, reference_track.shape[0])
# transform control points to correct dimensions (2 x num_points)
spline = ca.bspline(tau, c_points, knots, [degree], 2, {})
spline_function = ca.Function("spline", [tau, c_points], [spline])
spline_derivative = ca.jacobian(spline, tau)
spline_derivative_function = ca.Function("spline_derivative", [tau, c_points], [spline_derivative])
# errors
def get_errors(x,y,tau,control_points):
    phi =   ca.atan2((spline_derivative_function(tau, control_points.T)[1] + 1e-6), spline_derivative_function(tau, control_points.T)[0] + 1e-6)
    ec  =   ca.sin(phi) * (x - spline_function(tau, control_points.T)[0]) - ca.cos(phi) * (y - spline_function(tau, control_points.T)[1])
    el  = - ca.cos(phi) * (x - spline_function(tau, control_points.T)[0]) - ca.sin(phi) * (y - spline_function(tau,control_points.T)[1])
    return ec, el
control_points_spline = ca.reshape(control_points, 2, reference_track.shape[0]).T
ec, el = get_errors(model.x[0], model.x[1], model.x[5], control_points_spline)
# # test the errors ec and el
# x_test, y_test = -0.22954451, 0.047049
# tau_test = 0.2
# ec_test, el_test = get_errors(x_test, y_test, tau_test, reference_track[:,:2])
# print(f"ec_test = {ec_test}, el_test = {el_test}") 

# start cost
ocp.cost.cost_type_0 = 'NONLINEAR_LS'
ocp.model.cost_y_expr_0 = ca.vertcat(el, ec, model.u[:2], model.u[2])
# ocp.cost.yref_0 = np.zeros(2+nu)
ocp.cost.yref_0 = np.array([0.0, 0.0, 0.0, 0.0, -10])
ocp.cost.W_0 = W

# intermediate cost
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.model.cost_y_expr = ca.vertcat(el, ec, model.u[:2], model.u[2])
# ocp.cost.yref = np.zeros(2+nu)
ocp.cost.yref = np.array([0.0, 0.0, 0.0, 0.0, -10])
ocp.cost.W = W

# terminal cost
ocp.cost.cost_type_e = 'NONLINEAR_LS'
ocp.model.cost_y_expr_e = ca.vertcat(el, ec)
ocp.cost.yref_e = np.zeros(2)
ocp.cost.W_e = Qn

##############################
#set constraints
##############################
# set simple input constraints
velocity_limit = 5.0
wheelradius = 0.1
steering_limit = 0.5
ocp.constraints.idxbu = np.array([0, 1, 2])
ocp.constraints.lbu = np.array([-velocity_limit/wheelradius, -steering_limit, 0.001])
ocp.constraints.ubu = np.array([velocity_limit/wheelradius, steering_limit, 0.12])
# set simple state constraints
ocp.constraints.idxbx = np.array([3,4,5])
ocp.constraints.lbx = np.array([-np.pi/4, 0.0, 1e-8])
ocp.constraints.ubx = np.array([np.pi/4, 20, 1.0])
# TODO: implement Halfspace constraints 
# set initial condition
ocp.constraints.x0 = np.array([-0.22954451, 0.047049, 0.0, 0.16719286, 2.0, 1e-3])


##############################
# set solver
##############################
# ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
# ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.nlp_solver_type = 'SQP_RTI'
# ocp.solver_options.print_level = 1
# set max iterations
ocp.solver_options.qp_solver_iter_max = 2000

ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

# set the params
for i in range(N+1):
    ocp_solver.set(i, "p", reference_track[:,:2].flatten())

print(f"Initial cost: {ocp_solver.get_cost()}")

##############################
# solve and interpret
##############################
status = ocp_solver.solve()

# print states and inputs
X = np.zeros(N+1)
Y = np.zeros(N+1)
print("\n States:")
for i in range(N+1):
    x = ocp_solver.get(i,"x")
    X[i] = x[0]
    Y[i] = x[1]
    print(f"x: {x[0]}, y: {x[1]}, theta: {x[2]}, delta: {x[3]}, v: {x[4]}, tau: {x[5]}")
U_a = np.zeros(N)
U_s = np.zeros(N)
U_z = np.zeros(N)
print("\n Inputs:")
for i in range(N):
    u = ocp_solver.get(i,"u")
    U_a[i] = u[0]
    U_s[i] = u[1]
    U_z[i] = u[2]
    print(f'a: {U_a[0]}, delta: {U_s[1]}, zeta: {U_z[2]}')

# print the cost and its contibutions
print(f"\n cost: {ocp_solver.get_cost()}")
ec, el = get_errors(X, Y, 0.5, reference_track[:,:2])
cost_long = np.sum(ql*el**2)
cost_lat = np.sum(qc*ec**2)
cost_acc = np.sum(ra*U_a**2)
cost_steer = np.sum(rs*U_s[1]**2)
cost_zeta = np.sum(rz*U_z[2]**2)
print(f"longitudinal cost: {cost_long}")
print(f"lateral cost: {cost_lat}")
print(f"acceleration cost: {cost_acc}")
print(f"steering cost: {cost_steer}")
print(f"zeta cost: {cost_zeta}")
print(f'sum of costs: {cost_long + cost_lat + cost_acc + cost_steer + cost_zeta}')

# plot coordinates
plt.figure()
plt.plot(X,Y, 'o')
plt.plot(reference_track[:,0], reference_track[:,1], 'x')
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

