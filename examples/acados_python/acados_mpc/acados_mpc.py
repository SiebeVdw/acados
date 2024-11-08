import numpy as np
import matplotlib.pyplot as plt

from loader import loader
from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca


reference_track, current_state, u_prev = loader()
# print(f"reference_track = {reference_track}")
# print(f"current_state = {current_state}")

# x = reference_track[:, 0]
# y = reference_track[:, 1]
# plt.plot(x, y, 'o-', label='reference track')
# plt.axis('equal')
# plt.show()


# create an ocp to find follow the reference trajectory

ocp = AcadosOcp()

# set model
model = export_bicycle_model()
param = ca.MX.sym('p', 1)
model.p = param
ocp.model = model
ocp.dims.np = 1
ocp.parameter_values = np.zeros((1, ))

nx = model.x.rows()
nu = model.u.rows()
N = 20
dt = 0.1
Tf = N * dt

# set prediction horizon
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf


###### set cost #####
Qn = np.diag([8e-3, 8e-3, 0, 0, 0])
# R = np.diag([1e-5, 5e-2])
R = np.diag([0,0])
W = np.block([[Qn, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])

# start cost
ocp.cost.cost_type_0 = 'NONLINEAR_LS'
ocp.model.cost_y_expr_0 = ca.vertcat(model.x*param, model.u)
ocp.cost.yref_0 = np.zeros(nx+nu)
ocp.cost.W_0 = W

# intermediate cost
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.model.cost_y_expr = ca.vertcat(model.x*param, model.u)
ocp.cost.yref = np.zeros(nx+nu)
ocp.cost.W = W
# terminal cost
ocp.cost.cost_type_e = 'NONLINEAR_LS'
ocp.model.cost_y_expr_e = model.x*param
ocp.cost.yref_e = np.zeros(nx)
ocp.cost.W_e = Qn




###### set constraints #####
# set simple input constraints
velocity_limit = 5.0
wheelradius = 0.1
steering_limit = 0.5
ocp.constraints.idxbu = np.array([0, 1])
ocp.constraints.lbu = np.array([-velocity_limit/wheelradius, -steering_limit])
ocp.constraints.ubu = np.array([velocity_limit/wheelradius, steering_limit])
# set simple state constraints
ocp.constraints.idxbx = np.array([3,4])
ocp.constraints.lbx = np.array([-np.pi/4, 0.0])
ocp.constraints.ubx = np.array([np.pi/4, 20])
## Circular boundary constraints -> not used because I can't find how to set different constraints for each shooting node
# ocp.model.con_h_expr_0 = (ocp.model.x[0] - reference_track[0, 0]) ** 2 + (ocp.model.x[1] - reference_track[0, 1]) ** 2
# ocp.model.lh_0 = 0.0
# ocp.model.uh_0 = 1.2**2
# ocp.model.con_h_expr = (ocp.model.x[0] - reference_track[1, 0]) ** 2 + (ocp.model.x[1] - reference_track[1, 1]) ** 2
# ocp.model.lh = 0.0
# ocp.model.uh = 1.2**2
# ocp.model.con_h_expr_e = (ocp.model.x[0] - reference_track[-1, 0]) ** 2 + (ocp.model.x[1] - reference_track[-1, 1]) ** 2
# ocp.model.lh_e = 0.0
# ocp.model.uh_e = 1.2**2
# Halfspace constraints
ocp.constraints.C = np.array([[1, 1, 0, 0, 0]])
ocp.constraints.D = np.array([[0, 0]])
ocp.constraints.lg = -np.ones(1)
ocp.constraints.ug = np.ones(1)
ocp.constraints.C_e = np.array([[1, 1, 0, 0, 0]])
ocp.constraints.D_e = np.array([[0, 0]])
ocp.constraints.lg_e = -np.ones(1)
ocp.constraints.ug_e = np.ones(1)
# set initial condition
ocp.constraints.x0 = np.array([-0.22954451,  0.047049,  0.0,  0.16719286,  1.45000016])



###### set solver options #####
# set options, solve with qpoases
# ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
# ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.nlp_solver_type = 'SQP_RTI'
# ocp.solver_options.print_level = 1

ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
# set the params
for i in range(N+1):
    ocp_solver.set(i, "p", np.array([1.0]))

# set the cost reference for each node
for i in range(N):
    ocp_solver.cost_set(i, "yref", np.hstack((reference_track[i, :],np.zeros(nu))))
ocp_solver.cost_set(N, "yref", reference_track[N, :])
# set the halfspace constraints for each node
offset = 1.2
for i in range(N):
    # calculate normal to the reference track
    a = reference_track[i+1, 1] - reference_track[i, 1]
    b = reference_track[i, 0] - reference_track[i+1, 0]
    ocp_solver.constraints_set(i, "C", np.array([[a, b, 0, 0, 0]]))
    # calculate point (x0,y0) offset meters away from the reference track, in normal direction
    x0 = reference_track[i, 0] + offset * a / np.sqrt(a**2 + b**2)
    y0 = reference_track[i, 1] + offset * b / np.sqrt(a**2 + b**2)
    ocp_solver.constraints_set(i, "ug", np.array([x0*a + y0*b]))
    # calculate point (x1,y1) offset meters away from the reference track, in opposite direction of the normal
    x1 = reference_track[i, 0] - offset * a / np.sqrt(a**2 + b**2)
    y1 = reference_track[i, 1] - offset * b / np.sqrt(a**2 + b**2)
    ocp_solver.constraints_set(i, "lg", np.array([x1*a + y1*b]))
# set the terminal halfspace constraints
a = reference_track[N, 1] - reference_track[N-1, 1]
b = reference_track[N-1, 0] - reference_track[N, 0]
ocp_solver.constraints_set(N, "C", np.array([[a, b, 0, 0, 0]]))
x0 = reference_track[N, 0] + offset * a / np.sqrt(a**2 + b**2)
y0 = reference_track[N, 1] + offset * b / np.sqrt(a**2 + b**2)
ocp_solver.constraints_set(N, "ug", np.array([x0*a + y0*b]))
x1 = reference_track[N, 0] - offset * a / np.sqrt(a**2 + b**2)
y1 = reference_track[N, 1] - offset * b / np.sqrt(a**2 + b**2)
ocp_solver.constraints_set(N, "lg", np.array([x1*a + y1*b]))


# solve
status = ocp_solver.solve()

# get solution
simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
    # print(f"simX[{i}] = {simX[i,:]}")
    print(f"distance to reference = {np.sqrt((simX[i,0]-reference_track[i,0])**2+(simX[i,1]-reference_track[i,1])**2)}")
simX[N,:] = ocp_solver.get(N, "x")

# print statistics
time_tot = ocp_solver.get_stats("time_tot")
print(f"CPU-time = {time_tot}")

# plot results
plt.figure(1)
plt.plot(reference_track[:, 0], reference_track[:, 1], 'ro-', label='reference track')
plt.plot(simX[:, 0], simX[:, 1], 'go-', label='simulated trajectory')
plt.axis('equal')
plt.show()



# ##### plot to check the halfspace definition #####
# # Create a figure with 20 subplots arranged in a grid
# fig, axs = plt.subplots(4, 5, figsize=(15, 10))  # 4 rows, 5 columns

# # Visualize the halfspace constraints by coloring the area outside the constraints
# N = 20
# for i in range(N):
#     ax = axs[i // 5, i % 5]  # Get the correct subplot
#     ax.plot(reference_track[:, 0], reference_track[:, 1], 'ro-', label='reference track')
#     a = reference_track[i+1, 1] - reference_track[i, 1]
#     b = reference_track[i, 0] - reference_track[i+1, 0]
#     x0 = reference_track[i, 0] + offset * a / np.sqrt(a**2 + b**2)
#     y0 = reference_track[i, 1] + offset * b / np.sqrt(a**2 + b**2)
#     x1 = reference_track[i, 0] - offset * a / np.sqrt(a**2 + b**2)
#     y1 = reference_track[i, 1] - offset * b / np.sqrt(a**2 + b**2)
#     for m in np.linspace(-1, 4, 50):
#         for n in np.linspace(-2, 2, 50):
#             if a*m + b*n > x0*a + y0*b:
#                 ax.plot(m, n, 'b.')
#     ax.plot(reference_track[i, 0], reference_track[i, 1], 'go')
#     ax.axis('equal')

# plt.tight_layout()
# plt.show()