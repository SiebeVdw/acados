import numpy as np
import matplotlib.pyplot as plt
from loader import loader
from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca

reference_track, current_state, u_prev = loader()


ocp = AcadosOcp()

# set model
model = export_bicycle_model()
control_points = ca.MX.sym('control_points', 2*reference_track.shape[0], 1)
model.p = control_points
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


###### generate nonlinear function y ######
# spline through the control points

