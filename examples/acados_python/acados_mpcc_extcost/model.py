import casadi as ca
from acados_template import AcadosModel
import numpy as np
import yaml
from pathlib import Path

def export_bicycle_model(n_control_points):

    with open(Path(__file__).parent / "parameters.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    model_name = "bicycle_model"

    # parameters
    R = 0.2023
    L = 1.6

    # states
    posx = ca.MX.sym("posx")     # x position
    posy = ca.MX.sym("posy")     # y position
    theta = ca.MX.sym("theta")   # orientation
    delta = ca.MX.sym("delta")   # steering angle
    v = ca.MX.sym("v")           # velocity
    tau = ca.MX.sym("tau")       # arclength progress
    x = ca.vertcat(posx, posy, theta, delta, v, tau)

    # inputs
    alpha = ca.MX.sym("alpha")   # acceleration
    phi = ca.MX.sym("phi")       # steering angle rate
    zeta = ca.MX.sym("zeta")     # arclength rate
    u = ca.vertcat(alpha, phi, zeta)

    # parameters
    control_points = ca.MX.sym('control_points', 2*n_control_points, 1)

    # dynamics
    dx = v * ca.cos(theta)
    dy = v * ca.sin(theta)
    omega = v * ca.tan(delta) / L
    dv = alpha * R
    dtau = zeta
    f_expl = ca.vertcat(dx, dy, omega, phi, dv, dtau)


    # create the actual model
    model = AcadosModel()
    model.p = control_points
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = model_name

    # define spline functions
    degree = 2
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
    c_points = ca.MX.sym("c_points", 2, n_control_points)
    spline = ca.bspline(tau, c_points, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau, c_points], [spline])
    spline_derivative = ca.jacobian(spline, tau)
    spline_derivative_function = ca.Function("spline_derivative", [tau, c_points], [spline_derivative])

    # cost contributions
    ql, qc, ra, rs, rz = params['ql'], params['qc'], params['ra'], params['rs'], params['rz']

    control_points_spline = ca.reshape(control_points, 2, n_control_points).T
    phi =   ca.atan2((spline_derivative_function(x[5], control_points_spline.T)[1] + 1e-6), spline_derivative_function(tau, control_points_spline.T)[0] + 1e-6)
    ec  =   ca.sin(phi) * (x[0] - spline_function(x[5], control_points_spline.T)[0]) - ca.cos(phi) * (x[1] - spline_function(tau, control_points_spline.T)[1])
    el  = - ca.cos(phi) * (x[0] - spline_function(x[5], control_points_spline.T)[0]) - ca.sin(phi) * (x[1] - spline_function(tau,control_points_spline.T)[1])

    model.cost_expr_ext_cost = qc*ec**2 + ql*el**2 + ra*alpha**2 + rs*phi**2 - rz*zeta
    model.cost_expr_ext_cost_e = qc*ec**2 + ql*el**2

    # define constraints
    half_track_width = 2.0
    model.con_h_expr = ec**2 + el**2 - half_track_width**2

    return model
