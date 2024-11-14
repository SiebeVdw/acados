import casadi as ca
from acados_template import AcadosModel


def export_bicycle_model():
    model_name = "bicycle_model"

    # parameters
    R = 0.2
    L = 1.6

    # states
    x = ca.MX.sym("x", 6)  # [x, y, theta, delta, v, tau]
    # x_ = x[0]
    # y_ = x[1]
    theta_ = x[2]
    delta_ = x[3]
    v_ = x[4]
    tau_ = x[5]

    # inputs
    u = ca.MX.sym("u", 3)  # [alpha, phi, zeta]
    alpha_ = u[0]
    phi_ = u[1]
    zeta_ = u[2]

    # dynamics
    a = alpha_ * R
    omega = v_ * ca.tan(delta_) / L
    dx = v_ * ca.cos(theta_)
    dy = v_ * ca.sin(theta_)
    dv = a
    dtau = zeta_

    f_expl = ca.vertcat(dx, dy, omega, phi_, dv, dtau)

    # create the actual model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = model_name

    return model
