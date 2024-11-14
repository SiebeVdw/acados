import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from loader import loader

class Bspline():
    def __init__(self, n_control_points, degree=2):
        self.n_control_points = n_control_points
        self.degree = degree
        # create knots
        kk = np.linspace(0, 1, n_control_points - degree + 1)
        self.knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
        # create parametric bspline and derivative bspline
        self.tau = ca.MX.sym("tau")
        control_points_param = ca.MX.sym("control_points_param", 2, n_control_points)
        parametric_spline = ca.bspline(self.tau, control_points_param, self.knots, [degree], 2, {})
        parametric_spline_derivative = ca.jacobian(parametric_spline, self.tau)
        self.parametric_spline_function = ca.Function("spline", [self.tau, control_points_param], [parametric_spline])
        self.parametric_spline_derivative_function = ca.Function("spline_derivative", [self.tau, control_points_param], [parametric_spline_derivative])

    def set_control_points(self, control_points):
        """
        sets new control points to the parametric bspline and derivative

        input: control_points: (n_control_points X 2)
        """

        self.spline = ca.Function("spline", [self.tau], [self.parametric_spline_function(self.tau, control_points.T)])
        self.dspline = ca.Function("spline_derivative", [self.tau], [self.parametric_spline_derivative_function(self.tau, control_points.T)])


if __name__ == '__main__':
    reference_track, current_state, u_prev = loader()
    bspline = Bspline(reference_track.shape[0])
    bspline.set_control_points(reference_track[:,:2])

    xy = bspline.spline(ca.linspace(0,1,1000).T).T
    print(xy)
    figure = plt.figure()
    plt.plot(xy[:,0], xy[:,1])
    plt.plot(reference_track[:,0], reference_track[:,1], 'o')
    plt.show()