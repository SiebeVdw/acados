import numpy as np
import casadi as ca

def initial_guess(reference_track, N, dt, ):
    tau = ca.MX.sym("tau")
    degree = 2
    n_control_points = reference_track.shape[0]
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
    spline = ca.bspline(tau, reference_track.T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    spline_derivative = ca.jacobian(spline, tau)
    spline_derivative_function = ca.Function("spline_derivative", [tau], [spline_derivative])

    L  = 1.6
    R_wheel = 0.2023
    Tau0 = np.linspace(0.00001, 0.5, N + 1)
    X0 = []
    U0 = []
    for idx, tau in enumerate(Tau0):        
        if idx == 0:
            x0_state = [0, 0, 0, 0, 0, 0]
            X0.append(x0_state)
            continue
        x0 = np.array(spline_function(tau)).flatten()
        steering_angle = 0
        der_points = np.array(spline_derivative_function(tau)).flatten()
        phi = ca.atan2(der_points[1], der_points[0])
        heading = phi
        if len(X0) > 0:
            diff_with_prev = heading - X0[-1][2]
            diff_with_prev = ca.fmod(diff_with_prev + np.pi, 2 * np.pi) - np.pi
            heading = X0[-1][2] + diff_with_prev

        if len(X0) > 0:
            alpha = heading - X0[-1][2]
            alpha /= 2
            l_d = np.sqrt((x0[0] - X0[-1][0]) ** 2 + (x0[1] - X0[-1][1]) ** 2)
            # avoid division by zero
            if alpha > 1e-6:
                R = (l_d) / (2 * np.sin(alpha))
            else:
                R = 1e6
            steering_angle = np.arctan(L / R)
        else:
            steering_angle = 0

        if len(X0) > 0:
            velocity = (
                np.sqrt((x0[0] - X0[-1][0]) ** 2 + (x0[1] - X0[-1][1]) ** 2)
                / dt
            )
        else:
            velocity = 20

        x0_state = np.array([x0[0], x0[1], heading, steering_angle, velocity, tau])
        X0.append(x0_state)

    X0[0][4] = X0[1][4]

    for i in range(N):
        acceleration = (X0[i + 1][4] - X0[i][4]) / dt / R_wheel
        steering_angle_delta = X0[i + 1][3] - X0[i][3]
        steering_angle_delta /= dt
        dtau = X0[i + 1][5] - X0[i][5]
        U0.append([acceleration, steering_angle_delta, dtau])

    X0 = np.array(X0)
    U0 = np.array(U0)

    return X0, U0

from loader import loader
if __name__ == "__main__":
    reference_track, current_state, u_prev = loader()
    X0, U0 = initial_guess(reference_track[:,:2], N=20, dt=0.1)
    print(f'X0: {X0}')
    print(f'U0: {U0}')