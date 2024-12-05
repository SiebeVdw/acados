import numpy as np
import casadi as ca

def initial_guess(reference_track, N, dt, params):
    tau = ca.MX.sym("tau")
    degree = 2
    n_control_points = reference_track.shape[0]
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
    spline = ca.bspline(tau, reference_track.T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    spline_derivative = ca.jacobian(spline, tau)
    spline_derivative_function = ca.Function("spline_derivative", [tau], [spline_derivative])

    L  = params['wheelbase']
    R_wheel = params['wheel_radius']

    # create an array of tau values such that the distance between the points is equidistant
    tau_array = np.linspace(0.00001, 1, 1000)
    spline_points = np.array([spline_function(t) for t in tau_array])
    Tau0 = np.zeros(N + 1)
    Tau0[0] = 0.00001
    length = 0.0
    j = 1
    for i in range(1, len(spline_points)):
        length += np.linalg.norm(spline_points[i] - spline_points[i - 1])
        if length > 18.0/N:
            length = 0.0
            Tau0[j] = tau_array[i]
            j += 1
            if j > N:
                break

    X0 = []
    U0 = []
    for idx, tau in enumerate(Tau0):        
        if idx == 0:
            x0 = np.array(spline_function(tau)).flatten()

            der_points = np.array(spline_derivative_function(tau)).flatten()
            phi = ca.atan2(der_points[1], der_points[0])

            x0_state = [x0[0], x0[1], phi, 0, 200, tau]
            X0.append(x0_state)
            continue
        
        x0 = np.array(spline_function(tau)).flatten()
        steering_angle = 0
        der_points = np.array(spline_derivative_function(tau)).flatten()
        phi = ca.atan2(der_points[1], der_points[0])
        heading = phi

        diff_with_prev = heading - X0[-1][2]
        diff_with_prev = ca.fmod(diff_with_prev + np.pi, 2 * np.pi) - np.pi
        heading = X0[-1][2] + diff_with_prev

        alpha = heading - X0[-1][2]
        alpha /= 2
        l_d = np.sqrt((x0[0] - X0[-1][0]) ** 2 + (x0[1] - X0[-1][1]) ** 2)
        # avoid division by zero
        if alpha > 1e-6:
            R = (l_d) / (2 * np.sin(alpha))
        else:
            R = 1e6
        steering_angle = np.arctan(L / R)

        velocity = (
            np.sqrt((x0[0] - X0[-1][0]) ** 2 + (x0[1] - X0[-1][1]) ** 2)
            / dt
        )

        x0_state = np.array([x0[0], x0[1], heading, steering_angle, velocity, tau])
        X0.append(x0_state)

    # X0[0][2] = X0[1][2]
    X0[0][4] = X0[1][4]

    for i in range(N):
        acceleration = (X0[i + 1][4] - X0[i][4]) / dt / R_wheel
        steering_angle_delta = X0[i + 1][3] - X0[i][3]
        steering_angle_delta /= dt
        dtau = (X0[i + 1][5] - X0[i][5]) / dt
        U0.append([acceleration, steering_angle_delta, dtau])

    X0 = np.array(X0)
    U0 = np.array(U0)

    return X0, U0

