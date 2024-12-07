import numpy as np
import casadi as ca


def create_spline_function(ref_track, with_derivative=False, degree=2):
    degree = 2
    n_control_points = ref_track.shape[0]
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
    tau = ca.MX.sym("tau")
    spline = ca.bspline(tau, ref_track[:,:2].T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    if with_derivative:
        spline_derivative = ca.jacobian(spline, tau)
        spline_derivative_function = ca.Function("spline_derivative", [tau], [spline_derivative])
        return spline_function, spline_derivative_function
    return spline_function

def create_error_function(ref_track, with_splines=False):
    spline, dspline = create_spline_function(ref_track, with_derivative=True)
    xpos = ca.MX.sym("xpos")
    ypos = ca.MX.sym("ypos")
    tau = ca.MX.sym("tau")
    phi =   ca.atan2((dspline(tau)[1] + 1e-6), dspline(tau)[0] + 1e-6)
    ec  =   ca.sin(phi) * (xpos - spline(tau)[0]) - ca.cos(phi) * (ypos - spline(tau)[1])
    el  = - ca.cos(phi) * (xpos - spline(tau)[0]) - ca.sin(phi) * (ypos - spline(tau)[1])
    ec_function = ca.Function("ec", [xpos, ypos, tau], [ec])
    el_function = ca.Function("el", [xpos, ypos, tau], [el])
    if with_splines:
        return ec_function, el_function, spline, dspline
    return ec_function, el_function

def sample_equidistant_track_points(ref_track, distance):
    spline_function = create_spline_function(ref_track)
    ref_track = [[ref_track[0,0], ref_track[0,1]]]
    tau = 0.0
    while tau < 1.0:
        dist = np.linalg.norm(spline_function(tau).full().flatten() - ref_track[-1])
        if dist > distance:
            ref_track.append(spline_function(tau).full().flatten())
        tau += 0.00001
    return np.array(ref_track)

def extract_reference_track(ref_track, track_optimization_length):
    reference_track_length = 0.0
    for i in range(1, ref_track.shape[0]):
        reference_track_length += np.linalg.norm(ref_track[i, 0:2] - ref_track[i-1, 0:2])
        if reference_track_length > track_optimization_length:
            return ref_track[:i, :]
    print("Using the complete reference track for optimization")
    return ref_track

def print_cost_contributions(track, x, x_vars, u, u_vars, params):
    dt = params['dt']
    ql, qc, ra, rs, rz = params['ql'], params['qc'], params['ra'], params['rs'], params['rz']
    ec, el = create_error_function(track)
    longitudinal_cost, lateral_cost, acceleration_cost, steering_cost, zeta_cost = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(params['N']):
        ec_val = ec(x[i, x_vars.index('x')], x[i, x_vars.index('y')], x[i, x_vars.index('tau')]).full().flatten()
        el_val = el(x[i, x_vars.index('x')], x[i, x_vars.index('y')], x[i, x_vars.index('tau')]).full().flatten()
        longitudinal_cost += ql*el_val**2*dt
        lateral_cost += qc*ec_val**2*dt
        acceleration_cost += ra*u[i, u_vars.index('alpha')]**2*dt
        steering_cost += rs*u[i, u_vars.index('phi')]**2*dt
        zeta_cost -= rz*u[i, u_vars.index('zeta')]*dt
    print(f"longitudinal cost: {longitudinal_cost}")
    print(f"lateral cost: {lateral_cost}")
    print(f"acceleration cost: {acceleration_cost}")
    print(f"steering cost: {steering_cost}")
    print(f"zeta cost: {zeta_cost}")
    print(f'sum of costs: {longitudinal_cost + lateral_cost + acceleration_cost + steering_cost + zeta_cost}')


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

