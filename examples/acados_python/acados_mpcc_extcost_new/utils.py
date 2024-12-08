import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


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
    spline_function = create_spline_function(np.vstack((ref_track, ref_track[0,:])))
    prev_point = [[ref_track[0,0], ref_track[0,1]]]
    ref_track = []
    tau = 0.0
    dist=0.0
    while tau < 1.0:
        point = spline_function(tau).full().flatten()
        dist += np.linalg.norm(point - prev_point)
        if dist > distance:
            ref_track.append(spline_function(tau).full().flatten())
            dist = 0.0
        prev_point = point
        tau += 0.00001
    return np.array(ref_track)

def extract_reference_track(ref_track, track_optimization_length):
    reference_track_length = 0.0
    for i in range(1, ref_track.shape[0]):
        reference_track_length += np.linalg.norm(ref_track[i, 0:2] - ref_track[i-1, 0:2])
        if reference_track_length > track_optimization_length:
            return ref_track[:i, :]
    print("Using the complete reference track for optimization")
    # add the first point at the end
    ref_track = np.vstack((ref_track, ref_track[0,:]))
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

    Tau0 = np.zeros(N+1)
    velocity = np.zeros(N+1)
    acceleration = np.zeros(N)  
    x, y = np.zeros(N+1), np.zeros(N+1)
    theta = np.zeros(N+1)
    delta = np.zeros(N+1)
    ddelta = np.zeros(N)
    zeta = np.zeros(N)
    Tau0[0] = 0.00001
    v0 = 10.0
    a_max = params['alpha_max'] * params['wheel_radius']
    velocity[0] = v0
    x[0], y[0] = spline_function(Tau0[0]).full().flatten()
    dx0, dy0 = spline_derivative_function(Tau0[0]).full().flatten()
    theta[0] = ca.arctan2(dy0, dx0)
    for i in range(1, N+1):
        x0 = spline_function(Tau0[i-1]).full().flatten()
        tau = Tau0[i-1]
        x1 = x0
        while np.linalg.norm(x1-x0) < min(v0*dt+a_max*dt**2/2, params['v_max']*dt):
            tau += 0.000001
            x1 = spline_function(tau).full().flatten()
        Tau0[i] = tau
        v0 = min(v0 + a_max*dt, params['v_max'])
        velocity[i] = v0
        if i < N:
            acceleration[i] = min(a_max, (params['v_max'] - v0)/dt)
        x[i], y[i] = x1
        # heading
        der_points = np.array(spline_derivative_function(Tau0[i]).full().flatten())
        prev_heading = theta[i-1]
        phi = ca.arctan2(der_points[1], der_points[0])
        heading = phi
        diff_with_prev = heading - prev_heading
        diff_with_prev = ca.fmod(diff_with_prev + np.pi, 2 * np.pi) - np.pi
        theta[i] = prev_heading + diff_with_prev
        # steering angle -> choice to take 'i-1', then it is forwards differences to integrate omega. 
        delta[i-1] = ca.atan2(diff_with_prev*params['wheelbase'], dt*velocity[i])
        # zeta
        zeta[i-1] = (Tau0[i] - Tau0[i-1])/dt
    acceleration[0] = acceleration[1]
    delta[-1] = delta[-2]
    for i in range(N):
        ddelta[i] = (delta[i+1] - delta[i])/dt  
    X0 = np.array([x, y, theta, delta, velocity, Tau0]).T
    U0 = np.array([acceleration/params['wheel_radius'], ddelta, zeta]).T
    return X0, U0
