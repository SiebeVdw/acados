import matplotlib.pyplot as plt
import numpy as np
from utils import create_error_function


def plot_track_boundaries(track, width, closed_track=True, style='r--', lwidth=0.):  
    left_bp_prev, right_bp_prev = compute_boundary_points(track, 0, width)
    for i in range(1,track.shape[0]):
        left_bp, right_bp = compute_boundary_points(track, i, width)
        plt.plot([left_bp_prev[0], left_bp[0]], [left_bp_prev[1], left_bp[1]], style, linewidth=lwidth)
        plt.plot([right_bp_prev[0], right_bp[0]], [right_bp_prev[1], right_bp[1]], style, linewidth=lwidth)
        left_bp_prev, right_bp_prev = left_bp, right_bp
    if closed_track:
        left_bp_0, right_bp_0 = compute_boundary_points(track, 0, width)
        plt.plot([left_bp_prev[0], left_bp_0[0]], [left_bp_prev[1], left_bp_0[1]], style, linewidth=lwidth)
        plt.plot([right_bp_prev[0], right_bp_0[0]], [right_bp_prev[1], right_bp_0[1]], style, linewidth=lwidth)

def compute_boundary_points(track, idx, width):
    # compute normal vector
    normal = track[idx, :2] - track[(idx+1)%track.shape[0], :2]
    normal = np.array([-normal[1], normal[0]])
    normal = normal / np.linalg.norm(normal) * width
    # compute left and right boundary points
    left_bp, right_bp = track[idx, :2] - normal, track[idx, :2] + normal
    return left_bp, right_bp

def plot_result_time_series(x, x_guess, x_vars, u, u_guess, u_vars, params):
    N = params['N']
    # figure   
    fig, ax = plt.subplots(2,4)
    # inputs
    t = np.linspace(0,params['Tf'],params['N'])
    ax[0,0].plot(t, u[:,u_vars.index('alpha')])
    ax[0,0].step(t, u[:,u_vars.index('alpha')], where='post')
    ax[0,0].step(t, u_guess[:,u_vars.index('alpha')], 'g--', where='post')
    ax[0,0].plot(t, np.ones(N)*params['alpha_min'], 'r--')
    ax[0,0].plot(t, np.ones(N)*params['alpha_max'], 'r--')
    ax[0,0].set_ylabel('rad/s²')
    ax[0,0].set_title('alpha')
    ax[0,1].plot(t, u[:,u_vars.index('phi')]*180/np.pi)
    ax[0,1].step(t, u[:,u_vars.index('phi')]*180/np.pi, where='post')
    ax[0,1].step(t, u_guess[:,u_vars.index('phi')]*180/np.pi, 'g--', where='post')
    ax[0,1].plot(t, np.ones(N)*params['phi_min']*180/np.pi, 'r--')
    ax[0,1].plot(t, np.ones(N)*params['phi_max']*180/np.pi, 'r--')
    ax[0,1].set_ylabel('degrees/s')
    ax[0,1].set_title('phi')
    ax[0,2].plot(t, u[:,u_vars.index('zeta')])
    ax[0,2].step(t, u[:,u_vars.index('zeta')], where='post')
    ax[0,2].step(t, u_guess[:,u_vars.index('zeta')], 'g--', where='post')
    ax[0,2].plot(t, np.ones(N)*params['zeta_min'], 'r--')
    ax[0,2].plot(t, np.ones(N)*params['zeta_max'], 'r--')
    ax[0,2].set_title('zeta')
    # states
    t = np.linspace(0,params['Tf'],params['N']+1)
    ax[1,0].plot(t, x[:,x_vars.index('theta')]*180/np.pi)
    ax[1,0].plot(t, x_guess[:,x_vars.index('theta')]*180/np.pi, 'g--')
    ax[1,0].set_title('theta')
    ax[1,0].set_ylabel('degrees')
    ax[1,1].plot(t, x[:,x_vars.index('delta')]*180/np.pi)
    ax[1,1].plot(t, x_guess[:,x_vars.index('delta')]*180/np.pi, 'g--')
    # ax[1,1].plot(t, np.ones(N+1)*params['delta_min'], 'r--')
    # ax[1,1].plot(t, np.ones(N+1)*params['delta_max'], 'r--')
    ax[1,1].set_title('delta')
    ax[1,1].set_ylabel('degrees')
    ax[1,2].plot(t, x[:,x_vars.index('v')])
    ax[1,2].plot(t, x_guess[:,x_vars.index('v')], 'g--')
    ax[1,2].set_title('velocity')
    ax[1,2].plot(t, np.ones(N+1)*params['v_min'], 'r--')
    ax[1,2].plot(t, np.ones(N+1)*params['v_max'], 'r--')
    ax[1,2].set_ylabel('m/s')
    ax[1,3].plot(t, x[:,x_vars.index('tau')])
    ax[1,3].plot(t, x_guess[:,x_vars.index('tau')], 'g--')
    ax[1,3].plot(t, np.ones(N+1)*params['tau_min'], 'r--')
    ax[1,3].plot(t, np.ones(N+1)*params['tau_max'], 'r--')
    ax[1,3].set_title('tau')
    ax[1,3].set_ylabel('%')
    # position states at the left over plot 
    ax[0,3].plot(x[:,x_vars.index('x')], x[:,x_vars.index('y')])
    ax[0,3].plot(x_guess[:,x_vars.index('x')], x_guess[:,x_vars.index('y')], 'g--')

    return fig

def plot_result_xy(complete_track, track, x, x_guess, x_vars, params):
    ec, el, spline, dspline = create_error_function(track, with_splines=True)   
    N = params['N']
    fig = plt.figure()
    plt.plot(x[:,x_vars.index('x')], x[:,x_vars.index('y')], 'o-', label="MPCC prediction")
    plt.plot(np.append(complete_track[:,0],complete_track[0,0]), np.append(complete_track[:,1],complete_track[0,1]), '--', label="Centerline")

    # comment out the following line to unplot/plot the initial guess
    # plt.plot(x_guess[:,0], x_guess[:,1], 'o-', color='orange', label="Initial Guess")

    # plot initial position
    plt.plot(x_guess[0,0], x_guess[0,1], 'ro', label="Current position")
    # plot boundaries 
    plot_track_boundaries(complete_track, 1.5, lwidth=0.5)

    # plot dotted lines connecting the states (X,Y) with the spline points spline(tau)
    for i in range(N+1):
        # Spline point at current tau
        spline_xy = spline(x[i,x_vars.index('tau')]).full().flatten()
        
        # Draw line connecting state to spline
        plt.plot([x[i,x_vars.index('x')], spline_xy[0]], [x[i,x_vars.index('y')], spline_xy[1]], 'k--', linewidth=0.5)

        # draw constraint circle
        circle = plt.Circle((spline_xy[0], spline_xy[1]), params['circle_radius'], color='r', fill=False)
        plt.gca().add_artist(circle)

        # Compute errors
        ec_val = ec(x[i,x_vars.index('x')], x[i,x_vars.index('y')], x[i,x_vars.index('tau')]).full().flatten()
        el_val = el(x[i,x_vars.index('x')], x[i,x_vars.index('y')], x[i,x_vars.index('tau')]).full().flatten()

        # Compute spline tangent direction
        spline_der = dspline(x[i,x_vars.index('tau')]).full().flatten()
        tangent_dir = spline_der / (np.linalg.norm(spline_der) + 1e-6)  # Normalize tangent

        # Lateral error vector
        lateral_vector = np.array([-tangent_dir[1], tangent_dir[0]]) * ec_val

        # Longitudinal error vector
        longitudinal_vector = tangent_dir * el_val

        # Add arrows for longitudinal and lateral errors
        plt.arrow(x[i,x_vars.index('x')], x[i,x_vars.index('y')], lateral_vector[0], lateral_vector[1], color='blue', head_width=0.05, length_includes_head=True, label="Lateral Error" if i == 0 else "")
        plt.arrow(x[i,x_vars.index('x')], x[i,x_vars.index('y')], longitudinal_vector[0], longitudinal_vector[1], color='green', head_width=0.05, length_includes_head=True, label="Longitudinal Error" if i == 0 else "")

    plt.axis('equal')
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("XY plot of one MPCC iteration")

    return fig