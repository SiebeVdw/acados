import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import numpy as np
from utils import create_error_function
from pathlib import Path


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
    N = x.shape[0]-1
    # figure   
    fig, ax = plt.subplots(2,4)
    # time vectors
    t_x = np.linspace(0,params['Tf'],N+1)
    t_u = t_x[:-1]
    # inputs    
    # ax[0,0].plot(t_u, u[:,u_vars.index('alpha')])
    ax[0,0].step(t_u[:-1], u[:,u_vars.index('alpha')][:-1], where='post')
    ax[0,0].step(t_u[:-1], u_guess[:,u_vars.index('alpha')][:-1], 'g--', where='post')
    ax[0,0].plot(t_u[:-1], np.ones(N-1)*params['alpha_min'], 'r--')
    ax[0,0].plot(t_u[:-1], np.ones(N-1)*params['alpha_max'], 'r--')
    ax[0,0].set_ylabel('rad/s²', fontsize=12)
    ax[0,0].set_title('alpha', fontsize=14)
    # ax[0,1].plot(t_u, u[:,u_vars.index('phi')]*180/np.pi)
    ax[0,1].step(t_u[:-1], u[:,u_vars.index('phi')][:-1]*180/np.pi, where='post')
    ax[0,1].step(t_u[:-1], u_guess[:,u_vars.index('phi')][:-1]*180/np.pi, 'g--', where='post')
    ax[0,1].plot(t_u[:-1], np.ones(N-1)*params['phi_min']*180/np.pi, 'r--')
    ax[0,1].plot(t_u[:-1], np.ones(N-1)*params['phi_max']*180/np.pi, 'r--')
    ax[0,1].set_ylabel('degrees/s', fontsize=12)
    ax[0,1].set_title('phi', fontsize=14)
    # ax[0,2].plot(t_u, u[:,u_vars.index('zeta')])
    ax[0,2].step(t_u[:-1], u[:,u_vars.index('zeta')][:-1], where='post')
    ax[0,2].step(t_u[:-1], u_guess[:,u_vars.index('zeta')][:-1], 'g--', where='post')
    ax[0,2].plot(t_u[:-1], np.ones(N-1)*params['zeta_min'], 'r--')
    ax[0,2].plot(t_u[:-1], np.ones(N-1)*params['zeta_max'], 'r--')
    ax[0,2].set_title('zeta', fontsize=14)
    # states
    ax[1,0].plot(t_x[:-1], x[:,x_vars.index('theta')][:-1]*180/np.pi)
    ax[1,0].plot(t_x[:-1], x_guess[:,x_vars.index('theta')][:-1]*180/np.pi, 'g--')
    ax[1,0].set_title('theta', fontsize=14)
    ax[1,0].set_ylabel('degrees', fontsize=12)
    ax[1,1].plot(t_x[:-1], x[:,x_vars.index('delta')][:-1]*180/np.pi)
    ax[1,1].plot(t_x[:-1], x_guess[:,x_vars.index('delta')][:-1]*180/np.pi, 'g--')
    # ax[1,1].plot(t_x, np.ones(N+1)*params['delta_min'], 'r--')
    # ax[1,1].plot(t_x, np.ones(N+1)*params['delta_max'], 'r--')
    ax[1,1].set_title('delta', fontsize=14)
    ax[1,1].set_ylabel('degrees', fontsize=12)
    ax[1,2].plot(t_x[:-1], x[:,x_vars.index('v')][:-1])
    ax[1,2].plot(t_x[:-1], x_guess[:,x_vars.index('v')][:-1], 'g--')
    ax[1,2].set_title('velocity', fontsize=14)
    ax[1,2].plot(t_x[:-1], np.ones(N)*params['v_min'], 'r--')
    ax[1,2].plot(t_x[:-1], np.ones(N)*params['v_max'], 'r--')
    ax[1,2].set_ylabel('m/s', fontsize=12)
    ax[1,3].plot(t_x[:-1], x[:,x_vars.index('tau')][:-1])
    ax[1,3].plot(t_x[:-1], x_guess[:,x_vars.index('tau')][:-1], 'g--')
    ax[1,3].plot(t_x[:-1], np.ones(N)*params['tau_min'], 'r--')
    ax[1,3].plot(t_x[:-1], np.ones(N)*params['tau_max'], 'r--')
    ax[1,3].set_title('tau', fontsize=14)
    ax[1,3].set_ylabel('%', fontsize=12)
    # position states at the left over plot 
    ax[0,3].plot(x[:,x_vars.index('x')][:-1], x[:,x_vars.index('y')][:-1])
    ax[0,3].plot(x_guess[:,x_vars.index('x')][:-1], x_guess[:,x_vars.index('y')][:-1], 'g--')

    for a in ax.flat:
        a.tick_params(axis='both', which='major', labelsize=10)

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

def plot_solve_times_interactive(solve_times):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    def plot_hist(binsize, x_val):
        ax.hist(solve_times, bins=int(binsize))
        ax.set_title("Histogram of solver times")
        ax.set_xlabel("Time [msec]")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5)
        fig.canvas.draw_idle()
        fraction_below = np.sum(solve_times < x_val)/solve_times.shape[0]*100
        count_text = ax.text(0.55, 0.95, f'{fraction_below:.2f}% < {x_val}msec', 
                        transform=ax.transAxes, fontsize=12, verticalalignment='top')
    initial_bins = 100
    initial_x = 25
    plot_hist(initial_bins, initial_x)
    slider_binsize_ax = plt.axes([0.2, 0.15, 0.6, 0.03]) 
    slider_binsize = Slider(slider_binsize_ax, 'bins', 1, 200, valinit=initial_bins, valstep=1)
    slider_x_ax = plt.axes([0.2, 0.05, 0.6, 0.03]) 
    slider_x = Slider(slider_x_ax, 'X Line', 0, int(1.05*max(solve_times)), valinit=initial_x, valstep=1)
    
    def update(val):
        ax.clear()
        plot_hist(slider_binsize.val, slider_x.val)
    slider_binsize.on_changed(update)
    slider_x.on_changed(update)
    
    plt.show()
    return fig


if __name__ == '__main__':
    # plot solve times for a given run
    run_name = "08-12-2024__14-56"
    solve_times = np.load(Path(__file__).parent / "mpcc_results" / run_name / "solve_times.npy")
    fig = plot_solve_times_interactive(solve_times)
    fig.savefig(Path(__file__).parent / "mpcc_results" / run_name / "solve_times_hist", dpi=300)