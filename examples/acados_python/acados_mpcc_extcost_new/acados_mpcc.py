from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca
from utils import initial_guess, print_cost_contributions, sample_equidistant_track_points, extract_reference_track, create_spline_function
from utils_plotting import plot_track_boundaries, plot_result_time_series, plot_result_xy

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from datetime import datetime
import shutil
##########################################################################################

# state and input names for easy lookup
x_vars = ['x', 'y', 'theta', 'delta', 'v', 'tau']
u_vars = ['alpha', 'phi', 'zeta']



##############################
# load parameters and reference track
##############################
# load the parameters
with open(Path(__file__).parent / "parameters.yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# load the complete reference trajectory
ref_track = np.load(Path(__file__).parent / "maps" / f"{params['track_name']}.npy")[1:]
# apply a shift to the reference trajectory
ref_track = np.roll(ref_track, -5, axis=0)

##############################
# create ref/reference track and plot
##############################
ref_track = sample_equidistant_track_points(ref_track, params['distance_between_track_points'])
reference_track = extract_reference_track(ref_track, params['track_optimization_length'])
# plot the complete reference trajectory and the extracted part to optimize
plt.figure()
plt.plot(ref_track[:, 0], ref_track[:, 1], 'b')
plt.plot(reference_track[:, 0], reference_track[:, 1], 'go')
plot_track_boundaries(ref_track, 1.5, lwidth=0.5)
spline = create_spline_function(reference_track)
points = spline(ca.linspace(0, 1, 1000).T).full().T
plt.plot(points[:,0], points[:,1], 'r--')
plt.axis('equal')
plt.show()

##############################
# Initialize acados ocp
##############################
ocp = AcadosOcp()
n_control_points = reference_track.shape[0]
model = export_bicycle_model(n_control_points)
ocp.model = model
ocp.dims.np = 2*n_control_points
ocp.parameter_values = np.zeros((2*n_control_points, ))
# nonlinear circular constraints
ocp.dims.nh = 1
ocp.constraints.uh = np.array([0.0])
ocp.constraints.lh = np.array([-params['circle_radius']**2-0.1])
# dimensions and horizon
nx = model.x.rows()
nu = model.u.rows()
N = params['N']
dt = params['dt']
Tf = N * dt
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
# cost type
ocp.cost.cost_type = 'EXTERNAL'
ocp.cost.cost_type_e = 'EXTERNAL'

##############################
# set constraints to ocp
##############################
# simple input constraints
ocp.constraints.idxbu = np.array([0, 1, 2])
ocp.constraints.lbu = np.array([params['alpha_min'], params['phi_min'], params['zeta_min']])
ocp.constraints.ubu = np.array([params['alpha_max'], params['phi_max'], params['zeta_max']]) # 10 times higher then Zander's because ACADOS multiplies by dt=0.1
# set simple state constraints
ocp.constraints.idxbx = np.array([3,4,5])
ocp.constraints.lbx = np.array([params['delta_min']*np.pi/180, params['v_min'], params['tau_min']])
ocp.constraints.ubx = np.array([params['delta_max']*np.pi/180, params['v_max'], params['tau_max']])
ocp.constraints.idxbx_e = np.array([3,4,5])
ocp.constraints.lbx_e = np.array([params['delta_min']*np.pi/180, params['v_min'], params['tau_min']])
ocp.constraints.ubx_e = np.array([params['delta_max']*np.pi/180, params['v_max'], params['tau_max']])

##############################
# calculate initial guess and set initial state
##############################
if params['initial_guess'] == 'calculate':
    x_guess, u_guess = initial_guess(reference_track, N, dt, params)
    ocp.constraints.x0 = x_guess[0,:]
elif params['initial_guess'] == 'load':
    guess = np.load(Path(__file__).parent / "mpcc_results" / "last_solution.npz")
    x_guess, u_guess = guess['x'], guess['u']
    ocp.constraints.x0 = x_guess[0,:]
elif params['initial_guess'] == 'load_and_shift':
    guess = np.load(Path(__file__).parent / "mpcc_results" / "last_solution.npz")
    x_guess, u_guess = guess['x'], guess['u']
    # shift one step
    ocp.constraints.x0 = x_guess[1, :]
    x_guess = np.vstack((x_guess[1:, :], [x_guess[-1, :]]))
    u_guess = np.vstack((u_guess[1:,:], [u_guess[-1, :]]))

##############################
# set solver
##############################
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
# ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
# ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
# ocp.solver_options.nlp_solver_type = 'SQP_RTI'
ocp.solver_options.print_level = 2
# set max SQP iterations
# ocp.solver_options.qp_solver_iter_max = 100
# set max QP iterations
# ocp.solver_options.nlp_solver_max_iter = 200

ocp_solver = AcadosOcpSolver(ocp, json_file='my_acados_ocp.json')

##############################
# set parameters
##############################
# set reference track as parameters
for i in range(N+1):
    ocp_solver.set(i, "p", reference_track[:,:2].flatten())
# set initial guess
for i in range(N+1):
    ocp_solver.set(i, "x", x_guess[i, :])
ocp_solver.set(N, "x", x_guess[N, :])
for i in range(N):
    ocp_solver.set(i, "u", u_guess[i, :])


def solve_and_plot(save_path, with_plots=True):
    ##############################
    # solve and save
    ##############################
    ocp_solver.solve()
    solve_time = ocp_solver.get_stats('time_tot')*1e3
    print(f"total solve time: {solve_time} ms")
    # get x and u solutions
    x_sol = np.array([ocp_solver.get(i,"x") for i in range(N+1)])
    u_sol = np.array([ocp_solver.get(i,"u") for i in range(N)])
    # save solutions as .npz file
    np.savez(Path(__file__).parent / "mpcc_results" / "last_solution", x=x_sol, u=u_sol)
    np.savez(save_path / "data", x=x_sol, u=u_sol, x_guess=x_guess, u_quess=u_guess, ref_track=ref_track, reference_track=reference_track)

    ##############################
    # results + plots
    ##############################
    # calculate total cost and its contributions
    print(f"\n acados total cost: {ocp_solver.get_cost()}")
    print_cost_contributions(reference_track, x_sol, x_vars, u_sol, u_vars, params)

    if with_plots:
        # plot time series
        fig1 = plot_result_time_series(x_sol, x_guess, x_vars, u_sol, u_guess, u_vars, params)    
        # plot xy plot 
        fig2 = plot_result_xy(ref_track, reference_track, x_sol, x_guess, x_vars, params)
        # save the figures
        fig1.savefig(save_path / "time_series.png")
        fig2.savefig(save_path / "xy_plot.png")
        # show the plots
        # plt.show()
        plt.close("all")

    return x_sol, u_sol, solve_time



##############################
# solver loop
##############################
if __name__ == "__main__":
    # first save the params used for this run in ./mpcc_results/today/params.yaml
    result_path = Path(__file__).parent / "mpcc_results" / datetime.today().strftime('%d-%m-%Y__%H-%M')
    result_path.mkdir(parents=True, exist_ok=True)
    yaml_path = result_path / "parameters.yaml"
    shutil.copy(Path(__file__).parent / "parameters.yaml", yaml_path)
    if params['multi_solve']:
        solve_times = []
        for i in range(150):
            result_iteration_path = result_path / "iterations" / f"iteration{i}"
            result_iteration_path.mkdir(parents=True, exist_ok=True)
            print(f"initial_guess in loop: {x_guess[0,:]}") 
            x_sol, u_sol, solve_time = solve_and_plot(save_path=result_iteration_path, with_plots=True)
            solve_times.append(solve_time)
            # get previous solution
            x_guess, u_guess = x_sol, u_sol
            # shift one step and set as initial guess
            ocp_solver.set(0, 'lbx', x_guess[1, :])
            ocp_solver.set(0, 'ubx', x_guess[1, :])
            x_guess = np.vstack((x_guess[1:, :], [x_guess[-1, :]]))
            u_guess = np.vstack((u_guess[1:,:], [u_guess[-1, :]]))
            for i in range(N+1):
                ocp_solver.set(i, "x", x_guess[i, :])
            ocp_solver.set(N, "x", x_guess[N, :])
            for i in range(N):
                ocp_solver.set(i, "u", u_guess[i, :])

        np.save(result_path / 'solve_times', np.array(solve_times))
    else:
        solve_and_plot(save_path=result_path, with_plots=True)