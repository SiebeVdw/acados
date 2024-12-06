from model import export_bicycle_model
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca
from utils import initial_guess

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml



# state and input names for easy lookup
x_vars = ['x', 'y', 'theta', 'delta', 'v', 'tau']
u_vars = ['alpha', 'phi', 'zeta']

# load the parameters
with open(Path(__file__).parent / "parameters.yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# load the complete reference trajectory
ref_track = np.load(Path(__file__).parent / "maps" / f"{params['track_name']}.npy")[1:]
# apply a shift to the reference trajectory
ref_track = np.roll(ref_track, -5, axis=0)

# extract 20 meter from the reference trajectory
reference_track_length = 0.0
for i in range(1, ref_track.shape[0]):
    reference_track_length += np.linalg.norm(ref_track[i, 0:2] - ref_track[i-1, 0:2])
    if reference_track_length > params['track_optimization_length']:
        reference_track = ref_track[:i, :]
        break

# plot the complete reference trajectory and the extracted part to optimize
plt.figure()
plt.plot(ref_track[:, 0], ref_track[:, 1], 'b')
plt.plot(reference_track[:, 0], reference_track[:, 1], 'go')
plt.axis('equal')
plt.show()

##############################
# define ocp
##############################
ocp = AcadosOcp()

##############################
# set model
##############################
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
N = 20
dt = 0.1
Tf = N * dt
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
# cost type
ocp.cost.cost_type = 'EXTERNAL'
ocp.cost.cost_type_e = 'EXTERNAL'

##############################
# set constraints
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
    guess = np.load(Path(__file__).parent / "results" / "solution.npz")
    x_guess, u_guess = guess['x'], guess['u']
    ocp.constraints.x0 = x_guess[0,:]
elif params['initial_guess'] == 'load_and_shift':
    guess = np.load(Path(__file__).parent / "results" / "solution.npz")
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
ocp.solver_options.qp_solver_iter_max = 100
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


def solve_and_plot():
    print(f"initial_guess in function: {x_guess[0,:]}")
    ##############################
    # solve and save
    ##############################
    status = ocp_solver.solve()
    print(f"total solve time: {ocp_solver.get_stats('time_tot')*1e3} ms")
    # get x and u solutions
    x_sol = np.array([ocp_solver.get(i,"x") for i in range(N+1)])
    u_sol = np.array([ocp_solver.get(i,"u") for i in range(N)])
    # save solutions as .npz file
    path = Path(__file__).parent / "results" / "solution"
    np.savez(path, x=x_sol, u=u_sol)

    ##############################
    # results + interpretation
    ##############################
    # calculate total cost and its contributions
    ql, qc, ra, rs, rz = params['ql'], params['qc'], params['ra'], params['rs'], params['rz'], 
    # spline
    degree = 2
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]
    c_points = ca.MX.sym("c_points", 2, n_control_points)
    tau = ca.MX.sym("tau")
    spline = ca.bspline(tau, reference_track[:,:2].T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    spline_derivative = ca.jacobian(spline, tau)
    spline_derivative_function = ca.Function("spline_derivative", [tau], [spline_derivative])
    # errors
    xpos = ca.MX.sym("xpos")
    ypos = ca.MX.sym("ypos")
    phi =   ca.atan2((spline_derivative_function(tau)[1] + 1e-6), spline_derivative_function(tau)[0] + 1e-6)
    ec  =   ca.sin(phi) * (xpos - spline_function(tau)[0]) - ca.cos(phi) * (ypos - spline_function(tau)[1])
    el  = - ca.cos(phi) * (xpos - spline_function(tau)[0]) - ca.sin(phi) * (ypos - spline_function(tau)[1])
    ec_function = ca.Function("ec", [xpos, ypos, tau], [ec])
    el_function = ca.Function("el", [xpos, ypos, tau], [el])
    # calculate cost
    print(f"\n cost: {ocp_solver.get_cost()}")
    longitudinal_cost, lateral_cost, acceleration_cost, steering_cost, zeta_cost = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(N):
        ec_val = ec_function(x_sol[i, x_vars.index('x')], x_sol[i, x_vars.index('y')], x_sol[i, x_vars.index('tau')]).full().flatten()
        el_val = el_function(x_sol[i, x_vars.index('x')], x_sol[i, x_vars.index('y')], x_sol[i, x_vars.index('tau')]).full().flatten()
        longitudinal_cost += ql*el_val**2*dt
        lateral_cost += qc*ec_val**2*dt
        acceleration_cost += ra*u_sol[i, u_vars.index('alpha')]**2*dt
        steering_cost += rs*u_sol[i, u_vars.index('phi')]**2*dt
        zeta_cost -= rz*u_sol[i, u_vars.index('zeta')]*dt
    print(f"longitudinal cost: {longitudinal_cost}")
    print(f"lateral cost: {lateral_cost}")
    print(f"acceleration cost: {acceleration_cost}")
    print(f"steering cost: {steering_cost}")
    print(f"zeta cost: {zeta_cost}")
    print(f'sum of costs: {longitudinal_cost + lateral_cost + acceleration_cost + steering_cost + zeta_cost}')

    # plot inputs and state timeseries
    fig, ax = plt.subplots(2,4)
    # inputs
    t = np.linspace(0,Tf,N)
    ax[0,0].plot(t, u_sol[:,u_vars.index('alpha')])
    ax[0,0].step(t, u_sol[:,u_vars.index('alpha')], where='post')
    ax[0,0].step(t, u_guess[:,u_vars.index('alpha')], 'g--', where='post')
    ax[0,0].plot(t, np.ones(N)*params['alpha_min'], 'r--')
    ax[0,0].plot(t, np.ones(N)*params['alpha_max'], 'r--')
    ax[0,0].set_ylabel('rad/s²')
    ax[0,0].set_title('alpha')
    ax[0,1].plot(t, u_sol[:,u_vars.index('phi')]*180/np.pi)
    ax[0,1].step(t, u_sol[:,u_vars.index('phi')]*180/np.pi, where='post')
    ax[0,1].step(t, u_guess[:,u_vars.index('phi')]*180/np.pi, 'g--', where='post')
    ax[0,1].plot(t, np.ones(N)*params['phi_min']*180/np.pi, 'r--')
    ax[0,1].plot(t, np.ones(N)*params['phi_max']*180/np.pi, 'r--')
    ax[0,1].set_ylabel('degrees/s')
    ax[0,1].set_title('phi')
    ax[0,2].plot(t, u_sol[:,u_vars.index('zeta')])
    ax[0,2].step(t, u_sol[:,u_vars.index('zeta')], where='post')
    ax[0,2].step(t, u_guess[:,u_vars.index('zeta')], 'g--', where='post')
    ax[0,2].plot(t, np.ones(N)*params['zeta_min'], 'r--')
    ax[0,2].plot(t, np.ones(N)*params['zeta_max'], 'r--')
    ax[0,2].set_title('zeta')
    # states
    t = np.linspace(0,Tf,N+1)
    ax[1,0].plot(t, x_sol[:,x_vars.index('theta')]*180/np.pi)
    ax[1,0].plot(t, x_guess[:,x_vars.index('theta')]*180/np.pi, 'g--')
    ax[1,0].set_title('theta')
    ax[1,0].set_ylabel('degrees')
    ax[1,1].plot(t, x_sol[:,x_vars.index('delta')]*180/np.pi)
    ax[1,1].plot(t, x_guess[:,x_vars.index('delta')]*180/np.pi, 'g--')
    # ax[1,1].plot(t, np.ones(N+1)*params['delta_min'], 'r--')
    # ax[1,1].plot(t, np.ones(N+1)*params['delta_max'], 'r--')
    ax[1,1].set_title('delta')
    ax[1,1].set_ylabel('degrees')
    ax[1,2].plot(t, x_sol[:,x_vars.index('v')])
    ax[1,2].plot(t, x_guess[:,x_vars.index('v')], 'g--')
    ax[1,2].set_title('velocity')
    ax[1,2].plot(t, np.ones(N+1)*params['v_min'], 'r--')
    ax[1,2].plot(t, np.ones(N+1)*params['v_max'], 'r--')
    ax[1,2].set_ylabel('m/s')
    ax[1,3].plot(t, x_sol[:,x_vars.index('tau')])
    ax[1,3].plot(t, x_guess[:,x_vars.index('tau')], 'g--')
    ax[1,3].plot(t, np.ones(N+1)*params['tau_min'], 'r--')
    ax[1,3].plot(t, np.ones(N+1)*params['tau_max'], 'r--')
    ax[1,3].set_title('tau')
    ax[1,3].set_ylabel('%')

    # plot xy plot with extra features
    plt.figure()
    # plt.scatter(x_sol[:,x_vars.index('x')], x_sol[:,x_vars.index('y')], c=range(N+1), cmap='viridis', label="Vehicle States")
    plt.plot(x_sol[:,x_vars.index('x')], x_sol[:,x_vars.index('y')], 'o-', label="Vehicle States")
    # plt.scatter(ref_track[:,0], ref_track[:,1], c=range(ref_track.shape[0]), cmap='plasma', marker='x', label="Reference Track")
    plt.plot(ref_track[:,0], ref_track[:,1], '--', label="Reference Track")
    plt.plot([ref_track[-1,0], ref_track[0,0]], [ref_track[-1,1], ref_track[0,1]], 'k--')

    # plt.plot(x_guess[:,0], x_guess[:,1], 'o-', color='red', label="Initial Guess")
    # plot initial position
    plt.plot(x_guess[0,0], x_guess[0,1], 'ro', label="Initial Position")
    # plot boundaries = parallel lines to the reference track and at distance track width
    track_width = 1.5
    normal0 = ref_track[0, :2] - ref_track[1, :2]
    normal0 = np.array([-normal0[1], normal0[0]])
    normal0 = normal0 / np.linalg.norm(normal0) * track_width
    parallel1_prev = ref_track[0, :2] + normal0
    parallel2_prev = ref_track[0, :2] - normal0
    for i in range(1,ref_track.shape[0]):
        # compute normal vector
        normal = ref_track[i, :2] - ref_track[(i+1)%ref_track.shape[0], :2]
        normal = np.array([-normal[1], normal[0]])
        normal = normal / np.linalg.norm(normal) * track_width
        # compute parallel lines
        parallel1 = ref_track[i, :2] + normal
        parallel2 = ref_track[i, :2] - normal
        plt.plot([parallel1_prev[0], parallel1[0]], [parallel1_prev[1], parallel1[1]], 'r', linewidth=1.0)
        plt.plot([parallel2_prev[0], parallel2[0]], [parallel2_prev[1], parallel2[1]], 'r', linewidth=1.0)
        parallel1_prev = parallel1
        parallel2_prev = parallel2
    parallel1_0 = ref_track[0, :2] + normal0
    parallel2_0 = ref_track[0, :2] - normal0
    plt.plot([parallel1_prev[0], parallel1_0[0]], [parallel1_prev[1], parallel1_0[1]], 'r', linewidth=1.0)
    plt.plot([parallel2_prev[0], parallel2_0[0]], [parallel2_prev[1], parallel2_0[1]], 'r', linewidth=1.0)


    # plot dotted lines connecting the states (X,Y) with the spline points spline(tau)
    for i in range(N+1):
        # Spline point at current tau
        spline_xy = spline_function(x_sol[i,x_vars.index('tau')]).full().flatten()
        
        # Draw line connecting state to spline
        plt.plot([x_sol[i,x_vars.index('x')], spline_xy[0]], [x_sol[i,x_vars.index('y')], spline_xy[1]], 'k--', linewidth=0.5)

        # draw constraint circle
        circle = plt.Circle((spline_xy[0], spline_xy[1]), params['circle_radius'], color='r', fill=False)
        plt.gca().add_artist(circle)

        # Compute errors
        ec_val = ec_function(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], x_sol[i,x_vars.index('tau')]).full().flatten()
        el_val = el_function(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], x_sol[i,x_vars.index('tau')]).full().flatten()

        # Compute spline tangent direction
        spline_der = spline_derivative_function(x_sol[i,x_vars.index('tau')]).full().flatten()
        tangent_dir = spline_der / (np.linalg.norm(spline_der) + 1e-6)  # Normalize tangent

        # Lateral error vector
        lateral_vector = np.array([-tangent_dir[1], tangent_dir[0]]) * ec_val

        # Longitudinal error vector
        longitudinal_vector = tangent_dir * el_val

        # Add arrows for longitudinal and lateral errors
        plt.arrow(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], lateral_vector[0], lateral_vector[1], color='blue', head_width=0.05, length_includes_head=True, label="Lateral Error" if i == 0 else "")
        plt.arrow(x_sol[i,x_vars.index('x')], x_sol[i,x_vars.index('y')], longitudinal_vector[0], longitudinal_vector[1], color='green', head_width=0.05, length_includes_head=True, label="Longitudinal Error" if i == 0 else "")

    plt.axis('equal')
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Visualization of Longitudinal and Lateral Errors")

    # show all plots
    plt.show()

    return x_sol, u_sol



if params['multi_solve']:
    for _ in range(50):
        print(f"initial_guess in loop: {x_guess[0,:]}") 
        x_sol, u_sol = solve_and_plot()
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

else:
    solve_and_plot()