import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

def calc_times_casadi(num_points):
    # Parameters
    # num_points = 200 
    degree = 2

    angles = np.linspace(0, 4*np.pi, num_points)  
    def get_control_points(angles):
        return angles, np.sin(angles)*np.cos(angles)**2
        # return angles**2/np.sqrt((4*np.pi+1)**2 - angles**2), np.sin(angles)*np.cos(angles)**2
    control_points_x, control_points_y = get_control_points(angles)

    # nr knots, including 2*k duplicate knots at the ends
    kk = np.linspace(0, 1, len(control_points_x) - degree + 1)
    knots = [[float(i) for i in np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])]]

    # Define a clamped knot vector
    K = len(kk) + degree - 1

    # Define symbolic parameter tau
    tau = ca.MX.sym("tau")

    C1 = ca.DM.zeros(K,K)
    for i in range(K):
        C1[i,i] = 1

    # t0 = time.perf_counter()
    spline = ca.bspline(tau, C1, knots, [degree], K, {})
    spline_function = ca.Function("spline", [tau], [spline])

    points = ca.linspace(0, 1, K)
    A = spline_function(points.T).T


    ### R = AC and we need to solve for the control points C ###
    t0 = time.perf_counter()
    R = ca.DM([control_points_x, control_points_y]).T
    C = ca.solve(A, R, "csparse")


    # make spline with C as control points
    final_spline = ca.bspline(tau, C.T, knots, [degree], 2, {})
    final_spline_function = ca.Function("final_spline", [tau], [final_spline])
    dt_generate_spline = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = np.array(final_spline_function(0.5).full().flatten())
    dt_eval_spline = time.perf_counter() - t0

    t0 = time.perf_counter()
    # _ = np.array([final_spline_function(point).full().flatten() for point in np.linspace(0.0,1.0,1000)])
    _ = final_spline_function(ca.linspace(0,1,1000).T).T
    dt_eval_spline_1000 = time.perf_counter() - t0


    # calculate derivative
    t0 = time.perf_counter()
    final_spline_derivative = ca.jacobian(final_spline, tau)
    final_spline_derivative_function = ca.Function("final_spline_derivative", [tau], [final_spline_derivative])
    dt_generat_derivative = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = np.array(final_spline_derivative_function(0.5).full().flatten())
    dt_eval_derivative = time.perf_counter() - t0

    t0 = time.perf_counter()
    # _ = np.array([final_spline_derivative_function(point).full().flatten() for point in np.linspace(0.0,1.0,1000)])
    _ = np.array(final_spline_derivative_function(ca.linspace(0.0,1.0,1000).T).full().flatten())
    dt_eval_derivative_1000 = time.perf_counter() - t0

    return dt_generate_spline*1e3, dt_eval_spline*1e3, dt_eval_spline_1000*1e3, dt_generat_derivative*1e3, dt_eval_derivative*1e3, dt_eval_derivative_1000*1e3


if __name__ == "__main__":
    num_points_array = np.linspace(10,200,191)
    dt_generate_spline_array = np.zeros(len(num_points_array))
    dt_eval_spline_array = np.zeros(len(num_points_array))
    dt_eval_spline_1000_array = np.zeros(len(num_points_array))
    dt_generat_derivative_array = np.zeros(len(num_points_array))
    dt_eval_derivative_array = np.zeros(len(num_points_array))
    dt_eval_derivative_1000_array = np.zeros(len(num_points_array))

    for i,num_points in enumerate(num_points_array):
        dt_generate_spline_array[i], dt_eval_spline_array[i], dt_eval_spline_1000_array[i], dt_generat_derivative_array[i], dt_eval_derivative_array[i], dt_eval_derivative_1000_array[i] = calc_times_casadi(int(num_points))

    fig,ax = plt.subplots(2,3)
    ax[0,0].plot(num_points_array,dt_generate_spline_array)
    ax[0,0].set_title("generate spline")
    ax[0,1].plot(num_points_array,dt_eval_spline_array)
    ax[0,1].set_title("eval spline")
    ax[0,2].plot(num_points_array,dt_eval_spline_1000_array)
    ax[0,2].set_title("eval spline 1000 points")
    ax[1,0].plot(num_points_array,dt_generat_derivative_array)
    ax[1,0].set_title("generate derivative")
    ax[1,1].plot(num_points_array,dt_eval_derivative_array)
    ax[1,1].set_title("eval derivative")
    ax[1,2].plot(num_points_array,dt_eval_derivative_1000_array)
    ax[1,2].set_title("eval derivative 1000 points")
    plt.show()

