from loader import loader 
from Bspline import Bspline
import numpy as np 
import matplotlib.pyplot as plt
import time 
import casadi as ca 





solve_times = []
N0 = 10
N1 = 200
N_step = 1
for N in range(N0,N1,N_step):
    pass

def calc_times(N):
    # create an array with (x,y) coordinates, following a sine
    arr = np.array([[i,np.sin(i)] for i in np.linspace(0,50,N)])

    # create Bspline through the reference track
    degree = 2
    kk = np.linspace(0, 1, len(arr) - degree + 1)

    t1 = time.perf_counter()
    bspline = Bspline(kk, degree)
    control_points = bspline.compute_control_points(np.array(arr))
    curve = bspline.gen_curve(control_points)
    dt_generate_spline = (time.perf_counter()-t1)*1e3

    t1 = time.perf_counter()
    spline_arr = curve(0.5)
    dt_eval_spline = (time.perf_counter()-t1)*1e3

    t1 = time.perf_counter()
    spline_arr = curve(np.linspace(0, 1, 1000))
    dt_eval_spline_1000 = (time.perf_counter()-t1)*1e3
   
    # plt.plot(spline_arr[:, 0], spline_arr[:, 1], c='r')
    # plt.plot(arr[:, 0], arr[:, 1], 'o')
    # plt.show()

    return dt_generate_spline, dt_eval_spline, dt_eval_spline_1000

    
if __name__ == "__main__":
    num_points_array = np.linspace(10,200,191)
    dt_generate_spline_array = np.zeros(len(num_points_array))
    dt_eval_spline_array = np.zeros(len(num_points_array))
    dt_eval_spline_1000_array = np.zeros(len(num_points_array))

    for i,num_points in enumerate(num_points_array):
        dt_generate_spline_array[i], dt_eval_spline_array[i], dt_eval_spline_1000_array[i] = calc_times(int(num_points))

    fig,ax = plt.subplots(3,1)
    ax[0].plot(num_points_array,dt_generate_spline_array)
    ax[0].set_title("generate spline")
    ax[1].plot(num_points_array,dt_eval_spline_array)
    ax[1].set_title("eval spline")
    ax[2].plot(num_points_array,dt_eval_spline_1000_array)
    ax[2].set_title("eval spline 1000 points")

    plt.show()


