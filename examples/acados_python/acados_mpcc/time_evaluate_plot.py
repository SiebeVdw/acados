from bspline_boor_casadi_timer import calc_times_casadi
from test_splines import calc_times
import numpy as np

num_points_array = np.linspace(10,200,191)
dt_generate_spline_casadi_array = np.zeros(len(num_points_array))
dt_eval_spline_casadi_array = np.zeros(len(num_points_array))
dt_eval_spline_1000_casadi_array = np.zeros(len(num_points_array))
dt_generate_spline_array = np.zeros(len(num_points_array))
dt_eval_spline_array = np.zeros(len(num_points_array))
dt_eval_spline_1000_array = np.zeros(len(num_points_array))

for i,num_points in enumerate(num_points_array):
    print(i)
    dt_generate_spline_casadi_array[i], dt_eval_spline_casadi_array[i], dt_eval_spline_1000_casadi_array[i] = calc_times_casadi(int(num_points))[:3]
    dt_generate_spline_array[i], dt_eval_spline_array[i], dt_eval_spline_1000_array[i] = calc_times(int(num_points))

import matplotlib.pyplot as plt
fig,ax = plt.subplots(3,1)
ax[0].plot(num_points_array,dt_generate_spline_array, label="custom bspline")
ax[0].plot(num_points_array,dt_generate_spline_casadi_array, label="casadi bspline")
ax[0].set_xlabel("number of control points")
ax[0].set_ylabel("time [msec]")
ax[0].set_title("generate spline")
ax[0].legend()
ax[1].plot(num_points_array,dt_eval_spline_array, label="custom bspline")
ax[1].plot(num_points_array,dt_eval_spline_casadi_array, label="casadi bspline")
ax[1].set_xlabel("number of control points")
ax[1].set_ylabel("time [msec]")
ax[1].set_title("eval spline")
ax[1].legend()
ax[2].plot(num_points_array,dt_eval_spline_1000_array, label="custom bspline")
ax[2].plot(num_points_array,dt_eval_spline_1000_casadi_array, label="casadi bspline")
ax[2].set_title("eval spline 1000 points")
ax[2].set_xlabel("number of control points")
ax[2].set_ylabel("time [msec]")
ax[2].legend()
plt.tight_layout()
plt.show()
