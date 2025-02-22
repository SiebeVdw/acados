from bspline_boor_casadi_timer import calc_times_casadi, calc_times_casadi_simple
from test_splines import calc_times
import numpy as np

num_points_array = np.linspace(10,500,200)
dt_generate_spline_casadi_array = np.zeros(len(num_points_array))
dt_eval_spline_casadi_array = np.zeros(len(num_points_array))
dt_eval_spline_1000_casadi_array = np.zeros(len(num_points_array))
dt_generate_spline_array = np.zeros(len(num_points_array))
dt_eval_spline_array = np.zeros(len(num_points_array))
dt_eval_spline_1000_array = np.zeros(len(num_points_array))
dt_eval_spline_casadi_simple_array = np.zeros(len(num_points_array))
dt_eval_dspline_casadi_simple_array = np.zeros(len(num_points_array))


for i,num_points in enumerate(num_points_array):
    print(i)
    # dt_generate_spline_casadi_array[i], dt_eval_spline_casadi_array[i], dt_eval_spline_1000_casadi_array[i] = calc_times_casadi(int(num_points))[:3]
    dt_generate_spline_array[i], dt_eval_spline_array[i], dt_eval_spline_1000_array[i] = calc_times(int(num_points))
    dt_eval_spline_casadi_simple_array[i], dt_eval_dspline_casadi_simple_array[i] = calc_times_casadi_simple(int(num_points))
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2)
# eval spline
ax[0].plot(num_points_array, dt_eval_spline_array, label='eval spline')
ax[0].set_xlabel("number of control points", fontsize=16)
ax[0].set_ylabel("time [msec]", fontsize=16)
ax[0].set_title("Custom Bspline", fontsize=16)
ax[0].tick_params(axis='both', which='major', labelsize=14)

ax[1].plot(num_points_array, dt_eval_spline_casadi_simple_array, label='eval spline casadi simple')
ax[1].set_xlabel("number of control points", fontsize=16)
ax[1].set_ylabel("time [msec]", fontsize=16)
ax[1].set_title("CasADi Bspline", fontsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()
