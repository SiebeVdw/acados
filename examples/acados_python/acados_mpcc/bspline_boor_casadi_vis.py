import casadi as ca
import numpy as np
import matplotlib.pyplot as plt





# Parameters
num_points = 20 
degree = 2

angles = np.linspace(0, 4*np.pi, num_points)  
def get_control_points(angles):
    # return angles, np.sin(angles)
    return angles**2/np.sqrt((4*np.pi+1)**2 - angles**2), np.sin(angles)*np.cos(angles)**2
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


spline = ca.bspline(tau, C1, knots, [degree], K, {})
spline_function = ca.Function("spline", [tau], [spline])
res = np.array([spline_function(point).full().flatten() for point in np.linspace(0,1,100)])

points = np.linspace(0, 1, K)
A = np.array([spline_function(point).full().flatten() for point in points])


### R = AC and we need to solve for the control points C ###

R = ca.DM([control_points_x, control_points_y]).T
C = ca.solve(A, R, "csparse")


# make spline with C as control points
final_spline = ca.bspline(tau, C.T, knots, [degree], 2, {})
final_spline_function = ca.Function("final_spline", [tau], [final_spline])
final_res = np.array([final_spline_function(point).full().flatten() for point in np.linspace(0.0,1.0,1000)])

# calculate derivative
final_spline_derivative = ca.jacobian(final_spline, tau)
final_spline_derivative_function = ca.Function("final_spline_derivative", [tau], [final_spline_derivative])
final_res_for_derivative = np.array([final_spline_function(point).full().flatten() for point in np.linspace(0.0,1.0,10)])
final_res_derivative = np.array([final_spline_derivative_function(point).full().flatten() for point in np.linspace(0.0,1.0,10)])


plt.plot(final_res[:,0], final_res[:,1], label='Bspline with control points')
plt.plot(control_points_x, control_points_y, 'o', label='Control points')
plt.plot(get_control_points(np.linspace(0,4*np.pi,1000))[0], get_control_points(np.linspace(0,4*np.pi,1000))[1], label = 'reference curve')
plt.quiver(final_res_for_derivative[:,0], final_res_for_derivative[:,1], final_res_derivative[:,0], final_res_derivative[:,1], color='green', angles='xy', scale_units='xy', scale=1, width=0.002, label="Derivative Vectors")
plt.legend()
plt.show()

# plot basis functions
for i in range(len(res[0])):
    plt.plot(np.linspace(0,1,100),res[:,i], label=f'Spline basis function {i}')
    plt.plot(points, A[:,i], 'o', label = f'Basis function {i} evaluated at knots')
plt.legend()
plt.show()

