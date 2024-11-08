import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 10          # Number of control points on the half-circle
radius = 1.0             # Radius of the half-circle
degree = 2               # Degree of the B-spline

# Generate control points for a half-circle (theta from 0 to pi)
angles = np.linspace(0, 4*np.pi, num_points)
control_points_x = angles  # X-coordinates on the half-circle
control_points_y = radius * np.sin(angles)  # Y-coordinates on the half-circle

# Combine into a CasADi DM matrix for control points
control_points = ca.DM([control_points_x, control_points_y])

# Define a clamped knot vector
num_knots = num_points + degree + 1
knots = [0] * (degree + 1) + list(np.linspace(0, 1, num_knots - 2 * (degree + 1))) + [1] * (degree + 1)
knots = [float(k) for k in knots]  # Ensure all elements are float for CasADi compatibility
knots = [knots]  # Knot vector should be a list of lists

# Define symbolic parameter tau
tau = ca.MX.sym("tau")


###################
# Define the B-spline with CasADi's bspline function
# create C1 = matrix with all ones (num_points,num_points)
C1 = ca.DM.zeros(num_points,num_points)
points = np.linspace(0, 1, num_points)
R1 = []
for i in range(num_points):
    C1[i,i] = 1
    spline = ca.bspline(tau, C1, knots, [degree], num_points, {})
    spline_function = ca.Function("half_circle_spline", [tau], [spline])
    # plot spline function for tau = 0->1
    res = [spline_function(point).full().flatten() for point in np.linspace(0,1,1000)]
    plt.plot(np.linspace(0,1,1000),res)
plt.show()
    # for line in R1:
    # R1 = [spline_function(point).full().flatten() for point in points]
    # for line in R1:
    #     l = ''.join([f"{x:.2f} " for x in line])
    #     print(f"R1 = {l}")
    # C1[i,i] = 0
# for line in R1:
#     l = ''.join([f"{x:.2f} " for x in line])
#     print(f"R1 = {l}")
# C1[0,0] = 1
# print(f"C1 = {repr(C1)}")
# spline = ca.bspline(tau, C1, [[float(k) for k in np.linspace(0,1,num_knots)]], [degree], num_points, {})
# spline_function = ca.Function("half_circle_spline", [tau], [spline])
# # R1 is the B spline evaluated at 0,1/K,2/K,...,1
# points = np.linspace(0, 1, num_points)
# R1 = [spline_function(point).full().flatten() for point in points]
# for line in R1:
#     l = ''.join([f"{x:.2f} " for x in line])
#     print(f"R1 = {l}")
# # solve C from R1@C = C1
# A = np.linalg.solve(C1, R1)
# C = np.linalg.solve(C1, R1)@control_points
# print(f"C = {C}")
######################

spline = ca.bspline(tau, control_points, knots, [degree], 2, {})
# Define a CasADi function to evaluate the spline
spline_function = ca.Function("half_circle_spline", [tau], [spline])

# Take the derivative of the spline with respect to tau
spline_derivative = ca.jacobian(spline, tau)
spline_derivative_function = ca.Function("half_circle_spline_derivative", [tau], [spline_derivative])

# Sample the spline and its derivative at multiple values of tau to visualize the fit
tau_values = np.linspace(0, 1, 100)
fitted_points = [spline_function(tau_val).full().flatten() for tau_val in tau_values]
derivative_points = [spline_derivative_function(tau_val).full().flatten() for tau_val in tau_values]

# Convert lists to arrays for plotting
fitted_points = np.array(fitted_points)
derivative_points = np.array(derivative_points)

# Plot the control points, fitted spline, and derivatives
plt.figure(figsize=(10, 6))
plt.plot(control_points_x, control_points_y, 'ro-', label="Control Points")
plt.plot(fitted_points[:, 0], fitted_points[:, 1], 'b-', label="Fitted B-Spline Curve")
plt.quiver(fitted_points[:, 0], fitted_points[:, 1], 
           derivative_points[:, 0], derivative_points[:, 1], 
           color='green', angles='xy', scale_units='xy', scale=1, width=0.002, label="Derivative Vectors")

plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("B-Spline Fit to Half-Circle with Derivative Vectors")
plt.show()
