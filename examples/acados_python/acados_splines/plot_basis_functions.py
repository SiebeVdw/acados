import casadi as ca
import numpy as np
import matplotlib.pyplot as plt




def create_basis_functions(num_points, degree):
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
    res = spline_function(ca.linspace(0,1,1000).T).T

    return res, knots

max_degree = 4
fig, axs = plt.subplots(1, max_degree, figsize=(15, 4))

for degree in range(max_degree):
    basis_functions, knots = create_basis_functions(5, degree)
    axs[degree].plot(np.linspace(0, 1, 1000), basis_functions)
    axs[degree].set_title(f"p = {degree}")
    axs[degree].set_xlabel("Parameter $\\tau$")
    axs[degree].set_ylabel("Value")
    
    # Plot the knots
    for knot in knots[0]:
        axs[degree].axvline(knot, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

plt.tight_layout()
plt.show()
