import numpy as np
import casadi as ca
import matplotlib.pyplot as plt 

def create_spline_function(ref_track, with_derivative=False, degree=2):
    degree = 2
    n_control_points = ref_track.shape[0]
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [
        [
            float(i)
            for i in np.concatenate(
                [np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]]
            )
        ]
    ]
    tau = ca.MX.sym("tau")
    spline = ca.bspline(tau, ref_track[:, :2].T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    if with_derivative:
        spline_derivative = ca.jacobian(spline, tau)
        spline_derivative_function = ca.Function(
            "spline_derivative", [tau], [spline_derivative]
        )
        return spline_function, spline_derivative_function
    return spline_function


# plot spline through points sampled from a 2d circle
ref_track = np.array(
    [[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 100)]
)
spline_function = create_spline_function(ref_track, with_derivative=False)
# Evaluate the spline at many points
tau_values = np.linspace(0, 1, 1000)
spline_values = np.array([spline_function(tau).full().flatten() for tau in tau_values])

# Plot the original reference track and the spline
plt.figure(figsize=(8, 8))
plt.plot(ref_track[:, 0], ref_track[:, 1], 'o', label="Reference Points")
plt.plot(spline_values[:, 0], spline_values[:, 1], '-', label="Spline")
plt.axis("equal")
plt.legend()
plt.title("Spline Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()