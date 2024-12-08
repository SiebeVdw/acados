import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import yaml
from utils import create_spline_function
import casadi as ca



run_name = "08-12-2024__14-56"
# load the parameters
with open(Path(__file__).parent / "mpcc_results" / run_name / "parameters.yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)


num_files = params['n_solve_iterations']

# Initialize figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-', label='Prediction', markersize=2)  

######
# reference 
#####
# load reference track
ref_track = np.load(Path(__file__).parent / "maps" / f"{params['track_name']}.npy")[1:]
ax.plot(ref_track[:, 0], ref_track[:, 1], 'y--', label='Centerline')
# load the reference track used for optimization
data =  np.load(Path(__file__).parent / "mpcc_results" / run_name / "iterations" / f"iteration0" / "data.npz")
ref_track_optimzation = data['ref_track']
spline = create_spline_function(ref_track_optimzation)
points = spline(ca.linspace(0, 1, 1000).T).full().T
ax.plot(points[:, 0], points[:, 1], 'b--', label='centerline used for optimization')
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
    ax.plot([parallel1_prev[0], parallel1[0]], [parallel1_prev[1], parallel1[1]], 'r', linewidth=1.0)
    ax.plot([parallel2_prev[0], parallel2[0]], [parallel2_prev[1], parallel2[1]], 'r', linewidth=1.0)
    parallel1_prev = parallel1
    parallel2_prev = parallel2
parallel1_0 = ref_track[0, :2] + normal0
parallel2_0 = ref_track[0, :2] - normal0
ax.plot([parallel1_prev[0], parallel1_0[0]], [parallel1_prev[1], parallel1_0[1]], 'r', linewidth=1.0)
ax.plot([parallel2_prev[0], parallel2_0[0]], [parallel2_prev[1], parallel2_0[1]], 'r', linewidth=1.0)

#####
# mpcc applied states
#####
initial_states_path = []
for i in range(num_files):
    data = np.load(Path(__file__).parent / "mpcc_results" / run_name / "iterations" / f"iteration{i}" / "data.npz")
    initial_states_path.append([data['x'][0, 0], data['x'][0, 1]])
initial_states_path = np.array(initial_states_path)
ax.plot(initial_states_path[:,0], initial_states_path[:,1], 'g--', label='Actual driven raceline', markersize=2)

if params['track_name'] == 'fssim_fsi':
    ax.set_xlim(-60, 50)
    ax.set_ylim(-30, 15)
elif params['track_name'] == 'fssim_fsg':
    ax.set_xlim(-60, 50)
    ax.set_ylim(-90, 15)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend()
ax.grid()
ax.set_aspect('equal')



# List to hold data
x_data = []
y_data = []
reference_data = []

# Function to initialize the plot
def init():
    line.set_data([], [])
    return [line]

# Function to update the animation
def update(frame):
    global x_data, y_data
    
    # Load data from the .npz file
    data = np.load(Path(__file__).parent / "mpcc_results" / run_name / "iterations" / f"iteration{frame}" / "data.npz")
    
    x = data['x']
    x_data = x[:, 0]  # x position
    y_data = x[:, 1]  # y position
    
    # Set the car's position
    line.set_data(x_data, y_data)
      
    return line

# Create the animation
ani = FuncAnimation(fig, update, frames=num_files, init_func=init, blit=False)

# Save the animation as a GIF
path = Path(__file__).parent / "mpcc_results" / run_name /"mpcc_animation.gif"
ani.save(path, writer='pillow', fps=10, dpi=300)

# To save or display the animation
plt.show()
