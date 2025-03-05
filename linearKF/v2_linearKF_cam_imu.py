import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import yaml 

# Load parameters from config.yml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config:

# Simulation settings
model_path = config["simulation"]["model_path"]
sim_duration = config["simulation"]["duration"]
save_data = config["simulation"]["save_data"]
enable_random_control = config["simulation"]["enable_random_control"]

# Sensor indices
accelerometer_indices = config["sensors"]["accelerometer_indices"]
camera_pos_indices = config["sensors"]["camera_position_indices"]
camera_vel_indices = config["sensors"]["camera_velocity_indices"]

# Viewer settings
enable_contact_points = config["viewer"]["enable_contact_points"]
contact_toggle_interval = config["viewer"]["contact_toggle_interval"]

# Load the model and create data instance
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Calculate approximately how many needed steps based on the simulation timestep
estimated_steps = int(sim_duration / model.opt.timestep)

timestamps = np.zeros(estimated_steps)

# Store the ground truth state [xpos, ypos, zpos, xvel, yvel, zvel]
ground_truth_data = np.zeros((estimated_steps, 6))

# Store the camera data
camera_data = np.zeros((estimated_steps, 6))

# Store the integrated position and velocity
accelerometer_data = np.zeros((estimated_steps, 6))

# Set up the initial conditions for the accelerometer integration
initial_conditions = np.array(config["initial_conditions"]["position"] + config["initial_conditions"]["velocity"])  # [x0, y0, z0, vx0, vy0, vz0]
prev_accelerometer_state = None

# Initialize the data step counter
data_step = 0

def collect_ground_truth(data):
    """
    Collects the ground truth position and velocity.
    Returns: np.array [xpos, ypos, zpos, xvel, yvel, zvel]
    """
    return np.concatenate((data.qpos[0:3], data.qvel[0:3]))

def collect_camera_data(data):
    """
    Collects the simulated camera position and velocity with added Gaussian noise.

    Returns:
    np.array: Noisy [xpos, ypos, zpos, xvel, yvel, zvel]
    """
    camera_data = data.sensordata[camera_pos_indices[0]: camera_vel_indices[-1] + 1]
    
    # Add small Gaussian noise
    noise = np.random.normal(loc=0.0, scale=config["sensors"]["camera_noise"], size=camera_data.shape)  # Mean=0, Std=camera_noise
    return camera_data + noise

def collect_accelerometer_data(accel, initial_conditions, prev_integrated_state=None):
    """
    Integrates accelerometer readings to estimate position and velocity using RK4.

    Parameters:
    accel (np.array): The accelerometer readings [ax, ay, az]
    initial_conditions (np.array): Initial state [x0, y0, z0, vx0, vy0, vz0]
    prev_integrated_state (np.array, optional): Previous step's integrated state

    Returns:
    np.array: Estimated [xpos, ypos, zpos, xvel, yvel, zvel]
    """
    def dynamics(t, state):
        """State dynamics for integration: position and velocity updates."""
        x, y, z, vx, vy, vz = state
        ax, ay, az = accel  # Acceleration components
        return [vx, vy, vz, ax, ay, az]

    # Solve using RK4 (RK45) from t=0 to t=dt
    dt = model.opt.timestep
    
    # Decide on initial state
    if prev_integrated_state is not None:
        # Use previous step's result if available
        initial_state = prev_integrated_state
    else:
        # Use provided initial conditions for the first step
        initial_state = initial_conditions
    
    # Solve the integration
    sol = solve_ivp(dynamics, [0, dt], initial_state, method='RK45')

    # Extract final values
    return sol.y[:, -1]  # Returns updated [x, y, z, vx, vy, vz]

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after sim_duration wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < sim_duration and data_step < estimated_steps:
        step_start = time.time()

        # Apply random control signals if enabled in config
        if enable_random_control:
            data.ctrl[:] = np.random.uniform(-30, 30, data.ctrl.shape)

        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Record data at every simulation step
        if data_step < estimated_steps:
            # Get the ground truth state
            ground_truth_data[data_step] = collect_ground_truth(data)
            
            # Get the T265 camera data
            camera_data[data_step] = collect_camera_data(data)

            # Integrate accelerometer data to get estimated velocity and position
            accelerometer_data[data_step] = collect_accelerometer_data(data.sensordata[accelerometer_indices[0]:accelerometer_indices[-1] + 1], initial_conditions, prev_accelerometer_state)
        
            # Update the previous state for next iteration
            prev_accelerometer_state = accelerometer_data[data_step]

            # Store timestamp
            timestamps[data_step] = data.time
            
            # Print the data at this step for debugging
            '''print(f"Step #{data_step}")
            
            print(f"Ground Truth State:{ground_truth_data[data_step]}")
            print(f"Camera State: {camera_data[data_step]}")
            print(f"Accelerometer State: {accelerometer_data[data_step]}")

            print(f"Time: {timestamps[data_step]} s")
            print("-----------------------------\n")'''

            data_step += 1

        # Toggle contact points visualization based on config
        if enable_contact_points:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % contact_toggle_interval)

        viewer.sync()

        # Time keeping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Truncate arrays to the actual number of recorded steps
ground_truth_data = ground_truth_data[:data_step]
camera_data = camera_data[:data_step]
accelerometer_data = accelerometer_data[:data_step]
timestamps = timestamps[:data_step]

print(f"Data collection complete. Recorded {data_step} timesteps.")

# Plot function with subplots
def plot_subplots(fig, axes, row_idx, title, y_label, data_gt, data_cam, data_acc):
    """Plots data in a given subplot."""
    ax = axes[row_idx]
    ax.plot(timestamps, data_gt, label="Ground Truth")
    ax.plot(timestamps, data_cam, label="Camera", linestyle="--", alpha=0.5)
    ax.plot(timestamps, data_acc, label="Accelerometer", linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Create figure for position plots
fig_pos, axes_pos = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plot_subplots(fig_pos, axes_pos, 0, "Position X", "X Position (m)", ground_truth_data[:, 0], camera_data[:, 0], accelerometer_data[:, 0])
plot_subplots(fig_pos, axes_pos, 1, "Position Y", "Y Position (m)", ground_truth_data[:, 1], camera_data[:, 1], accelerometer_data[:, 1])
plot_subplots(fig_pos, axes_pos, 2, "Position Z", "Z Position (m)", ground_truth_data[:, 2], camera_data[:, 2], accelerometer_data[:, 2])
fig_pos.tight_layout()

# Create figure for velocity plots
fig_vel, axes_vel = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plot_subplots(fig_vel, axes_vel, 0, "Velocity X", "X Velocity (m/s)", ground_truth_data[:, 3], camera_data[:, 3], accelerometer_data[:, 3])
plot_subplots(fig_vel, axes_vel, 1, "Velocity Y", "Y Velocity (m/s)", ground_truth_data[:, 4], camera_data[:, 4], accelerometer_data[:, 4])
plot_subplots(fig_vel, axes_vel, 2, "Velocity Z", "Z Velocity (m/s)", ground_truth_data[:, 5], camera_data[:, 5], accelerometer_data[:, 5])
fig_vel.tight_layout()

plt.show()

# Save data if enabled in config
if save_data:
    np.savez("data.npz",
             timestamps=timestamps,
             ground_truth=ground_truth_data,
             camera_data=camera_data,
             accelerometer_data=accelerometer_data)
    fig_pos.savefig("position_plots.png")
    fig_vel.savefig("velocity_plots.png")

    print("Data and plots saved to 'data.npz'")
