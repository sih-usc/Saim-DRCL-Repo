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

# Initial conditions
initial_conditions = np.array(config["initial_conditions"]["position"] + config["initial_conditions"]["velocity"])  # [x0, y0, z0, vx0, vy0, vz0]

# Kalman filter settings
Q_pos_var = config["kalman"]["Q_position_variance"]
Q_vel_var = config["kalman"]["Q_velocity_variance"]
R_pos_var = config["kalman"]["R_position_variance"]
R_vel_var = config["kalman"]["R_velocity_variance"]

Q = np.diag([Q_pos_var]*3 + [Q_vel_var]*3)
R = np.diag([R_pos_var]*3 + [R_vel_var]*3)

P_0 = float(config["kalman"]["initial_state_covariance"])
kf_state = initial_conditions.copy()
kf_cov = np.eye(6) * P_0

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

# Store the Kalman Filter estimate
kf_data = np.zeros((estimated_steps, 6))

# Store RMSE values
kf_rmse = np.zeros(estimated_steps)

# Store the accelerometer raw data (for debugging)
# accel_raw_data = np.zeros((estimated_steps, 3))

# Initialize the previous state for accelerometer integration
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
    # noise = 0
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
    
    # Add Gaussian noise to the IMU readings
    accel_noise = np.random.normal(loc=0.0, scale=config["sensors"]["accelerometer_noise"], size=accel.shape)
    # accel_noise = 0
    noisy_accel = accel + accel_noise

    def dynamics(t, state):
        """State dynamics for integration: position and velocity updates."""
        x, y, z, vx, vy, vz = state
        ax, ay, az = noisy_accel  # Noisy acceleration components
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

def kalman_filter(x_prev, P_prev, a_input, z_measured, dt, Q, R):
    """
    Linear Kalman Filter for 6D state [pos, vel] using IMU acceleration and camera measurement.

    Parameters:
    - x_prev: Previous state estimate (6x1)
    - P_prev: Previous covariance estimate (6x6)
    - a_input: Acceleration input from IMU (3x1)
    - z_measured: Measurement from camera [pos, vel] (6x1)
    - dt: timestep
    - Q: Process noise covariance (6x6)
    - R: Measurement noise covariance (6x6)

    Returns:
    - x_new: Updated state estimate
    - P_new: Updated error covariance
    """
    # System matrices
    A = np.block([
        [np.eye(3), dt * np.eye(3)],
        [np.zeros((3, 3)), np.eye(3)]
    ])
    B = np.block([
        [0.5 * dt ** 2 * np.eye(3)],
        [dt * np.eye(3)]
    ])
    C = np.eye(6)  # Full state measurement

    # ---- I. A Priori State Estimate
    x_pred = A @ x_prev + B @ a_input

    # ---- II. A Priori Error Covariance
    P_pred = A @ P_prev @ A.T + Q

    # Kalman Gain
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.inv(S)

    # ---- III. A Posteriori State Estimate
    y = z_measured - C @ x_pred  # Innovation
    x_new = x_pred + K @ y

    # ---- IV. A Posteriori Error Covariance
    P_new = (np.eye(6) - K @ C) @ P_pred

    return x_new, P_new

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

            # Get the raw accelerometer data (for debugging)
            # accel_raw_data[data_step] = data.sensordata[accelerometer_indices[0]:accelerometer_indices[-1] + 1]
            
            # Integrate accelerometer data to get estimated velocity and position
            accelerometer_data[data_step] = collect_accelerometer_data(data.sensordata[accelerometer_indices[0]:accelerometer_indices[-1] + 1] + model.opt.gravity, initial_conditions, prev_accelerometer_state)
        
            # Apply Kalman Filter
            kf_state, kf_cov = kalman_filter(
                kf_state,
                kf_cov,
                data.sensordata[accelerometer_indices[0]:accelerometer_indices[-1] + 1] + model.opt.gravity,
                camera_data[data_step],
                model.opt.timestep,
                Q,
                R
            )
            kf_data[data_step] = kf_state
            
            # Calculate RMSE for the Kalman Filter
            # RMSE = sqrt(mean((estimated - ground_truth)^2))
            kf_error = kf_state - ground_truth_data[data_step]
            kf_rmse[data_step] = np.sqrt(np.mean(kf_error ** 2))

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
kf_data = kf_data[:data_step]
kf_rmse = kf_rmse[:data_step]
timestamps = timestamps[:data_step]

# accel_raw_data = accel_raw_data[:data_step]

print(f"Data collection complete. Recorded {data_step} timesteps.")

# Plot function with subplots
def plot_subplots(fig, axes, row_idx, title, y_label, data_gt, data_cam, data_acc, data_kf):
    """Plots data in a given subplot."""
    ax = axes[row_idx]
    ax.plot(timestamps, data_gt, label="Ground Truth", alpha=1.0)
    ax.plot(timestamps, data_cam, label="Camera", linestyle="--", alpha=0.5)
    ax.plot(timestamps, data_acc, label="Accelerometer", linestyle=":", alpha=0.5)
    ax.plot(timestamps, data_kf, label="Kalman Filter", linewidth=2, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Create figure for position plots
fig_pos, axes_pos = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plot_subplots(fig_pos, axes_pos, 0, "Position X", "X Position (m)", ground_truth_data[:, 0], camera_data[:, 0], accelerometer_data[:, 0], kf_data[:, 0])
plot_subplots(fig_pos, axes_pos, 1, "Position Y", "Y Position (m)", ground_truth_data[:, 1], camera_data[:, 1], accelerometer_data[:, 1], kf_data[:, 1])
plot_subplots(fig_pos, axes_pos, 2, "Position Z", "Z Position (m)", ground_truth_data[:, 2], camera_data[:, 2], accelerometer_data[:, 2], kf_data[:, 2])
fig_pos.tight_layout(pad=2.5)

# Create figure for velocity plots
fig_vel, axes_vel = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plot_subplots(fig_vel, axes_vel, 0, "Velocity X", "X Velocity (m/s)", ground_truth_data[:, 3], camera_data[:, 3], accelerometer_data[:, 3], kf_data[:, 3])
plot_subplots(fig_vel, axes_vel, 1, "Velocity Y", "Y Velocity (m/s)", ground_truth_data[:, 4], camera_data[:, 4], accelerometer_data[:, 4], kf_data[:, 4])
plot_subplots(fig_vel, axes_vel, 2, "Velocity Z", "Z Velocity (m/s)", ground_truth_data[:, 5], camera_data[:, 5], accelerometer_data[:, 5], kf_data[:, 5])
fig_vel.tight_layout(pad=2.5)

'''
# Plot raw accelerometer data (for debugging)
fig_accel, axes_accel = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
axes_accel[0].plot(timestamps, accel_raw_data[:, 0], label="Accel X", color="purple")
axes_accel[1].plot(timestamps, accel_raw_data[:, 1], label="Accel Y", color="purple")
axes_accel[2].plot(timestamps, accel_raw_data[:, 2], label="Accel Z", color="purple")

axes_accel[0].set_title("Accelerometer X")
axes_accel[1].set_title("Accelerometer Y")
axes_accel[2].set_title("Accelerometer Z")

for ax in axes_accel:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend()
    ax.grid(True)

fig_accel.tight_layout()
'''

# Plot RMSE for Kalman Filter
plt.figure()
plt.plot(timestamps, kf_rmse, color="black", label="KF RMSE")
plt.xlabel("Time (s)")
plt.ylabel("RMSE (m / m/s)")
plt.title("Kalman Filter RMSE Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# Save data if enabled in config
if save_data:
    np.savez("data.npz",
             timestamps=timestamps,
             ground_truth=ground_truth_data,
             camera_data=camera_data,
             accelerometer_data=accelerometer_data,
             kalman_data=kf_data,
             kf_rmse=kf_rmse)
    # Save the figures
    fig_pos.savefig("position_plots.png")
    fig_vel.savefig("velocity_plots.png")
    # fig_accel.savefig("accelerometer_plots.png")
    plt.savefig("kf_rmse_plot.png")
    
    print("Data and plots saved to 'data.npz'")

'''
# Print sensor information (for debugging)
for i in range(model.nsensor):
    name = model.sensor(i).name
    addr = model.sensor_adr[i]
    dim = model.sensor_dim[i]
    print(f"{i}: {name} -> indices {addr} to {addr + dim - 1}")'
'''