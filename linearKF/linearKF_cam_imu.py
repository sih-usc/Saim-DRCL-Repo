import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.integrate import solve_ivp

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("./models/biped_simple.xml")
data = mujoco.MjData(model)

# Simulation time settings
SIM_TIME = 30  # Run for 30 seconds
DT = model.opt.timestep  # Simulation timestep

# Time Data
time_series = []

# Ground Truth Data
p_true_series = []  # Ground truth position
v_true_series = []  # Ground truth velocity

# IMU Data
v_imu_series = []  # Velocity from single integration of accelerometer
p_imu_series = []  # Position from double integration of accelerometer

# Camera Data (T265 proxy)
p_cam_series = []  # Noisy camera position
v_cam_series = []  # Noisy camera velocity

# Initialize initial state for IMU integrator
v_imu = np.zeros(3)  # Assume zero initial velocity
p_imu = np.zeros(3)  # Assume zero initial position

def imu_rk4_integration(accel, v_init, p_init, dt):
    # Integrate IMU acceleration using RK4 to compute velocity and position
    
    def dynamics(t, y):
        # Defines the system of ODEs: y = [v_x, v_y, v_z, p_x, p_y, p_z]
        v = y[:3]  # Velocity components
        a = accel  # Acceleration remains constant over dt
        dpdt = v
        dvdt = a 
        return np.concatenate([dvdt, dpdt])  # Return [dv/dt, dp/dt]

    # Initial state [v_x, v_y, v_z, p_x, p_y, p_z]
    y0 = np.concatenate([v_init, p_init])
    
    # Solve the ODE using RK4
    sol = solve_ivp(dynamics, [0, dt], y0, method='RK45', t_eval=[dt])

    # Extract final velocity and position
    v_next = sol.y[:3, -1]
    p_next = sol.y[3:, -1]
    
    return v_next, p_next

# Simulate noisy T265 measurements (model noise as Gaussian distribution)
def get_noisy_camera_measurement():
    # Retrieve camera position & velocity from MuJoCo and apply Gaussian noise

    # Extract true position and velocity from MuJoCo
    p_cam_true = data.sensordata[2:5]  # True position from framepos
    v_cam_true = data.sensordata[5:8]  # True velocity from framevel

    # Define measurement noise covariance matrices
    R_p = np.diag([0.01, 0.01, 0.01])  # Position noise covariance
    R_v = np.diag([0.02, 0.02, 0.02])  # Velocity noise covariance

    # Generate Gaussian noise for position and velocity
    #v_p = np.random.multivariate_normal(mean=np.zeros(3), cov=R_p)  # Position noise
    #v_v = np.random.multivariate_normal(mean=np.zeros(3), cov=R_v)  # Velocity noise
    v_p = 0
    v_v = 0

    # Compute noisy measurements
    p_cam_noisy = p_cam_true + v_p
    v_cam_noisy = v_cam_true + v_v

    return p_cam_noisy, v_cam_noisy

# Define rotation from robot local frame to world frame of reference
def transform_to_world_frame(vec_local, quat_robot):
    
    # Convert a vector from the robot’s local frame to the global frame
    vec_world = np.zeros(3)
    
    # Convert MuJoCo quaternion (w, x, y, z) -> (x, y, z, w)
    quat_reordered = np.roll(quat_robot, -1)  
    
    mujoco.mju_rotVecQuat(vec_world, vec_local, quat_reordered)
    return vec_world

# Launch MuJoCo viewer and start simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    
    while viewer.is_running() and time.time() - start < SIM_TIME:
        step_start = time.time()

        # Get ground truth from MuJoCo -- FIX THIS!
        p_true = data.qpos[:3]  # True position
        v_true = data.qvel[:3]  # True velocity

        # Get IMU data
        imu_accel_local = data.sensordata[:3]  # IMU acceleration in robot frame

        # Get robot orientation quaternion from qpos
        q_robot_to_world = data.qpos[3:7]  # (w, x, y, z)

        #st = time.time()

        # Use RK4 for IMU-based integration in the local frame
        v_imu_local, p_imu_local = imu_rk4_integration(imu_accel_local, v_imu, p_imu, DT)

        #print("RK4 time: ", time.time() - st)

        # Convert IMU-integrated velocity and position to world frame
        v_imu = transform_to_world_frame(v_imu_local, q_robot_to_world)
        p_imu = transform_to_world_frame(p_imu_local, q_robot_to_world)

        # Retrieve noisy camera data (position & velocity in robot frame)
        p_cam_local, v_cam_local = get_noisy_camera_measurement() 

        # Transform camera data to world frame
        p_cam = transform_to_world_frame(p_cam_local, q_robot_to_world)
        v_cam = transform_to_world_frame(v_cam_local, q_robot_to_world)

        # Store data in arrays
        time_series.append(data.time)
        p_true_series.append(p_true)
        v_true_series.append(v_true)
        v_imu_series.append(v_imu)
        p_imu_series.append(p_imu)
        p_cam_series.append(p_cam)
        v_cam_series.append(v_cam)

        # Step the MuJoCo physics simulation
        mujoco.mj_step(model, data)

        # Print collected data for debugging
        print(f"Time: {data.time:.6f}")
        print(f"  p_true:  {np.round(p_true, 3)} | v_true:  {np.round(v_true, 3)}")
        print(f"  p_imu_world: {np.round(p_imu, 3)} | v_imu_world: {np.round(v_imu, 3)}")
        print(f"  p_cam_world: {np.round(p_cam, 3)} | v_cam_world: {np.round(v_cam, 3)}\n")

        # Sync viewer
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
        viewer.sync()

        # Timekeeping
        time_until_next_step = DT - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Convert to NumPy arrays for further analysis
p_true_series = np.array(p_true_series)
v_true_series = np.array(v_true_series)
v_imu_series = np.array(v_imu_series)
p_imu_series = np.array(p_imu_series)
p_cam_series = np.array(p_cam_series)
v_cam_series = np.array(v_cam_series)

# Plot data here
print("Data collection complete!")