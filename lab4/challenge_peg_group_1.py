from sympy.matrices import Matrix
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT
import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares
import time

# Joint offsets (assumed to be in radians)
offsets = [0, 0, 0, 0, 0, 0]

# Define symbolic joint variables
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

# Joint angle limits in degrees
lower_bounds_deg = np.array([-165, -90, -180, -160, -115, -175], dtype=float)
upper_bounds_deg = np.array([165,  90,   65,  160,  115,  175], dtype=float)

# Convert joint limits to radians for optimization
lower_bounds = np.deg2rad(lower_bounds_deg)
upper_bounds = np.deg2rad(upper_bounds_deg)

# Denavit-Hartenberg parameters (a, alpha, d, theta)
DH = [
    [0,  0,      114, q1],
    [0, -pi/2,     0, q2 - pi/2],
    [95, 0,        0, q3],
    [10, -pi/2,   95, q4],
    [0,  pi/2,     0, q5],
    [0, -pi/2,    63.55, q6],
]

# Compute a standard DH transformation matrix
def get_transformation_matrix(a, alpha, d, theta):
    return Matrix([
        [cos(theta),            -sin(theta),           0,            a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
        [0,                     0,                    0,            1]
    ])

# Build full forward kinematics transformation (base → end effector)
T = get_transformation_matrix(*DH[0])
for i in range(1, len(DH)):
    T = T @ get_transformation_matrix(*DH[i])

# Create symbolic joint vector (q1..q6)
q_sym = symbols('q1:7')

# Substitute joint values into symbolic FK matrix
def symbolic_forward_kinematics(q_values):
    subs_dict = {q: off + ang for q, off, ang in zip([q1, q2, q3, q4, q5, q6], offsets, q_values)}
    return T.subs(subs_dict)

# Convert symbolic FK into fast numerical function
forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), "numpy")

# Convert transformation matrix → pose [x,y,z,roll,pitch,yaw]
def transf_to_pose(t_matrix):
    t = np.array(t_matrix, dtype=float)

    X, Y, Z = t[0, 3], t[1, 3], t[2, 3]
    R = t[0:3, 0:3]

    # Extract Euler angles from rotation matrix
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw   = np.arctan2(R[1, 0], R[0, 0])

    return np.array([X, Y, Z, roll, pitch, yaw], dtype=float)

# Error function used by least squares IK solver
def target_pose_error(q, x_target, y_target, z_target, rx_d, ry_d, rz_d):
    current_fk = forward_kinematics_func(*q)
    current_pose = transf_to_pose(current_fk)
    target_pose = np.array([x_target, y_target, z_target, rx_d, ry_d, rz_d], dtype=float)
    return current_pose - target_pose

# Numerical inverse kinematics using least squares optimization
def inverse_kinematics(target_pose, init_pose, max_iter=1000, tolerance=1e-5, bounds=(lower_bounds, upper_bounds)):
    result = least_squares(
        target_pose_error,
        x0=np.array(init_pose, dtype=float),   # initial guess for joints
        args=tuple(target_pose),               # desired pose
        method='trf',
        max_nfev=max_iter,
        ftol=tolerance,
        bounds=bounds                          # joint limits
    )
    return result.x if result.success else None


if __name__ == "__main__":

    # Initialize robot connection
    mycobot = MechArm270(PI_PORT, PI_BAUD)
    mycobot.power_on()

    # Initial joint guess for IK solver
    init_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

    # Open gripper at start
    mycobot.set_gripper_value(100, 50, 1)
    time.sleep(2)

    # Sequence of target poses and gripper actions
    poses = [
        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
        ([190.6,85.6,122.5,np.deg2rad(-95.87),np.deg2rad(71.51),np.deg2rad(-60.74)], 50, 2), # pickup peg
        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 3), # origin
        ([237.1,41.3,150.5,np.deg2rad(-73.5),np.deg2rad(63.92),np.deg2rad(-58.04)], 50, 4), # peg1

        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
        ([190.6,85.6,122.5,np.deg2rad(-95.87),np.deg2rad(71.51),np.deg2rad(-60.74)], 50, 2), # pickup peg
        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 3), # origin
        ([214.3,9.8,154.0,np.deg2rad(-73.11),np.deg2rad(66.23),np.deg2rad(-61.79)], 50, 4), # peg 2

        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
        ([190.6,85.6,122.5,np.deg2rad(-95.87),np.deg2rad(71.51),np.deg2rad(-60.74)], 50, 2), # pickup peg
        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 3), # origin
        ([190.5,-27.7,110.5,np.deg2rad(-80.39),np.deg2rad(68.41),np.deg2rad(-69.79)], 50, 4), # peg3
    ]

    # Execute pose sequence
    for pose, speed, encoding in poses:

        # Pre-move gripper state
        if encoding == 1 or encoding == 2:
            print("Open gripper")
            mycobot.set_gripper_value(100, 50, 1)
            time.sleep(2)
        else:
            print("close gripper")
            mycobot.set_gripper_value(45, 50, 1)
            time.sleep(2)
        
        target_pose = np.array(pose)

        # Solve IK to obtain joint angles
        joint_angles = inverse_kinematics(target_pose, init_pose)

        # Convert radians → degrees for robot API
        joint_angles_deg = np.rad2deg(joint_angles)
        modified = joint_angles_deg.tolist()

        # Send joint commands to robot
        mycobot.send_angles(modified, speed)
            
        time.sleep(2)
        print(pose)

        # Post-move gripper state
        if encoding == 1 or encoding == 4:
            print("Open gripper")
            mycobot.set_gripper_value(100, 50, 1)
            time.sleep(2)
        else:
            print("close gripper")
            mycobot.set_gripper_value(45, 50, 1)
            time.sleep(2)


    # Return robot to origin pose
    target_pose = np.array([155.2, 0.9, 240.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], dtype=float)

    joint_angles = inverse_kinematics(target_pose, init_pose)
    joint_angles_deg = np.rad2deg(joint_angles)

    mycobot.send_angles(joint_angles_deg.tolist(), 30)

    time.sleep(5)