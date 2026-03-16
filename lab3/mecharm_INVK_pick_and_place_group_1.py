from sympy.matrices import Matrix
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT
import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares
import time

# Joint angle offsets applied before FK evaluation (radians)
offsets = [0, 0, 0, 0, 0, 0]

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

# Joint limits in degrees, converted to radians for the optimizer
lower_bounds_deg = np.array([-165, -90, -180, -160, -115, -175], dtype=float)
upper_bounds_deg = np.array([165,  90,   65,  160,  115,  175], dtype=float)
lower_bounds = np.deg2rad(lower_bounds_deg)
upper_bounds = np.deg2rad(upper_bounds_deg)

# DH parameters: [a, alpha, d, theta] for each joint
# Lengths in mm; alpha in radians using SymPy pi for symbolic compatibility
DH = [
    [0,  0,      114, q1],
    [0, -pi/2,     0, q2 - pi/2],
    [95, 0,        0, q3],
    [10, -pi/2,   95, q4],
    [0,  pi/2,     0, q5],
    [0, -pi/2,    63.55, q6],
]

def get_transformation_matrix(a, alpha, d, theta):
    """Standard DH homogeneous transformation matrix for a single joint."""
    return Matrix([
        [cos(theta),            -sin(theta),           0,            a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
        [0,                     0,                    0,            1]
    ])

# Build full symbolic transform T_0_6 by chaining each joint's matrix
T = get_transformation_matrix(*DH[0])
for i in range(1, len(DH)):
    T = T @ get_transformation_matrix(*DH[i])

q_sym = symbols('q1:7')  # (q1..q6)

def symbolic_forward_kinematics(q_values):
    """Substitute joint angles (+ offsets) into the symbolic transform."""
    subs_dict = {q: off + ang for q, off, ang in zip([q1, q2, q3, q4, q5, q6], offsets, q_values)}
    return T.subs(subs_dict)

# Compile symbolic FK to a fast numeric function
forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), "numpy")

def transf_to_pose(t_matrix):
    """Extract [X, Y, Z, roll, pitch, yaw] from a 4x4 homogeneous transform."""
    t = np.array(t_matrix, dtype=float)
    X, Y, Z = t[0, 3], t[1, 3], t[2, 3]
    R = t[0:3, 0:3]

    # ZYX Euler angles from rotation matrix
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw   = np.arctan2(R[1, 0], R[0, 0])

    return np.array([X, Y, Z, roll, pitch, yaw], dtype=float)

def target_pose_error(q, x_target, y_target, z_target, rx_d, ry_d, rz_d):
    """Residual between current FK pose and target pose — minimized by IK solver."""
    current_fk = forward_kinematics_func(*q)
    current_pose = transf_to_pose(current_fk)
    target_pose = np.array([x_target, y_target, z_target, rx_d, ry_d, rz_d], dtype=float)
    return current_pose - target_pose

def inverse_kinematics(target_pose, init_pose, max_iter=1000, tolerance=1e-5, bounds=(lower_bounds, upper_bounds)):
    """Solve IK numerically using trust-region least squares (TRF method)."""
    result = least_squares(
        target_pose_error,
        x0=np.array(init_pose, dtype=float),
        args=tuple(target_pose),
        method='trf',
        max_nfev=max_iter,
        ftol=tolerance,
        bounds=bounds
    )
    return result.x if result.success else None


if __name__ == "__main__":
    mycobot = MechArm270(PI_PORT, PI_BAUD)
    mycobot.power_on()

    init_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

    def move_to(target_pose, speed=50, wait=5):
        """Solve IK for target_pose, send joint angles to robot, and wait."""
        global init_pose
        joint_angles = inverse_kinematics(target_pose, init_pose)
        joint_angles_deg = np.rad2deg(joint_angles)
        mycobot.send_angles(joint_angles_deg.tolist(), speed)
        init_pose = joint_angles_deg
        time.sleep(wait)

    # Home/safe position above the workspace
    home = np.array([155.2, 0.9, 219.0,
                     np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], dtype=float)

    # 1. Move to home position
    move_to(home)

    # 2. Open gripper before pick
    mycobot.set_gripper_value(100, 50, 1)
    time.sleep(2)

    # 3. Lower to pick position
    pick = np.array([188.3,59.5,16.9,np.deg2rad(-166.86),np.deg2rad(56.97),np.deg2rad(-140.66)], dtype=float)
    move_to(pick)

    # 4. Close gripper to grasp object
    mycobot.set_gripper_value(69, 50, 1)
    time.sleep(2)

    # 5. Lift back to intermediate point
    intermediate = np.array([215.0,16.9,130.4,np.deg2rad(-121.4),np.deg2rad(64.61),np.deg2rad(-108.86)], dtype=float)
    move_to(intermediate)

    # 6. Lower to place position
    place = np.array([208.2,-32.7,35.3,np.deg2rad(-148.96),np.deg2rad(61.46),np.deg2rad(-136.43)], dtype=float)
    move_to(place)

    # 7. Release object slowly to avoid knocking it over
    mycobot.set_gripper_value(100, 10, 1)
    time.sleep(10)

    # 8. Move to another intermiedate point so that gripper doesn't knock block off.
    raise_z = np.array([208.2,-32.7,105.3,np.deg2rad(-148.96),np.deg2rad(61.46),np.deg2rad(-136.43)], dtype=float)
    move_to(raise_z)

    # 9. Return to home
    move_to(home)