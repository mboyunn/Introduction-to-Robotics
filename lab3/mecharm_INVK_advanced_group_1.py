import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares
import csv

# Joint angle offsets applied before FK evaluation (radians)
offsets = [0, 0, 0, 0, 0, 0]

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

# Joint limits in degrees, converted to radians for the optimizer
lower_bounds_deg = np.array([-165, -90, -180, -160, -115, -175], dtype=float)
upper_bounds_deg = np.array([165,  90,   65,  160,  115,  175], dtype=float)
lower_bounds = np.deg2rad(lower_bounds_deg)
upper_bounds = np.deg2rad(upper_bounds_deg)

# DH parameters: [a, alpha, d, theta] for each joint (lengths in mm)
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
    """Residual between current FK pose and target — minimized by IK solver."""
    current_fk = forward_kinematics_func(*q)
    current_pose = transf_to_pose(current_fk)
    target_pose = np.array([x_target, y_target, z_target, rx_d, ry_d, rz_d], dtype=float)
    return current_pose - target_pose

def inverse_kinematics(target_pose, init_pose, max_iter=1000, tolerance=1e-5, bounds=(lower_bounds, upper_bounds)):
    """
    Solve IK numerically using trust-region least squares (TRF).
    Unlike the basic solver, this enforces joint limits via bounds.
    """
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


# Initial joint angle guess for the solver (radians)
init_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

with open("val_ik_adv.csv", "w", newline="") as validation_file:
    writer = csv.writer(validation_file)
    writer.writerow([
        "J1_T", "J2_T", "J3_T", "J4_T", "J5_T", "J6_T",   # target joints (deg)
        "J1_IK", "J2_IK", "J3_IK", "J4_IK", "J5_IK", "J6_IK"  # IK solution (deg)
    ])

    with open("data.csv", "r", newline="") as file:
        csv_file = csv.reader(file)
        next(csv_file)  # skip header row

        for line in csv_file:
            # CSV columns 0-5: joint angles (degrees), 6-8: XYZ (mm), 9-11: RPY (degrees)
            j_target_deg = np.array([float(line[i]) for i in range(6)], dtype=float)
            j_target = np.deg2rad(j_target_deg)

            x_target = float(line[6])
            y_target = float(line[7])
            z_target = float(line[8])
            rx_d = np.deg2rad(float(line[9]))
            ry_d = np.deg2rad(float(line[10]))
            rz_d = np.deg2rad(float(line[11]))

            target_pose = [x_target, y_target, z_target, rx_d, ry_d, rz_d]
            joint_angles = inverse_kinematics(target_pose, init_pose)

            # Convert results to degrees, falling back to NaN if solver failed
            if joint_angles is None:
                ik_deg = [np.nan] * 6
            else:
                ik_deg = np.rad2deg(joint_angles).tolist()

            # Round both target and IK angles to 2 decimal places
            tgt_deg = [float(f"{v:.2f}") for v in j_target_deg.tolist()]
            ik_deg  = [float(f"{v:.2f}") if np.isfinite(v) else v for v in ik_deg]

            writer.writerow(tgt_deg + ik_deg)

            # Warm-start next iteration from current solution (or fall back to target)
            init_pose = joint_angles if joint_angles is not None else j_target