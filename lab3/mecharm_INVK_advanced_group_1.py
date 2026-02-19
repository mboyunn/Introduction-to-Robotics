import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares
import csv

# Offsets are assumed radians
offsets = [0, 0, 0, 0, 0, 0]

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

# Bounds in degrees -> convert to radians for optimization variables
lower_bounds_deg = np.array([-165, -90, -180, -160, -115, -175], dtype=float)
upper_bounds_deg = np.array([165,  90,   65,  160,  115,  175], dtype=float)
lower_bounds = np.deg2rad(lower_bounds_deg)
upper_bounds = np.deg2rad(upper_bounds_deg)

# Use SymPy values for alpha (no np.radians)
DH = [
    [0,  0,      114, q1],
    [0, -pi/2,     0, q2 - pi/2],
    [95, 0,        0, q3],
    [10, -pi/2,   95, q4],
    [0,  pi/2,     0, q5],
    [0, -pi/2,    63.55, q6],
]

def get_transformation_matrix(a, alpha, d, theta):
    # Standard DH (a, alpha, d, theta)
    return Matrix([
        [cos(theta),            -sin(theta),           0,            a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
        [0,                     0,                    0,            1]
    ])

T = get_transformation_matrix(*DH[0])
for i in range(1, len(DH)):
    T = T @ get_transformation_matrix(*DH[i])

# Lambdify a purely symbolic T(q)
q_sym = symbols('q1:7')  # (q1..q6)

def symbolic_forward_kinematics(q_values):
    subs_dict = {q: off + ang for q, off, ang in zip([q1, q2, q3, q4, q5, q6], offsets, q_values)}
    return T.subs(subs_dict)

forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), "numpy")

def transf_to_pose(t_matrix):
    # Ensure numeric ndarray
    t = np.array(t_matrix, dtype=float)

    X, Y, Z = t[0, 3], t[1, 3], t[2, 3]
    R = t[0:3, 0:3]

    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw   = np.arctan2(R[1, 0], R[0, 0])

    return np.array([X, Y, Z, roll, pitch, yaw], dtype=float)

def target_pose_error(q, x_target, y_target, z_target, rx_d, ry_d, rz_d):
    current_fk = forward_kinematics_func(*q)
    current_pose = transf_to_pose(current_fk)
    target_pose = np.array([x_target, y_target, z_target, rx_d, ry_d, rz_d], dtype=float)
    return current_pose - target_pose

def inverse_kinematics(target_pose, init_pose, max_iter=1000, tolerance=1e-5, bounds=(lower_bounds, upper_bounds)):
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

# init pose in radians
init_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

with open("val_ik.csv", "w", newline="") as file2:
    writer = csv.writer(file2)
    writer.writerow([
        "J1_T", "J2_T", "J3_T", "J4_T", "J5_T", "J6_T",
        "J1_IK", "J2_IK", "J3_IK", "J4_IK", "J5_IK", "J6_IK"
    ])

    with open("data.csv", "r", newline="") as file:
        csv_file = csv.reader(file)
        next(csv_file)

        for line in csv_file:
            # Targets from file: decide whether joints are degrees or radians in your CSV.
            # Here assuming CSV joints are degrees:
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

            
            if joint_angles is None:
                ik_deg = [np.nan]*6
            else:
                ik_deg = np.rad2deg(joint_angles).tolist()

            # target joints (your CSV targets are already degrees)
            tgt_deg = j_target_deg.tolist()

            # round both to 2 decimals
            tgt_deg = [float(f"{v:.2f}") for v in tgt_deg]
            ik_deg  = [float(f"{v:.2f}") if np.isfinite(v) else v for v in ik_deg]

            # write: target first, then IK (both in degrees)
            row = tgt_deg + ik_deg
            writer.writerow(row)

            # keep init_pose in radians for the solver
            init_pose = joint_angles if joint_angles is not None else j_target