import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares
import csv

# Offsets are assumed radians
offsets = [0, 0, 0, 0, 0, 0]

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

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

def position_error(q_position, x_target, y_target, z_target):
    # Calculate forward kinematics specifying angles for just the first three joints
    q_full = np.zeros(6)
    q_full[:3] = q_position
    q_full = q_full + np.radians(offsets)
    T_values = forward_kinematics_func(*q_full)
    # Extract the end-effector position from the transformation matrix
    X, Y, Z = T_values[0,3], T_values[1,3], T_values[2,3]
    #return an array of differences between calculated and target positions
    return np.array([x_target - X , y_target - Y, z_target - Z], dtype=float)

def orientation_error(q_orientation, rx_d, ry_d, rz_d):
    q_full = np.zeros(6)
    q_full[3:] = q_orientation
    q_full = q_full + np.radians(offsets)
    # Calculate forward kinematics specifying angles for just the last three joints
    T_values = forward_kinematics_func(*q_full)
    # Complete the trig expressions
    #  Extract the end-effector orientation from the transformation matrix
    R = T_values[0:3, 0:3]
    roll  = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    yaw   = np.arctan2(R[1,0], R[0,0])
    #return an array of differences between calculated and target orientations
    return np.array([rx_d - roll, ry_d - pitch, rz_d - yaw], dtype=float)

def inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, q_init, max_iterations=100, tolerance=1e-6):
    # Perform numerical inverse kinematics for position
    position_args = (x_target, y_target, z_target)
    q_position_solution = least_squares(position_error, 
                                       np.array(q_init[:3]), 
                                        args=position_args, 
                                        method='lm', 
                                        max_nfev=max_iterations, 
                                        ftol=tolerance).x
    
    # Perform numerical inverse kinematics for orientation
    orientation_args = (rx_d, ry_d, rz_d)
    q_orientation_solution = least_squares(orientation_error, 
                                           np.array(q_init[3:]), 
                                           args=orientation_args, 
                                           method='lm', 
                                           max_nfev=max_iterations, 
                                           ftol=tolerance).x

    # Combine the position and orientation components to get the final joint angles
    joint_angles = np.concatenate((q_position_solution, q_orientation_solution))
    return joint_angles


if __name__ == "__main__":
    # init pose in radians
    init_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

    with open("val_ik_basic.csv", "w", newline="") as validation_file:
        writer = csv.writer(validation_file)
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
                joint_angles = inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, init_pose)

                
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
