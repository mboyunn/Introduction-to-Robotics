import numpy as np
from sympy  import symbols, cos, sin, atan2, pi, Matrix, lambdify 
from scipy.optimize import least_squares

offsets = np.array([0, -90, 0, 0, 0, 0])
offsets_rad = np.deg2rad(offsets)

joint_limits = np.array([[-165, 165], [-90, 90], [-180, 65], [-160, 160], [-115, 115], [-175, 175]])
joint_limits = np.deg2rad(joint_limits)

lb_dh = joint_limits[:, 0]
ub_dh = joint_limits[:, 1]

# If FK uses (q + offsets), then bounds must be applied to q so that (q + offsets) stays within DH limits
lb = lb_dh - offsets_rad
ub = ub_dh - offsets_rad

bounds = (lb, ub)


def get_transformation_matrix(a, alpha, d, theta):
    M = Matrix([[cos(theta), -sin(theta), 0, a],
                [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
                [0, 0, 0, 1]])
    return M


# Converts the transformation matrix into a pose vector: [X, Y, Z, roll, pitch, yaw]
def transf_to_pose(t_matrix):
    # Position (fill in indices based on your matrix structure)
    X, Y, Z = t_matrix[0,3], t_matrix[1,3], t_matrix[2,3]

    # Breakdown the rotation matrix into axes
    R = t_matrix[0:3, 0:3]
    roll  = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    yaw   = np.arctan2(R[1,0], R[0,0])

    return X, Y, Z, roll, pitch, yaw

def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx  # matches your yaw-pitch-roll extraction convention



# Checks the current joint angles against the target pose and returns their delta
def target_pose_error(q, x_target, y_target, z_target, rx_d, ry_d, rz_d):
   q_full = np.zeros(6)
   q_full[:] = q + np.radians(offsets)

   current_fk = forward_kinematics_func(*q_full)

   # Convert translation matrix to pose vector
   current_pose = np.array(transf_to_pose(current_fk), dtype=float)

   error = np.array([x_target - current_pose[0], 
             y_target - current_pose[1], 
             z_target - current_pose[2], 
             rx_d - current_pose[3], 
             ry_d - current_pose[4],
             rz_d - current_pose[5]], dtype=float)
   return error


# Looks for a solution to inverse kinematics for a target pose and returns a transformation matrix
def inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, q_init, max_iter=1000, tolerance=1e-5):
    target_pose = (x_target, y_target, z_target, rx_d, ry_d, rz_d)

    result = least_squares(target_pose_error, 
                            q_init, 
                            args=target_pose, 
                            method='trf', 
                            max_nfev=max_iter, 
                            ftol=tolerance, 
                            bounds=bounds)

    if result.success:
        print(f"Inverse kinematics converged after {result.nfev} function evaluations.")
        return result.x
    else:
        print("Inverse kinematics did not converge.")
        return None


if __name__ == "__main__":
    q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
    DH = [
        # [a, alpha, d, theta] for joint 1, then do for joint 2, 3, 4, 5, 6
        [0, np.radians(0), 114, q1],
        [0, np.radians(-90), 0, q2],
        [95, np.radians(0), 0, q3],
        [10, np.radians(-90), 95, q4],
        [0, np.radians(90), 0, q5],
        [0, np.radians(-90), 60.55, q6]
    ]

    T_01 = get_transformation_matrix(DH[0][0], DH[0][1], DH[0][2], DH[0][3])
    T_12 = get_transformation_matrix(DH[1][0], DH[1][1], DH[1][2], DH[1][3])
    T_23 = get_transformation_matrix(DH[2][0], DH[2][1], DH[2][2], DH[2][3])
    T_34 = get_transformation_matrix(DH[3][0], DH[3][1], DH[3][2], DH[3][3])
    T_45 = get_transformation_matrix(DH[4][0], DH[4][1], DH[4][2], DH[4][3])
    T_56 = get_transformation_matrix(DH[5][0], DH[5][1], DH[5][2], DH[5][3])
    T_sym = T_01 * T_12 * T_23 * T_34 * T_45 * T_56

    forward_kinematics_func = lambdify((q1, q2, q3, q4, q5, q6), T_sym, "numpy")

    # Example (Replace these values with collected coords from Lab 2)

    x_target = 207.8
    y_target = 33.3
    z_target = 114.7
    rx_d = np.radians(88.2)  # Roll angle (in radians)
    ry_d = np.radians(5.16)  # Pitch angle (in radians)
    rz_d = np.radians(-28.38)  # Yaw angle (in radians)
    init_pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    joint_angles = inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, init_pose)
            
    # output the joint angles. You may save the output to a csv file
    if joint_angles is not None:
        print("Joint Angles (Degrees):", np.degrees(joint_angles))
    else:
        print("Joint Angles (Degrees): None")
