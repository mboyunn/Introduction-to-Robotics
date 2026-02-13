import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares


offsets = np.array([0, -90, 0, 0, 0, 0])
def get_transformation_matrix(a, alpha, d, theta):
    M = Matrix([[cos(theta), -sin(theta), 0, a],
                [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
                [0, 0, 0, 1]])
    return M

def position_error(q_position, x_target, y_target, z_target, link_lengths):
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

def inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, q_init, link_lengths, max_iterations=100, tolerance=1e-6):
    # Perform numerical inverse kinematics for position
    position_args = (x_target, y_target, z_target, link_lengths)
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
    x_target = 218.6
    y_target = 91.3
    z_target = 144.3
    link_lengths = [DH[0][0], DH[1][0], DH[2][0], DH[3][0], DH[4][0], DH[5][0]]
    rx_d = np.radians(-137.69)  # Roll angle (in radians)
    ry_d = np.radians(-68.97)  # Pitch angle (in radians)
    rz_d = np.radians(-28.38)  # Yaw angle (in radians)
    q_init = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    joint_angles = inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, q_init, link_lengths)
            
    #output the joint angles. You may save the output to a csv file
    print("Joint Angles (Degrees):", np.degrees(joint_angles))
