import numpy as np
from sympy  import symbols, cos, sin, atan2, pi, Matrix, lambdify 
from scipy.optimize import least_squares
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT


offsets = np.array([np.radians(0), -np.pi/2, np.radians(0), np.radians(0), np.radians(0), np.radians(0)])
lower_bounds = np.radians(np.array([-165, -90, -180, -160, -115, -175])) - offsets
upper_bounds = np.radians(np.array([165, 90, 65, 160, 115, 175])) - offsets


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

    return np.array([X, Y, Z, roll, pitch, yaw], dtype=float)

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# Checks the current joint angles against the target pose and returns their delta
def target_pose_error(q, x_target, y_target, z_target, rx_d, ry_d, rz_d):
   q_full = q + offsets

   current_fk = forward_kinematics_func(*q_full)

   # Convert translation matrix to pose vector
   current_pose = transf_to_pose(current_fk)
   target_pose = np.array([x_target, y_target, z_target, rx_d, ry_d, rz_d])

   error = target_pose - current_pose
   error[3:] = wrap_to_pi(error[3:])
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
                            bounds=(lower_bounds, upper_bounds))

    if result.success:
        print(f"Inverse kinematics converged after {result.nfev} function evaluations.")
        return result.x
    else:
        print("Inverse kinematics did not converge.")
        return None


def send_to(x_target, y_target, z_target, rx_d, ry_d, rz_d):
    angles = inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, init_pose).tolist()
    mycobot.send_angles(angles, 50)


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

    mycobot = MechArm270(PI_PORT, PI_BAUD)
    mycobot.power_on()
    mycobot.send_angles([0,0,0,0,0,0], 50)
    init_pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # open gripper
    mycobot.set_gripper_value(0, 50, "default")


    # send to origin
    send_to(0,0,0,0,0,0)

    # close gripper
    mycobot.set_gripper_value(100, 50, "default")

    # send to object
    send_to(0,0,0,0,0,0)

    # close gripper
    mycobot.set_gripper_value(0, 50, "default")

    # send to origin
    send_to(0,0,0,0,0,0)

    # place object
    send_to(0,0,0,0,0,0)

    # send to origin
    send_to(0,0,0,0,0,0)

    mycobot.power_off()
