import numpy as np
from sympy  import symbols, cos, sin, pi, Matrix, lambdify 
from scipy.optimize import least_squares
import csv

offsets = [0, 0, 0, 0, 0, 0]
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
lower_bounds = np.array([-165, -90, -180, -160, -115, -175])
upper_bounds = np.array([165, 90, 65, 160, 115, 175])
DH = [
    # [a, alpha, d, theta] for joint 1, then do for joint 2, 3, 4, 5, 6
    [0, np.radians(0), 114, q1],
    [0, np.radians(-90), 0, q2-pi/2],
    [95, np.radians(0), 0, q3],
    [10, np.radians(-90), 95, q4],
    [0, np.radians(90), 0, q5],
    [0, np.radians(-90), 58, q6]
]

def get_transformation_matrix(a, alpha, d, theta):
    M = Matrix([[cos(theta), -sin(theta), 0, a],
                [sin(theta)*np.cos(alpha), cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                [sin(theta)*np.sin(alpha), cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
                [0, 0, 0, 1]])
    return M

T = get_transformation_matrix(DH[0][0], DH[0][1], DH[0][2], DH[0][3])
for i in range(1, len(DH)):
    T = T @ get_transformation_matrix(DH[i][0], DH[i][1], DH[i][2], DH[i][3])

q_sym = symbols('q1:7')

q_sym = symbols('q1:7')
def symbolic_forward_kinematics(q_values):
    subs_dict = {q:offset + angle for q, offset, angle in zip([q1, q2, q3, q4, q5, q6], offsets, q_values)}
    T_symbolic = T.subs(subs_dict)
    return T_symbolic

forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), "numpy")

# Converts the transformation matrix into a pose vector: [X, Y, Z, roll, pitch, yaw]
def transf_to_pose(t_matrix):
    # Position (fill in indices based on your matrix structure)
    X, Y, Z = t_matrix[0][3], t_matrix[1][3], t_matrix[2][3]

    # Breakdown the rotation matrix into axes
    R = t_matrix[0:3, 0:3]
    roll  = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    yaw   = np.arctan2(R[1,0], R[0,0]) # look into this

    return X, Y, Z, roll, pitch, yaw


# Checks the current joint angles against the target pose and returns their delta
def target_pose_error(q, x_target, y_target, z_target, rx_d, ry_d, rz_d):
   current_fk = forward_kinematics_func(*q)

   # Convert translation matrix to pose vector
   current_pose = transf_to_pose(current_fk)
   target_pose = [x_target, y_target, z_target, rx_d, ry_d, rz_d]

   error = np.array(current_pose) - np.array(target_pose)
   return error


# Looks for a solution to inverse kinematics for a target pose and returns a transformation matrix
def inverse_kinematics(target_pose, init_pose, max_iter=1000, tolerance=1e-5, bounds=[lower_bounds,upper_bounds]):

    result = least_squares(target_pose_error, 
                            init_pose, 
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



x_target = 218.6
y_target = 91.3
z_target = 144.3
rx_d = np.radians(-137.69)  # Roll angle (in radians)
ry_d = np.radians(-68.97)  # Pitch angle (in radians)
rz_d = np.radians(-28.38)  # Yaw angle (in radians)

target_pose = [x_target, y_target, z_target, rx_d, ry_d, rz_d]
init_pose = [2.1,65.83,-51.32,-5.53,57.3,149.94]
joint_angles = inverse_kinematics(target_pose, init_pose)
        
# output the joint angles. You may save the output to a csv file
if joint_angles is not None:
    print("Joint Angles (Degrees):", joint_angles)
    print("True Angles (Degrees):", [24.69,60.02,-51.32,-51.32,10.89,-112.32])
else:
    print("Joint Angles (Degrees): None")

