from sympy.matrices import Matrix
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT
import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, lambdify
from scipy.optimize import least_squares
import time

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


if __name__ == "__main__":
    mycobot = MechArm270(PI_PORT, PI_BAUD)
    mycobot.power_on()

    # ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
    #     ([188.3,59.5,16.9,np.deg2rad(-166.86),np.deg2rad(56.97),np.deg2rad(-140.66)], 50, 2), # send to deconstruct loc
    #     ([215.0,16.9,130.4,np.deg2rad(-121.4),np.deg2rad(64.61),np.deg2rad(-108.86)], 30, 3), # send to intermiediate
    #     ([208.2,-32.7,35.3,np.deg2rad(-148.96),np.deg2rad(61.46),np.deg2rad(-136.43)], 10, 4), # send to end loc
    #     ([208.2,-32.7,105.3,np.deg2rad(-148.96),np.deg2rad(61.46),np.deg2rad(-136.43)], 10, 5), # raise z so EE doesnt bump

    init_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

    # open gripper
    mycobot.set_gripper_value(100, 50, 1)
    time.sleep(2)

    poses = [
        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
        ([180.4,41.4,135.0,np.deg2rad(-132.46),np.deg2rad(66.48),np.deg2rad(-108.17)], 20, 2), # send to deconstruct loc
        ([180.4,41.4,175.0,np.deg2rad(-132.46),np.deg2rad(66.48),np.deg2rad(-108.17)], 50, 3),
        ([190.8,-3.2,155.9,np.deg2rad(-110.31),np.deg2rad(70.7),np.deg2rad(-100.17)], 50, 3), # send to intermiediate
        ([208.2,-32.7,12.3,np.deg2rad(-148.96),np.deg2rad(61.46),np.deg2rad(-136.43)], 20, 4), # send to end loc

        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
        ([187.4,71.7,76.2,np.deg2rad(-147.25),np.deg2rad(73.37),np.deg2rad(-115.56)], 20, 2), # send to deconstruct loc
        ([172.2,3.6,119.3,np.deg2rad(-129.06),np.deg2rad(74.11),np.deg2rad(-105.62)], 50, 3), # send to intermiediate
        ([216.2,-16.3,67.0,np.deg2rad(-95.07),np.deg2rad(73.44),np.deg2rad(-92.5)], 10, 4), # send to end loc
        ([216.2,-16.3,107.0,np.deg2rad(-95.07),np.deg2rad(73.44),np.deg2rad(-92.5)], 10, 1),

        ([155.2, 0.9, 219.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], 50, 1), # origin
        ([188.3,59.5,16.9,np.deg2rad(-166.86),np.deg2rad(56.97),np.deg2rad(-140.66)], 50, 2), # send to deconstruct loc
        ([201.7,32.2,135.1,np.deg2rad(-96.27),np.deg2rad(64.05),np.deg2rad(-77.15)], 30, 3), # send to intermiediate
        ([200.1,-18.1,135.2,np.deg2rad(-83.85),np.deg2rad(65.59),np.deg2rad(-78.64)], 10, 3), # send to end loc
        ([205.1,-18.1,120.2,np.deg2rad(-83.85),np.deg2rad(65.59),np.deg2rad(-78.64)], 10, 4),
        ([205.1,-19.1,150.2,np.deg2rad(-83.85),np.deg2rad(65.59),np.deg2rad(-78.64)], 10, 1),
    ]

    for pose, speed, encoding in poses:
        if encoding == 1 or encoding == 2 or encoding == 5:
            print("Open gripper")
            mycobot.set_gripper_value(100, 50, 1)
            time.sleep(2)
        else:
            print("close gripper")
            mycobot.set_gripper_value(69, 50, 1)
            time.sleep(2)
            
        target_pose = np.array(pose)
        joint_angles = inverse_kinematics(target_pose, init_pose)
        joint_angles_deg = np.rad2deg(joint_angles)
        modified = joint_angles_deg.tolist()
        mycobot.send_angles(modified, speed)
        time.sleep(5)
        print(pose)

        if encoding == 1 or encoding == 4 or encoding == 5:
            print("Open gripper")
            mycobot.set_gripper_value(100, 50, 1)
            time.sleep(2)
        else:
            print("close gripper")
            mycobot.set_gripper_value(69, 50, 1)
            time.sleep(2)


    # send to origin
    target_pose = np.array([155.2, 0.9, 240.0, np.deg2rad(-172.88), np.deg2rad(88.58), np.deg2rad(-172.52)], dtype=float)
    joint_angles = inverse_kinematics(target_pose, init_pose)
    joint_angles_deg = np.rad2deg(joint_angles)
    mycobot.send_angles(joint_angles_deg.tolist(), 30)
    time.sleep(5)
