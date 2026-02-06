import numpy as np
from sympy import symbols, cos, sin, evalf, pi, simplify
from sympy.matrices import Matrix
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT
import csv
import time

# get_transformation_matrix will build the transformation matrix of a DH param row
def get_transformation_matrix(a, alpha, d, theta):
    M = Matrix([[cos(theta), -sin(theta), 0, a],
                [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
                [0, 0, 0, 1]])
    return M


#main
if __name__ == "__main__":
    mycobot = MechArm270(PI_PORT, PI_BAUD)


    mycobot.power_on()
    if mycobot.is_power_on():
        mycobot.send_angles([0,0,0,0,0,0], 50)

        q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6') # set of symbols for each joint angle
        # DH table for each axis that the robot has
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

        # testing with data from challange 2
        with open("data.csv", "r") as csvfile:
            file = csv.reader(csvfile)
            next(file)
            for row in file:
                joint_angles = [float(x) for x in row[0:6]]
                coords = [float(x) for x in row[6:]]

                offsets = [0, -90, 0, 0, 0, 0]

                subs_dict = {q:np.radians(offset + angle) for q, offset, angle in zip([q1, q2, q3, q4, q5, q6], offsets, joint_angles)}

                T_num = T_sym.subs(subs_dict)

                end_effector_position = T_num[:3, 3] # m = 3 and n = 3 because top left is the rot mat and the last col is the translation
                x = float(end_effector_position[0].evalf())
                y = float(end_effector_position[1].evalf())
                z = float(end_effector_position[2].evalf())

                # checking to see how close the calcualted positions are to real positions
                print("Recorded", coords)
                print("FK calculated coords: ", [x,y,z])

                mycobot.send_coords([x, y, z, coords[3], coords[4], coords[5]], 50)
                print("Moved")
                time.sleep(6)
                mycobot.send_angles([0, 0, 0, 0, 0, 0], 50)
                time.sleep(6)
        
        mycobot.send_angles([0,0,0,0,0,0], 50)


    mycobot.power_off()

