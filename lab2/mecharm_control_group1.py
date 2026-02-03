from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT
import csv
import time

# record_drag_points will write 10 datapoints to a csv file where each datapoint is a
# row in the csv file consisting of the 6 joint angles and the coordinates the end effector is at.
# additionally the method does angle validation to ensure that while the robot has all servos released,
# the angles drag taught by the user are valid.
def record_drag_points(mycobot, n_points=10):

    if mycobot.is_power_on():
        mycobot.release_all_servos()
        with open("data.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["j1", "j2", "j3", "j4", "j5", "j6", "x", "y", "z", "rx", "ry", "rz"])
            recording = 0
            while recording < 10:
                input(f"Move arm to point {recording+1}/{n_points}, then press ENTER to record...") # waits for user confirmation before recording
                angles = mycobot.get_angles()
                invalid = False
                if 165 < angles[0] < -165:
                    print("Invalid J1 Angle")
                    invalid = True
                if 90 < angles[0] < -90:
                    print("Invalid J2 Angle")
                    invalid = True
                if 65 < angles[0] < -180:
                    print("Invalid J3 Angle")
                    invalid = True
                if 160 < angles[0] < -160:
                    print("Invalid J4 Angle")
                    invalid = True
                if 115 < angles[0] < -115:
                    print("Invalid J5 Angle")
                    invalid = True
                if 175 < angles[0] < -175:
                    print("Invalid J6 Angle")
                    invalid = True
                if invalid:
                    continue
                coords = mycobot.get_coords() # returns a list of 6 elements containing x,y,z,rz,ry,rz
                row = angles[:6] + coords[:6] # concatening the angles and coords together in 1 array so we can write to csv 
                writer.writerow(row)
                print("Recorded")
                mycobot.send_angles([0,0,0,0,0,0], 50) # send robot to origin
                recording += 1
    else:
        print("Robot is not powering on.")


def send_angles(mycobot, speed=20, move_wait=6):
    if mycobot.is_power_on():
        with open("data.csv", "r") as csvfile: # read the csv file
            file = csv.reader(csvfile)
            next(file) # skip the first row in csv file which will be the column names
            for row in file:
                angles = [float(x) for x in row[0:6]]
                mycobot.send_angles(angles, speed)
                time.sleep(move_wait) # give the robot time to move to new location
                print(angles)
                print(mycobot.get_angles())
    else:
        print("Robot is not powering on.")


def send_coords(mycobot, speed=20, move_wait=6):

    if mycobot.is_power_on():
        with open("data.csv", "r") as csvfile:
            file = csv.reader(csvfile)
            next(file) # skip the first row in csv file which will be the column names
            for row in file:
                coords = [float(x) for x in row[6:]]
                mycobot.send_coords(coords, speed)
                time.sleep(move_wait)  # give the robot time to move to new location
    else:
        print("Robot is not powering on.") 



if __name__ == "__main__":
    stage = 1
    mycobot = MechArm270(PI_PORT, PI_BAUD)
    mycobot.power_on()
    mycobot.send_angles([0,0,0,0,0,0], 50)

    if stage == 1:
        record_drag_points(mycobot)
    elif stage == 2:
        for i in range(5):
            send_angles(mycobot)
    elif stage == 3:
        for i in range(5):
            send_coords(mycobot)
    
    mycobot.power_off()