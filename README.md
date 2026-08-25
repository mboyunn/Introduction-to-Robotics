# Introduction to Robotics

This repository contains my programming assignments and laboratory work from an **Introduction to Robotics** course. The course focused on the mathematical and programming foundations used to describe and control robotic manipulators, with hands-on implementation using a **6-DoF robotic arm**.

Throughout the course, I worked with concepts such as coordinate transformations, Denavit–Hartenberg (DH) parameters, forward and inverse kinematics, Jacobians, and robot motion control. These concepts were implemented in Python and applied to the physical robot to connect the mathematical models with real robotic motion.

## Topics Covered

* **Coordinate Frames and Transformations** – Representing the position and orientation of robot links using rotation and homogeneous transformation matrices.
* **Denavit–Hartenberg Parameters** – Modeling the geometry of a serial robotic arm by defining the relationship between consecutive joints and coordinate frames.
* **Forward Kinematics** – Computing the end-effector position and orientation from a given set of robot joint angles.
* **Jacobian Matrix** – Relating changes in joint angles and joint velocities to the resulting motion of the robot's end effector.
* **Inverse Kinematics** – Determining the joint angles required for the end effector to reach a desired position, including iterative Jacobian-based numerical methods.
* **Robot Motion and Joint Control** – Sending joint commands to the physical robot and observing how calculated trajectories translate into real-world motion.
* **Drag Teaching** – Recording robot configurations while manually moving the arm and replaying the recorded joint positions to reproduce the demonstrated motion.

## Implementation

The laboratory exercises primarily use **Python**, with libraries such as **NumPy, SymPy, and SciPy** for matrix operations, symbolic calculations, numerical methods, and kinematic computations.

A typical workflow used throughout the course was:

1. Define the geometry and joint configuration of the robotic arm.
2. Construct transformation matrices using the robot's DH parameters.
3. Calculate the forward kinematics to determine the end-effector pose.
4. Compute the Jacobian to describe how joint motion affects end-effector motion.
5. Use numerical inverse kinematics to determine joint configurations for desired end-effector positions.
6. Send the calculated joint commands to the physical robot and evaluate its motion.

This repository demonstrates how fundamental robotics theory can be translated into software that directly controls a real robotic system.

