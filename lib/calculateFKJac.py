import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout
        self.a = [0, 0, -0.0825, 0.0825, 0, 0.088, 0, 0.1, 0.2]
        self.alpha = [-pi/2, pi/2, -pi/2, pi/2, pi/2, pi/2, 0, 0, 0]
        self.d = [0.192, 0, 0.316, 0, 0.384, 0, 0.21, -0.105, 0]
        self.theta_offset = [0, 0, -pi, 0, -pi, 0, pi/4, -pi/2, pi]
        self.link0 = 0.141

        pass

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        q_new = np.append(q, [0, 0])
        T0e = np.zeros((10,4,4))
        Te = np.eye(4)
        for i in range(9):
            a = self.a[i]
            alpha = self.alpha[i]
            d = self.d[i]
            theta = q_new[i] - self.theta_offset[i]

            T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [0, np.sin(alpha), np.cos(alpha), d],
                      [0, 0, 0, 1]])
            Te= np.dot(Te, T)
            T0e[i+1] = Te
            frame_position = Te[:3, 3]
            z_axis_direction = Te[:3, 2]
            if i == 1:
                position = frame_position + 0.195 * z_axis_direction
            elif i == 3:
                position = frame_position + 0.125 * z_axis_direction
            elif i == 4:
                position = frame_position - 0.015 * z_axis_direction
            elif i == 5:
                position = frame_position + 0.051 * z_axis_direction
            else:
                position = frame_position
            jointPositions[i+1] = position
            
            
        jointPositions[:, 2] += self.link0 
        T0 = np.eye(4)
        T0[3, 3] = 0
        T0e[0] = T0
        T0e[:, 2, 3] += self.link0
        jointPositions = np.where(np.abs(jointPositions) < 0.00001, 0, jointPositions)
        T0e = np.where(np.abs(T0e) < 0.00001, 0, T0e)

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return()
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
