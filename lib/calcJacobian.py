import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))
    fk = FK()

    ## STUDENT CODE GOES HERE
    joint_positions, T0e_fk = fk.forward(q_in)
    T0e = np.identity(4)
    Jv = np.zeros((3, 7))
    Jw = np.zeros((3, 7))
    on = joint_positions[7, :]
    z0 = np.array([0, 0, 1])
    z = np.zeros((7, 3))
    z[0, :] = z0

    for i in range(7):
        a = fk.a[i]
        alpha = fk.alpha[i]
        d = fk.d[i]
        theta = q_in[i] - fk.theta_offset[i]

        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [0, np.sin(alpha), np.cos(alpha), d],
                      [0, 0, 0, 1]])
        T0e = np.dot(T0e, T)
        z_axis_direction = T0e[:3, 2]
        oi_1 = joint_positions[i, :] 
        #oi_1[2] += self.link0
        if i < 6:
            z[i + 1, :] = z_axis_direction
            Jw[:, i + 1] = z_axis_direction

        Jv[:, i] = np.cross(z[i, :], on - oi_1)

    Jv[:, 0] = np.cross(z0, T0e_fk[:3, 3])
    Jw[:, 0] = z0
    J = np.vstack((Jv, Jw))
    J = np.where(np.abs(J) < 0.00001, 0, J)

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
