import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))
    
    I = np.eye(7)
    mask_v = ~np.isnan(v_in).flatten()
    mask_omega = ~np.isnan(omega_in).flatten()
    J = calcJacobian(q_in)
    J = np.vstack((J[:3, :][mask_v, :], J[3:, :][mask_omega, :]))
    v_in = v_in[mask_v].reshape((-1, 1))
    omega_in = omega_in[mask_omega].reshape((-1, 1))
    input = np.vstack((v_in, omega_in))
    dq, residuals, rank, s= np.linalg.lstsq(J, input, rcond = None)
    null = (I - np.linalg.pinv(J) @ J) @ b

    return dq + null

