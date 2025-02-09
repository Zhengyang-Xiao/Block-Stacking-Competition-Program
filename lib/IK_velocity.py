import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    mask_v = ~np.isnan(v_in).flatten()
    mask_omega = ~np.isnan(omega_in).flatten()
    J = calcJacobian(q_in)
    J = np.vstack((J[:3, :][mask_v, :], J[3:, :][mask_omega, :]))
    v_in = v_in[mask_v].reshape((-1, 1))
    omega_in = omega_in[mask_omega].reshape((-1, 1))
    input = np.vstack((v_in, omega_in))
    dq, residuals, rank, s= np.linalg.lstsq(J, input, rcond = None)
    
    return dq
