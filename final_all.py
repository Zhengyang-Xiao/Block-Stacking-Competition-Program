import sys
import numpy as np
from math import pi
from lib.calculateFK import FK
from lib.IK_position_null import IK
import rospy
from core.interfaces import ObjectDetector
from core.interfaces import ArmController
import time
from core.utils import time_in_seconds
from lib.calcAngDiff import calcAngDiff

#This is the script to run the pick and place challenge for Group 15


'''Helper Functions'''

# Targets of the goals

def target_pos(dyn_num, team):
  end_effector_pose = np.array([[1,0,0],
                  [0,-1,0],
                  [0,0,-1],
                  [0,0,0]])
  if team == 'red':
    z = 2.55395038e-01 + (dyn_num-1)*5e-2
    target_vec = np.array([0.562, 0.134, z, 1]).reshape((4,1))
    target_H = np.append(end_effector_pose, target_vec, axis = 1)

  elif team == 'blue':
    z = 2.55395038e-01 + (dyn_num-1)*5e-2
    target_vec = np.array([0.562, -0.134, z, 1]).reshape((4,1))
    target_H = np.append(end_effector_pose, target_vec, axis = 1)

  return target_H

# Get max radius

def max_radius(Info, T0e, team):
  max_radius = 0
  for (name, pose) in Info:
    H_ee_camera = detector.get_H_ee_camera()
    pose_block_ef = np.dot(H_ee_camera, pose)
    pose_block_world = np.dot(T0e, pose_block_ef)
    x, y, z = pose_block_world[:3, 3]

    if team == 'red':
      y = -y + 0.99
      cur_radius = np.sqrt(x**2 + y**2)
    if team == 'blue':
      y = y + 0.99
      cur_radius = np.sqrt(x**2 + y**2)

    max_radius = max(cur_radius, max_radius)

    if team == 'red':
        max_radius = -(max_radius - 0.99)
    if team == 'blue':
        max_radius = max_radius - 0.99

    return max_radius

# Get armed position

def get_armed_position(max_radius, team):
    if team == 'red':
        x_pos = 0.05201
        y_pos = np.sqrt(max_radius**2 - x_pos**2)
    if team == 'blue':
        x_pos = -0.02317
        print(f"x_pos: {x_pos}, max_radius: {max_radius}")
        y_pos = np.sqrt(max_radius**2 - x_pos**2)

    return x_pos, y_pos

# Check gripper state

def check_grip_dyn(arm):
    # 获取夹爪的状态
    gripper_state = arm.get_gripper_state()

    # 获取夹爪位置
    gripper_position = gripper_state['position']

    # 计算夹爪的开合距离
    gripper_dist = abs(gripper_position[1] + gripper_position[0])
    print("Gripper distance: {:.3f} m".format(gripper_dist))

    # 判断夹爪是否抓住物体
    if gripper_dist <= 0.02:
        print("Gripper is too closed. Likely not catching the object.")
        return False
    else:
        print("Object successfully caught.")
        return True

 
'''Commands'''

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    fk = FK()
    ik = IK()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE
    
    # Dynamic blocks

    # Set detection positions
    if team == 'red':
        detection_pos = np.array([1.48897,0.85114,0.21863,-0.60305,-0.16518,1.444,0.82791]) # Adjust
    else:
        detection_pos = np.array([-1.48897,0.85114,0.21863,-0.60305,-0.16518,1.444,0.82791]) # Adjust

    _, T0e_dyn = fk.forward(detection_pos)

    arm.safe_move_to_position(detection_pos)

    # Detect dynamic blocks
    Info = detector.get_detections()
    dynamic_blocks_poses = np.array([]).reshape((0, 4))

    for (name, pose) in Info:
        dynamic_blocks_poses = np.append(dynamic_blocks_poses, pose, axis=0)

    print(dynamic_blocks_poses)

    # Obtain max radius
    max_radius = max_radius(Info, T0e_dyn, team)

    # Set armed position for max radius
    x_pos, y_pos = get_armed_position(max_radius, team)

    # Set armed robot configuration
    if team == 'red':
        R_matrix_EE = np.array([[-0.05849, 0.1992, -0.97821],[0.05501, 0.97904, 0.19608],[0.99677, -0.04234, -0.06823],[0, 0, 0]]) # Adjust
        offset = np.array([x_pos,y_pos,0.227,1]).reshape((4,1)) # Adjust Z
        H_matrix_EE = np.append(R_matrix_EE, offset, axis=1)
        pre_assumed_pos = np.array([0.53556664, 1.48332158, 1.29211466, -1.26160485, 0.19706884, 1.96706884, -0.97104849]) # Adjust
        target = H_matrix_EE
        armed_joints, qpath = ik.inverse_q(pre_assumed_pos, target)
    else:
        R_matrix_EE = np.array([[0.16127, -0.01425, 0.98681],[-0.01684, -0.99979, -0.01169],[0.98677, 0.01474, -0.16147],[0, 0, 0]]) # Adjust
        offset = np.array([x_pos,y_pos,0.22327,1]).reshape((4,1)) # Adjust Z
        H_matrix_EE = np.append(R_matrix_EE, offset, axis=1)
        pre_assumed_pos = np.array([-1.809, 1.257, -0.219, -0.69, 1.499, 1.286, -1.2045]) # Adjust
        target = H_matrix_EE
        armed_joints, qpath = ik.inverse_q(pre_assumed_pos, target)

    # Get target position for dynamics blocks
    # dyn_target_pos = target_pos(team)

    # Set the safe position for each team
    if team == 'red':
        safe_pos = np.array([0.15898, 0.09708, 0.14258, -1.50609, -0.01379, 1.60219, 1.08673]) # Adjust
    if team == 'blue':
        safe_pos = np.array([-0.14949, 0.09719, -0.15274, -1.50612, 0.01477, 1.60218, 0.48341]) # Adjust

    arm.safe_move_to_position(safe_pos)

    # Grasp statistics
    dyn_num = 0
    check = False
    t_starting = time_in_seconds()

    # Number of desired dynamic blocks
    num_dynblock = 1 # 抓三个先

    # Set gripper position before & right after slide in
    for i in range(num_dynblock):

        side_pos_blue = np.array([-1.7, 0.613, -0.31, -1.822, 1.326, 1.235, -1.697]) # Adjust
        side_pos_red = np.array([0.17404958, 1.43833732, 1.30138027, -1.63256299, 0.17443453, 1.74936236, -1.01136256]) # Adjust

        if team == 'red':
            arm.safe_move_to_position(side_pos_red)
        if team == 'blue':
            arm.safe_move_to_position(side_pos_blue)

        # Move gripper to armed position
        arm.exec_gripper_cmd(0.09, 0)  # 打开夹爪
        arm.safe_move_to_position(armed_joints)

        # Start grasping
        while True:
            time.sleep(3)
            arm.exec_gripper_cmd(0.01, 50)  # 关闭夹爪并尝试抓取
            gripper_state = check_grip_dyn(arm)

            # Check gripper state
            if gripper_state:
                print("Got Block")
                dyn_num += 1
                break

            # Check total time on dynamic blocks
            elif time_in_seconds()-t_starting > 10: # 这个时间暂定
                print("Time out")
                check = True
                break

            # If failed, try again
            else:
                print("Failed to grab, try again.")
                arm.exec_gripper_cmd(0.09, 0)

        # If timed out, exit grasping
        if check is True: # 超时直接退出
            if team == "blue":
                arm.safe_move_to_position(side_pos_blue)
                arm.safe_move_to_position(start_position)
            if team == "red":
                arm.safe_move_to_position(side_pos_red)
                arm.safe_move_to_position(start_position)
            break

        # If success, slide out to prevent collision
        if team == "blue":
            arm.safe_move_to_position(side_pos_blue)

        if team == "red":
            arm.safe_move_to_position(side_pos_red)

        # Move to safe position
        arm.safe_move_to_position(safe_pos)

        # Get target position of stacking
        target_H = target_pos(dyn_num, team)
        print(target_H)

        # Bring block to stack
        seed = safe_pos
        target = target_H
        q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)
        print("\n method: J_pseudo-inverse")
        print("   Success: ",success_pseudo, ":  ", message_pseudo)
        print("   Solution: ",q_pseudo)
        print("   #Iterations : ", len(rollout_pseudo))
        print("\n method: J_transpose")
        arm.safe_move_to_position(q_pseudo)
        arm.exec_gripper_cmd(pos = 9e-02, force = None)
        arm.safe_move_to_position(safe_pos)

    # Static blocks

    # Rotation Matrices for Valid Block Orientation
    R_local_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
        ])
    R_local_y_90 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
        ])
    R_local_z_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    R_local_x_minus_90 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
        ])
    R_local_y_minus_90 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
        ])

    ## Check current stack status
    Info_place = detector.get_detections()
    print(Info_place)
    if len(Info_place) == 0:
        if team == 'red':
            placing_target= np.array([
            [ 1.,      0.,      0.,      0.562 ],
            [ 0.,     -1.,      0.,      0.134 ],
            [ 0.,      0.,     -1.,      2.00e-01],
            [ 0.,      0.,      0.,      1.],
            ])
        else:
            placing_target= np.array([
            [ 1.,      0.,      0.,      0.562 ],
            [ 0.,     -1.,      0.,      -0.134 ],
            [ 0.,      0.,     -1.,      2.00e-01],
            [ 0.,      0.,      0.,      1.],
            ])
        z_max = 1.75e-01
    else:
        H_ee_camera = detector.get_H_ee_camera()
        list_height = []
        list_matrix = []
        _, T0e_safe = fk.forward(safe_pos)
        for (name, pose) in Info_place:
            pose_block_placing_ef = np.dot(H_ee_camera, pose)
            pose_block_placing_world = np.dot(T0e_safe, pose_block_placing_ef)
            list_height.append(pose_block_placing_world[2, 3])
            list_matrix.append(pose_block_placing_world)
        max_height_index = list_height.index(max(list_height))
        placing_target = list_matrix[max_height_index]
        R_block = placing_target[:3, :3]
        ## Adjust z axis of each block
        third_row = placing_target[2, :]
        index = np.where(np.isclose(np.abs(third_row), 1))[0][0]
        if index == 0:
            if np.isclose(third_row[index], 1):
                R_block = R_block @ R_local_y_minus_90
            else:
                R_block = R_block @ R_local_y_90
        elif index == 1:
            if np.isclose(third_row[index], 1):
                R_block = R_block @ R_local_x_90
            else:
                R_block = R_block @ R_local_x_minus_90
        else:
            if np.isclose(third_row[index], 1):
                R_block = R_block @ (R_local_x_90 @ R_local_x_90)
        ## Ensure R_block remains a valid rotation matrix
        U, _, Vt = np.linalg.svd(R_block)
        R_block = U @ Vt
        placing_target[:3, :3] = R_block
        print(name,'\n',placing_target)
        ##Adjust x and y to avoid false trajectory (Invalid grasping orientation)###
        ## Index refers to the minimum abs value of sin(rotation around z) 
        R_curr = T0e_safe[:3, :3]
        R_rotate_z_storage = np.zeros((4, 3, 3))
        omega = np.zeros(4)
        R_rotate_z_storage[0] = R_block
        diagonal = np.zeros((4, 3))
        R_des = R_block
        omega[0] = calcAngDiff(R_des, R_curr)[2]
        diagonal[0] = [R_block[0, 0], R_block[1, 1], R_block[2, 2]]
        for i in range (3):
            R_block = R_block @ R_local_z_90
            U, _, Vt = np.linalg.svd(R_block)
            R_block = U @ Vt
            R_rotate_z_storage[i + 1] = R_block
            R_des = R_block
            omega[i + 1] = calcAngDiff(R_des, R_curr)[2]
            diagonal[i + 1] = [R_block[0, 0], R_block[1, 1], R_block[2, 2]]
        index = np.argmin(np.abs(omega))
        ## Choose the matrix closest to ef
        diagonal_block = np.array([T0e_safe[0,0], T0e_safe[1,1], T0e_safe[2,2]])
        distance = np.linalg.norm(diagonal - diagonal_block, axis = 1)
        if index % 2 == 0:
            if distance[0] < distance[2]:
                index = 0
            else:
                index = 2
        else:
            if distance[1] < distance[3]:
                index = 1
            else:
                index = 3
        R_block = R_rotate_z_storage[index]
        ## Assign matrix and offset
        placing_target[:3, :3] = R_block
        z_max = max(list_height)




    # Move to static detection mode
    if team == 'red':
        block_pos = np.array([-pi/10,0,0,-pi/2,0,pi/2,pi/4])
        arm.safe_move_to_position(block_pos)
    if team == 'blue':
        block_pos = np.array([pi/10,0,0,-pi/2,0,pi/2,pi/4])
        arm.safe_move_to_position(block_pos)
    
    # Create Objects
    ik = IK(linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5)
    fk = FK()

    # Initialization
    q = block_pos
    block_offset = 1e-01

    # Get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()
    _, T0e = fk.forward(block_pos)
    Info = detector.get_detections()

    # Detect some blocks...
    j = 1
    for (name, pose) in Info:
        pose_block_ef = np.dot(H_ee_camera, pose)
        pose_block_world = np.dot(T0e, pose_block_ef)
        R_block = pose_block_world[:3, :3]

        print(name,'\n',pose)
        print('block world:', pose_block_world)

        # Adjust z axis of each block
        third_row = R_block[2, :]
        index = np.where(np.isclose(np.abs(third_row), 1))[0][0]
        if index == 0:
            if np.isclose(third_row[index], 1):
                R_block = R_block @ R_local_y_minus_90
            else:
                R_block = R_block @ R_local_y_90
        elif index == 1:
            if np.isclose(third_row[index], 1):
                R_block = R_block @ R_local_x_90
            else:
                R_block = R_block @ R_local_x_minus_90
        else:
            if np.isclose(third_row[index], 1):
                R_block = R_block @ (R_local_x_90 @ R_local_x_90)

        # Ensure R_block remains a valid rotation matrix
        U, _, Vt = np.linalg.svd(R_block)
        R_block = U @ Vt
        pose_block_world[:3, :3] = R_block

        print(name,'\n',pose_block_world)
        

        ###Adjust x and y to avoid false trajectory (Invalid grasping orientation)###
        # Index refers to the minimum abs value of sin(rotation around z) 
        R_curr = T0e[:3, :3]
        R_rotate_z_storage = np.zeros((4, 3, 3))
        omega = np.zeros(4)
        R_rotate_z_storage[0] = R_block
        diagonal = np.zeros((4, 3))
        R_des = R_block
        omega[0] = calcAngDiff(R_des, R_curr)[2]
        diagonal[0] = [R_block[0, 0], R_block[1, 1], R_block[2, 2]]
        for i in range (3):
            R_block = R_block @ R_local_z_90
            U, _, Vt = np.linalg.svd(R_block)
            R_block = U @ Vt
            R_rotate_z_storage[i + 1] = R_block
            R_des = R_block
            omega[i + 1] = calcAngDiff(R_des, R_curr)[2]
            diagonal[i + 1] = [R_block[0, 0], R_block[1, 1], R_block[2, 2]]
        index = np.argmin(np.abs(omega))

        # Choose the matrix closest to ef
        diagonal_block = np.array([1, -1, -1])
        distance = np.linalg.norm(diagonal - diagonal_block, axis = 1)
        if index % 2 == 0:
            if distance[0] < distance[2]:
                index = 0
            else:
                index = 2
        else:
            if distance[1] < distance[3]:
                index = 1
            else:
                index = 3
        R_block = R_rotate_z_storage[index]

        # Assign matrix and offset
        pose_block_world[:3, :3] = R_block
        pose_block_world[2, 3] += block_offset

        # Move to target pose
        seed = block_pos
        target = pose_block_world
        q_above_block, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)
        arm.safe_move_to_position(q_above_block)

        # Start grasping
        arm.exec_gripper_cmd(pos = 7e-02, force = None)
        pose_block_world[2, 3] -= block_offset
        seed = q_above_block
        target = pose_block_world
        q_center_block, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)
        arm.safe_move_to_position(q_center_block)
        arm.exec_gripper_cmd(pos = 4.9e-02, force = 50)
        arm.safe_move_to_position(block_pos)

        # Place the block
        q_vertical_motion = q_above_block - q_center_block
        placing_target[2, 3] = z_max + 2e-02 + j * 5e-02
        j += 1
        seed = block_pos
        target = placing_target
        q_placing, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)

        arm.safe_move_to_position(q_placing)
        arm.exec_gripper_cmd(pos = 7e-02, force = None)
        arm.safe_move_to_position(q_placing + 2 * q_vertical_motion)
#end
