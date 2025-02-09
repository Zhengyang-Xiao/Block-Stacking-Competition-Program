import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=5000, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE
        
        att_f = np.zeros((3, 1)) 
        d = 0.12
        distance = np.linalg.norm(target - current)
        zeta = 25
        if distance ** 2 > d:
            att_f = -(current - target) / distance
        else:
            att_f = -zeta * (current - target)

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        k_rep = 0.001
        dist, unitvec = PotentialFieldPlanner.dist_point2box(current.reshape(1, 3), obstacle)
        d_roi = 0.12


        if dist <= d_roi and dist > 0:
            rep_f = -k_rep * (1 / dist - 1 / d_roi) * (1 / dist ** 2) * unitvec.flatten()
            max_force = 100
            if np.linalg.norm(rep_f) > max_force:
                rep_f = (rep_f / np.linalg.norm(rep_f)) * max_force
        else:
            rep_f = np.zeros(3)



        ## END STUDENT CODE

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros((3, 9)) 
        obj = PotentialFieldPlanner
        for i in range(9):
            att_f = obj.attractive_force(target[:, i], current[:,i]).squeeze()
            rep_f = np.zeros(3)
            for obs in obstacle:
                rep_f += obj.repulsive_force(obs, current[:, i]).squeeze()
            joint_forces[:, i] = att_f + rep_f

        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros(9)
        fk = FK_Jac()
        jointPositions, T0e = fk.forward_expanded(q)
        jointPositions = np.delete(jointPositions, 7, axis = 0)
        for i in range(9):
            Fi = joint_forces[:, i]
            Jv_i = np.zeros((3, 9))
            pi = jointPositions[i, :3] 
            for j in range(i):          
                pj = jointPositions[j, :3]  
                zj = T0e[j, :3, 2]   
                Jv_i[:, j] = np.cross(zj, (pi - pj))
            
            joint_torques += Jv_i.T @ Fi

        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(current - target)
        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE

        obj = PotentialFieldPlanner
        fk = FK_Jac()
        jointPositions_current, _ = fk.forward_expanded(q)
        jointPositions_target, _ = fk.forward_expanded(target)
        joint_cur_without_joint0 = np.delete(jointPositions_current, 0, axis = 0)
        joint_tar_without_joint0 = np.delete(jointPositions_target, 0, axis = 0)
        joint_forces= obj.compute_forces(joint_tar_without_joint0.T, map_struct.obstacles, joint_cur_without_joint0.T)
        torques = obj.compute_torques(joint_forces, q)
        real_torques = torques[:7]
        norm_real_torques = np.linalg.norm(real_torques)
        if norm_real_torques > 0:
            dq = real_torques / norm_real_torques
        else:
            dq = np.zeros_like(real_torques)

        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """



        joint_positions_start, _ = self.fk.forward_expanded(start)
        linePt1_start = joint_positions_start[0:7]
        linePt2_start = joint_positions_start[1:8]

        joint_positions_goal, _ = self.fk.forward_expanded(goal)
        linePt1_goal = joint_positions_goal[0:7]
        linePt2_goal = joint_positions_goal[1:8]

        if np.any(start < self.lower) or np.any(start > self.upper) or np.any(goal < self.lower) or np.any(goal > self.upper):
            return np.empty((0, 7))
        for obstacle in map_struct.obstacles:
            if np.any(detectCollision(linePt1_start, linePt2_start, obstacle)) or np.any(detectCollision(linePt1_goal, linePt2_goal, obstacle)):
                return np.empty((0, 7))

        q = start
        qf = goal
        
        i = 0
        q_path = [q.copy()]

        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            dq = self.compute_gradient(q, qf, map_struct)
            alpha = 0.02
            distance_to_goal = np.linalg.norm(goal - q)
            beta = 1.0 * distance_to_goal
            q_new = q + alpha * (dq + beta * (goal - q))
            joint_positions_q, _ = self.fk.forward_expanded(q)
            linePt1_q_pose = joint_positions_q[0:8]
            joint_positions_q_new, _ = self.fk.forward_expanded(q_new)
            linePt2_q_new_pose = joint_positions_q_new[0:8]
            linePt1_q_new = joint_positions_q_new[0:7]
            linePt2_q_new = joint_positions_q_new[1:8]

            collision_between_links = False
            collision_between_poses = False

            for obstacle in map_struct.obstacles:
                if np.any(detectCollision(linePt1_q_new, linePt2_q_new, obstacle)):
                    collision_between_links = True
                    break
                elif np.any(detectCollision(linePt1_q_pose, linePt2_q_new_pose, obstacle)):
                    collision_between_poses = True
                    break
            q = np.clip(q_new, self.lower, self.upper)

            if collision_between_links or collision_between_poses or np.all(dq < self.min_step_size):
                while True:
                    random_step = (np.random.rand(*q.shape) - 0.5) * 2 * 0.3
                    q_rand = q + random_step

                    joint_positions_q_rand, _ = self.fk.forward_expanded(q_rand)
                    linePt1_q_rand = joint_positions_q_rand[0:7]
                    linePt2_q_rand = joint_positions_q_rand[1:8]

                    collision_detected = False
                    for obstacle in map_struct.obstacles:
                        if np.any(detectCollision(linePt1_q_rand, linePt2_q_rand, obstacle)):
                            collision_detected = True
                            break

                    if not collision_detected:
                        q = np.clip(q_rand, self.lower, self.upper)
                        print("Assigned q_rand due to collision or small step size.")
                        break
            else:
                q = np.clip(q_new, self.lower, self.upper)
                q_path.append(q.copy())
                print("Good path")

            # Termination Conditions
            if i >= self.max_steps:
                print("Maximum iterations reached.")
                break
            elif self.q_distance(qf, q) < self.tol:
                q_path.append(q.copy())
                print("Goal reached within tolerance.")
                break
            i += 1


        q_path = np.array(q_path)
            
            ## END STUDENT CODE

        return q_path
    
def evaluate_pf(map_file, start, goal, num_trials=10, num_samples=10000, step_limit=1.5):
    """
    Evaluate the RRT planner for a given map, start, and goal configuration.
    
    Parameters:
        map_file (str): Path to the map file.
        start (np.array): Start configuration.
        goal (np.array): Goal configuration.
        num_trials (int): Number of trials for evaluation.
        num_samples (int): Number of RRT samples.
        step_limit (float): Step limit for RRT expansion.
    
    Returns:
        dict: Contains success rate, average running time, and average path length.
    """
    map_struct = loadmap(map_file)
    successes = 0
    total_time = 0
    total_path_length = 0

    for _ in range(num_trials):
        start_time = time.perf_counter()
        planner = PotentialFieldPlanner()  # Create an instance of the class
        path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))  # Call the instance method
        end_time = time.perf_counter()

        if path.size > 0:  # Check if a valid path was found
            successes += 1
            total_time += end_time - start_time
            total_path_length += len(path)

    success_rate = successes / num_trials
    avg_time = total_time / successes if successes > 0 else float('inf')
    avg_path_length = total_path_length / successes if successes > 0 else 0

    return {
        "success_rate": success_rate,
        "avg_time": avg_time,
        "avg_path_length": avg_path_length
    }

################################
## Simple Testing Environment ##
################################


if __name__ == '__main__':
    starts = [
        np.array([0, 0, 0, -1.5, 0, 1.57, 0]),              # Map 1
        np.array([0, 0.3, 0, -2, 0, 1.57, 0]),              # Map 2
        np.array([-1.5, -0.2, 0.5, -2.2, 0.5, 1.3, 0.3]),   # Map 3
        np.array([0, -1.2, 0, -2, 0, 2, 0.5])               # Map 4
    ]

    goals = [
        np.array([0, 1.5, 0, -0.1, 0, 1.57, 0]),            # Map 1
        np.array([1.2, 0.3, 0, -2, 0, 1.57, 0]),            # Map 2
        np.array([0.5, -0.2, 0.5, -2.2, 0.5, 1.3, 0.3]),    # Map 3
        np.array([0, -0.2, 0, -2.4, 0, 2, 0.5])             # Map 4
    ]

    map_files = [
        "meam520_ws/src/meam520_labs/maps/map1.txt",  # Map 1
        "meam520_ws/src/meam520_labs/maps/map2.txt",  # Map 2
        "meam520_ws/src/meam520_labs/maps/map3.txt",  # Map 3
        "meam520_ws/src/meam520_labs/maps/map4.txt"   # Map 4
    ]

    num_trials = 10  # Number of trials for each map

    print("Evaluating RRT planner...\n")
    for i, (map_file, start, goal) in enumerate(zip(map_files, starts, goals)):
        print(f"Map {i + 1}:")
        results = evaluate_pf(map_file, start, goal, num_trials)
        print(f"  Success Rate: {results['success_rate']:.2f}")
        print(f"  Avg. Time to Find Path: {results['avg_time']:.2f} seconds")
        print(f"  Avg. Path Length: {results['avg_path_length']:.2f} configurations\n")