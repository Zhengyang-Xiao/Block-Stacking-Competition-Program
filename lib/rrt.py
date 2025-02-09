import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK
import itertools

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    tree = [start]
    parents = [-1]
    step_size = 1
    max_iter = 1000
    segments = 20
    fk = FK()

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    def sample_random_configuration():
        goal_bias = max(0.1, 1 - np.linalg.norm(tree[-1] - goal) / np.linalg.norm(start - goal))
        if np.random.rand() < goal_bias:
           return goal
        return np.random.uniform(lowerLim, upperLim)
    def find_nearest_node(random):
        distances = [np.linalg.norm(node - random) for node in tree]
        return np.argmin(distances)
    def extend_tree(nearest_node, random_point):
        direction = (random_point - tree[nearest_node]) / np.linalg.norm(random_point - tree[nearest_node])
        new_node = tree[nearest_node] + step_size * direction
        return new_node
    def is_path_valid(node_1, node_2):
        for i in np.linspace(0, 1, segments):
            intermediate_pos = node_1 + i * (node_2 - node_1)
            joint_positions, _ = fk.forward(intermediate_pos)
            line_start = joint_positions[:-1]
            line_end = joint_positions[1:]
            for box in map.obstacles:
                if any(detectCollision(line_start, line_end, box)):
                   return False
        return True
    def end_effector_distance(joint_config1, joint_config2):
        _, T1 = fk.forward(joint_config1)
        _, T2 = fk.forward(joint_config2)
        pos1 = T1[:3,3]
        pos2 = T2[:3,3]
        return np.linalg.norm(pos1-pos2)
    for i in range(max_iter):
        random_point = sample_random_configuration()
        nearest_node = find_nearest_node(random_point)
        new_node = extend_tree(nearest_node, random_point)
        if is_path_valid(new_node, tree[nearest_node]):
           tree.append(new_node)
           parents.append(nearest_node)
           if end_effector_distance(new_node, goal) < step_size:
              if is_path_valid(new_node, goal):
                 tree.append(goal)
                 parents.append(len(tree)-2)
                 path = []
                 current = len(tree) - 1
                 while current != -1:
                       path.append(tree[current])
                       current = parents[current]
                 path.reverse()
                 return np.array(path)
    return []

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    if len(path) == 0:
       print("No path found")    
    else:
       print("Path found:")
       print(path)
    
    
