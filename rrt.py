
from gym_env_discrete import PruningEnv
from args import args_dict
import pybullet_planning as pp
import numpy as np
# PARSE ARGUMENTS
import argparse
import time
# Create the ArgumentParser object
parser = argparse.ArgumentParser()

from random import random

from pybullet_planning.motion_planners.utils import irange, argmin, RRT_ITERATIONS, INF

__all__ = [
    'rrt',
]
def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)
env = PruningEnv(renders=False, tree_urdf_path= args.TREE_TEST_URDF_PATH, tree_obj_path=args.TREE_TEST_OBJ_PATH, name = "evalenv", num_points = args.EVAL_POINTS)
env.reset()
eval_env = PruningEnv(renders=True, tree_urdf_path= args.TREE_TEST_URDF_PATH, tree_obj_path=args.TREE_TEST_OBJ_PATH, name = "evalenv", num_points = args.EVAL_POINTS)
eval_env.reset()
start = env.get_current_pose(env.end_effector_index)[0]
goal_sample = env.tree_goal_pos
goal_orientation = env.con.getQuaternionFromEuler(perpendicular_vector(env.tree_goal_branch))
#get perpendicular to goal orientation


print(start, goal_sample)

def distance_fn(q1, q2):
    return np.linalg.norm(np.array(q1) - np.array(q2))

def difference_fn(q1, q2):
    return np.array(q1) - np.array(q2)

def sample_fn():
    global start

    """Sample a random pos between (1,-1)"""
    sample_pos = start+(np.random.rand(3)-0.5)*2
    # sample_or =  ((np.random.rand(3))-0.5)/0.5*2*np.pi#Yaw pitch roll
    # print(sample_pos, sample_or)
    return tuple(sample_pos)


#Add orientation to be perpendicular to the branch
def collision_fn(q):
    # print(q)
    if (q == env.tree_goal_pos).all():
        return False
    # print("checking collision")
    curr_angles = env.get_joint_angles()
    j_angles = env.calculate_ik(q, env.con.goal_orientation)
    env.set_joint_angles(j_angles)
    for j in range(100):
        env.con.stepSimulation()
    ret = env.check_collisions()
    env.set_joint_angles(curr_angles)
    for j in range(100):
        env.con.stepSimulation()
    # print(ret)
    # print(env.get_current_pose()[0])
    return ret[0]

def extend_fn(q1, q2):
    steps = np.abs(np.divide(difference_fn(q2, q1), 0.1))
    n = int(np.max(steps)) + 1
    # print(n)
    q = q1
    for i in range(n):
        # print(difference_fn(q2, q))
        q = tuple((1. / (n - i)) * np.array(difference_fn(q2, q)) + q)
        yield q
        # TODO: should wrap these joints
# for i in extend_fn(start, goal_sample):
#     print(i)

# print(res)
count = 0
for _ in range(50):
    env.reset()

    res = pp.rrt_connect(start, env.tree_goal_pos, distance_fn, sample_fn, extend_fn, collision_fn)#, radius = 0.3)#, verbose = True)
    if res == None:
        print("no path found")
        continue
# print(count)
    else:
        res.append(env.tree_goal_pos)
        eval_env.con.removeBody(eval_env.sphereUid)
        colSphereId = -1
        eval_env.tree_goal_pos = env.tree_goal_pos
        visualShapeId = eval_env.con.createVisualShape(eval_env.con.GEOM_SPHERE, radius=.02,rgbaColor =[1,0,0,1])
        eval_env.sphereUid = eval_env.con.createMultiBody(0.0, colSphereId, visualShapeId, [env.tree_goal_pos[0],env.tree_goal_pos[1],env.tree_goal_pos[2]], [0,0,0,1])
        count += 1
        for i in res:
            # print(i)
            j_angles = eval_env.calculate_ik(i, goal_orientation)
            eval_env.set_joint_angles(j_angles)
            for j in range(10):
                eval_env.con.stepSimulation()
                time.sleep(5/240)
            # env.render()
            # print(eval_env.get_current_pose()[0], env.tree_point_pos, eval_env.tree_point_pos)
        eval_env.reset()
print(count)