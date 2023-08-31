args_dict = {
    #File paths
    'TREE_TRAIN_URDF_PATH': {
        'type': str,
        'default': './meshes_and_urdf/urdf/trees/envy/train',
        'help': 'path to the train URDF file for the tree environment'
    },
    'TREE_TRAIN_OBJ_PATH': {
        'type': str,
        'default': './meshes_and_urdf/meshes/trees/envy/train',
        'help': 'path to the train OBJ file for the tree environment'
    },
    'TREE_TEST_URDF_PATH': {
        'type': str,
        'default': './meshes_and_urdf/urdf/trees/envy/test',
        'help': 'path to the test URDF file for the tree environment'
    },
    'TREE_TEST_OBJ_PATH': {
        'type': str,
        'default': './meshes_and_urdf/meshes/trees/envy/test',
        'help': 'path to the test OBJ file for the tree environment'
    },
    #PPO parameters
    'EMB_SIZE': {
        'type': int,
        'default': 128,
        'help': 'size of the embedding layer'
    },
    'ACTION_DIM_ACTOR': {
        'type': int,
        'default': 6,
        'help': 'dimension of the action space for the actor network'
    },
    'STATE_DIM': {
        'type': int,
        'default': 72+21,
        'help': 'dimension of the state space'
    },
    #Gym envirionment parameters
    'N_ENVS': {
        'type': int,
        'default': 4,
        'help': 'number of parallel environments to run'
    },
    'RENDER': {
        'type': bool,
        'default': False,
        'help': 'whether to render the environment'
    },
    'MAX_STEPS': {
        'type': int,
        'default': 400,
        'help': 'maximum number of steps per episode'
    },
    'EVAL_MAX_STEPS': {
        'type': int,
        'default': 400,
        'help': 'maximum number of steps per episode during evaluation'
    },
    'ACTION_SCALE': {
        'type': float,
        'default': 1,
        'help': 'scaling factor for the action space'
    },
    'MOVEMENT_REWARD_SCALE': {
        'type': float,
        'default': 5,
        'help': 'scaling factor for the movement reward'
    },
    'DISTANCE_REWARD_SCALE': {
        'type': float,
        'default': 0,#1/70,
        'help': 'scaling factor for the distance reward'
    },
    'CONDITION_REWARD_SCALE': {
        'type': float,
        'default': 0,
        'help': 'scaling factor for the condition reward'
    },
    'TERMINATE_REWARD_SCALE': {
        'type': float,
        'default': 5,
        'help': 'scaling factor for the terminate reward'
    },
    'COLLISION_REWARD_SCALE': {
        'type': float,
        'default': -0.0005,
        'help': 'scaling factor for the collision reward'
    },
    'SLACK_REWARD_SCALE': {
        'type': float,
        'default': -0.001,
        'help': 'scaling factor for the slack reward'
    },
    'POINTING_ORIENTATION_REWARD_SCALE': {
        'type': float,
        'default':2,
        'help': 'scaling factor for the orientation reward'
    },
    'PERPENDICULAR_ORIENTATION_REWARD_SCALE': {
        'type': float,
        'default': 1,
        'help': 'scaling factor for the orientation reward'
    },
    #Training parameters
    'TOTAL_TIMESTEPS': {
        'type': int,
        'default': 5000000,
        'help': 'total number of timesteps to train for'
    },
    'STEPS_PER_EPOCH': {
        'type': int,
        'default': 400,
        'help': 'number of timesteps per epoch'
    },
    'EPOCHS': {
        'type': int,
        'default': 10,
        'help': 'number of epochs to train for'
    },
    'BATCH_SIZE': {
        'type': int,
        'default': 128,
        'help': 'batch size'
    },
    'LEARNING_RATE': {
        'type': float,
        'default': 0.0000025,
        'help': 'learning rate'
    },
    'LEARNING_RATE_AE': {
        'type': float,
        'default': 0.0005,
        'help': 'learning rate for the autoencoder'
    },
    'LOG_STD_INIT': {
        'type': float,
        'default': -4,
        'help': 'initial value for the log standard deviation'
    },

    #Evaluation parameters
    'EVAL_FREQ': {
        'type': int,
        'default': 30000,
        'help': 'frequency of evaluation'
    },
    'EVAL_EPISODES': {
        'type': int,
        'default': 10,
        'help': 'number of episodes to run during evaluation'
    },
    'EVAL_POINTS': {
        'type': int,
        'default': 10,
        'help': 'number of points to sample in a tree during evaluation'
    },  
    'TESTING': {
        'type': bool,
        'default': False,
        'help': 'whether to run in testing mode'
    },
   'NAME' : {
	'type':str,
	'default': 'run'
	},
    'LOAD_PATH':{
        'type':str,
        'default':None
        },
    'USE_OPTICAL_FLOW':{
        'type':bool,
        'default':True
        },
}
