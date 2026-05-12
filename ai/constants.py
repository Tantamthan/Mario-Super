"""Constants used by the Mario RL environment."""

import numpy as np


FRAME_SKIP = 4
FPS = 60
MS_PER_FRAME = 1000 // FPS

ACTION_COUNT = 7
OBSERVATION_SIZE = 14

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MAP_WIDTH = 9000

MAX_VEL = 15.0
MAX_SCORE = 100000.0
MAX_TIME = 301.0

GROUND_SENSOR_OFFSETS = (40, 80, 120)
GROUND_SCAN_DEPTH = 120
GAP_LOOKAHEAD = 240
GAP_SCAN_STEP = 16

IDLE_DELTA_X = 0.1
IDLE_REWARD_STEPS = 200
IDLE_TRUNCATE_STEPS = 1000

OBS_LOW = np.array(
    [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
OBS_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
