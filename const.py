import torch
import math


##### CONFIG

torch.set_printoptions(precision=10)
## CUDA variable from Torch
CUDA = torch.cuda.is_available()
#torch.backends.cudnn.deterministic = True
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")

## Eps for log
EPSILON = 1e-6

## VAE
LATENT_VEC = 200
BETA = 3
VAE_LOSS = "bce"

## RNN
OFFSET = 1
HIDDEN_UNITS = 1024
HIDDEN_DIM = 1024
TEMPERATURE = 1.25
GAUSSIANS = 8
NUM_LAYERS = 1
SEQUENCE = 100
PARAMS_CONTROLLER = HIDDEN_UNITS * NUM_LAYERS * 2 + LATENT_VEC
MDN_CONST = 1.0 / math.sqrt(2.0 * math.pi)

## Controller
PARALLEL = 4
SIGMA_INIT = 4
POPULATION = 32
SCORE_CAP = 8000
REPEAT_ROLLOUT = 4
RENDER_TICK = 16
REWARD_BUFFER = 300
SAVE_SOLVER_TICK = 1
MIN_REWARD = 10

## Image size
HEIGHT = 128
WIDTH = 128

## Dataset
SIZE = 10000
MAX_REPLACEMENT = 0.1
REPEAT = 0

## Play
PARALLEL_PER_GAME = 2
PLAYOUTS = 2500
PLAYOUTS_PER_LEVEL = 10000
ACTION_SPACE = 6            #入力の種類
ACTION_SPACE_DISCRETE = 6   #行動の数

## Training
MOMENTUM = 0.9 ## SGD
ADAM = True
LR = 1e-3
L2_REG = 1e-4
LR_DECAY = 0.1
BATCH_SIZE_VAE = 300
BATCH_SIZE_LSTM = 2
SAMPLE_SIZE = 150

## Refresh
LOSS_TICK = 10
REFRESH_TICK = 200
SAVE_PIC_TICK = 20
SAVE_TICK = 1000
LR_DECAY_TICK = 100000

## Jerk
EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = 1e6




