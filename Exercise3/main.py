import torch.multiprocessing as mp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Networks import ValueNetwork
from Worker import Worker
from Environment import HFOEnv
import SharedAdam


# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.

parser = argparse.ArgumentParser()

parser.add_argument('--num_episodes', type = int, default = 5000)
parser.add_argument('--epsilon', type = float, default = 0.8)
parser.add_argument('--discount_factor', type = float, default = 0.99)
parser.add_argument('--learning_rate', type = float, default = 0.01)



if __name__ == "__main__" :

    args = parser.parse_args()

    stateDim = 2
    numActions = 4


    value_network = ValueNetwork(inputDim = stateDim+numActions, outputDim = 1)
    target_network = ValueNetwork(inputDim = stateDim+numActions, outputDim = 1)


    value_network.share_memory()
    target_value_network.share_memory()


    optimizer = SharedAdam.SharedAdam(value_network.parameters(), lr = args.lr)
    optimizer.share_memory()

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []

    for idx in range(0, args.num_processes):
        trainingArgs = (idx, args, value_network, target_value_network,
                        optimizer, lock, counter)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()

        processes.append(p)
    for p in processes:
        p.join()
