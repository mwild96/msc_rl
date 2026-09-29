import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from Networks import ValueNetwork

from torch.autograd import Variable
from Environment import HFOEnv
import random


#any version of partial credit is appreciated :)

def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
    """
    implement necessary computations required to train a single process
    of an agent
    """

    hfoEnv = HFOEnv(numOpponents = 0, numTeammates = 0)
    hfoEnv.connectToServer()

    criterion = nn.MSELoss()

    aysnc_update = 10
    target_update = 10

    possible_actions = ['MOVE','SHOOT','DRIBBLE','GO_TO_BALL']

    for j in range(num_episodes):

        state = hfoEnv.reset()
        localCounter = 0

        while done == False:

            #update global counter
            counter.value += 1

            #update local counter
            local_counter += 1


            #ACT
            c = np.random.choice(2,1,p=[epsilon, 1-epsilon])

            if c == 0:
                #pick a random action
                idx = np.random.randint(len(possibleActions))

            else:
                #choose the action with the maximum Q(s,a) value
                Q_sa = []
                for a in range(len(possible_actions)):
                    Q_spap.append(computePrediction(nextState, a, target_value_network))

                action_idx = np.argmax(np.asarray(Q_sa))
                action = possible_actions[action_idx]


            nextState, reward, done, status = hfoEnv.step(action)


            value = computeTargets(reward, nextState, discount_factor, done, target_value_network) #WHERE DO WE GET DONE?
            prediction = computePrediction(state, action, value_network)

            loss = criterion(prediction, value)
            loss.backward()

            #LEARN (LOCALLY)
            if local_counter % async_update == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()


            #LEARN (GLOBALLY)
            if counter.value % target_update == 0:
                target_value_network.parameters() = value_network.parameters()

                #self.saveModelNetwork(model, strDirectory)

            state = nextState



def computeTargets(reward, nextState, discount_factor, done, target_value_network):
    """
    implement the target value computation for Q-Learning
    Should be usable for any architecture being used
    """

    if done == True:
        y = reward
    else:
        Q_spap = []
        for a in self.possibleActions:
            Q_spap.append(computePrediction(nextState, a, target_value_network))
        #max_action_idx = np.argmax(np.asarray(Q_spap))
        #note this will give an INDEX not an action by name
        max_Qp = np.max(np.asarray(Q_spap))
        y = reward + discount_factor*max_Qp

    return y


def computePrediction(state, action, value_network):
    """
    implement a single call for the forward computation of Q-Network
    should be agnostic to any Q-Network architecture
    """

    #prepare the features
    inputs = np.empty((1,6))
    inputs[0,0] = state[0]
    inputs[0,1] = state[1]

    #one hot encode the action
    for i in range(2,6):
        if action = possible_actions[i]:
            #THIS TAKES ACTION STRING
            inputs[0,i] = 1
        else:
            inputs[0,i] = 0

    prediction = value_network#.forward(inputs).detach().squeeze(0).numpy()

    return prediction


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)
