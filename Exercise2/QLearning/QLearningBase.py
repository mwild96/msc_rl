#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import csv

class QLearningAgent(Agent):

    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(QLearningAgent, self).__init__()

        #initialize hyperparameters
        self.learningRate = learningRate
        self.learningRateZero = learningRate
        #self.learningRateDecay = 0.1
        self.epsilon = epsilon
        self.epsilonZero = epsilon
        #self.epsilonDecay = 0.00075
        self.discountFactor = discountFactor

        #define list of possible states
        self.possibleStates = [(x,y) for x in range(5) for y in range(6)]
        self.possibleStates.append("GOAL")
        self.possibleStates.append("OUT")

        #initialize Q values
        self.Q = np.empty((len(self.possibleStates), len(self.possibleActions)))
        self.Q.fill(initVals)

        #initialize state-action attributes
        self.curState = None
        self.nextState = None
        self.curAction = None
        self.maxAction = None
        self.reward = None

    def learn(self):

        curActIdx = self.possibleActions.index(self.curAction)

        #store the current Q(s,a) value for later use in the change calculation
        Q_sa = self.Q[self.curState,curActIdx]

        #determine the off policy action index, i.e. max action index
        maxActIdx = self.possibleActions.index(self.maxAction)

        Q_spap = self.Q[self.nextState, maxActIdx]

        #learn
        self.Q[self.curState, curActIdx] = Q_sa + self.learningRate*(self.reward + self.discountFactor*Q_spap - Q_sa)

        return self.Q[self.curState, curActIdx] - Q_sa


    def act(self):

        #"decide" which action to choose - should we explore (random) or exploit (max)?
        c = np.random.choice(2,1,p=[self.epsilon, 1-self.epsilon])

        if c == 0:
            #pick a random action
            idx = np.random.randint(len(self.possibleActions))
        else:
            #choose the action with the maximum Q(s,a) value
            idx = np.argmax(self.Q[self.curState,:])

        action = self.possibleActions[idx]
        return action


    def toStateRepresentation(self, state):

        if type(state[0]) is tuple:

            return self.possibleStates.index(state[0])

        elif type(state[0]) is int:

            return self.possibleStates.index(state)

        elif type(state[0]) is str:

            if state == "OUT_OF_BOUNDS":

                return self.possibleStates.index("OUT")

            elif state == "GOAL":

                return self.possibleStates.index("GOAL")


    def setState(self, state):
        self.curState = state


    def setExperience(self, state, action, reward, status, nextState):
        self.setState(state)
        self.nextState = nextState
        self.curAction = action
        self.maxAction = self.possibleActions[np.argmax(self.Q[self.nextState,:])]
        self.reward = reward


    def setLearningRate(self, learningRate):
        self.learningRate = learningRate


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon


    def reset(self):
        self.curState = None
        self.nextState = None
        self.curAction = None
        self.maxAction = None
        self.reward = None


    def computeHyperparameters(self, numTakenActions, episodeNumber):

        newLearningRate = self.learningRate

        if episodeNumber < 4000:
             #in the beginning we will mostly explore
            newEpsilon = 0.95
        else:
             #towards the end we will only exploit
            newEpsilon = 0.0

        #newEpsilon = self.epsilonZero*np.exp(-self.epsilonDecay*episodeNumber)

        return tuple((newLearningRate, newEpsilon))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default = 5000) #, default=500

    args=parser.parse_args()

    # Initialize connection with the HFO server
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learningRate = 0.01, discountFactor = 0.99, epsilon = 0.8)#= 1.0
    numEpisodes = args.numEpisodes


    numTakenActions = 0
    for episode in range(numEpisodes):
        status = 0
        observation = hfoEnv.reset()

        while status==0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
            update = agent.learn()

            observation = nextObservation
