#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
import itertools

class IndependentQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(IndependentQLearningAgent, self).__init__()

        #hyperparameters
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.discountFactor = discountFactor

        #list of possible states
        self.possibleStates = list(itertools.product([(x,y) for x in range(5) for y in range(5)], repeat = 2))
        self.possibleStates.append((('G',),))
        self.possibleStates.append((('O',),))
        self.possibleStates.append((('O',), ('U',)))
        self.possibleStates.append((('G',), ('O',)))
        self.stateRepresentations = dict(zip(self.possibleStates, list(range(len(self.possibleStates)))))

        #initialize Q
        self.Q = np.empty((len(self.possibleStates), len(self.possibleActions)))
        self.Q.fill(initVals)

        #initialize state-action
        self.curState = None
        self.nextState = None
        self.curAction = None
        self.maxAction = None #for the learn step, we know which off-policy (optimal) action to take for our learn calculations
        self.reward = None

    def setExperience(self, state, action, reward, status, nextState):
        self.setState(state)
        self.nextState = nextState
        self.curAction = action
        self.maxAction = self.possibleActions[np.argmax(self.Q[self.nextState,:])]
        self.reward = reward

    def learn(self):

        #store the current Q(s,a) value of the state-action pair for the change calculation
        curActIdx = self.possibleActions.index(self.curAction)
        Q_sa = self.Q[self.curState,curActIdx]

        #calculate the off policy (optimal) new Q(s,a)
        maxActIdx = self.possibleActions.index(self.maxAction)
        Q_spap = self.Q[self.nextState, maxActIdx]

        #learn
        self.Q[self.curState, curActIdx] = Q_sa + self.learningRate*(self.reward + self.discountFactor*Q_spap - Q_sa)

        #return the change in Q(s,a) for this state-action pair
        return self.Q[self.curState, curActIdx] - Q_sa #AM I DOING THIS RIGHT?


    def act(self):
        #"decide" whether we will exploit (maximum) or explore (random)
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

        #our dictionary keys are in the form of tuples, so we convert
        #the environment's state representation to tuples
        s1 = [tuple(l) for l in state[0]]
        s2 = [tuple(l) for l in state[1]]
        s = tuple((s1[0], s2[0]))

        #then return the value from our dictionary
        return self.stateRepresentations[s]

    def setState(self, state):
        self.curState = state

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def computeHyperparameters(self, numTakenActions, episodeNumber):

        #keep the learning rate constant
        newLearningRate = self.learningRate

        if episodeNumber < 40000:
            #in the beginning we will focus on exploring
            newEpsilon = 0.95
        else:
            #we can exploit towards the end
            newEpsilon = 0

        return (newEpsilon, newLearningRate)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type = int, default = 1)
    parser.add_argument('--numAgents', type = int, default = 2)
    parser.add_argument('--numEpisodes', type = int, default = 50000)

    args = parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
    agents = []
    for i in range(args.numAgents):
        agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
        agents.append(agent)

    numEpisodes = 50000
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0
        timeSteps = 0

        while status[0]=="IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies = []
            for agentIdx in range(args.numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(args.numAgents):
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx],
                                               reward[agentIdx],status[agentIdx],
                                               agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation
