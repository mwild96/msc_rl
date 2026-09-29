#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
import itertools

class WolfPHCAgent(Agent):
    def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
        super(WolfPHCAgent, self).__init__()

        #set hyperparameters
        self.learningRate = learningRate
        self.discountFactor = discountFactor#I don't think this one is being used
        self.winDelta = winDelta
        self.loseDelta = loseDelta

        #define list of possible states and state representation dictionary
        self.possibleStates = list(itertools.product([(x,y) for x in range(5) for y in range(5)], repeat = 2))
        self.possibleStates.append((('G',),))
        self.possibleStates.append((('O',),))
        self.stateRepresentations = dict(zip(self.possibleStates, list(range(len(self.possibleStates)))))

        #initialize policy
        self.policy = np.empty((len(self.possibleStates), len(self.possibleActions)))
        self.policy.fill(1/len(self.possibleActions))
        self.averagePolicy = np.zeros((len(self.possibleStates),len(self.possibleActions)))

        #initialize Q
        self.Q = np.zeros((len(self.possibleStates), len(self.possibleActions)))

        #initialize state counter
        self.C = np.zeros(len(self.possibleStates))

        #initialize state-action
        self.curState = None
        self.nextState = None
        self.curAction = None
        self.reward = None


    def setExperience(self, state, action, reward, status, nextState):
        self.setState(state)
        self.curAction = action
        self.reward = reward
        self.nextState = nextState

    def learn(self):
        """
        This function has no input parameters. On the other hand, this method must
        return a single scalar that specifies the change (value after update subtracted
        by value before training) in updated state-action value after you've trained
        your agents using WoLF-PHC's state-action update rule.
        """

        actionIdx = self.possibleActions.index(self.curAction)
        Q = self.Q[self.curState, actionIdx]

        #learn
        self.Q[self.curState, actionIdx] = Q + self.learningRate*(self.reward + np.max(self.Q[self.nextState,]) - self.Q[self.curState, actionIdx])

        return self.Q[self.curState, actionIdx] - Q

    def act(self):
        #select action according to policy; that is, according to the probabilities of our state-action distribution (policy), choose an action
        actionIdx = np.random.choice(np.arange(len(self.possibleActions)), p = self.policy[self.curState,])
        action = self.possibleActions[actionIdx]

        return action

    def calculateAveragePolicyUpdate(self):
        #update the state counter
        self.C[self.curState] += 1

        #the average policy gets updated by the difference in the true policy and the average policy, proportional to the # of times that state has been seen
        self.averagePolicy[self.curState,] += (1/self.C[self.curState])*(self.policy[self.curState,] - self.averagePolicy[self.curState,])

        return self.averagePolicy[self.curState,]

    def calculatePolicyUpdate(self):
        """
        This method should provide working implementation of the policy hill climbing
        approach to update agents' current policy.

        In the end it must return the updated policy of th state where th agent
        currently is.
        """

        #identify optimal and # of suboptimal actions:
        QMax = np.max(self.Q[self.curState,])
        AMaxIdx = np.argwhere(self.Q[self.curState,] == QMax)
        AMaxIdx = [item for sublist in AMaxIdx for item in sublist]
        numSubOpt = len(self.possibleActions) - len(AMaxIdx)

        #retrieve current policy and average policy expected payoffs
        curPolicyPayoff = np.dot(self.policy[self.curState,].T, self.Q[self.curState,])
        avgPolicyPayoff = np.dot(self.policy[self.curState,].T, self.Q[self.curState,])

        #decide the delta to used, based on whether the current policy payoff exceeds the average policy payoff
        if curPolicyPayoff >= avgPolicyPayoff:
            d = self.winDelta
        else:
            d = self.loseDelta

        #update policy for suboptimal actions (decide how much probability to take from suboptimal actions)
        pMoved = 0
        for i in range(len(self.possibleActions)):

            if i not in AMaxIdx:
                pMoved += min(d/numSubOpt, self.policy[self.curState, i])
                self.policy[self.curState,i] -= min(d/numSubOpt, self.policy[self.curState, i])

        #update policy for optimal actions
        for i in range(len(self.possibleActions)):
            if i in AMaxIdx:
                self.policy[self.curState,i] += pMoved/(abs(len(self.possibleActions)-numSubOpt))

        print(self.policy[self.curState,])
        return self.policy[self.curState,]


    def toStateRepresentation(self, state):
        #our dictionary keys are in the form of tuples, so we convert
        #the environment's state representation to tuples

        s = [tuple(l) for l in state[0]]
        s = tuple(s)

        return self.stateRepresentations[s]


    def setState(self, state):
        self.curState = state


    def setLearningRate(self,lr):
        self.learningRate = lr


    def setWinDelta(self, winDelta):
        self.winDelta = winDelta


    def setLoseDelta(self, loseDelta):
        self.loseDelta = loseDelta


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        """
        This function should return a tuple indiciating the losing delta, winning
        delta, and learning rate used at a certain timestep.
        """
        return self.loseDelta, self.winDelta, self.learningRate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=100000)

    args=parser.parse_args()

    numOpponents = args.numOpponents
    numAgents = args.numAgents
    MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

    agents = []
    for i in range(args.numAgents):
        agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()

        while status[0]=="IN_GAME":
            for agent in agents:
                loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
                agent.setLoseDelta(loseDelta)
                agent.setWinDelta(winDelta)
                agent.setLearningRate(learningRate)
            actions = []
            perAgentObs = []
            agentIdx = 0
            for agent in agents:
                obsCopy = deepcopy(observation[agentIdx])
                perAgentObs.append(obsCopy)
                agent.setState(agent.toStateRepresentation(obsCopy))
                actions.append(agent.act())
                agentIdx += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1

            agentIdx = 0
            for agent in agents:
                agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx],
                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agent.learn()
                agent.calculateAveragePolicyUpdate()
                agent.calculatePolicyUpdate()
                agentIdx += 1

            observation = nextObservation
