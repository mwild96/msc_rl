#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import csv

class MonteCarloAgent(Agent):
    def __init__(self, discountFactor, epsilon, initVals=0.0):
        super(MonteCarloAgent, self).__init__()

        #initialize hyperparameters
        self.epsilon = epsilon
        self.epsilonZero = epsilon
        #self.epsilonDecayRate = 0.00075
        self.discountFactor = discountFactor

        #define list of possible states
        self.possibleStates = [(x,y) for x in range(5) for y in range(6)]
        self.possibleStates.append("GOAL")
        self.possibleStates.append("OUT")

        #initialize Q values
        self.Q = np.empty((len(self.possibleStates),len(self.possibleActions)))
        self.Q.fill(initVals)

        #initialize state attribute
        self.curState = None
        self.G = 0 #um
        self.episodeSequence = []
        self.rewardSequence = []

        #issues with this data structure:
        #initialize returns
        self.SAReturns = [ [ [] for _ in range(len(self.possibleActions))] for _ in range(len(self.possibleStates))] #np.zeros(len(self.possibleActions),
        #


    def learn(self):
        """
        This is the most important function you need to implement in this task.
        This function has no input parameters. On the other hand, it should
        return a tuple of two entries. The first entry is the complete Q-value
        table of all states. The second entry is the Q-value estimate after
        update of the states you've encountered in the episode ordered by their
        first time appearance in the episode.

        In general, this function has similar functionality as line 9 - 16 of
        the pseudocode presented in the book
        """

        episodeQ = []

        for i in range(len(self.episodeSequence)-1, -1, -1):

            #for each episode we will learn an estimate of the total rewards
            self.G = self.discountFactor*self.G + self.rewardSequence[i]

            #store the state and action in the episode sequence log
            St = self.episodeSequence[i][0] #THIS NEEDS TO BE A LIST OF LISTS IS THAT OKAY?
            At = self.episodeSequence[i][1]
            Atidx = self.possibleActions.index(At)

            #order Q value estimate by first time appearance
            if self.episodeSequence[i] not in self.episodeSequence[0:i]:
                self.SAReturns[St][Atidx].append(self.G)
                self.Q[St, Atidx] = np.mean(self.SAReturns[St][Atidx])
                episodeQ.append(self.Q[St, Atidx])

            episodeQ = episodeQ[::-1]


        return tuple((self.Q, episodeQ))


    def toStateRepresentation(self, state):

        if type(state[0]) is tuple: #if it is their state representation
            return self.possibleStates.index(state[0])
        elif type(state[0]) is int:
            return self.possibleStates.index(state)
        elif type(state[0]) is str:
            if state == "OUT_OF_BOUNDS":
                return self.possibleStates.index("OUT")
            elif state == "GOAL":
                return self.possibleStates.index("GOAL")


    def setExperience(self, state, action, reward, status, nextState):
        self.episodeSequence.append(tuple((state, action)))
        self.rewardSequence.append(reward)


    def setState(self, state):
        self.curState = state

    def reset(self):
        self.G = 0 #um
        self.episodeSequence = []
        self.rewardSequence = []


    def act(self):
        """
        e-soft policy

        In general, this function has similar functionality as line 7 of the
        pseudocode presented in the book

        "all nongreedy actions are given the minimal probability of selection
        (epsilon/# of actions), and the remaining bulk of the probability
        (1 - epsilon + (epsilon/# of actions)), is given to the greedy action

        """
        maxAtidx = np.argmax(self.Q[self.curState, :])

        prob = [0]*len(self.possibleActions)

        for a in range(len(self.possibleActions)):
            if a != maxAtidx:
                prob[a] = self.epsilon/len(self.possibleActions)
            else:
                prob[a] = 1 - self.epsilon + (self.epsilon/len(self.possibleActions))

        actionIdx = np.random.choice(len(self.possibleActions), p = prob)

        return self.possibleActions[actionIdx]


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        """
        This function should return a tuple indicating the epsilon used at a certain
        timestep. This allows you to schedule the values of your hyperparameters and
        change them mid-training.

        ^^^I don't understand why this should be a tuple...
        """
        if episodeNumber < 4000:
                #in the beginning we will mostly explore
            newEpsilon = 0.95
        else:
             #towards the end we will only exploit
            newEpsilon = 0.0

        #newLearningRate = self.learningRate

        #newEpsilon = self.epsilonZero*np.exp(-self.epsilonDecayRate*episodeNumber)

        return newEpsilon



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=5000)

    args=parser.parse_args()

    #Init Connections to HFO Server
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a Monte-Carlo Agent
    agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 0.8)
    numEpisodes = args.numEpisodes
    numTakenActions = 0
    # Run training Monte Carlo Method
    for episode in range(numEpisodes):
        agent.reset()
        observation = hfoEnv.reset()
        status = 0

        while status==0:
            epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))

            observation = nextObservation

        agent.learn()
