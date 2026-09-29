#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import csv
import copy

class SARSAAgent(Agent):

    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(SARSAAgent, self).__init__()

        #set hyperparameters
        self.learningRate = learningRate
        self.learningRateZero = learningRate
        #self.learningRateDecay = 0.1
        self.epsilon = epsilon
        self.epsilonZero = epsilon
        #self.epsilonDecay = 0.0012
        self.discountFactor = discountFactor

        #define list of possible states
        self.possibleStates = [(x,y) for x in range(5) for y in range(6)]
        self.possibleStates.append("GOAL")
        self.possibleStates.append("OUT")
        #self.stateRepresentations = dict(zip(list(range(len(self.possibleStates))), self.possibleStates))

        #initialize Q
        self.Q = np.empty((len(self.possibleStates), len(self.possibleActions)))
        self.Q.fill(initVals)

        #initialize state-action
        #self.lastState = None
        #self.lastNextState = None
        self.state = None
        self.nextState = None#we're never using this though...
        #self.lastAction = None
        self.action = None
        #self.nextAction = None
        self.reward = None
        #self.lastReward = None


    def learn(self):
        """
        This is the most important function you need to implement in this task.
        This function has no input parameters. On the other hand, this method
        must return a single scalar that specifies the change (value after
        update subtracted by value before update) in updated state-action value
        after you've trained your agents using SARSA's update. This function
        will be used in the automarker to compare the correctness of your
        implementation against the solution. This function has the same functionality
        as line 9 in the books' SARSA pseudocode
        """

        lastActionIdx = self.possibleActions.index(self.lastAction)
        Q = self.Q[self.lastState, lastActionIdx].copy()

        if self.action == None:

            self.Q[self.lastState, lastActionIdx] += self.learningRate*(self.lastReward-self.Q[self.lastState, lastActionIdx])

        else:

            actionIdx = self.possibleActions.index(self.action)

            self.Q[self.lastState, lastActionIdx] += self.learningRate*(self.lastReward + self.epsilon*self.Q[self.state, actionIdx]- self.Q[self.lastState, lastActionIdx])

        return self.Q[self.lastState, lastActionIdx] - Q


    def act(self):
        """
        This function will be used to choose the actions that your agents will
        use when faced with a state. It should only return the action
        that should be taken by the agent at the current state.

        """
        c = np.random.choice(2,1,p=[self.epsilon, 1-self.epsilon])

        if c == 0:
            #pick a random action
            idx = np.random.randint(len(self.possibleActions))

        else:
            #choose the action with the maximum Q(s,a) value
            idx = np.argmax(self.Q[self.curState,:])

        action = self.possibleActions[idx]

        return action

    def setState(self, state):
        """
        This function will be used to provide the agents you're controlling
        with the current state information. It will receive the state
        representation from the environment as an input. On the other hand,
        this does not need to output anything.
        """
        self.curState = state


    def setExperience(self, state, action, reward, status, nextState):
        """
        Once an agent executes an action, it will receive the rewards, status,
        and next states resulting from that action. Use this method to set
        these data to prepare your agent to learn using the SARSA update.
        This function has the same functionality as line 10 in the books'
        SARSA pseudocode:
            s <- s'; A <- A'
        """

        print(self.reward)
        self.lastState = self.state
        self.lastAction = self.action
        self.lastReward = self.reward
        self.lastNextState = self.nextState

        print(self.reward)
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = state


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        """
        This function should return a tuple indicating the learning rate
        and epsilon used at a certain timestep. This allows you to schedule the
        values of your hyperparameters and change it midway of training.
        """

        if episodeNumber < 4000:
            #in the beginning we will mostly explore
            newEpsilon = 0.95
        else:
             #towards the end we will only exploit
            newEpsilon = 0.0

        newLearningRate = self.learningRate

        #newEpsilon = self.epsilonZero*np.exp(-self.epsilonDecay*episodeNumber)
        #newLearningRate = self.learningRateZero*np.exp(-self.learningRateDecay*episodeNumber)

        return (newLearningRate, newEpsilon)


    def toStateRepresentation(self, state):
        """
        You might want to use a different representation compared to the ones
        provided by the environment. This will provide a problem to the automarker.
        Therefore, you should implement a function that maps the raw state
        representation into the the state representation that you are using
        in your implementation. This function will receive a state and outputs
        it's value under the representations that you are using in your implementations.
        """

        if type(state[0]) is tuple:

             return self.possibleStates.index(state[0])

        elif type(state[0]) is int:

             return self.possibleStates.index(state)

        elif type(state[0]) is str:

            if state == "OUT_OF_BOUNDS":
                    return self.possibleStates.index("OUT")
            elif state == "GOAL":
                 return self.possibleStates.index("GOAL")

        #return self.stateRepresentations[state]


    def reset(self):
        self.curState = None
        self.nextState = None
        self.curAction = None
        self.nextAction = None
        self.reward = None

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=5000)

    args=parser.parse_args()

    numEpisodes = args.numEpisodes
    # Initialize connection to the HFO environment using HFOAttackingPlayer
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a SARSA Agent
    agent = SARSAAgent(0.1, 0.99, 0.8)

    # Run training using SARSA
    numTakenActions = 0
    for episode in range(numEpisodes):
        agent.reset()
        status = 0

        observation = hfoEnv.reset()
        nextObservation = None
        epsStart = True

        while status==0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()

            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)

            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward,
                                status, agent.toStateRepresentation(nextObservation))
            if not epsStart :
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
        agent.learn()
