#!/usr/bin/env python3
# encoding utf-8


import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
import itertools

class JointQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
        super(JointQLearningAgent, self).__init__()

        #initialize hyperparameters
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.numTeammates = numTeammates

        #define list of possible states & state representation dictionary
        self.possibleStates = list(itertools.product([(x,y) for x in range(5) for y in range(5)], repeat = 2))
        self.possibleStates.append((('G',),))
        self.possibleStates.append((('O',),))
        self.possibleStates.append((('O',), ('U',)))
        self.possibleStates.append((('G',), ('O',)))
        self.stateRepresentations = dict(zip(self.possibleStates,list(range(len(self.possibleStates)))))

        #define list of possible opponent action tuples (tuples are not necessary if there is only one teammate)
        if numTeammates > 1:
            self.possibleOppActions = list(itertools.product(self.possibleActions, repeat = numTeammates))
        else:
            self.possibleOppActions = self.possibleActions

        #define list of possible joint actions
        self.possibleJointActions = list(itertools.product(self.possibleActions, repeat = numTeammates + 1))

        #initialize Q
        self.Q = np.empty((len(self.possibleStates),len(self.possibleJointActions)))
        self.Q.fill(initVals)

        #initialize V
        self.V = [0]*len(self.possibleStates)

        #define counters of opponents' state log and state-action pair log
        self.oppCLog =  np.zeros((len(self.possibleStates), len(self.possibleOppActions)))
        self.oppNLog = [0]*len(self.possibleStates)

        #initialize state-action
        self.curState = None
        self.action = None
        self.oppAction = None
        self.nextState = None
        self.reward = None

    def setExperience(self, state, action, oppoActions, reward, status, nextState):
        self.setState(state)
        self.action = action
        self.oppAction = oppoActions[0]
        self.nextState = nextState
        self.reward = reward

    def learn(self):

        #find Vsp (value of next state)
        #the value of the next state will be the maximum expected value over actions in that state
        actionValues = []
        for i in range(len(self.possibleActions)):
            #so we'll loop over each action
            actionValue = 0

            for j in range(len(self.possibleOppActions)):
                #and then within that action we'll look at the value of each of the opponent's possible actions
                if self.numTeammates > 1:
                    a = (self.possibleActions[i],) + self.possibleOppActions[j]
                else:
                    a = (self.possibleActions[i],self.possibleOppActions[j])

                aIdx = self.possibleJointActions.index(a)

                if self.oppNLog[self.nextState] != 0:
                    n = self.oppNLog[self.nextState]
                    C = self.oppCLog[self.nextState,j]
                else:
                    n = 1
                    C = 1
                Q = self.Q[self.nextState, aIdx]

                #add the value of this opponents' action to total value of the agent's action
                actionValue += (1/n)*C*Q

            #each sum over opponents' actions is stored
            actionValues.append(actionValue)

        #so that we can choose the highest sum (expected value) to be our value of the new state
        Vsp = max(actionValues)

        #combine agent's action and opponents' actions to get the joint action
        if self.numTeammates > 1:
            a = (self.action,) + self.oppAction
        else:
            a = (self.action,self.oppAction)

        aIdx = self.possibleJointActions.index(a)

        #store current Q(s,a) value for the calculation of the change later
        Q = self.Q[self.curState,aIdx]

        #learn
        self.Q[self.curState,aIdx] = (1-self.learningRate)*self.Q[self.curState,aIdx] +self.learningRate*(self.reward + self.discountFactor*Vsp)

        #update opponent state,action and state counters
        self.oppCLog[self.nextState, self.possibleOppActions.index(self.oppAction)] += 1 #do we need to do this for EVERY opponent
        #do we need to store one for EVERY opponent?
        self.oppNLog[self.curState] += 1

        #return the change in Q
        return self.Q[self.curState,aIdx]-Q


    def act(self):

        #"decide" whether we will exploit (maximum) or explore (random)
        c = np.random.choice(2,1,p=[self.epsilon, 1-self.epsilon])

        if c == 0:
            #pick a random action
            idx = np.random.randint(len(self.possibleActions))

        else:
            #find and choose maximum action

            #again, max action is the action which maximizes the expected value of the state (the sum over the opponents' possible actions)
            actionValues = []
            for i in range(len(self.possibleActions)):

                actionValue = 0

                for j in range(len(self.possibleOppActions)):

                    if self.numTeammates > 1:
                        a = (self.possibleActions[i],) + self.possibleOppActions[j]
                    else:
                        a = (self.possibleActions[i],self.possibleOppActions[j])

                    aIdx = self.possibleJointActions.index(a)

                    C = self.oppCLog[self.curState,j]

                    if self.oppNLog[self.curState] != 0:
                        n = self.oppNLog[self.curState]
                    else:
                        n = 1

                    Q = self.Q[self.curState, aIdx]

                    actionValue += (1/n)*C.T*Q

                actionValues.append(actionValue)

                idx = np.argmax(actionValue)

        return  self.possibleActions[idx]


    def setEpsilon(self, epsilon) :
        self.epsilon = epsilon


    def setLearningRate(self, learningRate) :
        self.learningRate = learningRate


    def setState(self, state):
        self.curState = state


    def toStateRepresentation(self, rawState):

        s1 = [tuple(l) for l in rawState[0]]
        s2 = [tuple(l) for l in rawState[1]]
        s = tuple((s1[0], s2[0]))

        return self.stateRepresentations[s]


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        
        newLearningRate = self.learningRate

        if episodeNumber < 40000:
            #in the beginning we will focus on exploring
            newEpsilon = 0.95
        else:
            #we can exploit towards the end
            newEpsilon = 0

        return (newLearningRate, newEpsilon)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type = int, default = 1)
    parser.add_argument('--numAgents', type = int, default = 2)
    parser.add_argument('--numEpisodes', type = int, default = 50000)

    args = parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
    agents = []
    numAgents = args.numAgents
    numEpisodes = args.numEpisodes
    for i in range(numAgents):
        agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
        agents.append(agent)

    numEpisodes = numEpisodes
    numTakenActions = 0

    for episode in range(numEpisodes):
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()

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
                agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())


            #HOW ARE WE GOING TO PASS IN INFORMATION ABOUT THE OTHER AGENTS?
            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1

            for agentIdx in range(args.numAgents):
                oppoActions = actions.copy()
                del oppoActions[agentIdx]
                agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions,
                    reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation
