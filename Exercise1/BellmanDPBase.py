#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:44:24 2019

@author: s1834310
"""

from MDP import MDP
import numpy as np

class BellmanDPSolver(object):
    def __init__(self,discountRate):
        self.MDP = MDP()
        self.discountRate = discountRate
        self.policy = len(self.MDP.S)*[""]
        self.initVs()

    def initVs(self):
        self.V = len(self.MDP.S)*[0]

    def BellmanUpdate(self):

        #for every state
        for i in range(len(self.MDP.S)):
            s = self.MDP.S[i]
            vaMat = np.zeros(len(self.MDP.A))
            #create empty array to store values for each action

            #look at all possible actions for that state
            for j in range(len(self.MDP.A)):
                a = self.MDP.A[j]

                #and look at all possible next states given that action
                #(so that we can get the expected value of that action)

                nextStates = list(self.MDP.probNextStates(s, a).keys())

                for k in range(len(nextStates)):

                    sPrime = nextStates[k] #state representation
                    sPrimeIdx = self.MDP.S.index(sPrime)
                    p = self.MDP.probNextStates(s, a)[sPrime]
                    r = self.MDP.getRewards(s, a, sPrime)
                    vsPrime = self.V[sPrimeIdx]

                    vaMat[j] += p*(r + self.discountRate*vsPrime)
                    #we're going to add the values of all future states times their probabilities
                    #to the running total value of that action

            self.policy[i] = self.MDP.A[np.argmax(vaMat)]
            self.V[i] = np.max(vaMat)

        polDict = dict(zip(self.MDP.S, self.policy))
        valDict = dict(zip(self.MDP.S,self.V))
        return((valDict, polDict))


if __name__ == '__main__':
    solution = BellmanDPSolver(discountRate = 0.9)
    for i in range(20000):
        values, policy = solution.BellmanUpdate()
        #print("Values : ", values)
        #print("Policy : ", policy)
    print("Values : ", values)
    print("Policy : ", policy)
