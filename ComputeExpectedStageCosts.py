"""
 ComputeExpectedStageCosts.py

 Python function template to compute the expected stage cost.

 Dynamic Programming and Optimal Control
 Fall 2024
 Programming Exercise

 Contact: Antonio Terpin aterpin@ethz.ch

 Authors: Maximilian Stralz, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np
from utils import *


def compute_expected_stage_cost(Constants):
    """Computes the expected stage cost for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L)
    """
    Q = np.ones((Constants.K, Constants.L)) * np.inf
    for iState in range(0,Constants.K):
        for iInput in range(0,Constants.L):
            input_vec = idx2input(iInput)
            #print(f"State: {iState}, Input: {iInput}, Input_vec: {input_vec}")
            #print(f"h_fun: {h_fun(iState)}")
            state = idx2state(iState)
            #check if state is not valid i.e.drone is the same position of the swan:
            if state[0] == state[2] and state[1] == state[3]:
                Q[iState,iInput] = 0
            else:
                Q[iState,iInput] = Constants.TIME_COST +Constants.THRUSTER_COST*(np.abs(input_vec[0])+np.abs(input_vec[1]))+h_fun(iState,iInput)*Constants.DRONE_COST
        
    
    return Q
