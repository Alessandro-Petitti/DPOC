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
from ComputeTransitionProbabilities import *

def compute_expected_stage_cost(Constants):
    """Computes the expected stage cost for the given problem.
    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.

    Args:
        Constants: The Constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L)
    """
    Q = np.ones((Constants.K, Constants.L)) * np.inf

    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)
    respawn_indices = generate_respawn_indices(Constants)
    P  = compute_transition_probabilities(Constants)

    for i_state in range(Constants.K):
        for i_input in range(Constants.L):
            input_vec = idx2input(i_input)
            state = idx2state(i_state)

            # Invalid states (drone overlaps swan or static drone)
            if (state[0] == state[2] and state[1] == state[3]) or ((state[0], state[1]) in static_drones) or (state[0] == Constants.GOAL_POS[0] and state[1] == Constants.GOAL_POS[1]):
                Q[i_state, i_input] = 0
            else:
                # Calculate the probability of reaching a respawn state
        
                h_value = sum(P[i_state, respawn_indices, i_input])
                
                # Calculate the expected stage cost
                #h_value = h_with_disturbances_expected(i_state, i_input, Constants) + respawn_probability
                Q[i_state, i_input] = (Constants.TIME_COST +
                                       Constants.THRUSTER_COST * (abs(input_vec[0]) + abs(input_vec[1])) +
                                       h_value * Constants.DRONE_COST)

    
    return Q

        