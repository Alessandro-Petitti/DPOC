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
    count_over_05 = 0
    count_under_05 = 0
    for i_state in range(Constants.K):
        for i_input in range(Constants.L):
            input_vec = idx2input(i_input)
            state = idx2state(i_state)

            # Invalid states (drone overlaps swan or static drone)
            if (state[0] == state[2] and state[1] == state[3]) or (state[0], state[1]) in static_drones:
                Q[i_state, i_input] = 0
            else:
                # Calculate the probability of reaching a respawn state
        
                h_value = sum(P[i_state, respawn_indices, i_input])
               # print(f"h_value: {h_value}")
                if h_value >0.5:
                    h_value = 1
                    count_over_05 += 1  
                else:
                    h_value = 0
                    count_under_05 += 1
                # Calculate the expected stage cost
                #h_value = h_with_disturbances_expected(i_state, i_input, Constants) + respawn_probability
                Q[i_state, i_input] = (Constants.TIME_COST +
                                       Constants.THRUSTER_COST * (abs(input_vec[0]) + abs(input_vec[1])) +
                                       h_value * Constants.DRONE_COST)

    print(f"total over 0.5: {count_over_05}, total under 0.5: {count_under_05},total h_values: {np.shape(Q)}")
    return Q

        