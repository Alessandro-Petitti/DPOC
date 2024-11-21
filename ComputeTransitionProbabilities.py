"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

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

def compute_matrix_Piju(Constants):
    P = np.zeros((Constants.K, Constants.K,Constants.L))
    map_start = state2idx(Constants.START_POS[0],Constants.START_POS[1],Constants)
    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)  # Use a set for quick lookups
    for i in range(Constants.N):
        for j in range(Constants.M):
            for l in range(Constants.L):
                map_i = state2idx(i,j,Constants)
                no_current_i, no_current_j = compute_state_with_input(i,j,l, Constants)
                if 0 <= no_current_i < Constants.N and 0 <= no_current_j < Constants.M:
                    map_j = state2idx(no_current_i,no_current_j,Constants)
                    if tuple([no_current_i, no_current_j]) in static_drones:
                        P[map_i][map_start][l] = 1 - Constants.CURRENT_PROB[no_current_i][no_current_j]
                    else:
                        P[map_i][map_j][l] = 1 - Constants.CURRENT_PROB[no_current_i][no_current_j]
                else:
                    P[map_i][map_start][l] = 1 - Constants.CURRENT_PROB[no_current_i][no_current_j]
                current_i, current_j = compute_state_plus_currents(no_current_i,no_current_j, Constants)
                if 0 <= current_i < Constants.N and 0 <= current_j < Constants.M:
                    map_j = state2idx(current_i,current_j,Constants)
                    if tuple([current_i, current_j]) in static_drones:
                        P[map_i][map_start][l] = Constants.CURRENT_PROB[no_current_i][no_current_j]
                    else:
                        P[map_i][map_j][l] = Constants.CURRENT_PROB[no_current_i][no_current_j]
                elif no_current_i == -1 and no_current_j == -1:
                    continue
                else:
                    P[map_i][map_start][l] = Constants.CURRENT_PROB[no_current_i][no_current_j]
    return P

def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    P = compute_matrix_Piju(Constants)
    print()
    print("K ", Constants.K)
    print("L ", Constants.L)
    print("Result")
    print(P)
    # TODO fill the transition probability matrix P here

    return P
