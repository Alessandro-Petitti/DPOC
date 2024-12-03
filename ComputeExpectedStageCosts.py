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

                excess = 0
                i = int(state[0])
                j = int(state[1])
                i_swan = int(state[2])
                j_swan = int(state[3])
                no_current_i, no_current_j = compute_state_with_input(i,j,i_input, Constants)
                if no_current_i == Constants.START_POS[0] and no_current_j == Constants.START_POS[1]:
                    #check where the swan ends up if it moves
                    dx, dy = Swan_movment_to_catch_drone(i_swan, j_swan, i, j)
                    moved_swan_x = i_swan + dx
                    moved_swan_y = j_swan + dy
                    
                    #check if you end un inside the map and that the swan is not hitting the drone
                    if 0 <= no_current_i < Constants.M and 0 <= no_current_j < Constants.N:
                        #chek for static collision
                        if not tuple([no_current_i, no_current_j]) in static_drones:
                            #if not hitted, check for swan collision
                            # ---- moving swan ----
                            #if the swan is moving and is not going to hit the drone
                            if moved_swan_x != no_current_i or moved_swan_y != no_current_j:
                                #if no problem arises, you go to the designated state 
                                excess += (1 - Constants.CURRENT_PROB[i][j])* Constants.SWAN_PROB
                                
                            #if the swan is not moving and is not going to hit the drone
                            if (i_swan != no_current_i or j_swan != no_current_j):
                                excess += (1 - Constants.CURRENT_PROB[i][j]) *(1- Constants.SWAN_PROB)
                            
                            
                        
                # ––––– apply current –-------
                #check wherer you'd end up WITH current
                current_i_val, current_j_val = compute_state_plus_currents(i,j, Constants)
                current_input_state_i ,current_input_state_j = compute_state_with_input(current_i_val,current_j_val,i_input, Constants)
                if current_input_state_i == Constants.START_POS[0] and current_input_state_j == Constants.START_POS[1]:
                    dx, dy = Swan_movment_to_catch_drone(i_swan, j_swan, i, j)
                    moved_swan_x = i_swan + dx
                    moved_swan_y = j_swan + dy
                    # Genera la linea tra i punti di partenza e arrivo senza corrente
                    path = bresenham((i, j), (current_input_state_i, current_input_state_j))
                    # remove first element of path because is the starting position if the path is logner than 1
                    if len(path) > 1:
                        path = path[1:]
                    #check if you end up outside the map and that the swan is not hitting the drone
                    if 0 <= current_input_state_i < Constants.M and 0 <= current_input_state_j < Constants.N:
                        #check if collision with static drones
                        if not any(tuple(point) in static_drones for point in path):
                            #if the swan is moving and is not going to hit the drone
                            if current_input_state_i!=moved_swan_x or current_input_state_j != moved_swan_y:
                                #if no problem arises, you go to the designated x with probability p_current
                                excess += (Constants.CURRENT_PROB[i][j]) * Constants.SWAN_PROB
                                
                            #the swan is moving and hits the drone
                            #if the swan is not moving and is not going to hit the drone
                            if current_input_state_i != i_swan or current_input_state_j != j_swan:
                                excess += (Constants.CURRENT_PROB[i][j]) *(1- Constants.SWAN_PROB)

                h_value -= excess                      
                # if you are outside the map you go home.
                # Calculate the expected stage cost
                #h_value = h_with_disturbances_expected(i_state, i_input, Constants) + respawn_probability
                Q[i_state, i_input] = (Constants.TIME_COST +
                                       Constants.THRUSTER_COST * (abs(input_vec[0]) + abs(input_vec[1])) +
                                       h_value * Constants.DRONE_COST)

    
    return Q

        