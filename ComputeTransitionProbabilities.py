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
    # pre allocate
    P = np.zeros((Constants.K, Constants.K,Constants.L)) 
    #initial position as an index
    #map_start = state2idx(Constants.START_POS[0],Constants.START_POS[1],Constants)

    #set of admissible respawn states expressed as indices.
    respawn_indices = generate_respawn_indices(Constants)
    print(f"respawn_indices length: {len(respawn_indices)}")
    respawn_probability = 1/(Constants.M*Constants.N-1)
    # set of static drone positions  (Use a set for quick lookups)
    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)  
    for i in range(Constants.N):# iterate over all x
        for j in range(Constants.M): # iterate over all y
            for i_swan in range(Constants.N): # iterate over all x for the swan 
                for j_swan in range(Constants.M): # iterate over all y for the swan
                    # current state as index
                    map_i = state2idx([i,j,i_swan, j_swan]) 
                    if i == i_swan and j == j_swan:
                            continue
                    for l in range(Constants.L): # iterate over all input
                        print("The count has restarted", map_i, l)
                        #----- no current applied ----------
                        #check where you'd end up WITHOUTH current
                        no_current_i, no_current_j = compute_state_with_input(i,j,l, Constants)
                        #print(f"map_i: {map_i}")

                        #check where the swan ends up if it moves
                        dx, dy = Swan_movment_to_catch_drone(i_swan, j_swan, i, j)
                        moved_swan_x = i_swan + dx
                        moved_swan_y = j_swan + dy
                        
                        #check if you end un inside the map and that the swan is not hitting the drone
                        if 0 <= no_current_i < Constants.N and 0 <= no_current_j < Constants.M:
                            #chek for static collision
                            if not tuple([no_current_i, no_current_j]) in static_drones:
                                #if not hitted, check for swan collision
                                # ---- moving swan ----
                                #if the swan is moving and is not going to hit the drone
                                if moved_swan_x != no_current_i or moved_swan_y != no_current_j:
                                    #if no problem arises, you go to the designated state 
                                    P[map_i,state2idx([no_current_i,no_current_j,moved_swan_x, moved_swan_y]),l] += (1 - Constants.CURRENT_PROB[i][j])* Constants.SWAN_PROB
                                    
                                #the swan is moving and hits the drone
                                else: 
                                    for respawn_index in respawn_indices:
                                        P[map_i, respawn_index, l]  += (1 - Constants.CURRENT_PROB[i][j]) * Constants.SWAN_PROB*respawn_probability
                                        
                                #if the swan is not moving and is not going to hit the drone
                                if (i_swan != no_current_i or j_swan != no_current_j):
                                    P[map_i,state2idx([no_current_i,no_current_j,i_swan, j_swan]),l] += (1 - Constants.CURRENT_PROB[i][j]) *(1- Constants.SWAN_PROB)
                                    
                                #the swan is not moving and hits the drone
                                else:
                                    for respawn_index in respawn_indices:
                                        P[map_i, respawn_index, l] += (1 - Constants.CURRENT_PROB[i][j]) * (1- Constants.SWAN_PROB)*respawn_probability
                                        
                            else:
                                #if you hit a static drone, you go home
                                for respawn_index in respawn_indices:
                                    P[map_i, respawn_index, l]+= (1 - Constants.CURRENT_PROB[i][j])*respawn_probability
                                    
                        else:
                            #If you are outside the map, you go home with probability respawn_probability, 
                            for respawn_index in respawn_indices:
                                P[map_i, respawn_index, l] += (1-Constants.CURRENT_PROB[i][j])*respawn_probability
                                
                        # ––––– apply current –-------
                        #check wherer you'd end up WITH current
                        current_i, current_j = compute_state_plus_currents(no_current_i,no_current_j, Constants)
                        # Genera la linea tra i punti di partenza e arrivo senza corrente
                        path = bresenham((i, j), (current_i, current_j))
                        #check if you end up outside the map and that the swan is not hitting the drone
                        if 0 <= current_i < Constants.N and 0 <= current_j < Constants.M:
                            #check if collision with static drones
                            if not any(tuple(point) in static_drones for point in path):
                                #if the swan is moving and is not going to hit the drone
                                if all(point != (moved_swan_x, moved_swan_y) for point in path):
                                    #if no problem arises, you go to the designated x with probability p_current
                                    P[map_i,state2idx([current_i,current_j,moved_swan_x, moved_swan_y]),l] += (Constants.CURRENT_PROB[i][j]) * Constants.SWAN_PROB
                                    
                                #the swan is moving and hits the drone
                                else: 
                                    for respawn_index in respawn_indices:
                                        P[map_i, respawn_index, l] += (Constants.CURRENT_PROB[i][j]) * Constants.SWAN_PROB*respawn_probability
                                        
                                #if the swan is not moving and is not going to hit the drone
                                if all(point != (i_swan, j_swan) for point in path):
                                    P[map_i,state2idx([current_i,current_j,i_swan, j_swan]),l] += (Constants.CURRENT_PROB[i][j]) *(1- Constants.SWAN_PROB)
                                    
                                #the swan is not moving and hits the drone
                                else:
                                    for respawn_index in respawn_indices:
                                        P[map_i, respawn_index, l] += (Constants.CURRENT_PROB[i][j]) * (1- Constants.SWAN_PROB)*respawn_probability
                                        
                            else:
                                #if hitted by a static drone, you go home
                                for respawn_index in respawn_indices:
                                    P[map_i, respawn_index, l] += Constants.CURRENT_PROB[i][j]*respawn_probability
                                    
                        # if you are outside the map you go home.
                        else:
                            for respawn_index in respawn_indices:
                                P[map_i, respawn_index, l] += respawn_probability*Constants.CURRENT_PROB[i][j]
                                
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


    return P
