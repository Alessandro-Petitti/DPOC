"""
 test.py

 Python script implementing test cases for debugging.

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

import pickle

import numpy as np
from ComputeExpectedStageCosts import compute_expected_stage_cost
from ComputeTransitionProbabilities import compute_transition_probabilities
from Constants import Constants
from Solver import solution
from utils import *

if __name__ == "__main__":
    n_tests = 4
    for i in range(n_tests):
        print("-----------")
        print("Test " + str(i))
        with open("tests/test" + str(i) + ".pkl", "rb") as f:
            loaded_constants = pickle.load(f)
            for attr_name, attr_value in loaded_constants.items():
                if hasattr(Constants, attr_name):
                    setattr(Constants, attr_name, attr_value)

        file = np.load("tests/test" + str(i) + ".npz")
        # Begin tests
        P = compute_transition_probabilities(Constants)
        
        if not np.all(
            np.logical_or(np.isclose(P.sum(axis=1), 1), np.isclose(P.sum(axis=1), 0))
        ):
            print(
                "[ERROR] Transition probabilities do not sum up to 1 or 0 along axis 1!"
            )

        Q = compute_expected_stage_cost(Constants)
        passed = True
        P_true = file["P"]
        # Trova gli indici in cui le due matrici non sono uguali
        #different_indices = np.where(~np.isclose(P, P_true, rtol=1e-4, atol=1e-7))
        different_indices = np.where((P_true == 0) & (P != 0))

        print(f"number of different indices: {np.shape(different_indices)}")
        
        respawn_probability = 1/(Constants.M*Constants.N-1)
        # Trasforma gli indici in una lista di tuple
        list_of_indices = list(zip(different_indices[0], different_indices[1], different_indices[2]))
        #list_of_indices_tot = list(zip(different_indices_tot[0], different_indices_tot[1], different_indices_tot[2]))
        i = 0
        static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)
        starting_state = idx2state(list_of_indices[i][0])
        ending_state = idx2state(list_of_indices[i][1])
        print(f"index for input: {list_of_indices[i][2]}")
        input = idx2input(list_of_indices[i][2])
        dx, dy = Swan_movment_to_catch_drone(starting_state[2], starting_state[3], starting_state[0], starting_state[1])
        moved_swan_x = starting_state[2] + dx
        moved_swan_y = starting_state[3]+ dy
        #get the shape and the first 5 different indices
        print(f"index of arrival state: {list_of_indices[i][1]}")
        print(f"shape: {np.shape(list_of_indices)}")
        print(f"first 5 different indices: {list_of_indices[:5]}")
        #get the state corresponding to the first 5 different indices, both i,j and also the input values:
        #print stati drone positions
        print(f"drone positions: {Constants.DRONE_POS}")
        print(f"state di partenza: {starting_state}, stato di arrivo: {ending_state}, input: {input}")
        path_input = bresenham((int(starting_state[0]),int(starting_state[1])),(int(starting_state[0]+input[0]),int(starting_state[1]+input[1])))
        print(f"path from start to end with only input: {path_input}")
        print(f"is the path due input moving the drone onto static ones?",any(tuple(point) in static_drones for point in path_input))
        #get the ture probability of this scenario:
        print(f"true probability: {P_true[list_of_indices[i][0],list_of_indices[i][1],list_of_indices[i][2]]}")
        #get the computed probability of this scenario:
        print(f"computed probability: {P[list_of_indices[i][0],list_of_indices[i][1],list_of_indices[i][2]]}")
        #print the current values for initial sstate
        print(f"current values for initial state: {Constants.FLOW_FIELD[int(starting_state[0]),int(starting_state[1])]}")
        n_x,n_y = compute_state_plus_currents(int(starting_state[0]),int(starting_state[1]), Constants)
        no_curr_i,no_curr_j = compute_state_with_input(int(starting_state[0]),int(starting_state[1]),list_of_indices[i][2], Constants)
        #print(f"is the input and current moving the drone onto static ones?",tuple([no_curr_i, no_curr_j]) in static_drones)
        print(f"new state with input ( no swan movement ): {no_curr_i,no_curr_j,starting_state[2],starting_state[3]}")   
        print(f"new state with currend and input ( no swan movement ): {n_x+input[0],n_y+input[1],starting_state[2],starting_state[3]}")
        #print the moved swan
        print(f"moved swan: {moved_swan_x,moved_swan_y}")
        #print the path from initial state to final state plus current effect
        print(f"path from initial state to final state plus current effect: {bresenham((int(starting_state[0]),int(starting_state[1])),(int(n_x+input[0]),int(n_y+input[1])))}")
        #print respawn probability
        print(f"(p_current)*(p_swan): {(Constants.CURRENT_PROB[4][1])*(Constants.SWAN_PROB)}")
        
        print("--------------------")

        #calculate the probability of state = starting_state[0],int(starting_state[1]),moved_swan_x, moved_swan_y
        idx_moved_swan = state2idx([int(starting_state[0]),int(starting_state[1]),int(moved_swan_x), int(moved_swan_y)])
        movd_swan_state_probabilty = P_true[list_of_indices[i][0],idx_moved_swan,list_of_indices[i][2]]
        our_moved_swan_state_probability = P[list_of_indices[i][0],idx_moved_swan,list_of_indices[i][2]]
        print(f"going from initial state to ", [int(starting_state[0]),int(starting_state[1]),int(moved_swan_x), int(moved_swan_y)])
        print(f"probability of moved swan state TRUE: {movd_swan_state_probabilty}")
        print(f"our probability of moved swan state: {our_moved_swan_state_probability}")
        print("--------------------")
        #calculate the probability of state = [int(starting_state[0])+Constants.FLOW_FIELD[int(starting_state[0]),int(starting_state[1])+Constants.FLOW_FIELD[int(starting_state[2]),
        # int(starting_state[2]),int(starting_state[3]) ]
        idx_current_state = state2idx([n_x+input[0],n_y+input[1],int(starting_state[2]), int(starting_state[3])])
        current_state_probabilty = P_true[list_of_indices[i][0],idx_current_state,list_of_indices[i][2]]
        our_current_state_probability = P[list_of_indices[i][0],idx_current_state,list_of_indices[i][2]]
        print(f"going from initial state to ", [n_x+input[0],n_y+input[1],int(starting_state[2]), int(starting_state[3])])
        print(f"probability of current state TRUE: {current_state_probabilty}")
        print(f"our probability of current state: {our_current_state_probability}")
        print("--------------------")
        #calculate the probability of state = [int(starting_state[0])+Constants.FLOW_FIELD[int(starting_state[0]),int(starting_state[1])+Constants.FLOW_FIELD[int(starting_state[2]),
        swan_and_current_state = state2idx([n_x+input[0],n_y+input[1],int(moved_swan_x), int(moved_swan_y)])
        current_state_probabilty = P_true[list_of_indices[i][0],swan_and_current_state,list_of_indices[i][2]]
        our_current_state_probability = P[list_of_indices[i][0],swan_and_current_state,list_of_indices[i][2]]
        print(f"going from initial",starting_state,"state to ", idx2state(swan_and_current_state))
        print(f"probability of swand and current state TRUE: {current_state_probabilty}")
        print(f"our probability of swand and current state: {our_current_state_probability}")
        break

        
        if not np.allclose(P, file["P"], rtol=1e-4, atol=1e-7):
            print("Wrong transition probabilities")
            passed = False
        else:
            print("Correct transition probabilities")

        if not np.allclose(Q, file["Q"], rtol=1e-4, atol=1e-7):
            print("Wrong expected stage costs")
            passed = False
        else:
            print("Correct expected stage costs")

        # normal solution
        [J_opt, u_opt] = solution(P, Q, Constants)
        if not np.allclose(J_opt, file["J"], rtol=1e-4, atol=1e-7):
            print("Wrong optimal cost")
            passed = False
        else:
            print("Correct optimal cost")

    print("-----------")
