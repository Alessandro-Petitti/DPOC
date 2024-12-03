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
    
        if not np.allclose(P, file["P"], rtol=1e-4, atol=1e-7):
            print("Wrong transition probabilities")
            passed = False
        else:
            print("Correct transition probabilities")
        Q_true = file["Q"]
      
        if not np.allclose(Q, file["Q"], rtol=1e-4, atol=1e-7):
            respawn_indices = generate_respawn_indices(Constants)
            print("---- respawn states ----")
            for i in respawn_indices:
                print(idx2state(i))
            print("Wrong expected stage costs")
            passed = False
            # Get the indices where Q differs from Q_true
            idx = np.where(np.abs(Q - Q_true) > 1e-4)
            print("---- mismatches ----")
            print(f"Number of mismatches: {len(idx[0])}")
            print(f"cost for time: {Constants.TIME_COST}")
            print(f"cost for thruster: {Constants.THRUSTER_COST}")
            print(f"cost for drone: {Constants.DRONE_COST}")
            print("-----------------")
            # Convert the first few indices to states and inputs
            static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)  
            for i in range(min(1, len(idx[0]))):
                i =0   
                state = idx2state(idx[0][i]) # Convert state index to state
                i_state = state2idx(state)
                input_ = idx2input(idx[1][i])  # Convert input index to input
                i_input = input2idx(input_[0], input_[1])
                print(sum(P[int(i_state), respawn_indices, int(i_input) ]))
                print(f"Mismatch {i+1}: State: {state}, Input: {input_}, currents value {Constants.FLOW_FIELD[int(state[0]), int(state[1])]}")
                print(f"Expected Q: {Q_true[idx[0][i], idx[1][i]]}, Computed Q: {Q[idx[0][i], idx[1][i]]}")
                print(f"sum of probability of gettng a new drone from state {state} and input {input_}, is: {sum(P_true[idx[0][i], respawn_indices, idx[1][i]])}")            
        else:
            print("Correct expected stage costs")

        # normal solution
        [J_opt, u_opt] = solution(P, Q, Constants)
        if not np.allclose(J_opt, file["J"], rtol=1e-4, atol=1e-7):
            print("Wrong optimal cost")
            idx = np.where(np.abs(J_opt - file["J"]) > 1e-4)
            print("---- mismatches ----")
            print(f"Number of mismatches: {len(idx[0])}")
            passed = False
        else:
            print("Correct optimal cost")
        

    print("-----------")
