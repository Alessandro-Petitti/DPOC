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
import time 

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
        print(f"Map size: {Constants.M}x{Constants.N}")
        print(f"Number of drones: {Constants.N_DRONES}")
        print(f"Cost for each drone: {Constants.DRONE_COST}, cost for trhuster: {Constants.THRUSTER_COST}, cost for time: {Constants.TIME_COST}")
        print(f"swan probability {Constants.SWAN_PROB}")
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
            print("Wrong expected stage costs")
        else:
            print("Correct expected stage costs")

        
        timer = time.time()
        [J_opt, u_opt] = solution(P, Q, Constants, method="value_iteration")
        print("Time elapsed: ", time.time() - timer)
        if not np.allclose(J_opt, file["J"], rtol=1e-4, atol=1e-7):
            print("Wrong optimal cost")
            idx = np.where(np.abs(J_opt - file["J"]) > 1e-4)
            print("---- mismatches ----")
            print(f"Number of mismatches: {len(idx[0])}")
            passed = False
        else:
            print("Correct optimal cost")
        

    print("-----------")
