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
import prallel_Pij 

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
        timer = time.time()
        P = prallel_Pij.parallel_Pij(Constants)
        print("Time elapsed with parallel computation: ", time.time() - timer)

        #debug!
        """P_true = file["P"]
        tolerance = 1e-4

        # Differenza assoluta tra P e P_true
        diff = np.abs(P - P_true)

        # Maschera dei punti dove la differenza eccede la tolleranza
        mask = diff > tolerance

        # Trova gli indici (i, j, l) dei punti con differenze eccessive
        i_indices, j_indices, l_indices = np.where(mask)

        if len(i_indices) == 0:
            print("Non ci sono differenze superiori alla tolleranza.")
        else:
            static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)  
            respawn_probability = 1 / (Constants.M * Constants.N - 1)
            print(f"numero di differenze superiori alla tolleranza: {len(i_indices)}")
            print(f"static drones: {static_drones}")
            print(f"goal state: {Constants.GOAL_POS}")
            for idx in range(5):
                i = i_indices[idx]
                j = j_indices[idx]
                l = l_indices[idx]

                # Ricava gli stati di partenza e di arrivo se hai la funzione di decoding
                # Se non l'hai, commenta queste righe e stampa solo gli indici
                start_state = idx2state_vectorized(i)   # (x_drone_start, y_drone_start, x_swan_start, y_swan_start)
                next_state = idx2state_vectorized(j)    # (x_drone_next, y_drone_next, x_swan_next, y_swan_next)

                p_val = P[i, j, l]
                p_true_val = P_true[i, j, l]

                print(f"Diff > {tolerance} per i={i}, j={j}, l={l}")
                print("Stato di partenza (x_drone, y_drone, x_swan, y_swan):", start_state)
                print("Stato di arrivo   (x_drone, y_drone, x_swan, y_swan):", next_state)
                print(f"input: {idx2input(l)}")
                print(f"flow field value for starting state: {Constants.FLOW_FIELD[start_state[0][0],start_state[0][1]]} ")      
                print(f"swan movment: {Swan_movment_to_catch_drone_vectorized(np.array(start_state[0][2]), np.array(start_state[0][3]), np.array(start_state[0][0]), np.array(start_state[0][1]))}")
                print(f"path: {bresenham((start_state[0][0], start_state[0][1]), (next_state[0][0], next_state[0][1]))}")
                print("--- probability ---")              
                print(f"(1-pcurrent)*(1-pswan): {(1 - Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(1- Constants.SWAN_PROB)}")
                print(f"(pcurrent)*(1-pswan): {(Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(1- Constants.SWAN_PROB)}")
                print(f"(1-pcurrent)*(pswan): {(1 - Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(Constants.SWAN_PROB)}")
                print(f"(pcurrent)*(pswan): {(Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(Constants.SWAN_PROB)}")
                print(f"(1-pcurrent)*(1-pswan)*respawn probability: {(1 - Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(1- Constants.SWAN_PROB)*respawn_probability}")
                print(f"(pcurrent)*(1-pswan)*respawn probability: {(Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(1- Constants.SWAN_PROB)*respawn_probability}")
                print(f"(1-pcurrent)*(pswan)*respawn probability: {(1 - Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(Constants.SWAN_PROB)*respawn_probability}")
                print(f"(pcurrent)*(pswan)*respawn probability: {(Constants.CURRENT_PROB[start_state[0][0],start_state[0][1]])*(Constants.SWAN_PROB)*respawn_probability}")
                print(f"P: {p_val}, P_true: {p_true_val}")
                print("---------------------------------------------------")
        """
        
        
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
        [J_opt, u_opt] = solution(P, Q, Constants)
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
