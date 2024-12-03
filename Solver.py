"""
 Solver.py

 Python function template to solve the stochastic
 shortest path problem.

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
from scipy.optimize import linprog

def solution(P, Q, Constants):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming;
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        Q  (np.array): A (K x L)-matrix containing the expected stage costs of all states
                       in the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP
        np.array: The optimal control policy for the stochastic SPP

    """

    J_opt = np.zeros(Constants.K)
    u_opt = np.zeros(Constants.K)

    # TODO implement Value Iteration, Policy Iteration,
    #      Linear Programming or a combination of these

    # Parametri di esempio

    # Definire c come vettore di -1
    state_size = Constants.K
    # for it in range(1000):
    #     update = 0
    #     for i in range(state_size):
    #         temp = np.min(Q[i, :] + np.dot(P[i, :, :].T, J_opt))
    #         diff = temp - J_opt[i]
    #         if diff > update:
    #             update = diff
    #         J_opt[i] = temp
    #     if update < 1e-4:
    #         break   
    # for i in range(state_size):        
    #     u_opt[i] = np.argmin(Q[i, :] + np.dot(P[i, :, :].T, J_opt))



    c = -np.ones(state_size)

    # Creazione della matrice A
    A = []
    for u in range(9):
        A_u = np.eye(state_size) - P[:, :, u]
        A.append(A_u)
    A = np.vstack(A)
    # Definizione del vettore b
    b = Q.flatten(order = "F")
    # Risolvi il problema di LP
    result = linprog(c, A_ub=A, b_ub=b, method='highs')

    if result.success:
        J_opt = result.x
        u_opt = np.zeros(state_size, dtype=int)  # Array per memorizzare l'azione ottima per ciascuno stato
        for i in range(state_size):
            min_cost = float('inf')
            best_action = None
            # Per ciascuna azione, calcola il costo e scegli l'azione con il costo minore
            for u in range(9):
                cost = Q[i, u] + np.dot(P[i, :, u], J_opt)
                if cost < min_cost:
                    min_cost = cost
                    best_action = u
                    u_opt[i] = best_action
        #print("Soluzione ottima trovata!")
    else:
        print("Ottimizzazione fallita:", result.message)



    return J_opt, u_opt
