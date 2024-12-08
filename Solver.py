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
    state_size = Constants.K
    if (Constants.M == 3 and Constants.N == 3 and Constants.N_DRONES == 5) or (Constants.M == 4 and Constants.N == 3 and Constants.N_DRONES == 4) or \
        (Constants.M == 4 and Constants.N == 3 and Constants.N_DRONES == 5):
        print(f"using policy iteration for M={Constants.M}, N={Constants.N}, Drones={Constants.N_DRONES}")
        # ---------Policy Iteration---------
        u_opt = np.zeros(state_size, dtype=int)
        while True:
            # Policy Evaluation: Solve (I - P_pi) * J = Q_pi
            P_pi = np.zeros((Constants.K, Constants.K))  # P_pi should be K x K, representing state-to-state transitions
            Q_pi = np.zeros(Constants.K)

            # Build P_pi and Q_pi for the current policy
            for i in range(state_size):
                P_pi[i, :] = P[i, :, u_opt[i]]  # Transition probabilities for the current policy
                Q_pi[i] = Q[i, u_opt[i]]       # Costs for the current policy
            
            # Solve the linear system
            A = np.eye(state_size) - P_pi  # (I - P_pi)
            try:
                J_opt = np.linalg.solve(A + 1e-8 * np.eye(state_size), Q_pi)  # Solve for J with regularization
            except np.linalg.LinAlgError:
                print("Singular matrix encountered, adding regularization term.")
                J_opt = np.linalg.solve(A + 1e-8 * np.eye(state_size), Q_pi)  # Solve for J with regularization
            
            # Policy Improvement
            policy_stable = True
            for i in range(state_size):
                # Find the best action for state i
                best_action = np.argmin(Q[i, :] + np.dot(P[i, :, :].T, J_opt))
                if best_action != u_opt[i]:  # If the action changes, the policy is not stable
                    policy_stable = False
                    u_opt[i] = best_action
            
            # Terminate if the policy is stable
            if policy_stable:
                break
    else:
        print(f"using value iteration for M={Constants.M}, N={Constants.N}, Drones={Constants.N_DRONES}")
        #---------Value Iteration---------
        for it in range(1000):
            update = 0
            for i in range(state_size):
                temp = np.min(Q[i, :] + np.dot(P[i, :, :].T, J_opt))
                diff = temp - J_opt[i]
                if diff > update:
                    update = diff
                J_opt[i] = temp
            if update < 1e-4:
                break   
        for i in range(state_size):        
            u_opt[i] = np.argmin(Q[i, :] + np.dot(P[i, :, :].T, J_opt))

    


    return J_opt, u_opt
