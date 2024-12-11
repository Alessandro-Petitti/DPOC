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
def compute_transition_probabilities(Constants):
    P = np.zeros((Constants.K, Constants.K, Constants.L))

    y_swan, x_swan, y_drone, x_drone = np.meshgrid(
        np.arange(Constants.N),
        np.arange(Constants.M),
        np.arange(Constants.N),
        np.arange(Constants.M),
        indexing="ij"
    )

    x_drone = x_drone.ravel()
    y_drone = y_drone.ravel()
    x_swan = x_swan.ravel()
    y_swan = y_swan.ravel()

    start_states = np.stack([x_drone, y_drone, x_swan, y_swan], axis=1)
    map_i = state2idx_vectorialized(start_states)

    respawn_indices = generate_respawn_indices(Constants)
    respawn_probability = 1 / (Constants.M * Constants.N - 1)

    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)
    blocked = np.zeros((Constants.M, Constants.N), dtype=bool)
    for (xd, yd) in static_drones:
        blocked[xd, yd] = True

    starting_swan_collision_mask = (x_drone == x_swan) & (y_drone == y_swan)
    static_drone_collision_mask = (
        (0 <= x_drone) & (x_drone < Constants.M) &
        (0 <= y_drone) & (y_drone < Constants.N) &
        blocked[x_drone, y_drone]
    )
    goal_position_mask = (x_drone == Constants.GOAL_POS[0]) & (y_drone == Constants.GOAL_POS[1])

    not_valid_states_mask = (
        starting_swan_collision_mask |
        static_drone_collision_mask | 
        goal_position_mask
    )
    not_valid_map_i = map_i[not_valid_states_mask]

    for l in range(Constants.L):
        no_current_x_drone = x_drone + Constants.INPUT_SPACE[l][0]
        no_current_y_drone = y_drone + Constants.INPUT_SPACE[l][1]

        x_drone_with_current, y_drone_with_current = compute_state_plus_currents_vectorialized(x_drone, y_drone, Constants)
        current_x_drone = Constants.INPUT_SPACE[l][0] + x_drone_with_current
        current_y_drone = Constants.INPUT_SPACE[l][1] + y_drone_with_current

        starts = np.column_stack((x_drone, y_drone))
        ends = np.column_stack((current_x_drone, current_y_drone))
        paths = bresenham_fixed_length(starts, ends, max_len=3)
        paths_x = paths[:, :, 0]
        paths_y = paths[:, :, 1]

        dx, dy = Swan_movment_to_catch_drone_vectorized(x_swan, y_swan, x_drone, y_drone)
        new_x_swan = x_swan + dx
        new_y_swan = y_swan + dy

        valid_no_current_no_swan = (
            (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) &
            (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N)
        )

        valid_indices_no_current_no_swan = np.where(valid_no_current_no_swan)[0]
        valid_no_current_no_swan[valid_indices_no_current_no_swan] &= ~blocked[
            no_current_x_drone[valid_indices_no_current_no_swan],
            no_current_y_drone[valid_indices_no_current_no_swan]
        ] & ~(
            (no_current_x_drone[valid_indices_no_current_no_swan] == x_swan[valid_indices_no_current_no_swan]) &
            (no_current_y_drone[valid_indices_no_current_no_swan] == y_swan[valid_indices_no_current_no_swan])
        )

        valid_no_current_swan = (
            (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) &
            (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N)
        )

        valid_indices_no_current_swan = np.where(valid_no_current_swan)[0]
        valid_no_current_swan[valid_indices_no_current_swan] &= ~blocked[
            no_current_x_drone[valid_indices_no_current_swan],
            no_current_y_drone[valid_indices_no_current_swan]
        ] & ~(
            (no_current_x_drone[valid_indices_no_current_swan] == new_x_swan[valid_indices_no_current_swan]) &
            (no_current_y_drone[valid_indices_no_current_swan] == new_y_swan[valid_indices_no_current_swan])
        )

        valid_current_no_swan = (
            (0 <= current_x_drone) & (current_x_drone < Constants.M) &
            (0 <= current_y_drone) & (current_y_drone < Constants.N)
        )

        valid_indices_current_no_swan = np.where(valid_current_no_swan)[0]
        valid_current_no_swan[valid_indices_current_no_swan] &= ~blocked[
            current_x_drone[valid_indices_current_no_swan],
            current_y_drone[valid_indices_current_no_swan]
        ] & ~(
            (current_x_drone[valid_indices_current_no_swan] == x_swan[valid_indices_current_no_swan]) &
            (current_y_drone[valid_indices_current_no_swan] == y_swan[valid_indices_current_no_swan])
        )

        valid_current_swan = (
            (0 <= current_x_drone) & (current_x_drone < Constants.M) &
            (0 <= current_y_drone) & (current_y_drone < Constants.N)
        )

        valid_indices_current_swan = np.where(valid_current_swan)[0]
        valid_current_swan[valid_indices_current_swan] &= ~blocked[
            current_x_drone[valid_indices_current_swan],
            current_y_drone[valid_indices_current_swan]
        ] & ~(
            (current_x_drone[valid_indices_current_swan] == new_x_swan[valid_indices_current_swan]) &
            (current_y_drone[valid_indices_current_swan] == new_y_swan[valid_indices_current_swan])
        )

        valid_points_mask = (paths_x >= 0) & (paths_y >= 0) & (paths_x < Constants.M) & (paths_y < Constants.N)
        free_points = np.ones_like(valid_points_mask, dtype=bool)
        free_points[valid_points_mask] = ~blocked[paths_x[valid_points_mask], paths_y[valid_points_mask]]
        valid_paths = free_points.all(axis=1)

        valid_current_no_swan &= valid_paths
        valid_current_swan &= valid_paths

        next_states_no_current_no_swan = np.stack([
            no_current_x_drone[valid_no_current_no_swan], no_current_y_drone[valid_no_current_no_swan],
            x_swan[valid_no_current_no_swan], y_swan[valid_no_current_no_swan]
        ], axis=1)

        next_states_no_current_swan = np.stack([
            no_current_x_drone[valid_no_current_swan], no_current_y_drone[valid_no_current_swan],
            new_x_swan[valid_no_current_swan], new_y_swan[valid_no_current_swan]
        ], axis=1)

        next_states_current_no_swan = np.stack([
            current_x_drone[valid_current_no_swan], current_y_drone[valid_current_no_swan],
            x_swan[valid_current_no_swan], y_swan[valid_current_no_swan]
        ], axis=1)

        next_states_current_swan = np.stack([
            current_x_drone[valid_current_swan], current_y_drone[valid_current_swan],
            new_x_swan[valid_current_swan], new_y_swan[valid_current_swan]
        ], axis=1)

        P[map_i[valid_no_current_no_swan], state2idx_vectorialized(next_states_no_current_no_swan), l] += (
            (1 - Constants.CURRENT_PROB[x_drone[valid_no_current_no_swan], y_drone[valid_no_current_no_swan]]) *
            (1 - Constants.SWAN_PROB)
        )

        P[map_i[valid_no_current_swan], state2idx_vectorialized(next_states_no_current_swan), l] += (
            (1 - Constants.CURRENT_PROB[x_drone[valid_no_current_swan], y_drone[valid_no_current_swan]]) *
            Constants.SWAN_PROB
        )

        P[map_i[valid_current_no_swan], state2idx_vectorialized(next_states_current_no_swan), l] += (
            Constants.CURRENT_PROB[x_drone[valid_current_no_swan], y_drone[valid_current_no_swan]] *
            (1 - Constants.SWAN_PROB)
        )

        P[map_i[valid_current_swan], state2idx_vectorialized(next_states_current_swan), l] += (
            Constants.CURRENT_PROB[x_drone[valid_current_swan], y_drone[valid_current_swan]] *
            Constants.SWAN_PROB
        )

        invalid_no_current_no_swan = ~valid_no_current_no_swan
        invalid_no_current_swan = ~valid_no_current_swan
        invalid_current_no_swan = ~valid_current_no_swan
        invalid_current_swan = ~valid_current_swan

        invalid_indices = map_i[invalid_no_current_no_swan]
        P[invalid_indices[:, None], respawn_indices, l] += respawn_probability * (
            (1 - Constants.CURRENT_PROB[x_drone[invalid_indices], y_drone[invalid_indices]])[:, None] * (1 - Constants.SWAN_PROB)
        )

        invalid_indices_no_current_swan = map_i[invalid_no_current_swan]
        P[invalid_indices_no_current_swan[:, None], respawn_indices, l] += (
            respawn_probability *
            (1 - Constants.CURRENT_PROB[x_drone[invalid_indices_no_current_swan], y_drone[invalid_indices_no_current_swan]])[:, None] *
            Constants.SWAN_PROB
        )

        invalid_indices_current_no_swan = map_i[invalid_current_no_swan]
        P[invalid_indices_current_no_swan[:, None], respawn_indices, l] += (
            respawn_probability *
            Constants.CURRENT_PROB[x_drone[invalid_indices_current_no_swan], y_drone[invalid_indices_current_no_swan]][:, None] *
            (1 - Constants.SWAN_PROB)
        )

        invalid_indices_current_swan = map_i[invalid_current_swan]
        P[invalid_indices_current_swan[:, None], respawn_indices, l] += (
            respawn_probability *
            Constants.CURRENT_PROB[x_drone[invalid_indices_current_swan], y_drone[invalid_indices_current_swan]][:, None] *
            Constants.SWAN_PROB
        )

    P[not_valid_map_i, :, :] = 0

    return P