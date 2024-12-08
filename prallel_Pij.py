import numpy as np
from utils import *

def parallel_Pij(Constants):
    """
    Calcola la matrice delle probabilità di transizione P[i, j, u].

    Args:
        Constants: Classe contenente le costanti del problema.

    Returns:
        np.ndarray: Matrice di probabilità di transizione P di forma (K, K, L).
    """
    # Pre-allocazione della matrice delle probabilità di transizione
    P = np.zeros((Constants.K, Constants.K, Constants.L))

    # Genera griglie di coordinate per gli stati (drone e cigno)
    y_swan, x_swan, y_drone, x_drone = np.meshgrid(
        np.arange(Constants.N),
        np.arange(Constants.M),
        np.arange(Constants.N),
        np.arange(Constants.M),
        indexing="ij"
    )

    # Appiattisce le coordinate per operare con vettori
    x_drone = x_drone.ravel()
    y_drone = y_drone.ravel()
    x_swan = x_swan.ravel()
    y_swan = y_swan.ravel()

    # Stati di partenza come array di coordinate
    start_states = np.stack([x_drone, y_drone, x_swan, y_swan], axis=1)
    map_i = state2idx_vectorialized(start_states)  # Mappa degli indici per gli stati

    # Genera gli indici di respawn e la loro probabilità
    respawn_indices = generate_respawn_indices(Constants)
    respawn_probability = 1 / (Constants.M * Constants.N - 1)

    # Insieme dei droni statici (tuple per facile confronto)
    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)

    # Condizioni per stati non validi
    starting_swan_collision_mask = (x_drone == x_swan) & (y_drone == y_swan)  # Collisione con il cigno
    static_drone_collision_mask = np.array([(x, y) in static_drones for x, y in zip(x_drone, y_drone)])  # Collisione con drone statico
    goal_position_mask = (x_drone == Constants.GOAL_POS[0]) & (y_drone == Constants.GOAL_POS[1])  # Posizione del goal

    # Identifica stati iniziali non validi
    not_valid_states_mask = (starting_swan_collision_mask |
                              static_drone_collision_mask | 
                              goal_position_mask)
    not_valid_map_i = map_i[not_valid_states_mask]

    # Matrice di blocco per droni statici
    blocked = np.zeros((Constants.M, Constants.N), dtype=bool)
    for (xd, yd) in static_drones:
        blocked[xd, yd] = True

    # Itera su ogni input di controllo
    for l in range(Constants.L):
        # Calcola le nuove coordinate del drone senza corrente
        no_current_x_drone = x_drone + Constants.INPUT_SPACE[l][0]
        no_current_y_drone = y_drone + Constants.INPUT_SPACE[l][1]

        # Calcola le nuove coordinate del drone con corrente
        x_drone_with_current, y_drone_with_current = compute_state_plus_currents_vectorialized(x_drone, y_drone, Constants)
        current_x_drone = Constants.INPUT_SPACE[l][0] + x_drone_with_current
        current_y_drone = Constants.INPUT_SPACE[l][1] + y_drone_with_current

        # Calcola i percorsi usando Bresenham per la validità delle traiettorie
        starts = np.column_stack((x_drone, y_drone))
        ends = np.column_stack((current_x_drone, current_y_drone))
        paths = bresenham_fixed_length(starts, ends, max_len=3)
        paths_x = paths[:, :, 0]
        paths_y = paths[:, :, 1]

        # Movimento del cigno verso il drone
        dx, dy = Swan_movment_to_catch_drone_vectorized(x_swan, y_swan, x_drone, y_drone)
        new_x_swan = x_swan + dx
        new_y_swan = y_swan + dy

        # Condizioni di validità per i nuovi stati
        valid_no_current_no_swan = (
            (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) &
            (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N) &
            ~np.array([(x, y) in static_drones for x, y in zip(no_current_x_drone, no_current_y_drone)]) &
            ~((no_current_x_drone == x_swan) & (no_current_y_drone == y_swan))
        )

        valid_no_current_swan = (
            (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) &
            (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N) &
            ~np.array([(x, y) in static_drones for x, y in zip(no_current_x_drone, no_current_y_drone)]) &
            ~((no_current_x_drone == new_x_swan) & (no_current_y_drone == new_y_swan))
        )

        valid_current_no_swan = (
            (0 <= current_x_drone) & (current_x_drone < Constants.M) &
            (0 <= current_y_drone) & (current_y_drone < Constants.N) &
            ~np.array([(x, y) in static_drones for x, y in zip(current_x_drone, current_y_drone)]) &
            ~((current_x_drone == x_swan) & (current_y_drone == y_swan))
        )

        valid_current_swan = (
            (0 <= current_x_drone) & (current_x_drone < Constants.M) &
            (0 <= current_y_drone) & (current_y_drone < Constants.N) &
            ~np.array([(x, y) in static_drones for x, y in zip(current_x_drone, current_y_drone)]) &
            ~((current_x_drone == new_x_swan) & (current_y_drone == new_y_swan))
        )

        # Maschera dei punti validi nei percorsi
        valid_points_mask = (paths_x >= 0) & (paths_y >= 0) & (paths_x < Constants.M) & (paths_y < Constants.N)
        free_points = np.ones_like(valid_points_mask, dtype=bool)
        free_points[valid_points_mask] = ~blocked[paths_x[valid_points_mask], paths_y[valid_points_mask]]
        valid_paths = free_points.all(axis=1)  # Percorsi completamente validi

        # Integra la validità dei percorsi con le condizioni
        valid_current_no_swan &= valid_paths
        valid_current_swan &= valid_paths

        # Stati successivi validi
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

        # Aggiornamento della matrice P
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

        # Stati di respawn per input non validi
        invalid_no_current_no_swan = ~valid_no_current_no_swan
        invalid_no_current_swan = ~valid_no_current_swan
        invalid_current_no_swan = ~valid_current_no_swan
        invalid_current_swan = ~valid_current_swan

        for idx in map_i[invalid_no_current_no_swan]:
            P[idx, respawn_indices, l] += respawn_probability * (
                (1 - Constants.CURRENT_PROB[x_drone[idx], y_drone[idx]]) *
                (1 - Constants.SWAN_PROB)
            )

        for idx in map_i[invalid_no_current_swan]:
            P[idx, respawn_indices, l] += respawn_probability * (
                (1 - Constants.CURRENT_PROB[x_drone[idx], y_drone[idx]]) *
                Constants.SWAN_PROB
            )

        for idx in map_i[invalid_current_no_swan]:
            P[idx, respawn_indices, l] += respawn_probability * (
                Constants.CURRENT_PROB[x_drone[idx], y_drone[idx]] *
                (1 - Constants.SWAN_PROB)
            )

        for idx in map_i[invalid_current_swan]:
            P[idx, respawn_indices, l] += respawn_probability * (
                Constants.CURRENT_PROB[x_drone[idx], y_drone[idx]] *
                Constants.SWAN_PROB
            )

    # Azzeramento delle probabilità per stati non validi
    P[not_valid_map_i, :, :] = 0

    return P
