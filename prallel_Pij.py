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
    # Pre-allocazione della matrice P
    P = np.zeros((Constants.K, Constants.K, Constants.L))

    # Genera griglie di coordinate per gli stati
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

    # Stati di partenza come indici
    start_states = np.stack([x_drone, y_drone, x_swan, y_swan], axis=1)
    map_i = state2idx_vectorialized(start_states)
    # Stati di respawn
    respawn_indices = generate_respawn_indices(Constants)
    respawn_probability = 1 / (Constants.M * Constants.N - 1)

    # Set di droni statici
    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)

    # Condizione 1: x_drone != x_swan or y_drone != y_swan
    starting_swan_collision_mask = (x_drone == x_swan) & (y_drone == y_swan)
    # Condizione 2: (x_drone, y_drone) not in static_drones
    static_drone_collision_mask = np.array([(x, y) in static_drones for x, y in zip(x_drone, y_drone)])
    # Condizione 3: (x_drone, y_drone) != Constants.GOAL_POS
    goal_position_mask = (x_drone == Constants.GOAL_POS[0]) & (y_drone == Constants.GOAL_POS[1])
    # Tutti gli stati su cui lavorare.
    #stati in cui la probabilità di arrivare è zero
    not_valid_states_mask = (starting_swan_collision_mask |
                              static_drone_collision_mask | 
                              goal_position_mask)
    # Filtra gli stati iniziali e gli indici di mappatura
    invalid_start_states = start_states[not_valid_states_mask]
    #P(i,not_vald_ map_i,u) = 0

    not_valid_map_i = map_i[not_valid_states_mask]
    blocked = np.zeros((Constants.M, Constants.N), dtype=bool)
    for (xd, yd) in static_drones:
        blocked[xd, yd] = True
    
    # Per ogni ingresso, calcola le transizioni
    for l in range(Constants.L):
        # Calcola nuovi stati senza corrente
        no_current_x_drone = x_drone + Constants.INPUT_SPACE[l][0]
        no_current_y_drone = y_drone + Constants.INPUT_SPACE[l][1]
        # Calcola nuovi stati con corrente
        x_drone_with_current, y_drone_with_current = compute_state_plus_currents_vectorialized(x_drone, y_drone, Constants)
        current_x_drone = Constants.INPUT_SPACE[l][0]+x_drone_with_current
        current_y_drone = Constants.INPUT_SPACE[l][1]+y_drone_with_current
        starts = np.column_stack((x_drone, y_drone))  # shape (K, 2)
        ends = np.column_stack((current_x_drone, current_y_drone))      # shape (K, 2)

        paths = bresenham_fixed_length(starts, ends, max_len=3)
        #paths = bresenham_fixed_length((x_drone, y_drone), (current_x_drone, current_y_drone))
        
        # paths ha dimensione (N, max_len, 2)
        # paths[:,:,0] sono le x, paths[:,:,1] sono le y
        paths_x = paths[:, :, 0]
        paths_y = paths[:, :, 1]

        # Calcola la nuova posizione del cigno
        dx, dy = Swan_movment_to_catch_drone_vectorized(
            x_swan, y_swan, x_drone, y_drone
        )
        new_x_swan = x_swan + dx
        new_y_swan = y_swan + dy
        
        valid_no_current_no_swan = (
            (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) &
            (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N) &
            ~(np.array([(x, y) in static_drones for x, y in zip(no_current_x_drone, no_current_y_drone)])) &
            ~((no_current_x_drone == x_swan) & (no_current_y_drone == y_swan))
        )

        valid_no_current_swan = (
            (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) &
            (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N) &
            ~(np.array([(x, y) in static_drones for x, y in zip(no_current_x_drone, no_current_y_drone)])) &
            ~((no_current_x_drone == new_x_swan) & (no_current_y_drone == new_y_swan))
        )

        valid_current_no_swan = (
            (0 <= current_x_drone) & (current_x_drone < Constants.M) &
            (0 <= current_y_drone) & (current_y_drone < Constants.N) &
            ~(np.array([(x, y) in static_drones for x, y in zip(current_x_drone, current_y_drone)])) &
            ~((current_x_drone == x_swan) & (current_y_drone == y_swan))
        )

        valid_current_swan = (
            (0 <= current_x_drone) & (current_x_drone < Constants.M) &
            (0 <= current_y_drone) & (current_y_drone < Constants.N) &
            ~(np.array([(x, y) in static_drones for x, y in zip(current_x_drone, current_y_drone)])) &
            ~((current_x_drone == new_x_swan) & (current_y_drone == new_y_swan))
        )
        # Maschera dei punti validi (cioè non -1)
        valid_points_mask = (paths_x >= 0) & (paths_y >= 0) & (paths_x < Constants.M) & (paths_y < Constants.N)

        # Creiamo un array temporaneo per la libertà dei punti
        free_points = np.ones_like(valid_points_mask, dtype=bool)

        # Per tutti i punti validi controlliamo se sono liberi
        free_points[valid_points_mask] = ~blocked[paths_x[valid_points_mask], paths_y[valid_points_mask]]

        # Se vogliamo che l'intero percorso sia valido,
        # controlliamo che tutti i punti validi siano True
        valid_paths = free_points.all(axis=1)

        # Ora puoi integrare con le tue maschere
        valid_current_no_swan &= valid_paths
        valid_current_swan &= valid_paths
        # Stati successivi
        #seleziona quale stato è valido
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

        # Aggiorna la matrice P per i 4 casi
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
        # Stati di respawn specifici per ogni caso
        invalid_no_current_no_swan = ~valid_no_current_no_swan
        invalid_no_current_swan = ~valid_no_current_swan
        invalid_current_no_swan = ~valid_current_no_swan
        invalid_current_swan = ~valid_current_swan
        # Stati di respawn: gestisci ogni caso separatamente
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
    #set all the not valid states to 0 for each input
    P[not_valid_map_i, :, :] = 0

    return P

