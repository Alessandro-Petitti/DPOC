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
    x_drone, y_drone, x_swan, y_swan = np.meshgrid(
        np.arange(Constants.M), np.arange(Constants.N),
        np.arange(Constants.M), np.arange(Constants.N),
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
    not_valid_map_i = state2idx_vectorialized(invalid_start_states)

    # Per ogni ingresso, calcola le transizioni
    for l in range(Constants.L):
        # Calcola nuovi stati senza corrente
        no_current_x_drone = x_drone + Constants.INPUT_SPACE[l][0]
        no_current_y_drone = y_drone + Constants.INPUT_SPACE[l][1]
        #crea una maschera per gli stati validi
        valid_no_current = (0 <= no_current_x_drone) & (no_current_x_drone < Constants.M) & (0 <= no_current_y_drone) & (no_current_y_drone < Constants.N)
        # Calcola nuovi stati con corrente
        x_drone_with_current, y_drone_with_current = compute_state_plus_currents_vectorialized(x_drone, y_drone, Constants)
        current_x_drone = Constants.INPUT_SPACE[l][0]+x_drone_with_current
        current_y_drone = Constants.INPUT_SPACE[l][1]+y_drone_with_current
        
        #crea una maschera per gli stati validi
        valid_current = (0 <= current_x_drone) & (current_x_drone < Constants.M) & (0 <= current_y_drone) & (current_y_drone < Constants.N)
        
        # Calcola la nuova posizione del cigno
        dx, dy = Swan_movment_to_catch_drone_vectorized(
            x_swan, y_swan, x_drone, y_drone
        )
        new_x_swan = x_swan + dx
        new_y_swan = y_swan + dy
                
        # Crea una maschera che individua i movimenti invalidi
        mask_invalid_x = (new_x_swan < 0) | (new_x_swan >= Constants.M)
        mask_invalid_y = (new_y_swan < 0) | (new_y_swan >= Constants.N)
        mask_invalid = mask_invalid_x | mask_invalid_y

        if np.any(mask_invalid):
            # Ottieni gli indici degli elementi non validi
            invalid_indices = np.where(mask_invalid)[0]
            print("Stati con movimento invalido:")
            for idx in invalid_indices:
                print(
                    f"Indice: {idx}, "
                    f"x_drone: {x_drone[idx]}, y_drone: {y_drone[idx]}, "
                    f"x_swan: {x_swan[idx]}, y_swan: {y_swan[idx]}, "
                    f"dx: {dx[idx]}, dy: {dy[idx]}, "
                    f"new_x_swan: {new_x_swan[idx]}, new_y_swan: {new_y_swan[idx]}"
                )
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

        # Lista delle maschere e dei loro nomi per iterazioni dinamiche
        """masks = [valid_no_current_no_swan, valid_no_current_swan, valid_current_no_swan, valid_current_swan]
        mask_names = ["valid_no_current_no_swan", "valid_no_current_swan", "valid_current_no_swan", "valid_current_swan"]

        # Controllo globale della mutua esclusività
        overlap_check = np.sum([m.astype(int) for m in masks], axis=0)
        if np.any(overlap_check > 1):
            indices = np.where(overlap_check > 1)[0]
            print("ATTENZIONE: Ci sono stati che appartengono a più di una categoria!")
            print("Indici con sovrapposizione globale:", indices)
        else:
            print("Nessuno stato appartiene a più di una categoria (nessuna sovrapposizione globale).")

        # Controllo a coppie
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                overlap = masks[i] & masks[j]
                if np.any(overlap):
                    indices = np.where(overlap)[0]
                    print(f"Sovrapposizione tra {mask_names[i]} e {mask_names[j]}:")
                    print("Indici con sovrapposizione:", indices)
                else:
                    print(f"Nessuna sovrapposizione tra {mask_names[i]} e {mask_names[j]}")"""

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
        P[state2idx_vectorialized(start_states[valid_no_current_no_swan]), state2idx_vectorialized(next_states_no_current_no_swan), l] += (
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

