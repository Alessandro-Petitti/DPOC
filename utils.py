"""
 utils.py

 Helper functions that are used in multiple files. Feel free to add more functions.

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
from Constants import Constants

def bresenham(start, end):
    """
    Generates the coordinates of a line between two points using Bresenham's algorithm.

    Parameters:
        start (tuple or list): The starting point (x0, y0).
        end (tuple or list): The ending point (x1, y1).

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates.

    Example:
        >>> bresenham((2, 3), (10, 8))
        [(2, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 6), (8, 7), (9, 7), (10, 8)]
    """
    x0, y0 = start
    x1, y1 = end

    points = []

    dx = x1 - x0
    dy = y1 - y0

    x_sign = 1 if dx > 0 else -1 if dx < 0 else 0
    y_sign = 1 if dy > 0 else -1 if dy < 0 else 0

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = x_sign, 0, 0, y_sign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, y_sign, x_sign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        px = x0 + x * xx + y * yx
        py = y0 + x * xy + y * yy
        points.append((px, py))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

    return points

def idx2state(idx):
    """Converts a given index into the corresponding state.

    Args:
        idx (int): index of the entry whose state is required

    Returns:
        np.array: (x,y,x,y) state corresponding to the given index
    """
    state = np.empty(4)

    for i, j in enumerate(
        [
            Constants.M,
            Constants.N,
            Constants.M,
            Constants.N,
        ]
    ):
        state[i] = idx % j
        idx = idx // j
    return state


def state2idx(state):
    """Converts a given state into the corresponding index.

    Args:
        state (np.array): (x,y,x,y) entry in the state space

    Returns:
        int: index corresponding to the given state
    """
    idx = 0

    factor = 1
    for i, j in enumerate([Constants.M, Constants.N, Constants.M, Constants.N]):
        idx += state[i] * factor
        factor *= j

    return idx

def input2idx(ux, uy):
    mapping = {
        (-1, -1): 0,
        (0, -1): 1,
        (1, -1): 2,
        (-1, 0): 3,
        (0, 0): 4,
        (1, 0): 5,
        (-1, 1): 6,
        (0, 1): 7,
        (1, 1): 8,
    }
    result = mapping.get((ux, uy), -1) 
    return np.array(result) if result is not -1 else -1  # Converte in array NumPy
 
def idx2input(idx):
    reverse_mapping = {
        0: (-1, -1),
        1: (0, -1),
        2: (1, -1),
        3: (-1, 0),
        4: (0, 0),
        5: (1, 0),
        6: (-1, 1),
        7: (0, 1),
        8: (1, 1),
    }
    result = reverse_mapping.get(idx, None)  # Restituisce None se l'indice non è trovato
    return np.array(result) if result is not None else None  # Converte in array NumPy


def h_fun(idx):
    """
    Verifica se il drone è fuori dalla mappa o ha fatto collisione con un drone statico.

    Args:
        idx (int): indice dello stato.

    Returns:
        int: 1 se il drone è fuori dalla mappa o in collisione, 0 altrimenti.
    """
    # Ottieni lo stato corrente (x_drone, y_drone, x_swan, y_swan)
    state = idx2state(idx)

    x_drone, y_drone, _, _ = state

    #add disturbance
    flow = Constants.FLOW_FIELD[x_drone,y_drone] 
    x_drone += flow[0]
    y_drone += flow[1]
    
    # Controllo se il drone è fuori dalla griglia
    if not (0 <= x_drone < Constants.M and 0 <= y_drone < Constants.N):
        return 1  # Fuori dalla mappa

    # Controllo collisione con droni statici
    for drone_pos in Constants.DRONE_POS:
        if tuple(drone_pos) == (x_drone, y_drone):
            return 1 # Collisione con un drone statico
    

    return 0  # Nessuna collisione e il drone è nella mappa

def compute_state_plus_currents(i,j, Constants):
    if 0 <= i < Constants.N and 0 <= j < Constants.M:    
        current_i, current_j = Constants.FLOW_FIELD[i][j]
        new_i = i + current_i
        new_j = j + current_j
        return (new_i, new_j)
    return (-1,-1)

def compute_state_with_input(i,j,l, Constants):
    if l > len(Constants.INPUT_SPACE):
        return (i,j)
    return (i+Constants.INPUT_SPACE[l][0],j+Constants.INPUT_SPACE[l][1])

def current_disturbance_map():
    '''
    For every state, builds a map that for both cases with or without the current,
    checks if you'll go to a problematic state.

    Output:
        MxN map: 1 if the state is problematic (requires deploying a new drone),
                 0 otherwise.
    A state is problematic if, after the current disturbance, the drone ends up:
        - Outside the grid.
        - On a static drone.
    '''
    M = Constants.M
    N = Constants.N
    mappa = np.zeros((M, N))
    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)  # Use a set for quick lookups

    for iX in range(M):  # Include the full range
        for iY in range(N):  # Include the full range
            updated_x, updated_y = compute_state_plus_currents(iX, iY)
            # Check for outside of the map:
            if not (0 <= updated_x < M and 0 <= updated_y < N):
                mappa[iX, iY] = 1
                continue
            # Check for static drone collision:
            if (updated_x, updated_y) in static_drones:
                mappa[iX, iY] = 1
    return mappa





