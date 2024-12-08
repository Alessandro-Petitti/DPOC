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
def state2idx_vectorialized(states):
    """Converts an array of states into their corresponding indices.

    Args:
        states (np.ndarray): Array di forma (N, 4), dove ogni riga è (x, y, x, y).

    Returns:
        np.ndarray: Array di indici corrispondenti agli stati.
    """
    factors = np.array([1, Constants.M, Constants.M * Constants.N, Constants.M * Constants.N * Constants.M])
    return np.dot(states, factors)

def idx2state_vectorized(idx_array):
    """
    Converts a given array of indices into the corresponding states.

    Args:
        idx_array (np.ndarray): array of indices
        Constants: Class containing problem constants, with attributes:
            M (int), N (int)

    Returns:
        np.ndarray: array of states of shape (len(idx_array), 4)
                    Each row is [x_drone, y_drone, x_swan, y_swan].
    """
    idx_array = np.asarray(idx_array)

    # Allocazione array di output
    # Dimensione: numero di indici x 4 dimensioni dello stato
    states = np.empty((idx_array.size, 4), dtype=int)

    # Decodifica vettoriale
    # x_drone
    states[:, 0] = idx_array % Constants.M
    idx_tmp = idx_array // Constants.M

    # y_drone
    states[:, 1] = idx_tmp % Constants.N
    idx_tmp = idx_tmp // Constants.N

    # x_swan
    states[:, 2] = idx_tmp % Constants.M
    idx_tmp = idx_tmp // Constants.M

    # y_swan
    states[:, 3] = idx_tmp % Constants.N

    return states




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


def h_fun(idx_state, idx_input):
    """
    Verifica se il drone è fuori dalla mappa o ha fatto collisione con un drone statico.

    Args:
        idx (int): indice dello stato.

    Returns:
        int: 1 se il drone è fuori dalla mappa o in collisione, 0 altrimenti.
    """
    # Ottieni lo stato corrente (x_drone, y_drone, x_swan, y_swan)
    x_drone, y_drone, x_swan, y_swan = (int(value) for value in idx2state(idx_state))
    # Calcola la nuova posizione del drone con ingresso e corrente
    current_i, current_j = Constants.FLOW_FIELD[x_drone,y_drone]
    new_x_drone = x_drone + Constants.INPUT_SPACE[idx_input][0] + current_i
    new_y_drone = y_drone + Constants.INPUT_SPACE[idx_input][1] + current_j
    path = bresenham((x_drone, y_drone), (new_x_drone, new_y_drone))
    static_drones = set(tuple(pos) for pos in Constants.DRONE_POS)  

    #check if the moved drone is outside the map
    if not (0 <= new_x_drone < Constants.M and 0 <= new_y_drone < Constants.N):
        return 1  # outise of the map
    # Check for static drone collision in the path from start to end (input and current)
    if any(tuple(point) in static_drones for point in path):
        return 1
    
    #move the swan
    dx, dy = Swan_movment_to_catch_drone(x_swan, y_swan, x_drone, y_drone)
    moved_swan_x = x_swan + dx
    moved_swan_y = y_swan + dy
    #check collision between swan and moved drone
    if moved_swan_x == new_x_drone and moved_swan_y == new_y_drone:
        return 1
    
    return 0  # No need for a new drone

def compute_state_plus_currents(i,j, Constants):
    if 0 <= i < Constants.M and 0 <= j < Constants.N:    
        current_i, current_j = Constants.FLOW_FIELD[i,j]
        new_i = i + current_i
        new_j = j + current_j
        return (new_i, new_j)
    return (-1,-1)
def compute_state_plus_currents_vectorialized(i, j, Constants):
    """
    Calcola le nuove coordinate considerando la corrente.

    Args:
        i (np.ndarray): Array delle coordinate x.
        j (np.ndarray): Array delle coordinate y.
        Constants: Oggetto con le costanti del problema.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Nuove coordinate (new_i, new_j).
    """
    # Creazione maschera per validità degli stati
    valid_mask = (0 <= i) & (i < Constants.M) & (0 <= j) & (j < Constants.N)

    # Calcola gli spostamenti dovuti alla corrente
    current_i = np.zeros_like(i)
    current_j = np.zeros_like(j)
    current_i[valid_mask] = Constants.FLOW_FIELD[i[valid_mask], j[valid_mask], 0]
    current_j[valid_mask] = Constants.FLOW_FIELD[i[valid_mask], j[valid_mask], 1]

    # Applica gli spostamenti solo agli stati validi
    new_i = np.where(valid_mask, i + current_i, -1)
    new_j = np.where(valid_mask, j + current_j, -1)

    return new_i, new_j


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


def Swan_movment_to_catch_drone(x_swan, y_swan, x_drone, y_drone):
    # Calcolo dell'angolo θ usando atan2
    theta = np.arctan2(y_drone - y_swan, x_drone - x_swan)
    # Mappatura dell'angolo θ ai quadranti
    if -np.pi/8 <= theta < np.pi/8:
        return (+1, 0)  # East (E)
    elif np.pi/8 <= theta < 3*np.pi/8:
        return (+1, +1)  # North-East (NE)
    elif 3*np.pi/8 <= theta < 5*np.pi/8:
        return (0, +1)  # North (N)
    elif 5*np.pi/8 <= theta < 7*np.pi/8:
        return (-1, +1)  # North-West (NW)
    elif theta >= 7*np.pi/8 or theta < -7*np.pi/8:
        return (-1, 0)  # West (W)
    elif -7*np.pi/8 <= theta < -5*np.pi/8:
        return (-1, -1)  # South-West (SW)
    elif -5*np.pi/8 <= theta < -3*np.pi/8:
        return (0, -1)  # South (S)
    elif -3*np.pi/8 <= theta < -np.pi/8:
        return (+1, -1)  # South-East (SE)


def generate_respawn_indices(Constants):
    """
    Generates all the valid respawn states indices for the drone.
    """
    start_x, start_y = Constants.START_POS
    # all possible states but the starting one
    respawn_states = [
        state2idx([start_x, start_y, xswan, yswan])
        for xswan in range(Constants.M)
        for yswan in range(Constants.N)
        if not (xswan == start_x and yswan == start_y)
    ]

    return np.array(respawn_states)

import numpy as np

def Swan_movment_to_catch_drone_vectorized(x_swan, y_swan, x_drone, y_drone):
    # Converte i singoli valori in array monodimensionali nel caso vengano passati scalari
    x_swan = np.atleast_1d(x_swan)
    y_swan = np.atleast_1d(y_swan)
    x_drone = np.atleast_1d(x_drone)
    y_drone = np.atleast_1d(y_drone)
    
    # Calcolo dell'angolo theta
    theta = np.arctan2(y_drone - y_swan, x_drone - x_swan)
    
    # Creazione array per il movimento
    movement = np.zeros((len(theta), 2), dtype=int)

    # Mappatura dell'angolo theta ai movimenti nei vari quadranti
    movement[(theta >= -np.pi/8) & (theta < np.pi/8)] = [1, 0]    # E
    movement[(theta >= np.pi/8) & (theta < 3*np.pi/8)] = [1, 1]   # NE
    movement[(theta >= 3*np.pi/8) & (theta < 5*np.pi/8)] = [0, 1] # N
    movement[(theta >= 5*np.pi/8) & (theta < 7*np.pi/8)] = [-1, 1]# NW
    movement[(theta >= 7*np.pi/8) | (theta < -7*np.pi/8)] = [-1, 0] # W
    movement[(theta >= -7*np.pi/8) & (theta < -5*np.pi/8)] = [-1, -1] # SW
    movement[(theta >= -5*np.pi/8) & (theta < -3*np.pi/8)] = [0, -1]  # S
    movement[(theta >= -3*np.pi/8) & (theta < -np.pi/8)] = [1, -1]    # SE

    # Se cigno e drone sono nella stessa posizione, dx e dy = 0
    same_pos_mask = (x_swan == x_drone) & (y_swan == y_drone)
    movement[same_pos_mask] = [0, 0]

    dx, dy = movement[:, 0], movement[:, 1]
    
    # Se l'input era scalare, restituiamo valori scalari
    if dx.size == 1:
        return dx[0], dy[0]
    else:
        return dx, dy

def bresenham_fixed_length(starts, ends, max_len=3):
    """
    Applica l'algoritmo di Bresenham a più segmenti contemporaneamente e
    restituisce un array di dimensioni (N, max_len, 2) dove N è il numero
    di segmenti. Ogni riga corrisponde a un percorso, con esattamente `max_len`
    punti (x,y). Se il percorso è più corto, viene riempito con -1. 

    Parametri:
        starts (array_like): Nx2 array con i punti di start (x0, y0)
        ends (array_like): Nx2 array con i punti di end (x1, y1)
        max_len (int): Lunghezza massima del percorso (default=3)

    Ritorna:
        paths (np.ndarray): Array di shape (N, max_len, 2)
    """
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    if starts.shape[0] != ends.shape[0]:
        raise ValueError("Il numero di start deve coincidere con il numero di end.")

    N = starts.shape[0]
    # Array risultato: inizializzato a -1
    paths = np.full((N, max_len, 2), -1, dtype=int)

    for i, ((x0, y0), (x1, y1)) in enumerate(zip(starts, ends)):
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
        line_points = []
        for x in range(dx + 1):
            px = x0 + x * xx + y * yx
            py = y0 + x * xy + y * yy
            line_points.append((px, py))
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy

        # Se il percorso è più lungo di max_len, tronchiamo
        if len(line_points) > max_len:
            line_points = line_points[:max_len]

        # Se più corto, viene lasciato com'è: i restanti sono già -1
        for j, pt in enumerate(line_points):
            paths[i, j] = pt

    return paths