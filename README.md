# Drone Navigation for Dynamic Programming and Optimal Control

This repository contains a solution to "The Way of Water" programming exercise from the 2024 Dynamic Programming and Optimal Control course. The problem involves navigating a drone safely across a discretized lake grid to its designated position, avoiding collisions with static drones and a pursuing swan, while accounting for environmental disturbances such as lake currents.

## Problem Description

The goal is to compute an optimal policy for navigating a drone through a grid-like state space to reach a goal position with minimal cost. The cost is determined by:

1. Time taken to reach the goal.
2. Energy spent controlling the drone.
3. The number of drones deployed due to failures (e.g., collisions).

### Key Elements:

- **State Space:** Defined by the position of the drone and the swan.
- **Action Space:** Represents eight possible movement directions or staying still.
- **Disturbances:** Include random currents, swan movement, and spawning of a new drone after failure.
- **Dynamics:** Governed by deterministic inputs and probabilistic disturbances.

The solution implements the following features to achieve efficient and correct computations:

### Solution Features

1. **Parallelized Computation of Transition Probabilities (`P_ij`)**
   - Leveraging parallel processing, this feature accelerates the computation of state transition probabilities, essential for dynamic programming algorithms.

2. **Value Iteration with Gauss-Seidel Updates**
   - This implementation of the value iteration algorithm uses Gauss-Seidel updates for faster convergence. By updating states sequentially within each iteration, it reduces computational overhead compared to standard synchronous methods.

3. **Asynchronous Policy Iteration**
   - Used in carefully selected scenarios, asynchronous policy iteration allows state updates without strict adherence to sequential order. This approach improves performance in environments with sparse dependencies.

4. **Parallelized Stage Cost Computation**
   - The calculation of expected stage costs (`Q`) is parallelized to handle the large state space efficiently. This ensures scalability and reduces runtime significantly.

## Implementation

### Files

- **`ComputeTransitionProbabilities.py`**: Computes the transition probabilities for all state-action pairs.
- **`ComputeExpectedStageCosts.py`**: Computes the expected stage costs for all state-action pairs.
- **`Solver.py`**: Implements dynamic programming algorithms to compute the optimal cost and policy.
- **`utils.py`**: Contains shared utility functions such as state and action indexing.

### Usage

To run the solution:

1. Set up the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate dpoc_pe
   ```
2. Run the main script:
   ```bash
   python3 main.py
   ```
3. Use the visualization tool to debug and analyze results:
   ```bash
   python3 visualization.py
   ```

### Evaluation

- **Correctness:** The solution is validated against predefined test cases.
- **Performance:** Parallelization and algorithmic optimizations ensure efficient handling of large state spaces.
- **Scoring:** Submissions are ranked based on average runtime over multiple problem instances.

## Contributing

This repository is a submission for the programming exercise and adheres to the specified structure and requirements. If you wish to extend or adapt the solution, ensure compatibility with the provided framework.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
