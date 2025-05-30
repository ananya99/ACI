# Artificial Cambrian Intelligence (ACI)

This project is a fork of [cambrian-org/ACI](https://github.com/cambrian-org/ACI).

We extended the original framework to support predator-prey reinforcement learning and the evolution of agents' visual morphological traits.

## Usage Instructions

### Set up the Environment

Use the pyproject.toml file to install dependencies and set up the Python environment.

### Run Alternate Training

1. Execute scripts/rl.sh. Make sure to select the appropriate environment before running.

### Run Grid Search/Evolution

1. Grid search for visual traits - run scripts/evo.sh. To configure the search:
    - Set training_agent_name to specify which agent to evolve.
    - Adjust the visual trait parameters directly in the script.
    - Ensure the environment is correctly selected before execution.

2. CMA-ES evolution of visual traits - also run via scripts/evo.sh, with the same configuration steps as above.
