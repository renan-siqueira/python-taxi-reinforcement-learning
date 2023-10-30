## Taxi Reinforcement Learning

This project implements a simple Q-learning approach to train an agent to navigate the classic Taxi environment from OpenAI's Gym.

## Overview

1. **Environment Initialization:** We've created a function to initialize the Taxi-v3 environment. When initializing for training, the default rendering mode is used. When initializing for visualization, the 'human' rendering mode is enabled for better visualization.

2. **Agent Training:** A Q-learning based training method is used to train the agent. The agent learns to pick up a passenger and drop them at their desired location in the shortest time possible.

3. **Visualization:** Post training, we have a function to visualize the agent's behavior in the environment to understand its performance.

4. **Save and Load Models:** The trained Q-table can be saved to a file for later use and can also be loaded back when required.

## How to use

### Training:

- Initialize the environment in the training mode.
- Train the agent for a specified number of episodes.
- Save the trained Q-table if desired.

### Visualization:

- Load the trained Q-table.
- Initialize the environment in the visualization mode.
- Watch the trained agent in action.

## Requirements
- Python
- OpenAI's Gym
- NumPy
- (optional) Pygame, if using Gym's 'human' rendering mode.

## Code Structure

- `initialize_env()`: Initializes the Taxi-v3 environment.
- `train_agent()`: Trains the agent using Q-learning.
- `visualize_agent()`: Visualizes the agent's behavior post training.
- `save_model()`: Saves the trained Q-table.
- `load_model()`: Loads the saved Q-table.
- `main()`: The main driver function which prompts the user for choices and executes the corresponding tasks.

To run the code:

```bash
python taxi_rl_env.py
```

## Acknowledgments

This project uses the Taxi-v3 environment from OpenAI's Gym.
