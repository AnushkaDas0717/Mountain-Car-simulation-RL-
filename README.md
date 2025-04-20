# Mountain Car Continuous — Reinforcement Learning with SAC

This project is an interactive demo of a reinforcement learning agent trained to solve the MountainCarContinuous-v0 environment from Gymnasium.

## Overview

The objective of the environment is for the car to reach the top of a steep hill. However, the engine is not powerful enough to drive directly up the slope. The agent must learn to build momentum by moving back and forth to successfully reach the goal.

## The Agent

- Algorithm: Soft Actor-Critic (SAC)
- Action Type: Continuous
- Behavior: Learns to apply the correct amount of force at the right time to maximize long-term reward

## Environment Details

### Observation Space

- Position: Range from -1.2 to 0.6
- Velocity: Range from -0.07 to 0.07

### Action Space

- Force Applied: Continuous range from -1.0 to 1.0

### Reward Function

- Negative reward proportional to energy used (to encourage efficiency)
- Positive reward of +100 for successfully reaching the goal

## Technologies Used

- Python
- Gymnasium
- PyTorch (for SAC implementation)
- NumPy
- Imageo(for generating video simulation)
- Matplotlib (for visualizations)
- Streamlit(for web interface)


