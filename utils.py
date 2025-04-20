"""
Utility functions for the Mountain Car RL project.
"""

import numpy as np
import gymnasium as gym
import imageio
import os
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
from matplotlib import animation

def create_video(model_path, initial_state=None, video_path="videos/mountaincar_video.mp4"):
    """
    Create a video of the trained agent in action.
    
    Args:
        model_path: Path to the saved model
        initial_state: Optional initial state as [position, velocity]
        video_path: Path where to save the video
    
    Returns:
        video_path: Path to the generated video
        total_reward: Total reward achieved
        steps: Number of steps taken
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Load the model
    model = SAC.load(model_path)
    
    # Create environment with rendering
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    
    # Reset with custom initial state if provided
    if initial_state is not None:
        obs, _ = env.reset()
        env.unwrapped.state = np.array(initial_state, dtype=np.float32)
        obs = env.unwrapped.state
    else:
        obs, _ = env.reset()
    
    # Run the simulation
    frames = []
    total_reward = 0
    steps = 0
    
    # Capture frames and run the simulation
    done = False
    while not done and steps < 1000:  # Limit to 1000 steps to prevent infinite loops
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    # Save as video
    imageio.mimsave(video_path, frames, fps=30)
    
    return video_path, total_reward, steps

def plot_state_value_heatmap(model_path, resolution=50, save_path="logs/value_heatmap.png"):
    """
    Create a simplified heatmap visualization using just the policy's predicted actions.
    
    Args:
        model_path: Path to the saved model
        resolution: Resolution of the heatmap grid
        save_path: Path where to save the heatmap image
    
    Returns:
        save_path: Path to the generated heatmap
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load the model
        model = SAC.load(model_path)
        env = gym.make("MountainCarContinuous-v0")
        
        # Define the state space grid
        x_range = np.linspace(env.observation_space.low[0], env.observation_space.high[0], resolution)
        v_range = np.linspace(env.observation_space.low[1], env.observation_space.high[1], resolution)
        X, V = np.meshgrid(x_range, v_range)
        
        # Get action values for each state
        action_values = np.zeros(X.shape)
        
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], V[i, j]])
                action, _ = model.predict(state, deterministic=True)
                # Use the action magnitude as a proxy for state value
                action_values[i, j] = action[0]  # In MountainCar, positive actions push right
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, V, action_values, cmap='viridis')
        plt.colorbar(label='Action Value (Force)')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Action Value Heatmap')
        
        # Add environment boundaries
        plt.axvline(x=env.observation_space.low[0], color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=env.observation_space.high[0], color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=env.observation_space.low[1], color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=env.observation_space.high[1], color='k', linestyle='--', alpha=0.5)
        
        # Mark goal position
        plt.axvline(x=0.45, color='g', linestyle='--', alpha=0.7, label='Goal')
        
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None
    
def plot_trajectory(model_path, initial_state=None, save_path="logs/trajectory.png"):
    """
    Plot the trajectory of the car in the state space.
    
    Args:
        model_path: Path to the saved model
        initial_state: Optional initial state as [position, velocity]
        save_path: Path where to save the trajectory plot
    
    Returns:
        save_path: Path to the generated trajectory plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load the model
    model = SAC.load(model_path)
    
    # Create environment
    env = gym.make("MountainCarContinuous-v0")
    
    # Reset with custom initial state if provided
    if initial_state is not None:
        obs, _ = env.reset()
        env.unwrapped.state = np.array(initial_state, dtype=np.float32)
        obs = env.unwrapped.state
    else:
        obs, _ = env.reset()
    
    # Run the agent and track positions and velocities
    positions = [obs[0]]
    velocities = [obs[1]]
    done = False
    steps = 0
    
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        positions.append(obs[0])
        velocities.append(obs[1])
        steps += 1
        done = terminated or truncated
    
    # Plot the trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(positions, velocities, 'b-', alpha=0.7)
    plt.scatter(positions[0], velocities[0], c='g', s=100, label='Start')
    plt.scatter(positions[-1], velocities[-1], c='r', s=100, label='End')
    
    # Add environment boundaries
    plt.axvline(x=env.observation_space.low[0], color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=env.observation_space.high[0], color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=env.observation_space.low[1], color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=env.observation_space.high[1], color='k', linestyle='--', alpha=0.5)
    
    # Mark goal position
    plt.axvline(x=0.45, color='g', linestyle='--', alpha=0.7, label='Goal')
    
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Agent Trajectory in State Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    
    return save_path, positions, velocities
