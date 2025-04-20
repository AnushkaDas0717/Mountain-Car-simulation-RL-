"""
Train a SAC (Soft Actor-Critic) model on the MountainCarContinuous-v0 environment.
This script should be run once to train and save the model.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

def train_agent(total_timesteps=200000):
    """Train the SAC agent on the Mountain Car environment"""
    
    # Create and monitor environment
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env, "logs/")
    
    # Create evaluation environment
    eval_env = gym.make("MountainCarContinuous-v0")
    eval_env = Monitor(eval_env, "logs/eval/")
    
    # Setup noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/checkpoints/",
        name_prefix="sac_mountaincar"
    )
    
    # Create the model
    model = SAC(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        verbose=1, 
        tensorboard_log="logs/",
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[eval_callback, checkpoint_callback],
        log_interval=10
    )
    
    # Save the final model
    model.save("models/final_model")
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot learning curve
    plot_results(["logs/"], total_timesteps, results_plotter.X_TIMESTEPS, "MountainCarContinuous-v0")
    plt.savefig("logs/learning_curve.png")
    plt.close()
    
    return model

def evaluate_policy(model, env, n_eval_episodes=10):
    """Evaluate the trained policy"""
    episode_rewards = []
    
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward

if __name__ == "__main__":
    print("Training Mountain Car Continuous with SAC...")
    model = train_agent()
    print("Training complete! Model saved to 'models/final_model'")
