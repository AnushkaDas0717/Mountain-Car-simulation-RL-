"""
Streamlit app for the Mountain Car Continuous RL project.
This app allows users to visualize and interact with the trained model.
"""

import streamlit as st
import numpy as np
import gymnasium as gym
import os
import base64
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
from utils import create_video, plot_state_value_heatmap, plot_trajectory

# Page configuration
st.set_page_config(
    page_title="Mountain Car RL",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🚗 Mountain Car Continuous RL Demo")
st.markdown("""
This demo showcases a trained Soft Actor-Critic (SAC) reinforcement learning agent
solving the Mountain Car Continuous environment. The agent learns to apply the right
amount of force to drive up the steep mountain.
""")

# Check if model exists
MODEL_PATH = "models/final_model.zip"
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Trained model not found! Please run 'train_model.py' first.")

    if st.button("Train Model Now"):
        st.info("Training model... This might take a while (~10-15 minutes).")
        try:
            from train_model import train_agent
            with st.spinner("Training in progress..."):
                model = train_agent(total_timesteps=50000)  # Reduced timesteps for demo
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Error training model: {e}")
else:
    st.success("✅ Trained model found!")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Simulation", "State Value Analysis", "Trajectory Analysis"])

    with tab1:
        st.header("Mountain Car Simulation")

        # Simulation controls
        st.subheader("Simulation Settings")
        col1, col2 = st.columns(2)

        with col1:
            use_custom_state = st.checkbox("Use custom initial state")

        with col2:
            if use_custom_state:
                position = st.slider("Initial Position", -1.2, 0.6, -0.5, 0.01)
                velocity = st.slider("Initial Velocity", -0.07, 0.07, 0.0, 0.01)
                initial_state = [position, velocity]
            else:
                initial_state = None

        # Run simulation button
        if st.button("🚀 Run Simulation"):
            with st.spinner("Generating simulation..."):
                try:
                    video_path, reward, steps = create_video(
                        MODEL_PATH,
                        initial_state=initial_state
                    )

                    # Display metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Total Reward", f"{reward:.2f}")
                    col2.metric("Steps to Complete", steps)

                    # Display video
                    video_file = open(video_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)

                except Exception as e:
                    st.error(f"Error generating simulation: {e}")

    with tab2:
        st.header("State Value Analysis")
        st.markdown("""
        This visualization shows how the agent values different states in the environment.
        Brighter areas indicate states the agent considers more valuable.
        """)

        if st.button("Generate State Value Heatmap"):
            with st.spinner("Generating heatmap..."):
                try:
                    heatmap_path = plot_state_value_heatmap(MODEL_PATH)
                    if heatmap_path:
                        st.image(heatmap_path)
                        st.success("Heatmap generated successfully!")
                    else:
                        st.warning("Heatmap generation failed. This may happen if PyTorch is not properly installed.")
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")

    with tab3:
        st.header("Trajectory Analysis")
        st.markdown("""
        This visualization shows the path the agent takes through the state space.
        The green dot is the starting position, and the red dot is the ending position.
        """)

        # Custom initial state for trajectory
        use_custom_traj = st.checkbox("Use custom initial state for trajectory")
        if use_custom_traj:
            col1, col2 = st.columns(2)
            with col1:
                pos_traj = st.slider("Initial Position (Trajectory)", -1.2, 0.6, -0.9, 0.01)
            with col2:
                vel_traj = st.slider("Initial Velocity (Trajectory)", -0.07, 0.07, 0.0, 0.01)
            traj_initial_state = [pos_traj, vel_traj]
        else:
            traj_initial_state = None

        if st.button("Generate Trajectory Plot"):
            with st.spinner("Generating trajectory..."):
                try:
                    traj_path, positions, velocities = plot_trajectory(
                        MODEL_PATH,
                        initial_state=traj_initial_state
                    )

                    st.image(traj_path)

                    # Position and velocity over time
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                    # Position plot
                    ax1.plot(positions)
                    ax1.set_ylabel('Position')
                    ax1.set_title('Position over time')
                    ax1.grid(True, alpha=0.3)

                    # Velocity plot
                    ax2.plot(velocities)
                    ax2.set_xlabel('Time steps')
                    ax2.set_ylabel('Velocity')
                    ax2.set_title('Velocity over time')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error generating trajectory plot: {e}")

    # Add sidebar info
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This interactive demo showcases a reinforcement learning agent that has been trained
    to solve the Mountain Car Continuous environment from Gymnasium.

    ### The Challenge
    - The car must drive up a steep mountain
    - The car's engine is not strong enough to drive up directly
    - The agent must learn to build momentum by driving back and forth

    ### The Agent
    - Trained using SAC (Soft Actor-Critic) algorithm
    - Learns to apply the right amount of force at the right time
    - Outputs continuous actions (force applied to the car)
    """)

    # Add environment info
    st.sidebar.header("Environment Details")
    st.sidebar.markdown("""
    **Observation Space:**
    - Position: -1.2 to 0.6
    - Velocity: -0.07 to 0.07

    **Action Space:**
    - Force: -1.0 to 1.0

    **Reward:**
    - Negative reward for energy spent
    - +100 reward for reaching the goal
    """)


