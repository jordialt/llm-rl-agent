"""
train_agent.py
---------------
A minimal Reinforcement Learning agent using Stable-Baselines3 (PPO)
on the CartPole-v1 environment, with natural language feedback
from an LLM helper (google/flan-t5-small).
"""

import gymnasium as gym
from stable_baselines3 import PPO
from llm_helper import LLMHelper


def describe_state(observation):
    """
    Convert CartPole state into a simple English description.
    Observation = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    """
    cart_pos, cart_vel, pole_angle, pole_vel = observation
    desc = f"The pole angle is {pole_angle:.2f} radians and the cart position is {cart_pos:.2f}."
    if pole_angle > 0.05:
        desc += " The pole is leaning to the right."
    elif pole_angle < -0.05:
        desc += " The pole is leaning to the left."
    else:
        desc += " The pole is nearly vertical."
    return desc


def main():
    # 1️⃣ Initialize environment and LLM
    env = gym.make("CartPole-v1")
    llm = LLMHelper()

    # 2️⃣ Define PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu"
    )

    print("\n🚀 Starting training with LLM commentary...\n")

    num_episodes = 5  # keep small for demo
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        # 🧠 LLM generates commentary
        state_desc = describe_state(obs)
        prompt = f"{state_desc} Give one line of advice to improve performance."
        advice = llm.generate(prompt)

        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Reward: {total_reward:.2f}")
        print(f"LLM advice: {advice}\n")

        # Continue learning
        model.learn(total_timesteps=10_000)

    print("\n✅ Training completed with LLM commentary.\n")
    model.save("ppo_cartpole_llm")
    env.close()


if __name__ == "__main__":
    main()
