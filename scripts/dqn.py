import dataclasses
from datetime import datetime
from typing import List, Tuple, Deque
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import os
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip_module
import matplotlib.pyplot as plt
import collections
import random

nnx.set_graph_mode(False) # tree mode

# --- Configuration ---
@dataclasses.dataclass
class Config:
  env_name: str = "LunarLander-v3"
  learning_rate: float = 1e-3
  gamma: float = 0.99
  seed: int = 42
  num_episodes: int = 2000
  eval_interval: int = 20
  eval_num_episodes: int = 3
  hidden_dim: int = 128
  batch_size: int = 64
  buffer_size: int = 100000
  epsilon_start: float = 1.0
  epsilon_end: float = 0.01
  epsilon_decay: float = 0.995
  target_update_rate: float = 0.005  # Soft update tau
  train_frequency: int = 4
  start_learning_steps: int = 1000


# --- Replay Buffer ---
class ReplayBuffer:
  def __init__(self, capacity: int, seed: int):
    self.buffer = collections.deque(maxlen=capacity)
    self.rng = random.Random(seed)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
    batch = self.rng.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
      np.array(states),
      np.array(actions),
      np.array(rewards),
      np.array(next_states),
      np.array(dones, dtype=np.float32),
    )

  def __len__(self):
    return len(self.buffer)


# --- Q-Network ---
class QNetwork(nnx.Module):
  def __init__(self, action_dim: int, config: Config, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(8, config.hidden_dim, rngs=rngs)
    self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
    self.fc3 = nnx.Linear(config.hidden_dim, action_dim, rngs=rngs)

  @jax.jit
  def __call__(self, x: jax.Array) -> jax.Array:
    x = nnx.relu(self.fc1(x))
    x = nnx.relu(self.fc2(x))
    q_values = self.fc3(x)
    return q_values


# --- Training ---
@jax.jit
def train_step(
  online_net: QNetwork,
  target_net: QNetwork,
  optimizer: nnx.Optimizer,
  states: jax.Array,
  actions: jax.Array,
  rewards: jax.Array,
  next_states: jax.Array,
  dones: jax.Array,
  gamma: float,
  tau: float,
):
  def loss_fn(online_net):
    # Q(s, a)
    q_values = online_net(states)
    q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)

    # Target Q(s', a')
    # Double DQN could be implemented here, but sticking to standard DQN for now
    # Or standard DQN: max_a' Q_target(s', a')
    next_q_values = target_net(next_states)
    max_next_q = jnp.max(next_q_values, axis=-1)

    target = rewards + gamma * max_next_q * (1 - dones)

    # MSE Loss
    loss = jnp.mean((q_action - target) ** 2)
    return loss

  loss, grads = jax.value_and_grad(loss_fn)(nnx.as_immutable_vars(online_net))
  optimizer.update(online_net, grads)

  # soft update
  def interpolate(target_param, online_param):
    target_param[...] = (1 - tau) * target_param + tau * online_param
  jax.tree.map(interpolate, target_net, online_net)
  
  return loss


# --- Live Plotting ---
class LivePlotter:
  def __init__(self):
    self.episodes = []
    self.rewards = []
    self.ema_rewards = []
    self.alpha = 0.05

    plt.ion()
    self.fig, self.ax = plt.subplots()
    self.scatter = self.ax.scatter(
      [], [], c="b", alpha=0.3, s=10, label="Episode Reward"
    )
    (self.ema_line,) = self.ax.plot([], [], "k", linewidth=2, label="EMA")
    self.ax.set_xlabel("Episode")
    self.ax.set_ylabel("Reward")
    self.ax.set_title("Training Progress (DQN)")
    self.ax.legend()

    self.ax.spines["top"].set_visible(False)
    self.ax.spines["right"].set_visible(False)

  def update(self, episode, reward):
    self.episodes.append(episode)
    self.rewards.append(reward)

    # Calculate EMA
    if not self.ema_rewards:
      self.ema_rewards.append(reward)
    else:
      ema = self.alpha * reward + (1 - self.alpha) * self.ema_rewards[-1]
      self.ema_rewards.append(ema)

    # Update scatter plot
    if len(self.episodes) % 5 == 0:
      self.scatter.set_offsets(np.c_[self.episodes, self.rewards])

      # Update EMA line
      self.ema_line.set_xdata(self.episodes)
      self.ema_line.set_ydata(self.ema_rewards)

      # Update limits
      self.ax.set_xlim(0, len(self.episodes) + 1)

      min_reward = min(self.rewards)
      max_reward = max(self.rewards)
      padding = (max_reward - min_reward) * 0.1 if max_reward != min_reward else 1.0
      self.ax.set_ylim(min_reward - padding, max_reward + padding)

      self.fig.canvas.draw()
      self.fig.canvas.flush_events()


# --- Evaluation ---
def evaluate(
  model: QNetwork,
  config: Config,
  run_name: str,
  episode: int,
) -> float:
  env = gym.make(config.env_name, render_mode="rgb_array")
  rewards = []
  best_reward = -float("inf")
  best_frames = []

  for _ in range(config.eval_num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    frames = []

    while not done:
      state_jax = jnp.array(state)[None, ...]
      q_values = model(state_jax)
      action = q_values.argmax(axis=-1).item()  # Greedy action

      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      frames.append(env.render())
      total_reward += reward
      state = next_state

    rewards.append(total_reward)
    if total_reward > best_reward:
      best_reward = total_reward
      best_frames = frames

  env.close()

  # Save video of best episode
  if best_frames:
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    clip = ImageSequenceClip_module.ImageSequenceClip(best_frames, fps=30)
    clip.write_videofile(
      f"{video_dir}/episode_{episode}_reward_{best_reward:.2f}.mp4",
      codec="libx264",
      logger=None,
    )

  return np.mean(rewards)


# --- Main ---
def main():
  config = Config()
  np.random.seed(config.seed)
  run_name = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  print(f"Run name: {run_name}")

  env = gym.make(config.env_name)

  rngs = nnx.Rngs(config.seed)

  # Initialize Networks
  online_net = QNetwork(action_dim=4, config=config, rngs=rngs)
  target_net = QNetwork(
    action_dim=4, config=config, rngs=rngs
  )  # Independent initialization, will sync immediately

  # Sync target network
  nnx.update(target_net, nnx.state(online_net))

  optimizer = nnx.Optimizer(online_net, optax.adam(config.learning_rate), wrt=nnx.Param)

  buffer = ReplayBuffer(config.buffer_size, config.seed)
  plotter = LivePlotter()

  epsilon = config.epsilon_start
  steps = 0

  print("Starting training...")
  for episode in range(config.num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    loss = 0

    while not done:
      # Epsilon-Greedy Action Selection
      if random.random() < epsilon:
        action = env.action_space.sample()
      else:
        q_values = online_net(state)
        action = q_values.argmax(axis=-1).item()

      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      buffer.add(state, action, reward, next_state, done)

      state = next_state
      episode_reward += reward
      steps += 1

      # Training Step
      if steps > config.start_learning_steps and steps % config.train_frequency == 0:
        states, actions, rewards_batch, next_states, dones = buffer.sample(
          config.batch_size
        )

        loss = train_step(
          online_net,
          target_net,
          optimizer,
          states,
          actions,
          rewards_batch,
          next_states,
          dones,
          config.gamma,
          config.target_update_rate,
        )

    # Decay Epsilon
    epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    plotter.update(episode, episode_reward)

    if episode % 10 == 0:
      print(
        f"Episode {episode}, Steps: {steps}, Epsilon: {epsilon:.3f}, Loss: {loss:.4f}, Reward: {episode_reward:.2f}"
      )

    # Evaluation
    if (episode + 1) % config.eval_interval == 0:
      avg_reward = evaluate(online_net, config, run_name, episode + 1)
      print(f"Eval at episode {episode + 1}: Avg Reward = {avg_reward:.2f}")

  env.close()
  plt.ioff()
  print("Done. Close the plot window to exit.")
  plt.show()


if __name__ == "__main__":
  main()
