import dataclasses
from datetime import datetime
from typing import List
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import os
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip_module
import matplotlib.pyplot as plt


# --- Configuration ---
@dataclasses.dataclass
class Config:
  env_name: str = "LunarLander-v3"
  learning_rate: float = 1e-3
  gamma: float = 0.99
  seed: int = 42
  num_episodes: int = 5000
  eval_interval: int = 100
  eval_num_episodes: int = 1
  hidden_dim: int = 128
  batch_size: int = 500


# --- Policy Network ---
class PolicyNetwork(nnx.Module):
  def __init__(self, action_dim: int, config: Config, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(8, config.hidden_dim, rngs=rngs)
    self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
    self.fc3 = nnx.Linear(config.hidden_dim, action_dim, rngs=rngs)

  @jax.jit
  def __call__(self, x: jax.Array) -> jax.Array:
    x = nnx.relu(self.fc1(x))
    x = nnx.relu(self.fc2(x))
    logits = self.fc3(x)
    return logits

  @jax.jit
  def sample(self, x: jax.Array, rngs: nnx.Rngs):
    logits = self(x)
    action = rngs.categorical(logits)
    return action


# --- Value Network ---
class ValueNetwork(nnx.Module):
  def __init__(self, config: Config, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(8, config.hidden_dim, rngs=rngs)
    self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
    self.fc3 = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

  @jax.jit
  def __call__(self, x: jax.Array) -> jax.Array:
    x = nnx.relu(self.fc1(x))
    x = nnx.relu(self.fc2(x))
    value = self.fc3(x)
    return value.squeeze(-1)


# --- Training ---
@jax.jit
def train_step(
  policy_model: PolicyNetwork,
  value_model: ValueNetwork,
  policy_optimizer: nnx.Optimizer,
  value_optimizer: nnx.Optimizer,
  states: jax.Array,
  actions: jax.Array,
  returns: jax.Array,
):
  # Compute advantages using current value estimates (no gradient flow to value model from policy loss)
  values = value_model(states)
  advantages = returns - values

  # Policy Update
  def policy_loss_fn(policy_model):
    logits = policy_model(states)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(
      log_probs, actions[:, None], axis=-1
    ).squeeze(-1)
    # Loss is negative log likelihood scaled by advantages
    loss = -jnp.mean(action_log_probs * advantages)
    return loss

  policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(nnx.as_immutable_vars(policy_model))
  policy_optimizer.update(policy_model, policy_grads)

  # Value Update
  def value_loss_fn(value_model):
    values = value_model(states)
    loss = jnp.mean((values - returns) ** 2)
    return loss

  value_loss, value_grads = jax.value_and_grad(value_loss_fn)(nnx.as_immutable_vars(value_model))
  value_optimizer.update(value_model, value_grads)

  return policy_loss, value_loss


def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
  rewards = np.array(rewards, dtype=np.float32)
  discounts = gamma ** np.arange(len(rewards))
  discounted_rewards = rewards * discounts
  returns = np.cumsum(discounted_rewards[::-1])[::-1] / discounts

  # Normalize returns for stability
  if len(returns) > 1:
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
  return returns


# --- Live Plotting ---
class LivePlotter:
  def __init__(self):
    self.steps = []
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
    self.ax.set_title("Training Progress (REINFORCE + Baseline)")
    self.ax.legend()

    self.ax.spines["top"].set_visible(False)
    self.ax.spines["right"].set_visible(False)

  def update(self, step, reward):
    self.steps.append(step)
    self.rewards.append(reward)

    # Calculate EMA
    if not self.ema_rewards:
      self.ema_rewards.append(reward)
    else:
      ema = self.alpha * reward + (1 - self.alpha) * self.ema_rewards[-1]
      self.ema_rewards.append(ema)

    # Update scatter plot
    if len(self.steps) % 20 == 0:
      self.scatter.set_offsets(np.c_[self.steps, self.rewards])

      # Update EMA line
      self.ema_line.set_xdata(self.steps)
      self.ema_line.set_ydata(self.ema_rewards)

      # Update limits
      self.ax.set_xlim(0, len(self.steps) + 1)

      min_reward = min(self.rewards)
      max_reward = max(self.rewards)
      padding = (max_reward - min_reward) * 0.1 if max_reward != min_reward else 1.0
      self.ax.set_ylim(min_reward - padding, max_reward + padding)

      self.fig.canvas.draw()
      self.fig.canvas.flush_events()


# --- Evaluation ---
def evaluate(
  model: PolicyNetwork,
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
      logits = model(state_jax)
      action = logits.argmax(axis=-1).item()  # Greedy action for evaluation

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
  run_name = f"reinforce_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  print(f"Run name: {run_name}")

  env = gym.make(config.env_name)

  nnx.use_hijax(True)
  rngs = nnx.Rngs(config.seed)

  policy_model = PolicyNetwork(action_dim=4, config=config, rngs=rngs)
  value_model = ValueNetwork(config=config, rngs=rngs)
  
  policy_optimizer = nnx.Optimizer(policy_model, optax.adam(config.learning_rate), wrt=nnx.Param)
  value_optimizer = nnx.Optimizer(value_model, optax.adam(config.learning_rate), wrt=nnx.Param)
  
  plotter = LivePlotter()

  print("Starting training...")
  for episode in range(config.num_episodes):
    state, _ = env.reset()
    done = False

    states = []
    actions = []
    rewards = []

    # Collect trajectory
    while not done:
      # Sample action
      action = policy_model.sample(state, rngs).item()

      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      states.append(state)
      actions.append(action)
      rewards.append(reward)

      state = next_state

    # Compute returns
    returns = compute_returns(rewards, config.gamma)

    # Update policy
    states_jax = np.asarray(states)
    actions_jax = np.asarray(actions)
    returns_jax = np.asarray(returns)

    # Subsample to fixed batch size to avoid recompilation
    if len(states) > 0:
      indices = np.random.choice(
        len(states), config.batch_size, replace=len(states) < config.batch_size
      )
      states_jax = states_jax[indices]
      actions_jax = actions_jax[indices]
      returns_jax = returns_jax[indices]

      policy_loss, value_loss = train_step(
          policy_model, value_model, policy_optimizer, value_optimizer, 
          states_jax, actions_jax, returns_jax
      )
    else:
      policy_loss, value_loss = 0.0, 0.0

    total_reward = sum(rewards)
    plotter.update(episode, total_reward)

    if episode % 10 == 0:
      print(f"Episode {episode}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Total Reward: {total_reward:.2f}")

    # Evaluation
    if (episode + 1) % config.eval_interval == 0:
      avg_reward = evaluate(policy_model, config, run_name, episode + 1)
      print(f"Eval at episode {episode + 1}: Avg Reward = {avg_reward:.2f}")

  env.close()
  plt.ioff()
  print("Done. Close the plot window to exit.")
  plt.show()


if __name__ == "__main__":
  main()
