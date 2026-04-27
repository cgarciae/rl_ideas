from typing import Callable
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
from typing import Tuple
import dataclasses
from datetime import datetime
import tyro


# --- Configuration ---
@dataclasses.dataclass
class Config:
  env_name: str = "LunarLander-v3"
  hidden_dim: int = 256
  learning_rate: float = 1e-3
  batch_size: int = 256
  num_episodes_data: int = 5000  # Increased to get enough high-quality data
  train_steps: int = 50_000
  eval_interval: int = 1000
  data_collection_interval: int = 1000  # Collect new data every 1000 steps
  num_episodes_per_collection: int = 100  # Fixed number of episodes to collect each time
  max_dataset_size: int = 200_000  # Maximum dataset size (fixed buffer)
  eval_num_episodes: int = 10  # Number of episodes to evaluate
  target_return: float = 200.0  # Target return for LunarLander
  render: bool = True


# --- 1. Model Definition (RvS MLP) ---
class RvSMLP(nnx.Module):
  def __init__(self, action_dim: int, config: Config, rngs: nnx.Rngs):
    self.config = config
    self.fc1 = nnx.Linear(8 + 1, config.hidden_dim, rngs=rngs)  # State (8) + Return (1)
    self.ln1 = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
    self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
    self.ln2 = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
    self.fc3 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
    self.ln3 = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
    self.fc4 = nnx.Linear(config.hidden_dim, action_dim, rngs=rngs)

  def __call__(self, state: jax.Array, target_return: jax.Array) -> jax.Array:
    # Concatenate state and target return
    # state: (B, 8), target_return: (B, 1)
    x = jnp.concatenate([state, target_return], axis=-1)
    x = nnx.relu(self.ln1(self.fc1(x)))
    x = nnx.relu(self.ln2(self.fc2(x)))
    x = nnx.relu(self.ln3(self.fc3(x)))
    x = self.fc4(x)
    return x


# --- 2. Data Collection (Offline Data) ---
def collect_data(
  config: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
  print(f"Collecting data from {config.num_episodes_data} random episodes...")
  env = gym.make(config.env_name)

  all_episodes = []  # Store (states, actions, returns_to_go, total_return)

  for _ in range(config.num_episodes_data):
    state, _ = env.reset()
    done = False
    episode_states = []
    episode_actions = []
    episode_rewards = []

    while not done:
      action = env.action_space.sample()  # Random policy
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      episode_states.append(state)
      episode_actions.append(action)
      episode_rewards.append(reward)
      state = next_state

    # Compute Reward-to-Go
    # R_t = r_t + r_{t+1} + ... + r_T
    T = len(episode_rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running_sum = 0.0
    for t in reversed(range(T)):
      running_sum += episode_rewards[t]
      rtg[t] = running_sum

    total_return = sum(episode_rewards)
    all_episodes.append((episode_states, episode_actions, rtg, total_return))

  env.close()

  # Filter to keep at most max_dataset_size transitions (best episodes)
  all_episodes.sort(key=lambda x: x[3], reverse=True)
  
  top_episodes = []
  total_transitions = 0
  for ep in all_episodes:
    ep_len = len(ep[0])
    if total_transitions + ep_len > config.max_dataset_size:
      break
    top_episodes.append(ep)
    total_transitions += ep_len

  if not top_episodes:
      # Keep at least one if none fit (unlikely)
      top_episodes = [all_episodes[0]]

  print(
    f"Keeping top {len(top_episodes)} episodes ({total_transitions} transitions). Min return: {top_episodes[-1][3]:.2f}, Max return: {top_episodes[0][3]:.2f}"
  )

  # Flatten
  final_states = []
  final_actions = []
  final_rtgs = []

  for states, actions, rtgs, _ in top_episodes:
    final_states.extend(states)
    final_actions.extend(actions)
    final_rtgs.extend(rtgs)

  states_arr = np.array(final_states, dtype=np.float32)
  actions_arr = np.array(final_actions, dtype=np.int32)
  rtgs_arr = np.array(final_rtgs, dtype=np.float32).reshape(-1, 1)

  # Compute stats for normalization
  state_mean = np.mean(states_arr, axis=0)
  state_std = np.std(states_arr, axis=0) + 1e-6
  
  # Simple scaling for returns: divide by a fixed factor (e.g., 200 for LunarLander)
  # This keeps the scale manageable without distorting positive/negative returns too much
  rtg_scale = 200.0 

  # Normalize data
  states_arr = (states_arr - state_mean) / state_std
  rtgs_arr = rtgs_arr / rtg_scale

  return states_arr, actions_arr, rtgs_arr, state_mean, state_std, rtg_scale


# --- 2b. Collect Data with Policy ---
def collect_policy_data(
    forward: Callable,
    config: Config,
    state_mean,
    state_std,
    rtg_scale,
    num_episodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Collect data using the current policy."""
  print(f"Collecting {num_episodes} episodes with current policy...")
  env = gym.make(config.env_name)

  all_episodes = []

  for _ in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_states = []
    episode_actions = []
    episode_rewards = []

    current_target = config.target_return

    while not done:
      # Normalize state
      state_norm = (state - state_mean) / state_std
      target_norm = current_target / rtg_scale

      state_jax = jnp.array(state_norm)[None, ...]
      target_jax = jnp.array([target_norm], dtype=jnp.float32)[None, ...]

      logits = forward(state_jax, target_jax)
      # Stochastic sampling for exploration
      probs = jax.nn.softmax(logits, axis=-1)
      action = np.random.choice(probs.shape[-1], p=np.array(probs[0]))

      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      episode_states.append(state)
      episode_actions.append(action)
      episode_rewards.append(reward)

      current_target -= reward
      state = next_state

    # Compute returns-to-go
    T = len(episode_rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running_sum = 0.0
    for t in reversed(range(T)):
      running_sum += episode_rewards[t]
      rtg[t] = running_sum

    total_return = sum(episode_rewards)
    all_episodes.append((episode_states, episode_actions, rtg, total_return))

  env.close()

  # Flatten all episodes (no filtering here, filtering happens in main)
  final_states = []
  final_actions = []
  final_rtgs = []

  for states, actions, rtgs, _ in all_episodes:
    final_states.extend(states)
    final_actions.extend(actions)
    final_rtgs.extend(rtgs)

  states_arr = np.array(final_states, dtype=np.float32)
  actions_arr = np.array(final_actions, dtype=np.int32)
  rtgs_arr = np.array(final_rtgs, dtype=np.float32).reshape(-1, 1)

  # Normalize using existing stats
  states_arr = (states_arr - state_mean) / state_std
  rtgs_arr = rtgs_arr / rtg_scale

  return states_arr, actions_arr, rtgs_arr


# --- 3. Training ---
def get_train_step(model: RvSMLP, optimizer: nnx.Optimizer):
  @jax.jit
  def train_step(states, actions, returns_to_go):
    def loss_fn(model):
      logits = model(states, returns_to_go)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()
      return loss

    loss, grads  = jax.value_and_grad(loss_fn)(nnx.as_immutable_vars(model))
    optimizer.update(model, grads)

    return loss

  return train_step

def get_forward(model: RvSMLP):
  @jax.jit
  def forward(states, returns_to_go):
    return model(states, returns_to_go)

  return forward

# --- 4. Evaluation ---
def evaluate(
  forward: Callable,
  config: Config,
  state_mean,
  state_std,
  rtg_scale,
  run_name: str,
  step: int,
):
  """Evaluate the model and save video of the best episode."""
  env = gym.make(config.env_name, render_mode="rgb_array")

  all_episodes = []  # Store (reward, frames) for each episode

  for ep_idx in range(config.eval_num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0.0
    frames = []

    current_target = config.target_return

    while not done:
      # Normalize state
      state_norm = (state - state_mean) / state_std

      # Normalize target
      target_norm = current_target / rtg_scale

      state_jax = jnp.array(state_norm)[None, ...]  # Add batch dim
      target_jax = jnp.array([target_norm], dtype=jnp.float32)[None, ...]

      logits = forward(state_jax, target_jax)
      action = logits.argmax(axis=-1).item()

      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      # Capture frame
      frame = env.render()
      frames.append(frame)

      episode_reward += reward
      current_target -= reward

      state = next_state

    all_episodes.append((episode_reward, frames))

  env.close()

  # Find best episode
  best_idx = np.argmax([ep[0] for ep in all_episodes])
  best_reward, best_frames = all_episodes[best_idx]

  # Save video of best episode
  import os
  import moviepy.video.io.ImageSequenceClip as ImageSequenceClip_module

  video_dir = f"videos/{run_name}/step_{step}"
  os.makedirs(video_dir, exist_ok=True)

  clip = ImageSequenceClip_module.ImageSequenceClip(best_frames, fps=30)
  clip.write_videofile(
    f"{video_dir}/best_episode_reward_{best_reward:.2f}.mp4",
    codec="libx264",
    logger=None,
  )

  # Return average reward across all episodes
  avg_reward = np.mean([ep[0] for ep in all_episodes])
  return avg_reward


def main():
  config = tyro.cli(Config)
  run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
  print(f"Run name: {run_name}")

  # 1. Collect Data
  states, actions, rtgs, s_mean, s_std, r_scale = collect_data(config)
  print(f"Data collected: {len(states)} transitions.")
  print(f"Stats - State Mean: {s_mean[:2]}..., Std: {s_std[:2]}...")
  print(f"Stats - RTG Scale: {r_scale:.2f}")

  # 2. Setup Model & Optimizer
  nnx.use_hijax(True)
  rngs = nnx.Rngs(0)
  model = RvSMLP(action_dim=4, config=config, rngs=rngs)  # 4 actions for LunarLander
  optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate), wrt=nnx.Param)
  train_step = get_train_step(model, optimizer)
  forward = get_forward(model)

  # 3. Training Loop
  print("Starting training...")
  dataset_size = len(states)
  indices = np.arange(dataset_size)

  for step in range(config.train_steps):
    batch_idx = np.random.choice(indices, size=config.batch_size)
    batch_states = jnp.array(states[batch_idx])
    batch_actions = jnp.array(actions[batch_idx])
    batch_rtgs = jnp.array(rtgs[batch_idx])

    loss = train_step(batch_states, batch_actions, batch_rtgs)

    if (step + 1) % 1000 == 0:
      print(f"Step {step + 1}, Loss: {loss:.4f}")

    if (step + 1) % config.eval_interval == 0:
      avg_reward = evaluate(
        forward, config, s_mean, s_std, r_scale, run_name, step + 1
      )
      print(f"Eval at step {step + 1}: Avg Reward = {avg_reward:.2f}")

    # Periodic data collection
    if (step + 1) % config.data_collection_interval == 0:
      print(f"Collecting {config.num_episodes_per_collection} episodes...")
      
      new_states, new_actions, new_rtgs = collect_policy_data(
          forward,
          config,
          s_mean,
          s_std,
          r_scale,
          config.num_episodes_per_collection,
      )

      # Concatenate new data to dataset
      states = np.concatenate([states, new_states], axis=0)
      actions = np.concatenate([actions, new_actions], axis=0)
      rtgs = np.concatenate([rtgs, new_rtgs], axis=0)

      # If dataset exceeds max size, keep only the best transitions (by return-to-go)
      if len(states) > config.max_dataset_size:
        print(f"Dataset size {len(states)} exceeds max {config.max_dataset_size}. Filtering to keep best transitions...")
        
        # Sort by return-to-go (descending) and keep top max_dataset_size
        rtg_values = rtgs.flatten()
        sorted_indices = np.argsort(rtg_values)[::-1]
        top_indices = sorted_indices[:config.max_dataset_size]
        
        states = states[top_indices]
        actions = actions[top_indices]
        rtgs = rtgs[top_indices]
        
        print(f"Kept top {config.max_dataset_size} transitions. Min RTG: {rtgs.min():.2f}, Max RTG: {rtgs.max():.2f}")

      # Update indices
      dataset_size = len(states)
      indices = np.arange(dataset_size)
      print(f"Dataset size after collection: {dataset_size} transitions (added {len(new_states)})")

  print("Done.")


if __name__ == "__main__":
  main()
