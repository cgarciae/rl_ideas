import dataclasses
from datetime import datetime
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import os
import random
import tyro
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip_module
import matplotlib.pyplot as plt

nnx.set_graph_mode(False)  # tree mode


# --- Configuration ---
@dataclasses.dataclass
class Config:
  env_name: str = "LunarLander-v3"
  learning_rate: float = 3e-4
  seed: int = 42
  hidden_dim: int = 256
  batch_size: int = 512
  # Phase 1: initial random seed
  num_random_episodes: int = 200
  # Phase 3: SFT training per iteration
  train_steps_per_iter: int = 50
  dropout_rate: float = 0.1
  # Phase 4: rollout
  num_rollout_episodes: int = 50
  temperature: float = 1.0
  elite_fraction: float = 0.3
  percent_training: float = 0.3
  random_eviction_fraction: float = 0.0
  discard_previous: bool = False
  # Phase 5: buffer management
  cull_fraction: float = 0.05
  max_buffer_trajectories: int = 2000
  # Main loop
  num_iterations: int = 500
  # Eval
  eval_interval: int = 5
  eval_num_episodes: int = 3
  render: bool = True


# --- Trajectory ---
@dataclasses.dataclass
class Trajectory:
  states: list[np.ndarray]
  actions: list[int]
  rewards: list[float]
  rtgs: list[float] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if not self.rtgs:
      running = 0.0
      rtgs = []
      for r in reversed(self.rewards):
        running += r
        rtgs.append(running)
      self.rtgs = list(reversed(rtgs))

  @property
  def total_reward(self) -> float:
    return self.rtgs[0] if self.rtgs else 0.0


# --- Sorted Replay Buffer ---
class SortedBuffer:
  _flat_states: np.ndarray
  _flat_actions: np.ndarray
  _flat_rtgs: np.ndarray

  def __init__(self, max_trajectories: int):
    self.trajectories: list[Trajectory] = []
    self.max_trajectories = max_trajectories

  def add_batch(self, new_trajectories: list[Trajectory], np_rng: np.random.Generator = None, random_eviction_fraction: float = 0.0, discard_previous: bool = False):
    if discard_previous:
      self.trajectories = []
      self._dirty = True
    N = len(new_trajectories)
    S = self.max_trajectories - len(self.trajectories)
    if S < N:
      n_to_delete = N - S
      n_random = round(n_to_delete * random_eviction_fraction)
      n_worst = n_to_delete - n_random
      if n_random > 0 and np_rng is not None:
        random_indices = set(np_rng.choice(len(self.trajectories), size=n_random, replace=False).tolist())
        self.trajectories = [t for i, t in enumerate(self.trajectories) if i not in random_indices]
      if n_worst > 0:
        del self.trajectories[-n_worst:]
    self.trajectories.extend(new_trajectories)
    self.trajectories.sort(key=lambda t: t.total_reward, reverse=True)
    self._dirty = True

  def _rebuild_flat(self):
    if not self._dirty:
      return
    states, actions, rtgs = [], [], []
    for t in self.trajectories:
      states.extend(t.states)
      actions.extend(t.actions)
      rtgs.extend(t.rtgs)
    self._flat_states = np.array(states, dtype=np.float32)
    self._flat_actions = np.array(actions, dtype=np.int32)
    self._flat_rtgs = np.array(rtgs, dtype=np.float32)
    self._dirty = False

  def get_normalization_stats(self) -> tuple[float, float]:
    self._rebuild_flat()
    return float(self._flat_rtgs.min()), float(self._flat_rtgs.max())

  def get_total_rewards(self) -> np.ndarray:
    return np.array([t.total_reward for t in self.trajectories])

  def get_elite_total_rewards(self, elite_fraction: float) -> np.ndarray:
    n_elite = max(1, int(len(self.trajectories) * elite_fraction))
    return np.array([t.total_reward for t in self.trajectories[:n_elite]])

  def sample_transitions(
    self,
    batch_size: int,
    r_min: float,
    r_max: float,
    np_rng: np.random.Generator,
    top_fraction: float = 1.0,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_fraction < 1.0:
      n_elite = max(1, int(len(self.trajectories) * top_fraction))
      states, actions, rtgs = [], [], []
      for t in self.trajectories[:n_elite]:
        states.extend(t.states)
        actions.extend(t.actions)
        rtgs.extend(t.rtgs)
      flat_states = np.array(states, dtype=np.float32)
      flat_actions = np.array(actions, dtype=np.int32)
      flat_rtgs = np.array(rtgs, dtype=np.float32)
    else:
      self._rebuild_flat()
      flat_states, flat_actions, flat_rtgs = self._flat_states, self._flat_actions, self._flat_rtgs
    n = len(flat_states)
    indices = np_rng.choice(n, size=batch_size, replace=batch_size > n)
    denom = r_max - r_min if r_max != r_min else 1.0
    rtg_norms = np.clip((flat_rtgs[indices] - r_min) / denom, 0.0, 1.0)
    return flat_states[indices], flat_actions[indices], rtg_norms

  def __len__(self):
    return len(self.trajectories)


# --- Policy Network ---
class PolicyNetwork(nnx.Module):
  def __init__(self, state_dim: int, action_dim: int, config: Config, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(state_dim + 1, config.hidden_dim, rngs=rngs)
    self.drop1 = nnx.Dropout(config.dropout_rate, rngs=rngs)
    self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
    self.drop2 = nnx.Dropout(config.dropout_rate, rngs=rngs)
    self.fc3 = nnx.Linear(config.hidden_dim, action_dim, rngs=rngs)

  @nnx.jit
  def __call__(self, states: jax.Array, rtg_norms: jax.Array) -> jax.Array:
    # rtg_norms: (batch,) → (batch, 1) for concatenation
    x = jnp.concatenate([states, rtg_norms[:, None]], axis=-1)
    x = self.drop1(nnx.relu(self.fc1(x)))
    x = self.drop2(nnx.relu(self.fc2(x)))
    return self.fc3(x)


# --- Training Step ---
@nnx.jit
def train_step(
  model: PolicyNetwork,
  optimizer: nnx.Optimizer,
  states: jax.Array,
  actions: jax.Array,
  rtg_norms: jax.Array,
):
  
  def loss_fn(model):
    logits = model(states, rtg_norms)
    return optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()

  loss, grads = jax.value_and_grad(loss_fn, allow_int=True)(model)
  optimizer.update(model, grads)
  return loss


# --- Data Collection ---
def collect_random_episodes(env: gym.Env, num_episodes: int) -> list[Trajectory]:
  trajectories = []
  for _ in range(num_episodes):
    state, _ = env.reset()
    done = False
    states, actions, rewards = [], [], []
    while not done:
      action = env.action_space.sample()
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      states.append(state.copy())
      actions.append(int(action))
      rewards.append(float(reward))
      state = next_state
    trajectories.append(Trajectory(states=states, actions=actions, rewards=rewards))
  return trajectories


def rollout_episode(
  model: PolicyNetwork,
  env: gym.Env,
  rtg_target: float,
  temperature: float,
  r_min: float,
  r_max: float,
  np_rng: np.random.Generator,
  render: bool = False,
) -> tuple["Trajectory", list]:
  state, _ = env.reset()
  done = False
  states, actions, rewards, frames = [], [], [], []
  denom = r_max - r_min if r_max != r_min else 1.0
  rtg_remaining = r_min + rtg_target * denom

  while not done:
    rtg_norm = float(np.clip((rtg_remaining - r_min) / denom, 0.0, 1.0))
    state_jax = jnp.array(state, dtype=jnp.float32)[None]   # (1, 8)
    rtg_jax = jnp.array([rtg_norm], dtype=jnp.float32)       # (1,)
    logits = np.array(model(state_jax, rtg_jax)[0])

    # Numerically stable temperature sampling
    scaled = (logits - logits.max()) / temperature
    probs = np.exp(scaled)
    probs /= probs.sum()
    action = int(np_rng.choice(len(probs), p=probs))

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    states.append(state.copy())
    actions.append(action)
    rewards.append(float(reward))
    if render:
      frames.append(env.render())

    # Decrement and clip remaining RTG
    rtg_remaining = float(np.clip(rtg_remaining - reward, r_min, r_max))
    state = next_state

  return Trajectory(states=states, actions=actions, rewards=rewards), frames


# --- Evaluation ---
def evaluate(
  model: PolicyNetwork,
  config: Config,
  run_name: str,
  iteration: int,
  r_min: float,
  r_max: float,
  elite_total_rewards: np.ndarray,
) -> float:
  env = gym.make(config.env_name, render_mode="rgb_array")
  np_rng = np.random.default_rng(config.seed + iteration * 1000)

  denom = r_max - r_min if r_max != r_min else 1.0
  rtg_target = float(np.clip((np.max(elite_total_rewards) - r_min) / denom, 0.0, 1.0))
  rewards = []
  best_reward = -float("inf")
  best_frames: list = []

  for _ in range(config.eval_num_episodes):
    traj, frames = rollout_episode(
      model, env, rtg_target,
      temperature=0.01,
      r_min=r_min, r_max=r_max,
      np_rng=np_rng, render=True,
    )
    rewards.append(traj.total_reward)
    if traj.total_reward > best_reward:
      best_reward = traj.total_reward
      best_frames = frames

  env.close()

  if best_frames:
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    clip = ImageSequenceClip_module.ImageSequenceClip(best_frames, fps=30)
    clip.write_videofile(
      f"{video_dir}/iter_{iteration}_reward_{best_reward:.2f}.mp4",
      codec="libx264",
      logger=None,
    )

  return float(np.mean(rewards))


# --- Live Plotter ---
class LivePlotter:
  def __init__(self, video_dir: str, render: bool = True):
    self.video_dir = video_dir
    self.render = render
    self.iterations_scatter: list[int] = []
    self.rewards_scatter: list[float] = []
    self.ema_per_iter: list[float] = []
    self.alpha = 0.2

    if render:
      plt.ion()
    self.fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (self.ax_reward, self.ax_dist) = axes[0]
    (self.ax_loss, self.ax_curve) = axes[1]

    # Upper-left: rollout episode rewards with EMA
    self.scatter = self.ax_reward.scatter(
      [], [], c="b", alpha=0.3, s=10, label="Episode Reward"
    )
    (self.ema_line,) = self.ax_reward.plot([], [], "k", linewidth=2, label="EMA")
    self.ax_reward.set_xlabel("Iteration")
    self.ax_reward.set_ylabel("Reward")
    self.ax_reward.set_title("Training Progress (Iterative SFT)")
    self.ax_reward.legend()
    self.ax_reward.spines["top"].set_visible(False)
    self.ax_reward.spines["right"].set_visible(False)

    # Upper-right: RTG distribution of current buffer
    self.ax_dist.set_xlabel("Total Episode Reward (RTG₀)")
    self.ax_dist.set_ylabel("Count")
    self.ax_dist.set_title("Buffer RTG Distribution")
    self.ax_dist.spines["top"].set_visible(False)
    self.ax_dist.spines["right"].set_visible(False)

    # Lower-left: fine-tuning loss
    self.loss_iterations: list[int] = []
    self.loss_values: list[float] = []
    (self.loss_line,) = self.ax_loss.plot([], [], "r", linewidth=1.5)
    self.ax_loss.set_xlabel("Iteration")
    self.ax_loss.set_ylabel("Loss")
    self.ax_loss.set_title("Fine-tuning Loss")
    self.ax_loss.spines["top"].set_visible(False)
    self.ax_loss.spines["right"].set_visible(False)

    # Lower-right: per-step loss curve for current fine-tuning phase
    (self.loss_curve_line,) = self.ax_curve.plot([], [], "r", linewidth=1.0, alpha=0.8)
    self.ax_curve.set_xlabel("Step")
    self.ax_curve.set_ylabel("Loss")
    self.ax_curve.set_title("Fine-tuning Loss (current iteration)")
    self.ax_curve.spines["top"].set_visible(False)
    self.ax_curve.spines["right"].set_visible(False)

    self.fig.tight_layout()

  def update_loss_curve(self, losses: list[float]):
    steps = list(range(1, len(losses) + 1))
    self.loss_curve_line.set_xdata(steps)
    self.loss_curve_line.set_ydata(losses)
    self.ax_curve.set_xlim(1, len(losses))
    self.ax_curve.relim()
    self.ax_curve.autoscale_view()
    if self.render:
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()

  def update_losses(self, iteration: int, losses: list[float]):
    self.loss_iterations.append(iteration)
    self.loss_values.append(float(np.mean(losses)))
    self.loss_line.set_xdata(self.loss_iterations)
    self.loss_line.set_ydata(self.loss_values)
    self.ax_loss.relim()
    self.ax_loss.autoscale_view()
    if self.render:
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()

  def update_rewards(self, iteration: int, episode_rewards: list[float]):
    self.iterations_scatter.extend([iteration] * len(episode_rewards))
    self.rewards_scatter.extend(episode_rewards)

    mean_r = float(np.mean(episode_rewards))
    if not self.ema_per_iter:
      self.ema_per_iter.append(mean_r)
    else:
      self.ema_per_iter.append(
        self.alpha * mean_r + (1 - self.alpha) * self.ema_per_iter[-1]
      )

    self.scatter.set_offsets(np.c_[self.iterations_scatter, self.rewards_scatter])
    iter_axis = list(range(1, len(self.ema_per_iter) + 1))
    self.ema_line.set_xdata(iter_axis)
    self.ema_line.set_ydata(self.ema_per_iter)
    # relim() doesn't account for scatter offsets, so set limits manually
    y_min = min(self.rewards_scatter)
    y_max = max(self.rewards_scatter)
    y_pad = max(1.0, (y_max - y_min) * 0.05)
    x_max = max(self.iterations_scatter)
    self.ax_reward.set_xlim(0, x_max + 1)
    self.ax_reward.set_ylim(y_min - y_pad, y_max + y_pad)

    if self.render:
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()

  def update_distribution(
    self,
    total_rewards: np.ndarray,
    elite_cutoff: float | None = None,
    p50_elite: float | None = None,
  ):
    self.ax_dist.cla()
    self.ax_dist.hist(
      total_rewards, bins=50, color="steelblue", alpha=0.7, edgecolor="none"
    )

    p_max = float(np.max(total_rewards))
    self.ax_dist.axvline(
      p_max, color="green", linestyle="--", linewidth=1.5, label=f"Max={p_max:.0f}"
    )
    if elite_cutoff is not None:
      self.ax_dist.axvline(
        elite_cutoff,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Elite cutoff={elite_cutoff:.0f}",
      )
    if p50_elite is not None:
      self.ax_dist.axvline(
        p50_elite,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"P50 elite={p50_elite:.0f}",
      )

    self.ax_dist.set_xlabel("Total Episode Reward (RTG₀)")
    self.ax_dist.set_ylabel("Count")
    self.ax_dist.set_title(f"Buffer RTG Distribution (n={len(total_rewards)})")
    self.ax_dist.legend(fontsize=8)
    self.ax_dist.spines["top"].set_visible(False)
    self.ax_dist.spines["right"].set_visible(False)

    if self.render:
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()
    self.fig.savefig(os.path.join(self.video_dir, "0_plot.png"), bbox_inches="tight")

  def save_and_close(self, path: str):
    self.fig.tight_layout()
    dir_ = os.path.dirname(path)
    if dir_:
      os.makedirs(dir_, exist_ok=True)
    self.fig.savefig(path)
    plt.close(self.fig)


# --- Main ---
def main():
  config = tyro.cli(Config)
  np.random.seed(config.seed)
  random.seed(config.seed)
  np_rng = np.random.default_rng(config.seed)
  run_name = f"iterative_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  video_dir = f"videos/{run_name}"
  os.makedirs(video_dir, exist_ok=True)
  print(f"Run name: {run_name}")

  env = gym.make(config.env_name)
  state_dim: int = env.observation_space.shape[0]  # 8 for LunarLander
  action_dim: int = env.action_space.n              # 4 for LunarLander

  rngs = nnx.Rngs(config.seed)

  model = PolicyNetwork(
    state_dim=state_dim, action_dim=action_dim, config=config, rngs=rngs
  )
  model_eval = nnx.with_attributes(model, only=nnx.Dropout, deterministic=True)
  optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate), wrt=nnx.Param)

  buffer = SortedBuffer(max_trajectories=config.max_buffer_trajectories)
  plotter = LivePlotter(video_dir, render=config.render)

  # --- Phase 1: Random Data Collection ---
  print(f"Phase 1: Collecting {config.num_random_episodes} random episodes...")
  random_trajectories = collect_random_episodes(env, config.num_random_episodes)
  buffer.add_batch(random_trajectories)
  mean_random = float(np.mean([t.total_reward for t in random_trajectories]))
  print(f"  Random policy mean reward: {mean_random:.2f}")
  plotter.update_distribution(buffer.get_total_rewards())

  # --- Main Iterative Loop ---
  for iteration in range(1, config.num_iterations + 1):

    # Phase 2: Normalization stats from current buffer
    r_min, r_max = buffer.get_normalization_stats()

    # Phase 3: SFT Training
    losses = []
    for _ in range(config.train_steps_per_iter):
      states_b, actions_b, rtg_norms_b = buffer.sample_transitions(
        config.batch_size, r_min, r_max, np_rng, top_fraction=config.percent_training
      )
      loss = train_step(
        model,
        optimizer,
        jnp.array(states_b),
        jnp.array(actions_b),
        jnp.array(rtg_norms_b),
      )
      losses.append(float(loss))
    mean_loss = float(np.mean(losses))

    # Phase 4: Optimistic Rollout
    elite_rewards = buffer.get_elite_total_rewards(config.elite_fraction)
    p50_elite = float(np.percentile(elite_rewards, 50))

    new_trajectories: list[Trajectory] = []
    episode_rewards: list[float] = []
    denom = r_max - r_min if r_max != r_min else 1.0
    p50_norm = float(np.clip((p50_elite - r_min) / denom, 0.0, 1.0))
    for i in range(config.num_rollout_episodes):
      if i < int(config.num_rollout_episodes * 0.2):
        rtg_target = 1.0
      else:
        rtg_target = float(np_rng.uniform(p50_norm, 1.0))
      traj, _ = rollout_episode(
        model_eval, env, rtg_target, config.temperature, r_min, r_max, np_rng
      )
      new_trajectories.append(traj)
      episode_rewards.append(traj.total_reward)

    # Phase 5: Append and sort
    buffer.add_batch(new_trajectories, np_rng=np_rng, random_eviction_fraction=config.random_eviction_fraction, discard_previous=config.discard_previous)

    # Update plots
    elite_cutoff = float(np.min(elite_rewards))
    plotter.update_loss_curve(losses)
    plotter.update_losses(iteration, losses)
    plotter.update_rewards(iteration, episode_rewards)
    plotter.update_distribution(buffer.get_total_rewards(), elite_cutoff=elite_cutoff, p50_elite=p50_elite)

    mean_ep = float(np.mean(episode_rewards))
    print(
      f"Iter {iteration:3d}/{config.num_iterations} | "
      f"Loss: {mean_loss:.4f} | "
      f"Rollout Mean: {mean_ep:.2f} | "
      f"Buffer: {len(buffer)} trajs | "
      f"RTG [{r_min:.0f}, {r_max:.0f}]"
    )

    # Evaluation
    if iteration % config.eval_interval == 0:
      elite_rewards_eval = buffer.get_elite_total_rewards(config.elite_fraction)
      avg_reward = evaluate(
        model_eval, config, run_name, iteration, r_min, r_max, elite_rewards_eval
      )
      print(f"  → Eval Iter {iteration}: Avg Reward = {avg_reward:.2f}")

  env.close()
  plotter.save_and_close(f"plots/{run_name}.png")
  print("Done.")


if __name__ == "__main__":
  main()
