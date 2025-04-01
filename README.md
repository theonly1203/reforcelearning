#1
# Import necessary libraries
!pip install equinox
# If using CUDA, uncomment this line
#!pip install jax[cuda12] jaxlib
import gymnasium as gym
import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
import numpy as np
import imageio
import matplotlib.pyplot as plt
from IPython.display import Image
import optax
import cv2
import tqdm
from jaxtyping import Array, Float, Int, Bool, PyTree
from typing import Callable, Tuple, Union, List, Dict



#2
# Visualize the environment
# You must balance an inverted pendulum on a moving cart
# Letting the pole fall too far or moving too far to the side results in losing!
env = gym.make("CartPole-v1", render_mode="rgb_array")
np.random.seed(0)
frames = []
state, info = env.reset()
for _ in range(200):  # Run for a set number of steps
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

    if terminated or truncated:
        state, info = env.reset()

gif_filename = "pole_test.gif"
imageio.mimsave(gif_filename, frames, fps=30, loop=0)
display(Image(filename=gif_filename))



#3
# Some types to help you
State = Float[Array, "4"]
Action = int
Reward = Float[Array, ""]
Terminated = Bool[Array, ""]

BatchState = Float[Array, "B 4"]
BatchAction = Int[Array, "B"]
BatchReward = Float[Array, "B"]
BatchReturn = Float[Array, "B"]
BatchTerminated = Bool[Array, "B"]

Params = PyTree[Float[Array, "?"]]


#4
class Model(eqx.Module):
  policy_network: eqx.Module

  def __init__(self, action_dim: int, key):
    key1, key2, key3 = jax.random.split(key, 3)
    self.policy_network = nn.Sequential([
        nn.Linear(4, 64, key=key1),  # 第一层: 4 -> 64
        jax.nn.relu,                # 激活函数
        nn.Linear(64, 32, key=key2), # 第二层: 64 -> 32
        jax.nn.relu,                # 激活函数
        nn.Linear(32, action_dim, key=key3), # 输出层: 32 -> action_dim
        jax.nn.softmax             # 转换为概率分布
      ])
    #raise NotImplementedError("TODO: Create your own policy network")
    # Personally, I like to use LayerNorm but others think it makes learning harder

  def __call__(self, state: State) -> Float[Array, "A"]:
    return self.policy_network(state)
    #"""Equivalent to torch module forward().
    #Compute model outputs. """
    #raise NotImplementedError("TODO: Implement model forward/call")



#5
# Implement policies and losses
def sample_action_new(model, state, rng):
  #"""Given the model, sample an action from the policy distribution"""
  z = model.policy_network(state)
  action = jax.random.choice(rng, a=jnp.arange(len(z)), p=z)
  #return jax.random.choice(key, a=jnp.arange(len(z)), p=z).item()
  return action.item()
  #raise NotImplementedError("TODO: Using the model, map a state to randomly-sampled action")

def compute_returns(rewards: BatchReward, terminateds: BatchTerminated, gamma: float) -> BatchReturn:
    """Given a sequence of rewards and terminateds, returns the corresponding sequence of returns"""
    reversed_rewards = jnp.flip(rewards)
    reversed_dones = jnp.flip(terminateds)

    def scan_fn(carry, x):
        reward, done = x
        current_return = reward + gamma * carry * (1 - done)
        return current_return, current_return

    _, returns = jax.lax.scan(scan_fn, 0.0, (reversed_rewards, reversed_dones))
    returns = jnp.flip(returns)
    return returns

def pg_loss(
    model: Callable,
    states: BatchState,
    actions: BatchAction,
    returns: BatchReward,
  ) -> Tuple[Float[Array, ""], Dict[str, float]]:
  """The loss for REINFORCE policy gradient.

  Returns the loss and dictionary of scalars (metrics) for debugging
  """
  B = actions.shape[0]
  probs = jax.vmap(lambda s: model.policy_network(s))(states)
  log_probs = jnp.log(probs[jnp.arange(B), actions])
  loss = -jnp.mean(log_probs * returns)
  metrics = {"loss": loss, "avg_log_prob": jnp.mean(log_probs), "avg_return": jnp.mean(returns)}
  return loss, metrics
  #metrics = {"example_metric": 0}
  #raise NotImplementedError("TODO: Implement policy gradient REINFORCE loss")



# TODO: Uncomment the below line after your code does not crash
# It will make training much faster!
#@eqx.filter_jit
def update_pi(
    model: Callable,
    X: Tuple[BatchState, BatchAction, BatchReward, BatchTerminated],
    opt_state: optax.OptState,
    gamma: Float
  ) -> Tuple[optax.OptState, Float[Array, ""], Dict[str, float]]:
  """After verifying that this function works, you will want to uncomment
  @eqx.filter_jit above. This will make the function run 10x faster."""
  states, actions, rewards, terminateds = X
  returns = compute_returns(rewards, terminateds, gamma)
  grad_fn = jax.value_and_grad(pg_loss, has_aux=True)
  (loss, metrics), grads = grad_fn(model, states, actions, returns)


  updates, new_opt_state = optimizer.update(grads, opt_state)
  model = eqx.tree_at(lambda m: m.policy_network, model, updates)
  return model, new_opt_state, loss, metrics
  #raise NotImplementedError("TODO: Implement the function that updates your policy")
  # NOTE: You should use has_aux=True with filter_grad to return metrics



#6
seed = 0
alpha = 1e-3 # Learning rate
epochs = 1_000 # How long to train the policy for
gamma = 0.99 # Decay
batch_size = 1_000
optimizer = optax.adamw(learning_rate=alpha)



#7
key = jax.random.key(seed)
# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
class Model:
    def __init__(self, action_dim: int, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.policy_network = eqx.nn.Sequential([
            eqx.nn.Linear(4, 64, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(64, 32, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(32, action_dim, key=key3),
            jax.nn.softmax
        ])

# Create models
key, model_key = jax.random.split(key)
model = Model(env.action_space.n, model_key)
opt_state = optimizer.init(eqx.filter(model.policy_network, eqx.is_array))

# Training loop
# Once you see an episode return above 30, your model is learning
# This means your code is probably correct
# You just need to wait for a while for the agent to learn to generalize
# To all possible states
num_episodes = 0
# Progress bar for logging
pbar = tqdm.tqdm(range(epochs), dynamic_ncols=True, position=0, leave=True)
terminated = True
truncated = False
best_return = -float('inf')
best_model = model
# Train for max_env_steps
for epoch in pbar:
  # For the progress bar, you might want to use something like this:
  # pbar.set_description(f"Episode {epoch} return {episode_return:.1f}, Loss: {loss:.3f}")
  # It will print pretty information to help you track your experiment

  rewards: Union[List, jax.Array] = []
  states: Union[List, jax.Array] = []
  actions: Union[List, jax.Array] = []
  terminateds: Union[List, jax.Array] = []
  total_return: List[float] = []
  episode_return = 0.0
  state, _ = env.reset()

  for t in range(batch_size):
    if (terminated or truncated):
      state, _ = env.reset()
      total_return.append(episode_return)
      episode_return = 0
      num_episodes += 1
      terminated = truncated = False


    key, subkey = jax.random.split(key)
    action = sample_action_new(model, state, subkey)------------------------------------------error line



    next_state, reward, terminated, truncated, _ = env.step(action)


    states.append(state)
    actions.append(action)
    rewards.append(reward)
    terminateds.append(terminated)
    episode_return += reward
    state = next_state


  states = jnp.array(states)
  actions = jnp.array(actions)
  rewards = jnp.array(rewards)
  terminateds = jnp.array(terminateds)


  X = (states, actions, rewards, terminateds)
  model, opt_state, loss, metrics = update_pi(model, X, opt_state, gamma)


  avg_return = jnp.mean(jnp.array(total_return)) if total_return else episode_return


  if avg_return > best_return:
    best_return = avg_return
    best_model = model

    pbar.set_description(f"Episode {num_episodes} return {avg_return:.1f}, Loss: {loss:.3f}")

  #raise NotImplementedError("TODO: Implement the rest of the train loop")



  error

  TypeError                                 Traceback (most recent call last)
<ipython-input-7-2eba356c5ee1> in <cell line: 0>()
     55 
     56     key, subkey = jax.random.split(key)
---> 57     action = sample_action_new(model, state, subkey)
     58 
     59 

4 frames
    [... skipping hidden 3 frame]

/usr/lib/python3.11/inspect.py in _bind(self, args, kwargs, partial)
   3182                 arguments[kwargs_param.name] = kwargs
   3183             else:
-> 3184                 raise TypeError(
   3185                     'got an unexpected keyword argument {arg!r}'.format(
   3186                         arg=next(iter(kwargs))))

TypeError: got an unexpected keyword argument 'key'




  #8
  # Visualize your policy
# Use for debugging, and also for the final result!
env = gym.make("CartPole-v1", render_mode="rgb_array")

# You can try changing the seed to view different episodes
seed = 0
np.random.seed(seed)
env.action_space.seed(seed)
frames = []
state, info = env.reset(seed=0)
the_return = 0
terminated = truncated = False
key = jax.random.key(seed)
while not (terminated or truncated):  # Run for a set number of steps
    key, _ = jax.random.split(key)
    action = sample_policy(pi, state, key).item()
    next_state, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    state = next_state
    the_return += reward

gif_filename = "pole_trained.gif"
imageio.mimsave(gif_filename, frames, fps=30, loop=0)
display(Image(filename=gif_filename))
print(f"Return: {the_return:.1f}")





