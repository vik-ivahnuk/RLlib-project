import gym
from gym import register
import pygame

import tensorflow as tf
from ray.rllib import TFPolicy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from DoublePendulum.DoublePendulum import DoublePendulumEnv
from models import DoublePendulumModelV1
from models import DoublePendulumModelV2
from models import DoublePendulumModelV4

tf.compat.v1.disable_eager_execution()

register(
    id='DoublePendulum-v0',
    entry_point='DoublePendulum.DoublePendulum:DoublePendulumEnv',
    max_episode_steps=10000
)

register_env("DP", lambda _: DoublePendulumEnv())
ModelCatalog.register_custom_model("DoublePendulumModelV2", DoublePendulumModelV2)
my_restored_policy = TFPolicy.from_checkpoint("policy_checkpoint_v2")
CartpoleEnv = gym.make("DoublePendulum-v0")
obs = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
done = False
while not done:
    actions = my_restored_policy.compute_single_action(obs)
    obs, rew, done, info = CartpoleEnv.step(actions[0])
    total_rev += rew
    CartpoleEnv.render()
print(total_rev)
CartpoleEnv.close()
