import keyboard
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from gym.envs.registration import register
import gym
import matplotlib.pyplot as plt


from models import DoublePendulumModelV1

ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit = True


def create_my_env():
    register(
        id='DoublePendulum-v0',
        entry_point='DoublePendulum.DoublePendulum:DoublePendulumEnv',
        max_episode_steps=2500
    )
    return gym.make("DoublePendulum-v0")


env_creator = lambda config: create_my_env()
ray.init(include_dashboard=False)
register_env("D", env_creator)
ModelCatalog.register_custom_model("DoublePendulumModelV1", DoublePendulumModelV1)
trainer = ppo.PPOTrainer(env="DP", config={"model": {"custom_model": "DoublePendulumModelV1"}, 'create_env_on_driver': True})
keyboard.on_press_key("q", press_key_exit)

len_episodes = []
reward_episodes = []

while True:
    if ready_to_exit:
        break
    rest = trainer.train()
    cur_reward = rest["episode_reward_mean"]
    reward_episodes.append(cur_reward)
    len_episodes.append(reward_episodes)
    print("avg. reward:", cur_reward, "episode len:", rest["episode_len_mean"])

trainer.save("pendulum_checkpoints_v1")
default_policy = trainer.get_policy(policy_id="default_policy")
default_policy.export_checkpoint("policy_checkpoint_v1")

x = range(1, len(reward_episodes) + 1)

plt.plot(x, reward_episodes)
plt.title('rewards per episode for a neural network with 3 hidden layers of 256 neurons')
plt.savefig('diagrams/reward_v1')

plt.plot(x, len_episodes)
plt.title('episode lengths for a neural network with 3 hidden layers of 256 neurons')
plt.savefig('diagrams/lens_v1')

ray.shutdown()