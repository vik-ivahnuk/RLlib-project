import keyboard
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from gym.envs.registration import register
import gym
import matplotlib.pyplot as plt

from models import DoublePendulumModelV1
from models import DoublePendulumModelV2
from models import DoublePendulumModelV3
from models import DoublePendulumModelV4
from models import DoublePendulumModelV5


ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit = True


def save_graph_train(y_label, values, path):
    plt.figure()
    x = range(1, len(values) + 1)
    plt.plot(x, values)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.savefig(path)


def create_my_env():
    register(
        id='DoublePendulum-v0',
        entry_point='DoublePendulum.DoublePendulum:DoublePendulumEnv',
        max_episode_steps=2500
    )
    return gym.make("DoublePendulum-v0")


env_creator = lambda config: create_my_env()
ray.init(include_dashboard=False)
register_env("DP", env_creator)
ModelCatalog.register_custom_model("DoublePendulumModelV5", DoublePendulumModelV5)
trainer = ppo.PPOTrainer(env="DP",
                         config={"model": {"custom_model": "DoublePendulumModelV5"},
                                 'create_env_on_driver': True
                                 }
                         )

# 'optimizer': {
#  'lr': 0.0005
# }

keyboard.on_press_key("q", press_key_exit)

len_episodes = []
reward_episodes = []

while True:
    if ready_to_exit:
        break
    rest = trainer.train()

    cur_reward = rest["episode_reward_mean"]
    cur_len = rest["episode_len_mean"]

    reward_episodes.append(cur_reward)
    len_episodes.append(cur_len)

    print("avg. reward:", cur_reward, "episode len:", cur_len)

trainer.save("training_models/pendulum_checkpoints_v5")
default_policy = trainer.get_policy(policy_id="default_policy")
default_policy.export_checkpoint("training_models/policy_checkpoint_v5")

save_graph_train('Reward',
                 reward_episodes,
                 'diagrams/reward_v5'
                 )

save_graph_train('Length',
                 len_episodes,
                 'diagrams/mean_lens_v5'
                 )

ray.shutdown()
