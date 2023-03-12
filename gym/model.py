import tensorflow as tf
from abc import ABC
from ray.rllib.models.tf import TFModelV2


class DoublePendulumModel(TFModelV2, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DoublePendulumModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')(self.inputs)
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')(self.fc1)
        self.out = tf.keras.layers.Dense(num_outputs, activation='tanh')(self.fc2)
        self.value_out = tf.keras.layers.Dense(1, activation='tanh')(self.fc2)
        self.base_model = tf.keras.Model(self.inputs, [self.out, self.value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self.value_out = self.base_model(input_dict['obs'])
        return model_out, state

    def value_function(self):
        return tf.reshape(self.out, [-1])
