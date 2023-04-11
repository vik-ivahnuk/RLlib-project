import sys
from time import sleep, time
import gym
from gym import spaces, logger
import pygame
import math
import numpy as np
from gym.utils import seeding


class DoublePendulumEnv(gym.Env):
    def __init__(self):
        self.last_rew = None
        self.FPS = 60
        self.dt = 1 / self.FPS

        self.width = 800
        self.height = 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.WIDTH = 800
        self.HEIGHT = 600
        self.clock = pygame.time.Clock()
        self.screen = None

        self.x = self.width // 2
        self.y = self.height // 2
        self.L1 = 100
        self.L2 = 130
        self.D = 2 * (self.L1 + self.L2)
        self.m1 = 0.1
        self.m2 = 0.2
        self.g = 98.1
        self.max_av = 5 * np.pi

        self.steps_count = 0
        self._init_param()

        self.high_obs = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.high_obs, high=self.high_obs, dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([3, 3])

    def _init_param(self):
        self.steps_count = 0
        self.theta1 = 0
        self.theta2 = 0
        self.angular_velocity1 = 0
        self.angular_velocity2 = 0
        self._calc_coordinates()

        self.zero_rew = math.exp(-1)
        self.last_rew = -1

    def seed(self, seed=int(time())):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps_count += 1

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        denominator = (2 * self.m1 + self.m2 - self.m2 * math.cos(2 * self.theta1 - 2 * self.theta2))

        numerator1 = -self.g * (2 * self.m1 + self.m2) * math.sin(self.theta1) - self.m2 * self.g * math.sin(
            self.theta1 - 2 * self.theta2) - 2 * math.sin(self.theta1 - self.theta2) * self.m2 * (
                             self.angular_velocity2 ** 2 * self.L2 + self.angular_velocity1 ** 2 * self.L1 * math.cos(
                         self.theta1 - self.theta2))
        denominator1 = self.L1 * denominator
        angular_acceleration1 = numerator1 / denominator1

        numerator2 = 2 * math.sin(self.theta1 - self.theta2) * (
                self.angular_velocity1 ** 2 * self.L1 * (self.m1 + self.m2) + self.g * (
                self.m1 + self.m2) * math.cos(
            self.theta1) + self.angular_velocity2 ** 2 * self.L2 * self.m2 * math.cos(self.theta1 - self.theta2))
        denominator2 = self.L2 * denominator

        angular_acceleration2 = numerator2 / denominator2

        if action[0] == 2:
            angular_acceleration1 += 0.2
        elif action[0] == 1:
            angular_acceleration1 -= 0.2

        if action[1] == 2:
            angular_acceleration2 += 0.2
        elif action[1] == 1:
            angular_acceleration2 -= 0.2

        self.angular_velocity1 += angular_acceleration1 * self.dt
        self.angular_velocity2 += angular_acceleration2 * self.dt

        self.theta1 += self.angular_velocity1 * self.dt
        self.theta2 += self.angular_velocity2 * self.dt

        obs = self._observation()
        info = {}

        self._calc_coordinates()

        distance = math.sqrt((self.x2 - self.x) ** 2 + (self.y2 - (self.y - self.L1 - self.L2)) ** 2)

        done = abs(distance) < 1

        if done:
            rew = (self.D * 2500 / self.steps_count) ** 2
            return obs, rew, done, info

        reward = self.D - distance

        if reward > self.last_rew:
            self.last_rew = reward
        else:
            reward = 0

        # reward = math.exp(-distance / self.D) - self.zero_rew

        return obs, reward/100, done, info

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width, self.height)
            )
        self.screen.fill(self.WHITE)
        pygame.display.set_caption("Frame " + str(self.steps_count))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                sys.exit(0)

        pygame.draw.line(self.screen, self.RED, (self.x, self.y), (self.x1, self.y1), 5)
        pygame.draw.line(self.screen, self.RED, (self.x1, self.y1), (self.x2, self.y2), 5)
        pygame.draw.circle(self.screen, self.BLACK, (int(self.x1), int(self.y1)), 10)
        pygame.draw.circle(self.screen, self.BLACK, (int(self.x2), int(self.y2)), 10)

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def reset(self):
        self._init_param()
        return self._observation()

    def _calc_coordinates(self):
        self.x1 = self.x + self.L1 * math.sin(self.theta1)
        self.y1 = self.y + self.L1 * math.cos(self.theta1)
        self.x2 = self.x1 + self.L2 * math.sin(self.theta2)
        self.y2 = self.y1 + self.L2 * math.cos(self.theta2)

    def _observation(self):
        return np.array([
            math.cos(self.theta1),
            math.sin(self.theta1),
            math.cos(self.theta2),
            math.sin(self.theta2),
            np.clip(self.angular_velocity1 / self.max_av, -1, 1),
            np.clip(self.angular_velocity2 / self.max_av, -1, 1)
        ], dtype=np.float32)


if __name__ == "__main__":
    double_pendulum = DoublePendulumEnv()
    q = [0, 0]
    running = True
    double_pendulum.render()
    while running:

        reset = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    q[0] = 1
                    print(" --> top")
                elif event.key == pygame.K_s:
                    q[0] = 0
                    print(" 0 top")
                elif event.key == pygame.K_q:
                    q[1] = 1
                    print(" <-- bottom")
                elif event.key == pygame.K_e:
                    q[1] = 2
                    print(" --> bottom")
                elif event.key == pygame.K_w:
                    q[1] = 0
                    print(" 0 bottom")
                elif event.key == pygame.K_r:
                    reset = True
                    q = [0, 0]

        if reset:
            double_pendulum.reset()
            is_done = False
        else:
            _, _, is_done, _ = double_pendulum.step(q)
        double_pendulum.render()

        if is_done:
            running = False
