import pygame
import math
import numpy as np


class DoublePendulum:
    def __init__(self, width, height, L1, L2, m1, m2, theta1, theta2):
        self.FPS = 60
        self.dt = 1 / self.FPS

        self.width = width
        self.height = height
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.WIDTH = 800
        self.HEIGHT = 600
        self.clock = pygame.time.Clock()
        self.screen = None

        self.x = width // 2
        self.y = height // 2
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.theta1 = theta1
        self.theta2 = theta2
        self.angular_velocity1 = 0
        self.angular_velocity2 = 0
        self.g = 98.1

        self.steps_count = 0

    def step(self, action):
        self.steps_count += 1

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
        done = abs(obs[0] + 1) < 0.001 and abs(obs[2] + 1) < 0.001
        reward = 0
        info = {}

        return obs, done, reward, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width, self.height)
            )
        self.screen.fill(self.WHITE)
        pygame.display.set_caption("Frame " + str(self.steps_count))
        x1 = self.x + self.L1 * math.sin(self.theta1)
        y1 = self.y + self.L1 * math.cos(self.theta1)
        x2 = x1 + self.L2 * math.sin(self.theta2)
        y2 = y1 + self.L2 * math.cos(self.theta2)

        pygame.draw.line(self.screen, self.RED, (self.x, self.y), (x1, y1), 5)
        pygame.draw.line(self.screen, self.RED, (x1, y1), (x2, y2), 5)
        pygame.draw.circle(self.screen, self.BLACK, (int(x1), int(y1)), 10)
        pygame.draw.circle(self.screen, self.BLACK, (int(x2), int(y2)), 10)

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def reset(self):
        pass

    def _observation(self):
        return np.array([
            math.cos(self.theta1),
            math.sin(self.theta1),
            math.cos(self.theta2),
            math.cos(self.theta2),
            np.clip(self.angular_velocity1 / (5 * np.pi), -1, 1),
            np.clip(self.angular_velocity2 / (5 * np.pi), -1, 1)
        ], dtype=np.float32)


double_pendulum = DoublePendulum(800, 600, 100, 130, 0.1, 0.2, 0, 0)

q = [0, 0]
running = True
double_pendulum.render()
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                q[0] = 1
            elif event.key == pygame.K_d:
                q[0] = 2
            elif event.key == pygame.K_s:
                q[0] = 0
            elif event.key == pygame.K_q:
                q[1] = 1
            elif event.key == pygame.K_e:
                q[1] = 2
            elif event.key == pygame.K_w:
                q[1] = 0

    _, is_done, _, _ = double_pendulum.step(q)
    double_pendulum.render()

    if is_done:
        running = False
        print("kek")
