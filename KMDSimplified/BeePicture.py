import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    

class Bee(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("bee.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_w, self.icon_h))
  
        
class Food(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("food.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_w, self.icon_h))
        

class TaskEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        
        # Define a 2-D observation space
        self.observation_shape = (600, 800, 3)  # h:600, w:800
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(5)

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []
        
        # Maximum fuel chopper can take at once
        # self.max_fuel = 1000

        # Permissible area of helicper to be
        # self.y_min = int(self.observation_shape[0] * 0.1)
        self.y_min = 0
        self.x_min = 0
        # self.y_max = int(self.observation_shape[0] * 0.9)
        self.y_max = self.observation_shape[0]
        self.x_max = self.observation_shape[1]
    
    def get_action_meanings(self, action):
        action_dict = {0: "Down", 1: "Up", 2: "Right", 3: "Left",  4: "Do Nothing"}
        return action_dict[action]
    
    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the heliopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape  # (h,w)
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[0], x: x + elem_shape[1]] = elem.icon

        # text = 'Fuel Left: {} | Rewards: {}'.format(self.fuel_left, self.ep_return)

        text = 'Rewards:{}'.format(self.ep_return)

        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.8, (0, 0, 0), 1, cv2.LINE_AA)
        
    def reset(self):
        return super().reset()
    
    def render(self, mode='human'):
        return super().render(mode)
    
    def step(self, action):
        return super().step(action)
    
    def close(self):
        return super().close()