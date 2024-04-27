import random
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
from tqdm import tqdm


MAP = [[
    "    +---+",
    "bee | : |",
    "food| : |",
    "grip+---+",
],[
    "    +-----+",
    "bee | : : |",
    "food| : : |",
    "grip+-----+",
],[
    "    +-------+",
    "bee | : : : |",
    "food| : : : |",
    "grip+-------+",
],[
    "    +---------+",
    "bee | : : : : |",
    "food| : : : : |",
    "grip+---------+",
],[
    "    +-----------+",
    "bee | : : : : : |",
    "food| : : : : : |",
    "grip+-----------+",
],]


class BeeFoodEnv(discrete.DiscreteEnv):
    """
    The bee must be at the far right in order to grip. 
    Bee must grip before you can pull .
    
    """
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, num_locations = 3):
        assert 2 <= num_locations <= 6, 'The number of locations is between 2 and 6'
        self.num_locations = num_locations
        self.reward_options = (0, 1)
        # bee的state * food的state * 是否握住绳子
        num_states = self.num_locations * self.num_locations * 2
        
        initial_state_distrib = np.zeros(num_states)
        initial_state_distrib[0] = 1  
        initial_state_distrib /= initial_state_distrib.sum()

        action_dict = {0: 'grip', 1: 'not move', 2: 'right', 3: 'left', 4: 'stand up', 5: 'pull'}
        num_actions = len(action_dict)  # 0:握住 1:不动 2:右移 3:左移 4:站立 5:拽
        
        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        
        for is_grip in (False, True):
            for food_location in range(num_locations):
                for bee_location in range(num_locations):
                    for action in range(num_actions):
                        state = self.encode(bee_location, food_location, is_grip)
                        prob, reward, done = 1, self.reward_options[0], False
                        
                        if action == 0:  # grip
                            next_state = self.encode(bee_location, food_location, True)
                        elif action == 1:  # not move
                            next_state = state 
                        elif action == 2:  # right
                            bee_loc = min(bee_location + 1, num_locations - 1)
                            next_state = self.encode(bee_loc, food_location, is_grip) 
                            prob = .88  # 1-prob概率移动失败
                        elif action == 3:  # left
                            bee_loc = max(bee_location - 1, 0)
                            next_state = self.encode(bee_loc, food_location, is_grip) 
                            prob = .88  # 1-prob概率移动失败
                        elif action == 4:  # stand up
                            next_state = state
                            if state == self.encode(num_locations - 1, num_locations - 1, is_grip):
                                reward = self.reward_options[1]
                                done = True 
                        else:  # pull
                            if state >= num_states / 2:  # is gripping
                                food_loc = min(food_location + 1, num_locations - 1)
                                next_state = self.encode(bee_location, food_loc, is_grip) 
                            else:    
                                next_state = state        
                        P[state][action].append((prob, next_state, reward, done))
                        if prob != 1:
                            P[state][action].append((1-prob, state, reward,done))  # 动作执行失败，状态不变

        # for key, value in P.items():
        #     print(key,value)      
        # print(P[0][0])
        self.mapInit()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib
        )
    
    def encode(self, bee_loc, food_loc, is_grip):
        i = food_loc
        i *= self.num_locations
        i += bee_loc
        if is_grip == True:
            i += self.num_locations * self.num_locations
        return i
    
    def decode(self, i):
        return i % self.num_locations, i // self.num_locations, bool(i - self.num_locations * self.num_locations + 1)
            
    def mapInit(self):
        self.desc = np.asarray(MAP[self.num_locations-2], dtype="c")
        self.layer_name_length = 3

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        
        out = self.desc.copy().tolist()    
        out = [[c.decode("utf-8") for c in line] for line in out]
        bee_location, food_location, is_grip = self.decode(self.s)
        
        out[1][self.layer_name_length + (bee_location + 1) * 2] = utils.colorize('B', "yellow", highlight=True)
        out[2][self.layer_name_length + (food_location + 1) * 2] = utils.colorize('F', "green", highlight=True)
        out[3][self.layer_name_length + self.num_locations + 1] = utils.colorize(str(int(is_grip)), "red", highlight=False)
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
            
            
if __name__ == "__main__":
    env = BeeFoodEnv(6)
    obs = env.reset()
    
    done_step = 0
    average_step = 0
    eps = 1000000
    with tqdm(total=eps) as pbar:
        for _ in range(eps):
            done_step = 0
            obs = env.reset()
            while True:
                action = env.action_space.sample()
            
                obs, reward, done, info = env.step(action)

                # print('action:%d, state:%d, reward:%d, done:%s, info:%s' % (action, obs, reward, done, info))
            
                # env.render()
                done_step += 1
                if done:
                    # print('done')
                    average_step += done_step
                    break
            pbar.update()
    print(average_step/eps)
    env.close()