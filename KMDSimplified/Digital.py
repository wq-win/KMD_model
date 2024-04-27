import random
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
from tqdm import tqdm


MAPD = [[
    "  +-----+  ",
    "  |  _  |  ",
    "  | | | |  ",
    "  | |_| |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |     |  ",
    "  |   | |  ",
    "  |   | |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  |  _| |  ",
    "  | |_  |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  |  _| |  ",
    "  |  _| |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |     |  ",
    "  | |_| |  ",
    "  |   | |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  | |_  |  ",
    "  |  _| |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  | |_  |  ",
    "  | |_| |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  |   | |  ",
    "  |   | |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  | |_| |  ",
    "  | |_| |  ",
    "  |     |  ",
    "  +-----+  ",
], [
    "  +-----+  ",
    "  |  _  |  ",
    "  | |_| |  ",
    "  |  _| |  ",
    "  |     |  ",
    "  +-----+  ",
]]

MAPS = [
    "           ",
    "-----------",
]

class DiditalEnv(discrete.DiscreteEnv):
    def __init__(self, num_locations=10):
        self.num_locations = num_locations
        self.reward_options = (0, 1)
        num_states = self.num_locations
        
        initial_state_distrib = np.zeros(num_states)
        initial_state_distrib[0] = 1  
        initial_state_distrib /= initial_state_distrib.sum()

        action_dict = {0: 'no move', 1: 'right', 2: 'left'}
        num_actions = len(action_dict)

        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        
        for state in range(num_states):
            for action in range(num_actions):
                prob, reward, done = 1, self.reward_options[0], False

                if action == 0:
                    next_state = state
                    if state == num_states - 1:
                        reward = self.reward_options[1]
                        done = True
                elif action == 1:
                    next_state = state + 1
                    prob = .88
                    if  next_state > num_states-1:
                        next_state = num_states-1
                else:
                    next_state = state - 1
                    prob = .88 
                    if next_state < 0:
                        next_state = 0
                        
                P[state][action].append((prob, next_state, reward, done))
                if prob != 1:
                    P[state][action].append((1-prob, state, reward,done))  # 动作执行失败，状态不变

        # for key, value in P.items():
        #     print(key,value) 
        self.mapInit()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib
        )

    def mapInit(self):
        self.desc = np.asarray(MAPD[0]+MAPS, dtype="c")
        self.layer_name_length = 0
    
    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        self.desc = np.asarray(MAPD[self.s]+MAPS, dtype="c")
        out = self.desc.copy().tolist()    
        out = [[c.decode("utf-8") for c in line] for line in out]
        colorpara1 = str(self.s)
        out[7][self.layer_name_length+self.s] = utils.colorize(colorpara1,'yellow',highlight=True)
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
            
if __name__ == "__main__":
    env = DiditalEnv()
    
    
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