#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Each agent can observe itself (it's own identity) i.e. s_j = j and vision, path ahead of it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

# core modules
import random
import math
import curses
import time

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from ic3net_envs.traffic_helper import *


def nPr(n,r):
    f = math.factorial
    return f(n)//f(n-r)

class TrafficJunctionEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10

        self.FRONT_PENALTY = -0.5 #Penalty for not having a car directly in front of you (excluding first car)
        self.BACK_PENALTY = -0.2 #Penalty for not having a car directly behind you (excluding last car)
        self.ACC_PENALTY = -0.05 #Penalty for accelerating (so that the cars are led to stay at constant speed)
        self.EBRAKE_PENALTY = -.1 #Penalty for emergency braking

        self.episode_over = False
        self.has_failed = 0

        self.counter = 0




        self.isFull = False

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Traffic Junction task')
        env.add_argument('--dim', type=int, default=30,
                         help="Dimension of box (i.e length of road) ")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of car")
        env.add_argument('--add_rate_min', type=float, default=0.5,
                         help="rate at which to add car (till curr. start)")
        env.add_argument('--add_rate_max', type=float, default=0.5,
                         help=" max rate at which to add car")
        env.add_argument('--curr_start', type=float, default=0,
                         help="start making harder after this many epochs [0]")
        env.add_argument('--curr_end', type=float, default=0,
                         help="when to make the game hardest [0]")
        env.add_argument('--difficulty', type=str, default='easy',
                         help="Difficulty level, easy|medium|hard")
        env.add_argument('--vocab_type', type=str, default='bool',
                         help="Type of location vector to use, bool|scalar")


    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'add_rate_min', 'add_rate_max', 'curr_start', 'curr_end',
                  'difficulty', 'vocab_type']

        for key in params:
            setattr(self, key, getattr(args, key))



        self.ncar = args.nagents
        self.h = self.dim
        self.w = 2
        self.dims = dims = (self.h, self.w)

        #TODO: find way to change interval based on max steps
        self.b = random.randint(20, 40)
        difficulty = args.difficulty
        vision = args.vision

        #self.speeds = np.zeros(self.ncar)
        self.speeds = np.full(self.ncar, 1) #THIS IS JUST FOR EASE OF SEEING THE SIMULATION, SET SPEEDS TO 0 IN TRAINING



        if difficulty in ['medium','easy']:
            assert dims[0]%2 == 0, 'Only even dimension supported for now.'

            assert dims[0] >= 4 + vision, 'Min dim: 4 + vision' #THEY HAVE A MINIMUM FOR DIMENSIONS

        if difficulty == 'hard':
            assert dims[0] >= 9, 'Min dim: 9'
            assert dims[0]%3 ==0, 'Hard version works for multiple of 3. dim. only.'

        # Add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do -
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY) - 2 = move two spaces ACCELERATE
        self.naction = 4 #0 - emergency brake, 1 - decelerate, 2 - maintain, 3 - accelerate
        self.action_space = spaces.Discrete(self.naction)

        # make no. of dims odd for easy case.
        if difficulty == 'easy':
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        nroad = {'easy':2,
                'medium':4,
                'hard':8}

        dim_sum = dims[0] + dims[1]
        base = {'easy':   dim_sum,
                'medium': 2 * dim_sum,
                'hard':   4 * dim_sum}

        self.npath = nPr(nroad[difficulty],2)

        # Setting max vocab size for 1-hot encoding
        if self.vocab_type == 'bool':
            self.BASE = base[difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            # car_type + base + outside + 0-index
            self.vocab_size = 1 + self.BASE + 1 + 1
            self.observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, len(dims), vision * vision * vocab)
            self.observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiDiscrete(dims),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
            # Actual observation will be of the shape 1 * ncar * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        self._set_grid()



        if difficulty == 'easy':
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

        return

    def reset(self, epoch=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.ncar)
        self.wait = np.zeros(self.ncar)
        self.cars_in_sys = 0

        #self.speeds = np.zeros(self.ncar)
        self.speeds = np.full(self.ncar, 1) #THIS IS JUST FOR EASE OF MAKING THE SIMULATION, USE ABOVE IN TRAINING

        # Chosen path for each car:
        self.chosen_path = [0] * self.ncar
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.ncar

        # self.cars = np.zeros(self.ncar)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS,self.CAR_CLASS + self.ncar)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.ncar, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.ncar, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.ncar, - 1)

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = (self.curr_end - self.curr_start)
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be ncar * vision * vision ndarray
        obs = self._get_obs()

        self.isFull = False
        self.counter = 0
        self.b = random.randint(self.h // 2, self.h)
        print(self.b)

        return obs

    def step(self, action):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either ncar or ncar x 1

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (ncar x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over.
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either ncar or ncar x 1
        action = np.array(action).squeeze()



        print("Chosen Actions ", action)
        #This is an array of size ncars, with each element either 0 or 1 - I think each index refers to a single car

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        assert len(action) == self.ncar, "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.ncar)

        for i, a in enumerate(action):
            action[i] = self._take_action(i, a)

        print("Taken Actions ", action)
        self._add_cars()

        obs = self._get_obs()
        reward = self._get_reward() #Getting reward for every single car

        for i in range(len(reward)):
            if action[i] == 3:
                reward[i] += self.ACC_PENALTY
            if action[i] == 0:
                reward[i] += self.EBRAKE_PENALTY



        debug = {'car_loc':self.car_loc,
                'alive_mask': np.copy(self.alive_mask),
                'wait': self.wait,
                'cars_in_sys': self.cars_in_sys,
                'is_completed': np.copy(self.is_completed)}

        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate

        self.counter += 1


        for i in range(self.ncar):
            if self.alive_mask[i] == 1:
                self.episode_over = False
                break
            elif self.alive_mask[i] == 0:
                self.episode_over = True

        #print("Speeds", self.speeds)
        #print("Actions", action)
        #print("Alive cars", self.alive_mask)
        #print("Rewards", reward)

        return obs, reward, self.episode_over, debug

    #FIXME - when running through the script, once cars start exiting the simulation trainer has no action element to act on. Once all cars exit the simulation
    #FIXME - the trainer has no action to look at so the running just stops. I am wondering if we want to make this a continuous exercise, where there is no end
    #FIXME - but rather when a car exits the simulation it just starts back up at the top with a reset reward and everything? I don't know if that would be too much

    def render(self, mode='human', close=False):


        grid = self.grid.copy().astype(object)
        # grid = np.zeros(self.dims[0]*self.dims[1], dtypeobject).reshape(self.dims)
        grid[grid != self.OUTSIDE_CLASS] = '_'
        grid[grid == self.OUTSIDE_CLASS] = ''
        self.stdscr.clear()
        for i, p in enumerate(self.car_loc):
            if self.car_last_act[i] == 0: # GAS
                if grid[p[0]][p[1]] != 0: #EMPTY SPACE IN FRONT
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<'+str(i)+'>' #Added the str(i) to keep track of which car is which
                else:
                    grid[p[0]][p[1]] = '<'+str(i)+'>'
            elif self.car_last_act[i] == 2:
                if grid[p[0]][p[1]] != 0: #EMPTY SPACE TWO SPACES IN FRONT
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1] + 1]).replace('_', '') + '<a'+str(i)+'>'
                else:
                    grid[p[0]][p[1]] = '<a'+str(i)+'>'
            else: # BRAKE
                if grid[p[0]][p[1]] == 1:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<b'+str(i)+'>'
                else:
                    grid[p[0]][p[1]] = '<b'+str(i)+'>'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if row_num == idx == 0:
                    continue
                if item != '_': #This doesn't do a good job at indicating when a crash happens with acceleration
                    if '<'+str(i)+'>' in item and len(item) > 3: #CRASH, one car gas
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2)) #Yellow
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', '').center(3), curses.color_pair(2))
                    elif '<'+str(i)+'>' in item: #GAS
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1)) #RED
                    elif 'a'+str(i)+'' in item and len(item) > 3:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', ''), curses.color_pair(2))
                    elif 'a'+str(i)+'' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', ''), curses.color_pair(3))
                    elif 'b'+str(i)+'' in item and len(item) > 3: #CRASH
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2)) #Yellow
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', '').center(3), curses.color_pair(2))
                    elif 'b'+str(i)+'' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(5)) #BLUE
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2)) #Yellow
                else:
                    self.stdscr.addstr(row_num, idx * 4, '_'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self):
        return

    def _set_grid(self):
        self.grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        h, w = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty) #Roads are gotten from this function from traffic_helper - returns an array of length two of arrays
        for road in roads: #Figure out how exactly this for loop works and what it accesses
            # looks like it accesses the first and second element of the tuple returned by get_road_blocks
            self.grid[road] = self.ROAD_CLASS
        if self.vocab_type == 'bool':
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(self.grid[road].shape)
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_obs(self):
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1


        # remove the outside class.
        if self.vocab_type == 'scalar':
            self.bool_base_grid = self.bool_base_grid[:,:,1:]


        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.naction - 1)

            # route id
            r_i = self.route_id[i] / (self.npath - 1)

            # loc
            p_norm = p / (h-1, w-1)

            # vision square
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = self.bool_base_grid[slice_y, slice_x]

            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)

            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = tuple(obs)

        return obs


    def _add_cars(self):
        for r_i, routes in enumerate(self.routes):
            if self.cars_in_sys >= self.ncar: #IF WERE ALREADY AT MAX NAGENTS
                self.isFull = True
                return

            # Add car to system and set on path
            #if np.random.uniform() <= self.add_rate:
            if self.counter < self.ncar * 2:
                if self.counter % 2 == 0:
                # chose dead car on random
                    idx = self.counter // 2 #THIS IMPLEMENTS EACH CAR AS THE SUCCESSIVE INDEX
                # make it alive
                    self.alive_mask[idx] = 1

                # choose path randomly & set it
                    p_i = np.random.choice(len(routes))
                # make sure all self.routes have equal len/ same no. of routes
                    self.route_id[idx] = p_i + r_i * len(routes)
                    self.chosen_path[idx] = routes[p_i]

                # set its start loc
                    self.car_route_loc[idx] = 0
                    self.car_loc[idx] = routes[p_i][0]

                # increase count
                    self.cars_in_sys += 1

    def _set_paths_easy(self): #CHANGE THIS FOR PATHS
        h, w = self.dims
        self.routes = {
            'TOP': []
        }

        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2) for i in range(h)] #MIGHT NEED TO CHANGE SOMETHING HERE FOR THE ROUTES ARRAY
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        #full = [(h//2, i) for i in range(w)]
        #self.routes['LEFT'].append(np.array([*full]))

        self.routes = list(self.routes.values())


    def _set_paths_medium_old(self):
        h,w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': [],
            'RIGHT': [],
            'DOWN': []
        }

        # type 0 paths: go straight on junction
        # type 1 paths: take right on junction
        # type 2 paths: take left on junction


        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2-1) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to UP to LEFT, type 1
        first_half = full[:h//2]
        second_half = [(h//2 - 1, i) for i in range(w//2 - 2,-1,-1) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))

        # 2 refers to UP to RIGHT, type 2
        second_half = [(h//2, i) for i in range(w//2-1, w) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))


        # 3 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        # 4 refers to LEFT to DOWN, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2 - 1) for i in range(h//2+1, h)]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))

        # 5 refers to LEFT to UP, type 2
        second_half = [(i, w//2) for i in range(h//2, -1,-1) ]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))


        # 6 refers to DOWN to UP, type 0
        full = [(i, w//2) for i in range(h-1,-1,-1)]
        self.routes['DOWN'].append(np.array([*full]))

        # 7 refers to DOWN to RIGHT, type 1
        first_half = full[:h//2]
        second_half = [(h//2, i) for i in range(w//2+1,w)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))

        # 8 refers to DOWN to LEFT, type 2
        second_half = [(h//2-1, i) for i in range(w//2,-1,-1)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))


        # 9 refers to RIGHT to LEFT, type 0
        full = [(h//2-1, i) for i in range(w-1,-1,-1)]
        self.routes['RIGHT'].append(np.array([*full]))

        # 10 refers to RIGHT to UP, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2) for i in range(h//2 -2, -1,-1)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))

        # 11 refers to RIGHT to DOWN, type 2
        second_half = [(i, w//2-1) for i in range(h//2-1, h)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))


        # PATHS_i: 0 to 11
        # 0 refers to UP to down,
        # 1 refers to UP to left,
        # 2 refers to UP to right,
        # 3 refers to LEFT to right,
        # 4 refers to LEFT to down,
        # 5 refers to LEFT to up,
        # 6 refers to DOWN to up,
        # 7 refers to DOWN to right,
        # 8 refers to DOWN to left,
        # 9 refers to RIGHT to left,
        # 10 refers to RIGHT to up,
        # 11 refers to RIGHT to down,

        # Convert to routes dict to list of paths
        paths = []
        for r in self.routes.values():
            for p in r:
                paths.append(p)

        # Check number of paths
        # assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)

    def _set_paths(self, difficulty):
        route_grid = self.route_grid if self.vocab_type == 'bool' else self.grid
        self.routes = get_routes(self.dims, route_grid, difficulty)

        # Convert/unroll routes which is a list of list of paths
        paths = []
        for r in self.routes:
            for p in r:
                paths.append(p)

        # Check number of paths
        assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)


    def _unittest_path(self,paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis =1)
            if np.any(step_jump != 1):
                print("Any", p, i)
                return False
            if not np.all(step_jump == 1):
                print("All", p, i)
                return False
        return True


    def _take_action(self, idx, act): #Actions - 0: Emergency brake, 1: Decelerate, 2: Maintain, 3:Accelerate
        # non-active car
        #time.sleep(.5)

        #print(self.car_loc[idx, 0])
        #NOTE: self.car_loc is two dimensional. The first column is the position of the ith car, the second column is dim / 2

        #Forces action to be brake if car is crashed with car in front, and it is not the first car. This means that the cars cannot pass each other
        i = idx

        #Sinusoidal movement of first car
        if idx == 0 and (self.counter % 6 == 0 or self.counter % 6 == 1):
            act = 3
        elif idx == 0 and (self.counter % 6 == 3 or self.counter % 6 == 4):
            act = 1
        elif idx == 0:
            act = 2

        #this is causing them all to emergency break almost every time??
        while i != 0:
            if self.car_loc[idx - i, 0] == self.car_loc[idx, 0]:
                act = 0 #Emergency brake
                break

            i = i - 1


        if self.alive_mask[idx] == 0:
            return act

        # add wait time for active cars
        self.wait[idx] += 1

            # action Emergency brake
        if act == 0:
            self.car_last_act[idx] = 0
            self.speeds[idx] = 0
            return act

        if self.speeds[idx] == 0 and act == 1: #if speed is 0 and action is decelerate, change action to be maintain
            act = 2

            # Maintain
        if act == 2:
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += self.speeds[idx]
            curr = self.car_route_loc[idx]

            #So cars don't jump each other
            for i, l in enumerate(self.car_loc):
                if i < idx:
                    if curr > l[0] and self.alive_mask[i] != 0:
                        print("CRASH")
                        self.car_route_loc[idx] = l[0]
                        curr = self.car_route_loc[idx]

            # car/agent has reached end of its path
            if curr >= len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
                self.is_completed[idx] = 1
                return act

            elif curr > len(self.chosen_path[idx]):
                print(curr)
                raise RuntimeError("Out of bound car path")

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 2
            return act

        if act == 1: #This is for decelerate


            self.speeds[idx] -= 1
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += self.speeds[idx]
            curr = self.car_route_loc[idx]

            #for i, l in enumerate(self.car_loc): #I THINK THIS FOR LOOP ACCOUNTS FOR ONE CAR TRYING TO JUMP ANOTHER ONE, AND RETURNS THEM TO SAME POSITION
                #if i != idx and l[0] == curr - 1:
                    #print("CRASH")
                    #self.car_route_loc[idx] -= 1
                    #curr = self.car_route_loc[idx]

            #This is the new loop for making sure cars don't jump each other
            for i, l in enumerate(self.car_loc):
                if i < idx:
                    if curr > l[0] and self.alive_mask[i] != 0:
                        #print("CRASH")
                        self.car_route_loc[idx] = l[0]
                        curr = self.car_route_loc[idx]




            if curr >= len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                self.car_loc[idx] = np.zeros(len(self.dims), dtype = int)
                self.is_completed[idx] = 1
                return act

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            self.car_loc[idx] = curr
            self.car_last_act[idx] = 1
            return act

        if act == 3: #This is for acceleration
            self.speeds[idx] += 1
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += self.speeds[idx]
            curr = self.car_route_loc[idx]


            for i, l in enumerate(self.car_loc):
                if i < idx:
                    if curr > l[0] and self.alive_mask[i] != 0:
                        #print("CRASH")
                        self.car_route_loc[idx] = l[0]
                        curr = self.car_route_loc[idx]

            if curr >= len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0
                self.car_loc[idx] = np.zeros(len(self.dims), dtype = int) #DOES THIS LINE MEAN THAT ALL CAR LOCS ARE RESET WHEN ONE FINISHES? I DON'T THINK SO
                self.is_completed[idx] = 1
                return act

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            self.car_loc[idx] = curr
            self.car_last_act[idx] = 3
            return act




    def _get_reward(self): #Need to add a negative reward both for an empty space behind a car, as well as an empty space in front of car
        reward = np.full(self.ncar, self.TIMESTEP_PENALTY) * self.wait #self.ncar is a number (nagents), timestep_penalty = -.01, self.wait = number of actions a car has taken
        #Negative reward from amount of actions taken - if car has taken n actions, negative timestep reward is n * -.01

        #Reward is in this function resets every time step, but is added outside this class
        #Except for time step, which keeps incrementing every turn. First turn, penalty is .01, second it is .02, the penalty increases every time step



        for i, l in enumerate(self.car_loc): #Iterate through every car (i) and give the location of the car (l)
            if (len(np.where(np.all(self.car_loc[:i] == l,axis=1))[0]) or #OR PART - If any car has the same position (first branch is cars after current car, second branch is cars before)
               len(np.where(np.all(self.car_loc[i+1:] == l,axis=1))[0])) and l.any(): #Any basically means the car is in bounds
               reward[i] += self.CRASH_PENALTY
               self.has_failed = 1

        #NEED TO CODE SO THAT THERE IS A NEGATIVE REWARD FOR HAVING EMPTY SPOTS IN FRONT AND IN BACK



        for i, l in enumerate(self.car_loc):
            currCar = i
            currCarLoc = self.car_loc[i, 0]
            frontCar = i - 1
            backCar = i + 1

            if currCar != 0: #Don't include first car in this because no front car
                frontCarLoc = self.car_loc[frontCar, 0]

                if frontCarLoc - currCarLoc > 3 or frontCarLoc - currCarLoc < 3:
                    reward[i] += self.FRONT_PENALTY
            #No negative reward for space behind for now
            #if currCar != self.ncar - 1: #Don't include last car in this because no back car
                #backCarLoc = self.car_loc[backCar, 0]

                #if currCarLoc - backCarLoc > 1:
                    #reward[i] += self.BACK_PENALTY

        #This is so if the car accelerates it gets a penalty




       #print(self.alive_mask)
       #print(actions)



        reward = self.alive_mask * reward
        #print(reward)
        return reward

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1 # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])

    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)
