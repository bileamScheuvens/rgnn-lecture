"""
Simple control task where an agent should be moved to a goal position
"""
import gymnasium as gym
from gymnasium import spaces, logger
import my_rendering as rendering
import numpy as np
import pyglet
import random
from pyglet.gl import *
from pyglet.image.codecs.png import PNGImageDecoder
import os

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'

class SimpleControlGym(gym.Env):


    def __init__(self, r_seed=42):
        super().__init__()
        np.random.seed(r_seed)
        self.agent_pos = (np.random.rand(2) - 0.5) * 2
        self.goal_pos = np.array([0.5, 0.5], dtype=np.float64)
        self.r_seed = r_seed

        self.agent_pos_upper_limits = np.array([0.95, 0.95], dtype=np.float64)
        self.agent_pos_lower_limits = np.array([-0.95, -0.95], dtype=np.float64)
        action_limits = np.array([1, 1])
        obs_limits = np.array([0.95, 0.95])
        self.action_space = spaces.Box(-1 * action_limits, action_limits, dtype=np.float64)
        self.observation_space = spaces.Box(-1 * obs_limits, obs_limits, dtype=np.float64)

        # VISUALIZATION
        self.viewer = None
        # all entities are composed of a Geom, a Transform (determining position) and a sprite
        self.agent_sprite = None
        self.agent_sprite_trans = None
        self.goal_sprite = None
        self.goal_sprite_trans = None

        # background image is treated the same way
        self.background_sprite = None
        self.background_geom = None
        self.background_trans = None

        # Threshold for reaching the goal
        self.goal_threshold = 0.1

        # Scaling of action effect
        self.action_factor = 0.1

        self.last_action = np.array([-0.1, 0], dtype=np.float64)
        self.t = 0
        self.t_render = 0

    def seed(self, seed):
        self.r_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------- STEP -------------
    def step(self, action):
        """
        Performs one step of the simulation
        :param action: next action to perform
        :return: next information, obtained reward, end of sequence?, additional inf
        """

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # Save the action
        self.last_action = action * self.action_factor

        # Move agent (and robot) based on action
        self.agent_pos = self.agent_pos + self.last_action

        # check boundaries of agent and patient position
        self._clip_positions()

        # Check if agent reached goal
        distance_agent_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        done = distance_agent_goal < self.goal_threshold
        reward = 0.0
        if done:
            reward = 1.0
        return np.copy(self.agent_pos), reward, done, [done], None

    def _clip_positions(self):
        np.clip(self.agent_pos, self.agent_pos_lower_limits, self.agent_pos_upper_limits, self.agent_pos)

    # ------------- RESET -------------
    def reset(self, seed):
        """
        Randomly reset the simulation.
        :return: first observation, additional info
        """
        self.seed(seed)
        self.agent_pos = (np.random.rand(2) - 0.5) * 2
        self.goal_pos = np.array([0.5, 0.5], dtype=np.float64)
        return self.agent_pos, None

    # ------------- RENDERING -------------

    def _determine_agent_sprite(self, action, t):
        """
        Finds the right sprite for the agent depending on the last actions
        :param action: last action
        :param t: time
        :return: sprite number
        """
        if abs(action[1]) > abs(action[0]):
            # Left right dimension is stronger than up/down
            if action[1] < 0:
                # down:
                return 0 + t % 2
            else:
                # up
                return 2 + t % 2
        else:
            if action[0] < 0:
                # left:
                return 4 + t % 2
            else:
                # right
                return 6 + t % 2

    def render(self, store_video=False, video_identifier=1, mode='human'):
        """
        Renders the simulation
        :param store_video: bool to save screenshots or not
        :param video_identifier: number to label video of this simulation
        :param mode: inherited from gym, currently not used
        """

        # Constant values of window size, sprite sizes, etc...
        screen_width = 600  # pixels
        screen_height = 630  # pixels
        agent_sprite_width = 70  # pixels of sprite
        wall_pixel_width = 12
        goal_sprite_width = 16
        goal_sprite_height = 16
        scale = 300.0  # to compute from positions -> pixels
        foreground_sprite_scale = 2  # constant scaling of foreground sprites
        background_sprite_scale = 3  # constant scaling of walls, floor, etc...

        self.t += 1
        if self.t % 10 == 0:
            self.t_render += 1

        if self.viewer is None:
            # we create a new viewer (window)
            self.viewer = rendering.Viewer(screen_width, screen_height)

            glEnable(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

            # Agents sprite list [d, u, l, r] *[0, 1]
            agent_sprite_list = []
            robot_sprite_list = []
            agent_sprite_names = ["d", "u", "l", "r"]
            for i in range(4):
                for j in range(2):
                    agent_sprite_file = SCRIPT_PATH + "sprites/" + agent_sprite_names[i] + str(j + 1) + ".png"
                    agent_image = pyglet.image.load(agent_sprite_file, decoder=PNGImageDecoder())
                    agent_pyglet_sprite = pyglet.sprite.Sprite(img=agent_image)
                    agent_sprite_list.append(agent_pyglet_sprite)

            self.agent_sprite = rendering.SpriteGeom(agent_sprite_list)
            self.agent_sprite_trans = rendering.Transform()
            self.agent_sprite.add_attr(self.agent_sprite_trans)
            self.viewer.add_geom(self.agent_sprite)

            goal_image = pyglet.image.load(SCRIPT_PATH + "sprites/target.png", decoder=PNGImageDecoder())
            goal_pyglet_sprite = pyglet.sprite.Sprite(img=goal_image)
            goal_sprite_list = [goal_pyglet_sprite]
            self.goal_sprite = rendering.SpriteGeom(goal_sprite_list)
            self.goal_sprite_trans = rendering.Transform()
            self.goal_sprite.add_attr(self.goal_sprite_trans)
            self.viewer.add_geom(self.goal_sprite)




            wall_2image = pyglet.image.load(SCRIPT_PATH + "sprites/long_wall.png", decoder=PNGImageDecoder())
            wall_2pyglet_sprite = pyglet.sprite.Sprite(img=wall_2image)
            wall_2sprite_list = [wall_2pyglet_sprite]
            wall_2sprite = rendering.SpriteGeom(wall_2sprite_list)
            wall_2sprite_trans = rendering.Transform()
            wall_2sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_2sprite_trans.set_translation(0 - wall_pixel_width / 2.0 * background_sprite_scale, 0)
            wall_2sprite.set_z(3)
            wall_2sprite.add_attr(wall_2sprite_trans)
            self.viewer.add_geom(wall_2sprite)

            wall_1image = pyglet.image.load(SCRIPT_PATH + "sprites/long_wall.png", decoder=PNGImageDecoder())
            wall_1pyglet_sprite = pyglet.sprite.Sprite(img=wall_1image)
            wall_1sprite_list = [wall_1pyglet_sprite]
            wall_1sprite = rendering.SpriteGeom(wall_1sprite_list)
            wall_1sprite_trans = rendering.Transform()
            wall_1sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_1sprite_trans.set_translation(screen_width - wall_pixel_width / 2.0 * background_sprite_scale, 0)
            wall_1sprite.set_z(3)
            wall_1sprite.add_attr(wall_1sprite_trans)
            self.viewer.add_geom(wall_1sprite)

            back_wall_image = pyglet.image.load(SCRIPT_PATH + "sprites/back_wall.png", decoder=PNGImageDecoder())
            back_wall_pyglet_sprite = pyglet.sprite.Sprite(img=back_wall_image)
            back_wall_sprite_list = [back_wall_pyglet_sprite]
            back_wall_sprite = rendering.SpriteGeom(back_wall_sprite_list)
            back_wall_sprite_trans = rendering.Transform()
            back_wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            back_wall_pixel_height = 20
            back_wall_sprite_trans.set_translation(0, screen_height - back_wall_pixel_height)
            back_wall_sprite.add_attr(back_wall_sprite_trans)
            self.viewer.add_geom(back_wall_sprite)

            front_wall_image = pyglet.image.load(SCRIPT_PATH + "sprites/grey_line.png", decoder=PNGImageDecoder())
            front_wall_pyglet_sprite = pyglet.sprite.Sprite(img=front_wall_image)
            front_wall_sprite_list = [front_wall_pyglet_sprite]
            front_wall_sprite = rendering.SpriteGeom(front_wall_sprite_list)
            front_wall_sprite_trans = rendering.Transform()
            front_wall_sprite_trans.set_translation(0, 0)
            front_wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            front_wall_sprite.add_attr(front_wall_sprite_trans)
            self.viewer.add_geom(front_wall_sprite)

            background_image = pyglet.image.load(SCRIPT_PATH + "sprites/grey_wood_background.png",
                                                 decoder=PNGImageDecoder())
            background_pyglet_sprite = pyglet.sprite.Sprite(img=background_image)
            background_sprite_list = [background_pyglet_sprite]
            background_sprite = rendering.SpriteGeom(background_sprite_list)
            background_sprite_trans = rendering.Transform()
            background_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            background_sprite.set_z(-1)
            background_sprite.add_attr(background_sprite_trans)
            self.viewer.add_geom(background_sprite)

        # during video recording images of the simulation are saved
        if store_video:
            self.viewer.activate_video_mode("Video" + str(video_identifier) + "/")

        # determine the sprite position and size for
        # 1.  ... agent
        agent_x = (self.agent_pos[0] + 1) * scale
        agent_y = (self.agent_pos[1] + 1) * scale
        self.agent_sprite.set_z(1)
        self.agent_sprite.alter_sprite_index(self._determine_agent_sprite(self.last_action, self.t_render))
        self.agent_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        sprite_center = np.array([foreground_sprite_scale * agent_sprite_width / 2.0, 0.0])
        self.agent_sprite_trans.set_translation(agent_x - sprite_center[0], agent_y - sprite_center[1])

        # 2. ... the goal
        goal_x = (self.goal_pos[0] + 1) * scale
        goal_y = (self.goal_pos[1] + 1) * scale
        self.goal_sprite.set_z(0)
        self.goal_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
        goal_sprite_center = np.array(
            [background_sprite_scale * goal_sprite_width / 2.0, background_sprite_scale * goal_sprite_height / 2.0])
        self.goal_sprite_trans.set_translation(goal_x - goal_sprite_center[0], goal_y - goal_sprite_center[1])

        return self.viewer.render(mode == 'rgb_array')

    # ------------- CLOSE -------------
    def close(self):
        """
        Shut down the gym
        """
        if self.viewer:
            self.viewer.deactivate_video_mode()
            self.viewer.close()
            self.viewer = None
