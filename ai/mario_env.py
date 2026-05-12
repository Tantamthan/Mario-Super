"""Gymnasium environment wrapper for the Pygame Mario clone."""

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.constants import (
    ACTION_COUNT,
    FRAME_SKIP,
    GAP_LOOKAHEAD,
    GAP_SCAN_STEP,
    GROUND_SCAN_DEPTH,
    GROUND_SENSOR_OFFSETS,
    IDLE_DELTA_X,
    IDLE_REWARD_STEPS,
    IDLE_TRUNCATE_STEPS,
    MAP_WIDTH,
    MAX_SCORE,
    MAX_TIME,
    MAX_VEL,
    MS_PER_FRAME,
    OBS_HIGH,
    OBS_LOW,
    OBSERVATION_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from ai.fake_keys import action_to_keys


class MarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, headless=True):
        super().__init__()
        self.headless = headless
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"

        import pygame as pg
        from source import constants as game_constants
        from source import setup
        from source.states import level

        self.pg = pg
        self.c = game_constants
        self.setup = setup
        self.level_cls = level.Level

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.surface = self.setup.SCREEN
        self.current_time = 0
        self.level = None
        self.prev_x = 0
        self.prev_score = 0
        self.idle_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.level = self.level_cls()
        persist = {
            self.c.COIN_TOTAL: 0,
            self.c.SCORE: 0,
            self.c.LIVES: 3,
            self.c.TOP_SCORE: 0,
            self.c.CURRENT_TIME: 0.0,
            self.c.LEVEL_NUM: 1,
            self.c.PLAYER_NAME: self.c.PLAYER_MARIO,
        }
        self.level.startup(self.current_time, persist)
        self.prev_x = self.level.player.rect.x
        self.prev_score = self.level.game_info[self.c.SCORE]
        self.idle_steps = 0
        return self._get_observation(), {}

    def step(self, action):
        keys = action_to_keys(int(action))
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(FRAME_SKIP):
            self.current_time += MS_PER_FRAME
            self.pg.event.pump()
            self.level.update(
                self.surface,
                keys,
                self.current_time,
                headless=self.headless,
            )
            total_reward += self._compute_reward()

            terminated = self._mario_died() or self._level_cleared()
            truncated = self._time_over() or self.idle_steps > IDLE_TRUNCATE_STEPS
            if terminated or truncated:
                break

        info = {
            "x": self.level.player.rect.x,
            "score": self.level.game_info[self.c.SCORE],
            "time": self._time_remaining(),
            "idle_steps": self.idle_steps,
        }
        return self._get_observation(), float(total_reward), terminated, truncated, info

    def render(self):
        if self.headless:
            return np.transpose(
                self.pg.surfarray.array3d(self.surface),
                axes=(1, 0, 2),
            )
        self.pg.display.update()
        return None

    def close(self):
        self.pg.display.quit()

    def _get_observation(self):
        mario = self.level.player
        score = self.level.game_info[self.c.SCORE]

        # Observation gom cac dac trung so da normalize de PPO hoc nhanh hon anh tho.
        ground_sensors = self._ground_sensors()
        obs = np.array(
            [
                self._clip01(mario.rect.x / MAP_WIDTH),  # vi tri ngang cua Mario
                self._clip01(mario.rect.y / SCREEN_HEIGHT),  # vi tri doc cua Mario
                self._clip_signed(mario.x_vel / MAX_VEL),  # van toc ngang
                self._clip_signed(mario.y_vel / MAX_VEL),  # van toc doc
                1.0 if mario.state == self.c.JUMP else 0.0,  # Mario dang nhay
                self._nearest_enemy_distance() / SCREEN_WIDTH,  # enemy gan nhat
                self._nearest_obstacle_distance() / SCREEN_WIDTH,  # vat can phia truoc
                self._clip01(score / MAX_SCORE),  # diem hien tai
                self._clip01(self._time_remaining() / MAX_TIME),  # thoi gian con lai
                1.0 if self._is_on_ground() else 0.0,  # Mario dang dung tren nen
                ground_sensors[0],  # co nen o diem gan phia truoc
                ground_sensors[1],  # co nen o diem trung binh phia truoc
                ground_sensors[2],  # co nen o diem xa phia truoc
                self._gap_distance() / GAP_LOOKAHEAD,  # khoang cach toi ho gan nhat
            ],
            dtype=np.float32,
        )
        return np.clip(obs, OBS_LOW, OBS_HIGH).astype(np.float32)

    def _compute_reward(self):
        mario = self.level.player
        score = self.level.game_info[self.c.SCORE]
        delta_x = mario.rect.x - self.prev_x

        if abs(delta_x) <= IDLE_DELTA_X:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        # Reward khuyen khich di sang phai, tang diem, qua man; phat chet va dung yen.
        reward = delta_x * 0.1
        reward += (score - self.prev_score) * 0.5
        if self._mario_died():
            reward -= 15.0
        if self._level_cleared():
            reward += 50.0
        reward -= 0.01
        if self.idle_steps > IDLE_REWARD_STEPS:
            reward -= 5.0

        self.prev_x = mario.rect.x
        self.prev_score = score
        return reward

    def _nearest_enemy_distance(self):
        mario_x = self.level.player.rect.centerx
        distances = []
        for group in (self.level.enemy_group, self.level.shell_group):
            for sprite in group:
                distance = sprite.rect.centerx - mario_x
                if distance >= 0:
                    distances.append(distance)
        if not distances:
            return float(SCREEN_WIDTH)
        return float(min(min(distances), SCREEN_WIDTH))

    def _nearest_obstacle_distance(self):
        mario = self.level.player
        groups = (
            self.level.pipe_group,
            self.level.step_group,
            self.level.brick_group,
            self.level.box_group,
        )
        distances = []
        for group in groups:
            for sprite in group:
                vertical_overlap = sprite.rect.bottom > mario.rect.top and sprite.rect.top < mario.rect.bottom
                distance = sprite.rect.left - mario.rect.right
                if vertical_overlap and distance >= 0:
                    distances.append(distance)
        if not distances:
            return float(SCREEN_WIDTH)
        return float(min(min(distances), SCREEN_WIDTH))

    def _ground_groups(self):
        return (
            self.level.ground_group,
            self.level.step_group,
            self.level.pipe_group,
            self.level.slider_group,
            self.level.brick_group,
            self.level.box_group,
        )

    def _has_ground_below(self, x, foot_y):
        scan_rect = self.pg.Rect(int(x), int(foot_y), 4, GROUND_SCAN_DEPTH)
        for group in self._ground_groups():
            for sprite in group:
                if sprite.rect.colliderect(scan_rect):
                    return True
        return False

    def _is_on_ground(self):
        mario = self.level.player
        if mario.state in (self.c.JUMP, self.c.FALL, self.c.DEATH_JUMP):
            return False
        return self._has_ground_below(mario.rect.centerx, mario.rect.bottom + 1)

    def _ground_sensors(self):
        mario = self.level.player
        direction = 1 if mario.facing_right else -1
        foot_y = mario.rect.bottom + 2
        sensors = []
        for offset in GROUND_SENSOR_OFFSETS:
            x = mario.rect.centerx + (direction * offset)
            sensors.append(1.0 if self._has_ground_below(x, foot_y) else 0.0)
        return sensors

    def _gap_distance(self):
        mario = self.level.player
        direction = 1 if mario.facing_right else -1
        foot_y = mario.rect.bottom + 2
        for distance in range(GAP_SCAN_STEP, GAP_LOOKAHEAD + GAP_SCAN_STEP, GAP_SCAN_STEP):
            x = mario.rect.centerx + (direction * distance)
            if not self._has_ground_below(x, foot_y):
                return float(min(distance, GAP_LOOKAHEAD))
        return float(GAP_LOOKAHEAD)

    def _mario_died(self):
        return bool(self.level.player.dead)

    def _level_cleared(self):
        return bool(
            self.level.player.state == self.c.IN_CASTLE
            or (self.level.done and not self.level.player.dead and self.level.next == self.c.LOAD_SCREEN)
        )

    def _time_over(self):
        return self._time_remaining() <= 0

    def _time_remaining(self):
        return getattr(self.level.overhead_info, "time", MAX_TIME)

    @staticmethod
    def _clip01(value):
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _clip_signed(value):
        return float(np.clip(value, -1.0, 1.0))
