"""
MIT License

Copyright (c) [2023] [m0rniac]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__author__ = 'm0rniac'

import pygame as pg
from .. import setup, tools
from .. import constants as c
from . import coin, powerup

# Class for a Box sprite that represents a block in the game world
class Box(pg.sprite.Sprite):
    def __init__(self, x, y, type, group=None, name=c.MAP_BOX):
        # Initialize the Box sprite
        pg.sprite.Sprite.__init__(self)
        
        self.frames = []    # List to store animation frames for the box
        self.frame_index = 0    # Index of the current animation frame
        self.load_frames()    # Load animation frames
        self.image = self.frames[self.frame_index]    # Set the current image to the first frame
        self.rect = self.image.get_rect()    # Rectangle for the sprite's position
        self.rect.x = x    # Set the initial x position
        self.rect.y = y    # Set the initial y position

        self.rest_height = y    # Store the resting height of the box
        self.animation_timer = 0    # Timer for animation updates
        self.first_half = True    # Flag for the first half of the animation cycle
        self.state = c.RESTING    # State of the box (resting, bumped, or opened)
        self.y_vel = 0    # Vertical velocity (used when bumped)
        self.gravity = 1.2    # Gravity value (used when bumped)
        self.type = type    # Type of the box (e.g., c.TYPE_COIN, c.TYPE_MUSHROOM, etc.)
        self.group = group    # Group to which the box belongs (used for adding power-ups)
        self.name = name    # Name of the box (default: c.MAP_BOX)
        
    # Load animation frames for the box from the tile sheet
    def load_frames(self):
        sheet = setup.GFX['tile_set']    # Load the tile sheet image
        # List of frame rectangles in the tile sheet
        frame_rect_list = [(384, 0, 16, 16), (400, 0, 16, 16), 
            (416, 0, 16, 16), (400, 0, 16, 16), (432, 0, 16, 16)]
        # Extract frames from the tile sheet and store them in self.frames
        for frame_rect in frame_rect_list:
            self.frames.append(tools.get_image(sheet, *frame_rect, 
                            c.BLACK, c.BRICK_SIZE_MULTIPLIER))
    
    # Update the box's animation and state based on the game information
    def update(self, game_info):
        self.current_time = game_info[c.CURRENT_TIME]    # Get the current game time
        if self.state == c.RESTING:
            self.resting()
        elif self.state == c.BUMPED:
            self.bumped()

    # Function to handle the resting state animation of the box
    def resting(self):
        time_list = [375, 125, 125, 125]    # Time intervals for each frame of the animation
        if (self.current_time - self.animation_timer) > time_list[self.frame_index]:
            self.frame_index += 1    # Move to the next frame
            if self.frame_index == 4:    # Reset frame index to loop the animation
                self.frame_index = 0
            self.animation_timer = self.current_time    # Reset the animation timer

        self.image = self.frames[self.frame_index]    # Set the current image to the current frame
    
    # Function to handle the bumped state of the box (when it's hit from below)
    def bumped(self):
        self.rect.y += self.y_vel    # Adjust the y position based on the vertical velocity
        self.y_vel += self.gravity    # Apply gravity to the vertical velocity
        
        # When the box falls back to its resting height, change its state to OPENED
        if self.rect.y > self.rest_height + 5:
            self.rect.y = self.rest_height
            self.state = c.OPENED
            # Based on the type of the box, add the corresponding power-up to the group
            if self.type == c.TYPE_MUSHROOM:
                self.group.add(powerup.Mushroom(self.rect.centerx, self.rect.y))
            elif self.type == c.TYPE_FIREFLOWER:
                self.group.add(powerup.FireFlower(self.rect.centerx, self.rect.y))
            elif self.type == c.TYPE_LIFEMUSHROOM:
                self.group.add(powerup.LifeMushroom(self.rect.centerx, self.rect.y))
        self.frame_index = 4    # Set the frame index to the last frame for the bumped state
        self.image = self.frames[self.frame_index]    # Set the current image to the bumped frame
    
    # Function to start the bumping animation of the box (when it's hit from below by the player)
    def start_bump(self, score_group):
        self.y_vel = -6    # Set the initial vertical velocity for the bump
        self.state = c.BUMPED    # Change the state of the box to BUMPED
        
        # If the box is of type c.TYPE_COIN, add a coin to the group with the score_group information
        if self.type == c.TYPE_COIN:
            self.group.add(coin.Coin(self.rect.centerx, self.rect.y, score_group))
