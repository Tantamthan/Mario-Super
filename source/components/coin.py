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

# Class for a Coin sprite that represents a spinning coin in the game world
class Coin(pg.sprite.Sprite):
    def __init__(self, x, y, score_group):
        # Initialize the Coin sprite
        pg.sprite.Sprite.__init__(self)
        
        self.frames = []    # List to store animation frames for the spinning coin
        self.frame_index = 0    # Index of the current animation frame
        self.load_frames()    # Load animation frames
        self.image = self.frames[self.frame_index]    # Set the current image to the first frame
        self.rect = self.image.get_rect()    # Rectangle for the sprite's position
        self.rect.centerx = x    # Set the initial x position
        self.rect.bottom = y - 5    # Set the initial y position
        self.gravity = 1    # Gravity value (used when the coin is spinning)
        self.y_vel = -15    # Initial vertical velocity for the spinning motion
        self.animation_timer = 0    # Timer for animation updates
        self.initial_height = self.rect.bottom - 5    # Store the initial y position (bottom) of the coin
        self.score_group = score_group    # Group to which the coin belongs (used for scoring purposes)
        
    # Load animation frames for the spinning coin from the item sheet
    def load_frames(self):
        sheet = setup.GFX[c.ITEM_SHEET]    # Load the item sheet image
        # List of frame rectangles in the item sheet
        frame_rect_list = [(52, 113, 8, 14), (4, 113, 8, 14), 
                        (20, 113, 8, 14), (36, 113, 8, 14)]
        # Extract frames from the item sheet and store them in self.frames
        for frame_rect in frame_rect_list:
            self.frames.append(tools.get_image(sheet, *frame_rect, 
                            c.BLACK, c.BRICK_SIZE_MULTIPLIER))
    
    # Update the coin's spinning animation based on the game information
    def update(self, game_info):
        self.current_time = game_info[c.CURRENT_TIME]
        self.spinning()

    # Function to handle the spinning animation of the coin
    def spinning(self):
        self.image = self.frames[self.frame_index]    # Set the current image to the current frame
        self.rect.y += self.y_vel    # Adjust the y position based on the vertical velocity
        self.y_vel += self.gravity    # Apply gravity to the vertical velocity
        
        # Update the animation frame based on the timer
        if (self.current_time - self.animation_timer) > 80:
            if self.frame_index < 3:    # Move to the next frame if not the last frame
                self.frame_index += 1
            else:    # Reset frame index to loop the animation
                self.frame_index = 0
            self.animation_timer = self.current_time    # Reset the animation timer
        
        # When the coin falls back to its initial height, remove it from the score_group
        if self.rect.bottom > self.initial_height:
            self.kill()
            
# Class for a FlashCoin sprite that represents a flashing coin in the game world
class FlashCoin(pg.sprite.Sprite):
    def __init__(self, x, y):
        # Initialize the FlashCoin sprite
        pg.sprite.Sprite.__init__(self)
        self.frame_index = 0    # Index of the current animation frame
        self.frames = []    # List to store animation frames for the flashing coin
        self.load_frames()    # Load animation frames
        self.image = self.frames[self.frame_index]    # Set the current image to the first frame
        self.rect = self.image.get_rect()    # Rectangle for the sprite's position
        self.rect.x = x    # Set the initial x position
        self.rect.y = y    # Set the initial y position
        self.animation_timer = 0    # Timer for animation updates
        
    # Load animation frames for the flashing coin from the item sheet
    def load_frames(self):
        sheet = setup.GFX[c.ITEM_SHEET]    # Load the item sheet image
        # List of frame rectangles in the item sheet
        frame_rect_list = [(1, 160, 5, 8), (9, 160, 5, 8),
                        (17, 160, 5, 8), (9, 160, 5, 8)]
        # Extract frames from the item sheet and store them in self.frames
        for frame_rect in frame_rect_list:
            self.frames.append(tools.get_image(sheet, *frame_rect, 
                            c.BLACK, c.BRICK_SIZE_MULTIPLIER))

    # Update the flashing coin's animation based on the current time
    def update(self, current_time):
        time_list = [375, 125, 125, 125]    # Time intervals for each frame of the animation
        if self.animation_timer == 0:
            self.animation_timer = current_time    # Initialize the animation timer
        elif (current_time - self.animation_timer) > time_list[self.frame_index]:
            self.frame_index += 1    # Move to the next frame based on the timer
            if self.frame_index == 4:    # Reset frame index to loop the animation
                self.frame_index = 0
            self.animation_timer = current_time    # Reset the animation timer
        
        self.image = self.frames[self.frame_index]    # Set the current image to the current frame

# Class for a StaticCoin sprite that represents a static coin in the game world
class StaticCoin(pg.sprite.Sprite):
    def __init__(self, x, y):
        # Initialize the StaticCoin sprite
        pg.sprite.Sprite.__init__(self)
        self.frame_index = 0    # Index of the current animation frame
        self.frames = []    # List to store animation frames for the static coin
        self.load_frames()    # Load animation frames
        self.image = self.frames[self.frame_index]    # Set the current image to the first frame
        self.rect = self.image.get_rect()    # Rectangle for the sprite's position
        self.rect.x = x    # Set the initial x position
        self.rect.y = y    # Set the initial y position
        self.animation_timer = 0    # Timer for animation updates

    # Load animation frames for the static coin from the item sheet
    def load_frames(self):
        sheet = setup.GFX[c.ITEM_SHEET]    # Load the item sheet image
        # List of frame rectangles in the item sheet
        frame_rect_list = [(3, 98, 9, 13), (19, 98, 9, 13),
                        (35, 98, 9, 13), (51, 98, 9, 13)]
        # Extract frames from the item sheet and store them in self.frames
        for frame_rect in frame_rect_list:
            self.frames.append(tools.get_image(sheet, *frame_rect, 
                            c.BLACK, c.BRICK_SIZE_MULTIPLIER))

    # Update the static coin's animation based on the game information
    def update(self, game_info):
        self.current_time = game_info[c.CURRENT_TIME]

        time_list = [375, 125, 125, 125]    # Time intervals for each frame of the animation
        if self.animation_timer == 0:
            self.animation_timer = self.current_time    # Initialize the animation timer
        elif (self.current_time - self.animation_timer) > time_list[self.frame_index]:
            self.frame_index += 1    # Move to the next frame based on the timer
            if self.frame_index == 4:    # Reset frame index to loop the animation
                self.frame_index = 0
            self.animation_timer = self.current_time    # Reset the animation timer
        
        self.image = self.frames[self.frame_index]    # Set the current image to the current frame
