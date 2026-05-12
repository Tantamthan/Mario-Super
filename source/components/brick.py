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
from . import coin, stuff, powerup

# Function to create a single brick object or a group of bricks
def create_brick(brick_group, item, level):
    # Check if the brick has a color specified, else set it to orange
    if c.COLOR in item:
        color = item[c.COLOR]
    else:
        color = c.COLOR_TYPE_ORANGE

    # Extract x, y, and type values from the item dictionary
    x, y, type = item['x'], item['y'], item['type']
    
    # Check the type of the brick and add it to the appropriate group
    if type == c.TYPE_COIN:
        brick_group.add(Brick(x, y, type, color, level.coin_group))
    elif (type == c.TYPE_STAR or
        type == c.TYPE_FIREFLOWER or
        type == c.TYPE_LIFEMUSHROOM):
        brick_group.add(Brick(x, y, type, color, level.powerup_group))
    else:
        # If the brick has multiple parts (e.g., a group of bricks), create each part
        if c.BRICK_NUM in item:
            create_brick_list(brick_group, item[c.BRICK_NUM], x, y, type, color, item['direction'])
        else:
            brick_group.add(Brick(x, y, type, color))
            
# Function to create a group of bricks in a specific direction (horizontal or vertical)
def create_brick_list(brick_group, num, x, y, type, color, direction):
    ''' direction:horizontal, create brick from left to right, direction:vertical, create brick from up to bottom '''
    size = 43 # 16 * c.BRICK_SIZE_MULTIPLIER is 43
    tmp_x, tmp_y = x, y
    for i in range(num):
        if direction == c.VERTICAL:
            tmp_y = y + i * size
        else:
            tmp_x = x + i * size
        brick_group.add(Brick(tmp_x, tmp_y, type, color))
        
# Class for a Brick sprite that represents a block in the game world
class Brick(stuff.Stuff):
    def __init__(self, x, y, type, color=c.ORANGE, group=None, name=c.MAP_BRICK):
        orange_rect = [(16, 0, 16, 16), (432, 0, 16, 16)]
        green_rect = [(208, 32, 16, 16), (48, 32, 16, 16)]
        # Set the frame rectangles based on the brick's color
        if color == c.COLOR_TYPE_ORANGE:
            frame_rect = orange_rect
        else:
            frame_rect = green_rect
        # Initialize the Brick sprite using the parent class Stuff
        stuff.Stuff.__init__(self, x, y, setup.GFX['tile_set'],
                        frame_rect, c.BRICK_SIZE_MULTIPLIER)

        self.rest_height = y    # Store the resting height of the brick
        self.state = c.RESTING    # State of the brick (resting, bumped, or opened)
        self.y_vel = 0    # Vertical velocity (used when bumped)
        self.gravity = 1.2    # Gravity value (used when bumped)
        self.type = type    # Type of the brick (e.g., c.TYPE_COIN, c.TYPE_STAR, etc.)
        if self.type == c.TYPE_COIN:
            self.coin_num = 10    # Number of coins that will come out of the brick when bumped
        else:
            self.coin_num = 0
        self.group = group    # Group to which the brick belongs (used for adding power-ups)
        self.name = name    # Name of the brick (default: c.MAP_BRICK)
    
    def update(self):
        if self.state == c.BUMPED:
            self.bumped()
    
    # Function to handle the bumped state of the brick (when it's hit from below)
    def bumped(self):
        self.rect.y += self.y_vel    # Adjust the y position based on the vertical velocity
        self.y_vel += self.gravity    # Apply gravity to the vertical velocity
        
        # When the brick falls back to its resting height, change its state based on the type
        if self.rect.y >= self.rest_height:
            self.rect.y = self.rest_height
            if self.type == c.TYPE_COIN:
                if self.coin_num > 0:
                    self.state = c.RESTING
                else:
                    self.state = c.OPENED
            elif self.type in [c.TYPE_STAR, c.TYPE_FIREFLOWER, c.TYPE_LIFEMUSHROOM]:
                self.state = c.OPENED
                if self.type == c.TYPE_STAR:
                    self.group.add(powerup.Star(self.rect.centerx, self.rest_height))
                elif self.type == c.TYPE_FIREFLOWER:
                    self.group.add(powerup.FireFlower(self.rect.centerx, self.rest_height))
                elif self.type == c.TYPE_LIFEMUSHROOM:
                    self.group.add(powerup.LifeMushroom(self.rect.centerx, self.rest_height))
            else:
                self.state = c.RESTING
        
    # Function to start the bumping animation of the brick (when it's hit from below by the player)
    def start_bump(self, score_group):
        self.y_vel -= 7    # Set the initial vertical velocity for the bump
        
        # Handle different types of bricks when bumped
        if self.type == c.TYPE_COIN:
            if self.coin_num > 0:
                self.group.add(coin.Coin(self.rect.centerx, self.rect.y, score_group))
                self.coin_num -= 1
                if self.coin_num == 0:
                    self.frame_index = 1
                    self.image = self.frames[self.frame_index]
        elif self.type in [c.TYPE_STAR, c.TYPE_FIREFLOWER, c.TYPE_LIFEMUSHROOM]:
            self.frame_index = 1
            self.image = self.frames[self.frame_index]
        
        self.state = c.BUMPED    # Change the state of the brick to BUMPED
    
    # Function to change the brick into pieces (used when it's destroyed by certain events)
    def change_to_piece(self, group):
        arg_list = [(self.rect.x, self.rect.y - (self.rect.height/2), -2, -12),
                    (self.rect.right, self.rect.y - (self.rect.height/2), 2, -12),
                    (self.rect.x, self.rect.y, -2, -6),
                    (self.rect.right, self.rect.y, 2, -6)]
        
        # Create brick pieces and add them to the provided group
        for arg in arg_list:
            group.add(BrickPiece(*arg))
        self.kill()    # Remove the original brick from the group
        
# Class for a BrickPiece sprite that represents a piece of a destroyed brick
class BrickPiece(stuff.Stuff):
    def __init__(self, x, y, x_vel, y_vel):
        # Initialize the BrickPiece sprite using the parent class Stuff
        stuff.Stuff.__init__(self, x, y, setup.GFX['tile_set'],
            [(68, 20, 8, 8)], c.BRICK_SIZE_MULTIPLIER)
        self.x_vel = x_vel    # Horizontal velocity
        self.y_vel = y_vel    # Vertical velocity
        self.gravity = .8    # Gravity value
    
    def update(self, *args):
        self.rect.x += self.x_vel    # Update the x position based on horizontal velocity
        self.rect.y += self.y_vel    # Update the y position based on vertical velocity
        self.y_vel += self.gravity    # Apply gravity to the vertical velocity
        if self.rect.y > c.SCREEN_HEIGHT:
            self.kill()    # Remove the BrickPiece sprite when it goes off the screen
