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

class Collider(pg.sprite.Sprite):
    """
    Class representing a collider sprite.
    """

    def __init__(self, x, y, width, height, name):
        """
        Initialize a collider sprite.

        Args:
            x (int): The x-coordinate of the collider's initial position.
            y (int): The y-coordinate of the collider's initial position.
            width (int): The width of the collider.
            height (int): The height of the collider.
            name (str): The name of the collider.
        """
        pg.sprite.Sprite.__init__(self)
        self.image = pg.Surface((width, height)).convert()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.name = name
        if c.DEBUG:
            self.image.fill(c.RED)

class Checkpoint(pg.sprite.Sprite):
    """
    Class representing a checkpoint sprite.
    """

    def __init__(self, x, y, width, height, type, enemy_groupid=0, map_index=0, name=c.MAP_CHECKPOINT):
        """
        Initialize a checkpoint sprite.

        Args:
            x (int): The x-coordinate of the checkpoint's initial position.
            y (int): The y-coordinate of the checkpoint's initial position.
            width (int): The width of the checkpoint.
            height (int): The height of the checkpoint.
            type (str): The type of the checkpoint.
            enemy_groupid (int): ID of the enemy group related to the checkpoint (default is 0).
            map_index (int): Index of the checkpoint on the map (default is 0).
            name (str): The name of the checkpoint (default is MAP_CHECKPOINT).
        """
        pg.sprite.Sprite.__init__(self)
        self.image = pg.Surface((width, height))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.type = type
        self.enemy_groupid = enemy_groupid
        self.map_index = map_index
        self.name = name

class Stuff(pg.sprite.Sprite):
    """
    Base class for various stuff in the game.
    """

    def __init__(self, x, y, sheet, image_rect_list, scale):
        """
        Initialize a Stuff sprite.

        Args:
            x (int): The x-coordinate of the stuff's initial position.
            y (int): The y-coordinate of the stuff's initial position.
            sheet (pygame.Surface): The sprite sheet containing the stuff's images.
            image_rect_list (list): List of image rectangles (x, y, width, height) for animation frames.
            scale (float): Scale factor for resizing the stuff's images.
        """
        pg.sprite.Sprite.__init__(self)
        
        self.frames = []
        self.frame_index = 0
        for image_rect in image_rect_list:
            self.frames.append(tools.get_image(sheet, *image_rect, c.BLACK, scale))
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    
    def update(self, *args):
        """Update the stuff (abstract method)."""
        pass

class Pole(Stuff):
    """
    Class representing a pole stuff.
    """

    def __init__(self, x, y):
        """
        Initialize a pole stuff object.

        Args:
            x (int): The x-coordinate of the pole's initial position.
            y (int): The y-coordinate of the pole's initial position.
        """
        Stuff.__init__(self, x, y, setup.GFX['tile_set'], [(263, 144, 2, 16)], c.BRICK_SIZE_MULTIPLIER)

class PoleTop(Stuff):
    """
    Class representing the top part of a pole stuff.
    """

    def __init__(self, x, y):
        """
        Initialize a pole top stuff object.

        Args:
            x (int): The x-coordinate of the pole top's initial position.
            y (int): The y-coordinate of the pole top's initial position.
        """
        Stuff.__init__(self, x, y, setup.GFX['tile_set'], [(228, 120, 8, 8)], c.BRICK_SIZE_MULTIPLIER)

class Flag(Stuff):
    """
    Class representing a flag stuff.
    """

    def __init__(self, x, y):
        """
        Initialize a flag stuff object.

        Args:
            x (int): The x-coordinate of the flag's initial position.
            y (int): The y-coordinate of the flag's initial position.
        """
        Stuff.__init__(self, x, y, setup.GFX[c.ITEM_SHEET], [(128, 32, 16, 16)], c.SIZE_MULTIPLIER)
        self.state = c.TOP_OF_POLE
        self.y_vel = 5

    def update(self):
        """Update the flag stuff's behavior."""
        if self.state == c.SLIDE_DOWN:
            self.rect.y += self.y_vel
            if self.rect.bottom >= 485:
                self.state = c.BOTTOM_OF_POLE

class CastleFlag(Stuff):
    """
    Class representing a castle flag stuff.
    """

    def __init__(self, x, y):
        """
        Initialize a castle flag stuff object.

        Args:
            x (int): The x-coordinate of the castle flag's initial position.
            y (int): The y-coordinate of the castle flag's initial position.
        """
        Stuff.__init__(self, x, y, setup.GFX[c.ITEM_SHEET], [(129, 2, 14, 14)], c.SIZE_MULTIPLIER)
        self.y_vel = -2
        self.target_height = y
    
    def update(self):
        """Update the castle flag stuff's behavior."""
        if self.rect.bottom > self.target_height:
            self.rect.y += self.y_vel

class Digit(pg.sprite.Sprite):
    """
    Class representing a digit sprite.
    """

    def __init__(self, image):
        """
        Initialize a digit sprite.

        Args:
            image (pygame.Surface): The image of the digit.
        """
        pg.sprite.Sprite.__init__(self)
        self.image = image
        self.rect = self.image.get_rect()

class Score():
    """
    Class representing the score in the game.
    """

    def __init__(self, x, y, score):
        """
        Initialize the score object.

        Args:
            x (int): The x-coordinate of the score's position.
            y (int): The y-coordinate of the score's position.
            score (int): The current score value.
        """
        self.x = x
        self.y = y
        self.y_vel = -3
        self.create_images_dict()
        self.score = score
        self.create_score_digit()
        self.distance = 130 if self.score == 1000 else 75
        
    def create_images_dict(self):
        """Create a dictionary of digit images for the score."""
        self.image_dict = {}
        digit_rect_list = [(1, 168, 3, 8), (5, 168, 3, 8),
                            (8, 168, 4, 8), (0, 0, 0, 0),
                            (12, 168, 4, 8), (16, 168, 5, 8),
                            (0, 0, 0, 0), (0, 0, 0, 0),
                            (20, 168, 4, 8), (0, 0, 0, 0)]
        digit_string = '0123456789'
        for digit, image_rect in zip(digit_string, digit_rect_list):
            self.image_dict[digit] = tools.get_image(setup.GFX[c.ITEM_SHEET], *image_rect, c.BLACK, c.BRICK_SIZE_MULTIPLIER)
    
    def create_score_digit(self):
        """Create the digit sprites for the score."""
        self.digit_group = pg.sprite.Group()
        self.digit_list = []
        for digit in str(self.score):
            self.digit_list.append(Digit(self.image_dict[digit]))
        
        for i, digit in enumerate(self.digit_list):
            digit.rect = digit.image.get_rect()
            digit.rect.x = self.x + (i * 10)
            digit.rect.y = self.y
    
    def update(self, score_list):
        """Update the score and remove it if it has moved out of the screen."""
        for digit in self.digit_list:
            digit.rect.y += self.y_vel
        
        if (self.y - self.digit_list[0].rect.y) > self.distance:
            score_list.remove(self)
            
    def draw(self, screen):
        """Draw the score digits on the screen."""
        for digit in self.digit_list:
            screen.blit(digit.image, digit.rect)

class Pipe(Stuff):
    """
    Class representing a pipe stuff.
    """

    def __init__(self, x, y, width, height, type, name=c.MAP_PIPE):
        """
        Initialize a pipe stuff object.

        Args:
            x (int): The x-coordinate of the pipe's initial position.
            y (int): The y-coordinate of the pipe's initial position.
            width (int): The width of the pipe.
            height (int): The height of the pipe.
            type (str): The type of the pipe.
            name (str): The name of the pipe (default is MAP_PIPE).
        """
        if type == c.PIPE_TYPE_HORIZONTAL:
            rect = [(32, 128, 37, 30)]
        else:
            rect = [(0, 160, 32, 30)]
        Stuff.__init__(self, x, y, setup.GFX['tile_set'], rect, c.BRICK_SIZE_MULTIPLIER)
        self.name = name
        self.type = type
        if type != c.PIPE_TYPE_HORIZONTAL:
            self.create_image(x, y, height)

    def create_image(self, x, y, pipe_height):
        """Create an image for the pipe based on its height."""
        img = self.image
        rect = self.image.get_rect()
        width = rect.w
        height = rect.h
        self.image = pg.Surface((width, pipe_height)).convert()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        top_height = height//2 + 3
        bottom_height = height//2 - 3
        self.image.blit(img, (0,0), (0, 0, width, top_height))
        num = (pipe_height - top_height) // bottom_height + 1
        for i in range(num):
            y = top_height + i * bottom_height
            self.image.blit(img, (0,y), (0, top_height, width, bottom_height))
        self.image.set_colorkey(c.BLACK)

    def check_ignore_collision(self, level):
        """
        Check if the collision with the pipe should be ignored.

        Args:
            level (Level): The current level object.

        Returns:
            bool: True if the collision should be ignored, False otherwise.
        """
        if self.type == c.PIPE_TYPE_HORIZONTAL:
            return True
        elif level.player.state == c.DOWN_TO_PIPE:
            return True
        return False

class Slider(Stuff):
    """
    Class representing a slider stuff.
    """

    def __init__(self, x, y, num, direction, range_start, range_end, vel, name=c.MAP_SLIDER):
        """
        Initialize a slider stuff object.

        Args:
            x (int): The x-coordinate of the slider's initial position.
            y (int): The y-coordinate of the slider's initial position.
            num (int): The number of slider images to be created.
            direction (str): The direction of the slider (VERTICAL or HORIZONTAL).
            range_start (int): The start position of the slider's movement range.
            range_end (int): The end position of the slider's movement range.
            vel (int): The velocity of the slider's movement.
            name (str): The name of the slider (default is MAP_SLIDER).
        """
        Stuff.__init__(self, x, y, setup.GFX[c.ITEM_SHEET], [(64, 128, 15, 8)], 2.8)
        self.name = name
        self.create_image(x, y, num)
        self.range_start = range_start
        self.range_end = range_end
        self.direction = direction
        if self.direction == c.VERTICAL:
            self.y_vel = vel
        else:
            self.x_vel = vel

    def create_image(self, x, y, num):
        """Create an image for the slider with multiple copies."""
        if num == 1:
            return
        img = self.image
        rect = self.image.get_rect()
        width = rect.w
        height = rect.h
        self.image = pg.Surface((width * num, height)).convert()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        for i in range(num):
            x = i * width
            self.image.blit(img, (x,0))
        self.image.set_colorkey(c.BLACK)

    def update(self):
        """Update the slider's movement based on its direction and movement range."""
        if self.direction == c.VERTICAL:
            self.rect.y += self.y_vel
            if self.rect.y < -self.rect.h:
                self.rect.y = c.SCREEN_HEIGHT
                self.y_vel = -1
            elif self.rect.y > c.SCREEN_HEIGHT:
                self.rect.y = -self.rect.h
                self.y_vel = 1
            elif self.rect.y < self.range_start:
                self.rect.y = self.range_start
                self.y_vel = 1
            elif self.rect.bottom > self.range_end:
                self.rect.bottom = self.range_end
                self.y_vel = -1
        else:
            self.rect.x += self.x_vel
            if self.rect.x < self.range_start:
                self.rect.x = self.range_start
                self.x_vel = 1
            elif self.rect.left > self.range_end:
                self.rect.left = self.range_end
                self.x_vel = -1
