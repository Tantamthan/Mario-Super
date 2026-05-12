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

import os
import pygame as pg
from . import constants as c
from . import tools


# Initialize pygame
pg.init()

# Set allowed events for the game
pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

# Set the caption for the game window
pg.display.set_caption(c.ORIGINAL_CAPTION)

# Create the game window/screen
SCREEN = pg.display.set_mode(c.SCREEN_SIZE)

# Get the rectangle representing the dimensions of the game window
SCREEN_RECT = SCREEN.get_rect()

# Load all the graphics from the specified directory
GFX = tools.load_all_gfx(os.path.join("resources", "graphics"))
