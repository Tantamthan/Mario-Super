"""Convert discrete RL actions into pygame-like key states."""

from collections import defaultdict

import pygame as pg

from source import tools


class FakeKeys(defaultdict):
    """Dict gia lap pg.key.get_pressed(), tra False cho phim khong bam."""

    def __init__(self):
        super().__init__(bool)


def action_to_keys(action: int) -> dict:
    """Map action 0-6 thanh trang thai phim ma player.py dang doc."""
    keys = FakeKeys()

    if action == 1:
        keys[tools.keybinding["right"]] = True
    elif action == 2:
        keys[tools.keybinding["jump"]] = True
    elif action == 3:
        keys[tools.keybinding["right"]] = True
        keys[tools.keybinding["jump"]] = True
    elif action == 4:
        keys[tools.keybinding["right"]] = True
        keys[tools.keybinding["action"]] = True
        keys[pg.K_LSHIFT] = True
    elif action == 5:
        keys[tools.keybinding["right"]] = True
        keys[tools.keybinding["jump"]] = True
        keys[tools.keybinding["action"]] = True
        keys[pg.K_LSHIFT] = True
    elif action == 6:
        keys[tools.keybinding["left"]] = True

    return keys
