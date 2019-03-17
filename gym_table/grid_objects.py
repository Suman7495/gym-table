import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_table.utils import *


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def _set_color(self, r):
        """Set the color of this object as the active drawing color"""
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])


class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c/2)
        r.drawPolygon([
            (1          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           1),
            (1          ,           1)
        ])



class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])


class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2], 50 if self.is_locked else 0)

        if self.is_open:
            r.drawPolygon([
                (CELL_PIXELS-2, CELL_PIXELS),
                (CELL_PIXELS  , CELL_PIXELS),
                (CELL_PIXELS  ,           0),
                (CELL_PIXELS-2,           0)
            ])
            return

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
        r.drawPolygon([
            (2            , CELL_PIXELS-2),
            (CELL_PIXELS-2, CELL_PIXELS-2),
            (CELL_PIXELS-2,           2),
            (2            ,           2)
        ])

        if self.is_locked:
            # Draw key slot
            r.drawLine(
                CELL_PIXELS * 0.55,
                CELL_PIXELS * 0.5,
                CELL_PIXELS * 0.75,
                CELL_PIXELS * 0.5
            )
        else:
            # Draw door handle
            r.drawCircle(CELL_PIXELS * 0.75, CELL_PIXELS * 0.5, 2)


class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)

        # Vertical quad
        r.drawPolygon([
            (16, 10),
            (20, 10),
            (20, 28),
            (16, 28)
        ])

        # Teeth
        r.drawPolygon([
            (12, 19),
            (16, 19),
            (16, 21),
            (12, 21)
        ])
        r.drawPolygon([
            (12, 26),
            (16, 26),
            (16, 28),
            (12, 28)
        ])

        r.drawCircle(18, 9, 6)
        r.setLineColor(0, 0, 0)
        r.setColor(0, 0, 0)
        r.drawCircle(18, 9, 2)


class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)


class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)
        r.setLineWidth(2)

        r.drawPolygon([
            (4            , CELL_PIXELS-4),
            (CELL_PIXELS-4, CELL_PIXELS-4),
            (CELL_PIXELS-4,             4),
            (4            ,             4)
        ])

        r.drawLine(
            4,
            CELL_PIXELS / 2,
            CELL_PIXELS - 4,
            CELL_PIXELS / 2
        )

        r.setLineWidth(1)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True