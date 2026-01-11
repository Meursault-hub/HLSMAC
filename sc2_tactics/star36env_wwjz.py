from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.sc2_tactics.maps import get_map_params
from smac.env.sc2_tactics.utils import map_specific_utils
from smac.env.sc2_tactics.utils import common_utils

import atexit
from warnings import warn
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

import smac.env.sc2_tactics.sc2_tactics_env as te

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsWWJZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("----------------------")
        print("You create a WWJZ env!")
        print("----------------------")
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.unit_type == self.rlunit_ids.get("nexus") and unit.health > 0:
            avail_actions = [0] * self.n_actions
            avail_actions[1] = 1
            return avail_actions
        else:
            return super().get_avail_agent_actions(agent_id)
        
    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 59:
                type_id = 0 # nexus
            elif unit.unit_type == 73:
                type_id = 1 # nexus
            else:
                type_id = 99 # error
        else:  # use default SC2 unit types
            if unit.unit_type == 18:
                type_id = 0 # commandCenter
            elif unit.unit_type == 48:
                type_id = 1 # marine
            elif unit.unit_type == 53:
                type_id = 1 # Hellion
            else:
                type_id = 99 # error
        return type_id

    def check_structure(self, ally = True):
        """Check if the enemy's Nexus unit is killed."""
        if ally == True:
            for a in self.agents.values():
                if a.unit_type == self.rlunit_ids.get("nexus") and a.health <= 0:
                    return True
        
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type == 18 and e.health <= 0:
                    return True
        return False
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 18 and e.health > 0:
                    return False
            return True
        
        if ally == True:
            for a in self.agents.values():
                if a.unit_type != self.rlunit_ids.get("nexus") and a.health > 0:
                    return False
            return True