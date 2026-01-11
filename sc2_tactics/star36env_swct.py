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

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "WarpPrismLoad": 911,
    "WarpPrismUnload": 913,
    "ForceField": 1526,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

fx = 44
fy = 44
fPoint = sc_common.Point2D(x=fx, y=fy)

class SC2TacticsSWCTEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load = {}
        self.n_actions += 1
        self.n_actions_no_attack += 1
        print("----------------------")
        print("You create a SWCT env!")
        print("----------------------")
    
    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        
        elif a_id in self.load:
            return None

        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))

        elif action == 6:
            if unit.unit_type == self.rlunit_ids.get("warpPrism"):
                # Load all units in the sight range
                target_tag = 0
                for t_id, t_unit in self.agents.items():
                    if (t_unit.unit_type == self.rlunit_ids.get("sentry") and 
                        self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                        <= self.unit_load_range(t_id) and
                        t_id not in self.load):
                        target_tag = t_unit.tag
                        self.load[t_id] = t_unit
                        break
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["WarpPrismLoad"],
                    target_unit_tag=target_tag,
                    unit_tags=[tag],
                    queue_command=False,
                )
            elif unit.unit_type == self.rlunit_ids.get("sentry"):
                # use force field at the specific location
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["ForceField"],
                    target_world_space_pos=fPoint,
                    unit_tags=[tag],
                    queue_command=False,
                )

        elif action == 7 and unit.unit_type == self.rlunit_ids.get("warpPrism"):
            # Unload an agent
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["WarpPrismUnload"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )

        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            target_unit = self.enemies[target_id]
            action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(
                        a_id, action_name, target_id
                    )
                )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            if unit.unit_type == self.rlunit_ids.get("warpPrism"):
                # check if WarpPrism can load units
                for t_id, t_unit in self.agents.items():
                    if (len(self.load) < self.get_load_max() and
                        t_unit.unit_type == self.rlunit_ids.get("sentry") and 
                        t_id not in self.load):
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                        )
                        load_range = self.unit_load_range(agent_id)
                        if dist <= load_range:
                            avail_actions[6] = 1
                            break
                if self.load != {} and self.pathing_grid[int(unit.pos.x), int(unit.pos.y)]:
                    avail_actions[7] = 1
                return avail_actions

            dist = self.distance(unit.pos.x, unit.pos.y, fx, fy)
            cast_range = self.spell_cast_range(agent_id)
            if dist <= cast_range and unit.energy >= 50:
                avail_actions[6] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 77:
                type_id = 0 # sentry
            elif unit.unit_type == 81:
                type_id = 1 # warpPrism
        else:
            if unit.unit_type == 86:
                type_id = 0 # hatchery
            elif unit.unit_type == 105:
                type_id = 1 # zergling
        return type_id
    
    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        # Check if roach is unloaded in dhls
        self.clean_load()

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            updated = self.check_load(al_unit.tag, updated)

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
            n_ally_alive == 0
            and n_enemy_alive > 0
            or self.check_end_code(ally=True)
        ):
            return -1  # lost
        if (
            n_ally_alive > 0
            and n_enemy_alive == 0
            or self.check_end_code(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None
        
    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit != None and unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit.health > 0] + [
            unit.tag for unit in self.load.values() if unit.health > 0
        ] + [unit.tag for unit in self._obs.observation.raw_data.units if unit.owner == 16]
        self.load = {}
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if unit.unit_type == self.rlunit_ids.get("warpPrism"):
            if self.check_bounds(x, y):
                return True
            return False
        
        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def unit_load_range(self, a_id):
        """Returns the load range for the WarpPrism."""
        return 5

    def get_load_max(self):
        """Returns the max load of the WarpPrism, which is 4"""
        return 4
    
    def spell_cast_range(self, a_id):
        """Returns the range of the sentry, which is 8"""
        return 8

    def init_units(self):
        """Initialise the units."""
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}
            self.load = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if (unit.owner == 1)   # not larva
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values() if unit is not None
                )
                self._init_assign_aliases(min_unit_type)
                self.cooldown_map = common_utils.build_cooldown_map(self.rlunit_ids)

            all_agents_created = len(self.agents) == self.n_agents
            all_enemies_created = len(self.enemies) == self.n_enemies

            self._unit_types = [
                unit.unit_type for unit in ally_units_sorted
            ] + [
                unit.unit_type
                for unit in self._obs.observation.raw_data.units
                if (unit.owner == 2 and unit.unit_type != 151)
            ]

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def check_structure(self, ally = True):
        """Check if the enemy's Hatchery unit is killed."""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type == 86 and e.health <= 0:
                    return True
        return False
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 86 and e.health > 0:
                    return False
            return True
        
        if ally == True:
            for a in self.agents.values():
                if self.load != {} or (a.unit_type != self.rlunit_ids.get("warpPrism") and a.health > 0):
                    return False
            return True
        
    def clean_load(self):
        """If the self.last_action for WarpPrism is unloading, remove the respective unit from self.load"""
        for a_id, a_unit in self.agents.items():
            if a_unit.unit_type == self.rlunit_ids.get("warpPrism"):
                if self.last_action[a_id][7] == 1:
                    self.find_unload_unit()
                if self.last_action[a_id][0] == 1:
                    self.load = {}
                return
        return
            
    def find_unload_unit(self):
        """find the unload unit and remove it from the load"""
        for a_id, a_unit in self.load.items():
            for o_unit in self._obs.observation.raw_data.units:
                if o_unit.tag == a_unit.tag:
                    # finded the unload unit
                    del self.load[a_id]
                    return
    
    def check_load(self, a_tag, updated):
        """make sure the units in load is not dead"""
        for t_unit in self.load.values():
            if a_tag == t_unit.tag:
                return True
        return updated