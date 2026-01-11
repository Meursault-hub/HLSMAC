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
    "BurrowDown": 1444,
    "BurrowUp": 1446,
    "SpawnEgg": 247,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsADCCEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        map_params = get_map_params(self.map_name)
        # self.n_actions += 1
        self.n_agents_max = map_params["n_agents_max"]
        self.last_action = np.zeros((self.n_agents_max, self.n_actions))
        self.death_tracker_ally = np.zeros(self.n_agents_max)
        self.n_agents_egg_temp = 0
        print("----------------------")
        print("You create a ADCC env!")
        print("----------------------")

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents_max, self.n_actions))
        self.death_tracker_ally = np.zeros(self.n_agents_max)

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        return self.get_obs(), self.get_state()
    
    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        if unit == None:
            return None
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
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

        elif unit.unit_type == self.rlunit_ids.get("infestor"):
            if action == 6:
                # burrowDown
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["BurrowDown"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                if self.debug:
                    logging.debug("Agent {}: burrowDown".format(a_id))
            elif action == 8 and self.n_agents <= self.n_agents_max - 1 - self.n_agents_egg_temp:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["SpawnEgg"],
                    target_world_space_pos=sc_common.Point2D(
                        x=x, y=y
                    ),
                    unit_tags=[tag],
                    queue_command=False,
                )
                self.n_agents_egg_temp += 1
                if self.debug:
                    logging.debug("Agent {}: Spawn Egg".format(a_id))
            else:
                return None
        
        elif unit.unit_type == self.rlunit_ids.get("infestorBurrowed"):
            if action == 7:
                # burrowUp
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["BurrowUp"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                if self.debug:
                    logging.debug("Agent {}: burrowUp".format(a_id))
            elif action == 8 and self.n_agents <= self.n_agents_max - 1 - self.n_agents_egg_temp:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["SpawnEgg"],
                    target_world_space_pos=sc_common.Point2D(
                        x=x, y=y
                    ),
                    unit_tags=[tag],
                    queue_command=False,
                )
                self.n_agents_egg_temp += 1
                if self.debug:
                    logging.debug("Agent {}: Spawn Egg".format(a_id))
            else:
                return None

        else:
            # attack units that are in range
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
    
    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents_max)]
        return agents_obs
    
    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - enemies: numpy array containing enemies and their attributes
        - last_action: numpy array of previous actions for each agent
        - timestep: current no. of steps divided by total no. of steps

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes()
        nf_en = self.get_enemy_num_attributes()

        ally_state = np.zeros((self.n_agents_max, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit != None and al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                #max_cd = self.unit_max_cooldown(al_unit)
                max_cd = self.cooldown_map.get(al_unit.unit_type, 15)
                
                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                ally_state[al_id, 1] = (
                    al_unit.weapon_cooldown / max_cd
                )  # cooldown
                ally_state[al_id, 2] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                if self.shield_bits_ally > 0:
                    #max_shield = self.unit_max_shield(al_unit)
                    max_shield = common_utils.unit_max_shield(al_unit.unit_type, self.rlunit_ids)
                    ally_state[al_id, 4] = (
                        al_unit.shield / max_shield
                    )  # shield

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, type_id - self.unit_type_bits] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                if self.shield_bits_enemy > 0:
                    #max_shield = self.unit_max_shield(e_unit)
                    max_shield = common_utils.unit_max_shield(e_unit.unit_type, self.rlunit_ids)
                    enemy_state[e_id, 3] = e_unit.shield / max_shield  # shield

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, type_id - self.unit_type_bits] = 1

        state = {"allies": ally_state, "enemies": enemy_state}

        if self.state_last_action:
            state["last_action"] = self.last_action
        if self.state_timestep_number:
            state["timestep"] = self._episode_steps / self.episode_limit

        return state
    
    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return self.n_agents_max - 1, nf_al
    
    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents_max * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents_max * self.n_actions

        if self.state_timestep_number:
            size += 1

        return size
    
    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 86:
                type_id = 0 # hatchery
            elif unit.unit_type == 111:
                type_id = 1 # infestor
            elif unit.unit_type == 127:
                type_id = 1 # infestorBurrowed
            elif unit.unit_type == 150:
                type_id = 2 # infestedTerranEgg
            elif unit.unit_type == 7:
                type_id = 3 # infestedTerran
        else:  # use default SC2 unit types
            if unit.unit_type == 18:
                type_id = 0 # commandCenter
            elif unit.unit_type == 48:
                type_id = 1 # marine
            elif unit.unit_type == 33:
                type_id = 2 # siegeTank
        return type_id
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit != None and unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if the unit is Hatchery or Egg
            if (unit.unit_type == self.rlunit_ids.get("hatchery") or
                unit.unit_type == self.rlunit_ids.get("infestedEgg")):
                return avail_actions

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # if the unit is infestor or burrowed infestor, then it can spawn egg or change the status
            if unit.unit_type == self.rlunit_ids.get("infestor"):
                avail_actions[6] = 1    # burrow
                if unit.energy >= 50 and self.n_agents <= self.n_agents_max - 1:
                    avail_actions[8] = 1
                return avail_actions
            
            if unit.unit_type == self.rlunit_ids.get("infestorBurrowed"):
                avail_actions[7] = 1    #unburrow
                if unit.energy >= 50 and self.n_agents <= self.n_agents_max - 1:
                    avail_actions[8] = 1
                return avail_actions

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
        
    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        
        for agent_id in range(self.n_agents, self.n_agents_max):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions
    
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
                if (unit.owner == 1 and unit.unit_type != 151)   # not larva
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

            self.n_agents = get_map_params(self.map_name)["n_agents"]
            for i in range(self.n_agents, self.n_agents_max):
                self.agents[i] = None

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

            all_agents_created = len(self.agents) == self.n_agents_max
            all_enemies_created = len(self.enemies) == self.n_enemies

            self._unit_types = [
                unit.unit_type for unit in ally_units_sorted
            ] + [
                unit.unit_type
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 2
            ]

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()
    
    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        env_info["n_agents"] = self.n_agents_max
        return env_info
    
    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def update_units(self):
        self.update_infestedEgg()
        self.update_infestedTerran()
        return super().update_units()

    def check_structure(self, ally = True):
        """Check if the enemy's CommandCenter or the agent's Hatchery is destroyed."""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type == 18 and e.health <= 0:
                    return True
        if ally == True:
            for a in self.agents.values():
                if a != None and a.unit_type == self.rlunit_ids.get("hatchery") and a.health <= 0:
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
                if a != None and a.unit_type != self.rlunit_ids.get("hatchery") and a.health > 0:
                    return False
            return True
        return False
    
    def update_infestedEgg(self):
        """find the new infestedEgg on the obs and put them into self.agents"""
        # self.n_agents_egg_temp = 0
        for unit in self._obs.observation.raw_data.units:
            if unit.owner == 1 and unit.unit_type == self.rlunit_ids.get("infestedEgg"):
                find_same = False
                for al_unit in self.agents.values():
                    if al_unit != None and unit.tag == al_unit.tag:
                        find_same = True
                        break
                if find_same == False:
                    self.agents[self.n_agents] = unit
                    self.n_agents += 1
                    self.n_agents_egg_temp -= 1
        return
    
    def update_infestedTerran(self):
        """find the new infestedTerran on the obs"""
        for unit in self._obs.observation.raw_data.units:
            if unit.owner == 1 and unit.unit_type == self.rlunit_ids.get("infestedTerran"):
                find_same = False
                for al_unit in self.agents.values():
                    if al_unit != None and unit.tag == al_unit.tag:
                        find_same = True
                        break
                if find_same == False:
                    corresponding_egg = self.find_corresponding_egg(unit.pos.x, unit.pos.y)
                    self.agents[corresponding_egg] = unit
        return
    
    def find_corresponding_egg(self, unit_x, unit_y):
        """find the egg that is closest to the new infestedTerran's pos"""
        min_dist = 99999
        corresponding_egg = 99999
        for t_id, t_unit in self.agents.items():
            if t_unit != None and t_unit.unit_type == self.rlunit_ids.get("infestedEgg"):
                dist = self.distance(unit_x, unit_y, t_unit.pos.x, t_unit.pos.y)
                if dist < min_dist:
                    min_dist = dist
                    corresponding_egg = t_id
        return corresponding_egg
