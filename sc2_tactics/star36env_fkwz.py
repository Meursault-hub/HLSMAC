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
import random
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
    "warpgateTrain": 1413,
    "WarpPrismLoad": 911,
    "WarpPrismUnload": 913,
    "PhasingMode": 1528,
    "TransportMode": 1530,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

COOL_VAL = 536
R_PYLON = 6.5
R_PRISM = 3.75
R_LOAD = 4

class SC2TacticsFKWZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        map_params = get_map_params(self.map_name)
        self.n_agents_max = map_params["n_agents_max"]
        self.n_enemies_max = map_params["n_enemies_max"]
        self.resource = map_params["resource_start"]
        self.resource_ssh = map_params["resource_ssh"]
        self.unit_killed_ssh = map_params["unit_killed_ssh"]
        self.n_actions = self.n_actions_no_attack + self.n_enemies_max
        self.last_action = np.zeros((self.n_agents_max, self.n_actions))
        self.death_tracker_ally = np.zeros(self.n_agents_max)
        self.death_tracker_enemy = np.zeros(self.n_enemies_max)
        self.cooldown_dict = {}
        self.load = {}
        self.n_temp = 0
        print("----------------------")
        print("You create a FKWZ env!")
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
        # self.death_tracker_ally = np.zeros(self.n_agents)
        # self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents_max, self.n_actions))
        self.death_tracker_ally = np.zeros(self.n_agents_max)
        self.death_tracker_enemy = np.zeros(self.n_enemies_max)

        # === [新增] 初始化上一帧距离记录 ===
        # 用于计算差分奖励 (Delta Reward)
        self.prism_last_dist = {} 
        # ================================

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
        self.init_cooldown()
        return self.get_obs(), self.get_state()

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} with type {} cannot perform action {} on target {} with type {}".format(a_id, self.get_unit_by_id(a_id).unit_type, action, action - self.n_actions_no_attack, self.get_unit_by_id(action - self.n_actions_no_attack).unit_type)

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

        elif unit.unit_type == self.rlunit_ids.get("warpgate"):
            target_id = action - self.n_actions_no_attack
            if self.resource >= 100 and self.n_agents + self.n_temp < self.n_agents_max:
                t_unit = self.get_unit_by_id(target_id)
                if (t_unit != None and
                    (t_unit.unit_type == self.rlunit_ids.get("pylon") or
                    t_unit.unit_type == self.rlunit_ids.get("warpPrismPhasing"))):
                    tx, ty = self.get_valid_point(t_unit)
                    if tx == None and ty == None:
                        return None
                    else:
                        self.n_temp += 1
                        cmd = r_pb.ActionRawUnitCommand(
                            ability_id=actions["warpgateTrain"],
                            target_world_space_pos=sc_common.Point2D(
                                x=tx, y=ty
                            ),
                            unit_tags=[tag],
                            queue_command=False,
                        )
                        # self.resource -= 100
                        self.cooldown_dict[a_id] = COOL_VAL
                else:
                    return None
                    # print(f"agent {a_id} cannot use action {action} for target type {t_unit.unit_type}")
                    # exit(1)
            else:
                return None
        
        elif (unit.unit_type == self.rlunit_ids.get("warpPrism") or
            unit.unit_type == self.rlunit_ids.get("warpPrismPhasing")):
            if action == 6:
            # Load all units in the sight range
                target_tag = 0
                for t_id, t_unit in self.agents.items():
                    if (t_unit.unit_type == self.rlunit_ids.get("zealot") and 
                        self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                        <= self.unit_load_range(t_id) and
                        t_id not in self.load and t_unit.build_progress == 1.0):
                        target_tag = t_unit.tag
                        self.load[t_id] = t_unit
                        break
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["WarpPrismLoad"],
                    target_unit_tag=target_tag,
                    unit_tags=[tag],
                    queue_command=False,
                )
            elif action == 7:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["WarpPrismUnload"],
                    target_world_space_pos=sc_common.Point2D(
                        x=x, y=y
                    ),
                    unit_tags=[tag],
                    queue_command=False,
                )
            elif action == 8:
                if unit.unit_type == self.rlunit_ids.get("warpPrism"):
                    cmd = r_pb.ActionRawUnitCommand(
                        ability_id=actions["PhasingMode"],
                        unit_tags=[tag],
                        queue_command=False,
                    )
                elif unit.unit_type == self.rlunit_ids.get("warpPrismPhasing"):
                    cmd = r_pb.ActionRawUnitCommand(
                        ability_id=actions["TransportMode"],
                        unit_tags=[tag],
                        queue_command=False,
                    )
            else:
                print(f"Unknown action {action}!")
                exit(1)
            
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
        enemy_state = np.zeros((self.n_enemies_max, nf_en))

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
            if e_unit != None and e_unit.health > 0:
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
    
    def get_obs_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies_max, nf_en

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

        enemy_state = self.n_enemies_max * nf_en
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
            if unit.unit_type == 133:
                type_id = 0 # warpgate
            elif unit.unit_type == 60:
                type_id = 1 # pylon
            elif unit.unit_type == 73:
                type_id = 2 # zealot
            elif unit.unit_type == 81:
                type_id = 3 # warpPrism
            elif unit.unit_type == 136:
                type_id = 4 # warpPrismPhasing
            else:
                type_id = 99
        else:  # use default SC2 unit types
            if unit.unit_type == 62:
                type_id = 0 # gateway
            elif unit.unit_type == 60:
                type_id = 1 # pylon
            elif unit.unit_type == 74:
                type_id = 2 # stalker
            elif unit.unit_type == 66:
                type_id = 3 # photonCannon
            else:
                type_id = 99
        return type_id
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit != None and unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            if unit.unit_type == self.rlunit_ids.get("pylon") or unit.build_progress != 1.0:
                return avail_actions
            
            # === [修改重点] 折跃门逻辑 ===
            if unit.unit_type == self.rlunit_ids.get("warpgate"):
                
                # --- 新增：硬性人口锁 (Hard Limit Check) ---
                # 统计当前实际存在的智能体数量 (排除 None)
                current_active_agents = 0
                for u in self.agents.values():
                    if u is not None:
                        current_active_agents += 1
                
                # 如果当前单位数 >= 9 (训练时的上限)，强制禁止造兵
                # 直接返回当前 avail_actions (此时只有 stop 是 1，造兵动作全是 0)
                if current_active_agents >= 9:
                    return avail_actions
            # ------------------------------------------
            
            if unit.unit_type == self.rlunit_ids.get("warpgate"):
                if self.resource >= 100 and self.cooldown_dict[agent_id] == 0:
                    for a_id, a_unit in self.agents.items():
                        if (a_unit != None and 
                            (a_unit.unit_type == self.rlunit_ids.get("pylon") or 
                            a_unit.unit_type == self.rlunit_ids.get("warpPrismPhasing"))):
                            if a_unit.health > 0:
                                avail_actions[self.n_actions_no_attack + a_id] = 1
                return avail_actions

            # see if we can move
            if unit.unit_type != self.rlunit_ids.get("warpPrismPhasing"):
                if self.can_move(unit, Direction.NORTH):
                    avail_actions[2] = 1
                if self.can_move(unit, Direction.SOUTH):
                    avail_actions[3] = 1
                if self.can_move(unit, Direction.EAST):
                    avail_actions[4] = 1
                if self.can_move(unit, Direction.WEST):
                    avail_actions[5] = 1

            if (unit.unit_type == self.rlunit_ids.get("warpPrism") or
                unit.unit_type == self.rlunit_ids.get("warpPrismPhasing")):
                # check if WarpPrism can load units
                for t_id, t_unit in self.agents.items():
                    if (t_unit != None and len(self.load) < self.get_load_max() and
                        t_unit.unit_type == self.rlunit_ids.get("zealot") and 
                        t_id not in self.load and t_unit.build_progress == 1.0):
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                        )
                        load_range = self.unit_load_range(agent_id)
                        if dist <= load_range:
                            avail_actions[6] = 1
                            break
                if self.load != {} and self.pathing_grid[int(unit.pos.x), int(unit.pos.y)]:
                    avail_actions[7] = 1
                if unit.unit_type == self.rlunit_ids.get("warpPrism") or not self.is_warping():
                    avail_actions[8] = 1
                return avail_actions

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()

            for t_id, t_unit in target_items:
                if t_unit != None and t_unit.health > 0:
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

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1#(unit.owner == 1 and unit.type != 151)   # not larva
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
            self.n_enemies = get_map_params(self.map_name)["n_enemies"]
            for i in range(self.n_agents, self.n_agents_max):
                self.agents[i] = None

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2 or unit.owner == 16:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            for i in range(self.n_enemies, self.n_enemies_max):
                self.enemies[i] = None

            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values() if unit is not None
                )
                self._init_assign_aliases(min_unit_type)
                self.cooldown_map = common_utils.build_cooldown_map(self.rlunit_ids)

            all_agents_created = len(self.agents) == self.n_agents_max
            all_enemies_created = len(self.enemies) == self.n_enemies_max

            self._unit_types = [
                unit.unit_type for unit in ally_units_sorted
            ] + [
                unit.unit_type
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 2
            ]

            self.resource = self._obs.observation.player_common.minerals

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

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit != None and unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit != None and unit.health > 0] + [
            unit.tag for unit in self.load.values() if unit.health > 0
        ] + [unit.tag for unit in self._obs.observation.raw_data.units if unit.owner == 16]
        self.load = {}
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    # def can_move(self, unit, direction):
    #     """Whether a unit can move in a given direction."""
    #     m = self._move_amount / 2

    #     if direction == Direction.NORTH:
    #         x, y = int(unit.pos.x), int(unit.pos.y + m)
    #     elif direction == Direction.SOUTH:
    #         x, y = int(unit.pos.x), int(unit.pos.y - m)
    #     elif direction == Direction.EAST:
    #         x, y = int(unit.pos.x + m), int(unit.pos.y)
    #     else:
    #         x, y = int(unit.pos.x - m), int(unit.pos.y)

    #     if unit.unit_type == self.rlunit_ids.get("warpPrismPhasing"):
    #         return False
    #     elif unit.unit_type == self.rlunit_ids.get("warpPrism"):
    #         if self.check_bounds(x, y):
    #             return True
    #         else:
    #             return False
    #     elif self.check_bounds(x, y) and self.pathing_grid[x, y]:
    #         return True

    #     return False

    def unit_load_range(self, a_id):
        """Returns the load range for the WarpPrism."""
        return R_LOAD
    
    def get_load_max(self):
        """Returns the max load of the WarpPrism, which is 4"""
        return 4

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def update_units(self):
        self.resource = self._obs.observation.player_common.minerals
        self.n_temp = 0
        for unit in self._obs.observation.raw_data.units:
            if unit.owner == 1:
                find_same = False
                for al_unit in self.agents.values():
                    if al_unit != None and unit.tag == al_unit.tag:
                        find_same = True
                        break
                if find_same == False:
                    self.agents[self.n_agents] = unit
                    self.n_agents += 1
            else:
                find_same = False
                for e_unit in self.enemies.values():
                    if e_unit != None and unit.tag == e_unit.tag:
                        find_same = True
                        break
                if find_same == False:
                    self.enemies[self.n_enemies] = unit
                    self.n_enemies += 1
        for a_id in self.cooldown_dict:
            if self.cooldown_dict[a_id] > 0:
                self.cooldown_dict[a_id] -= self._step_mul
            if self.cooldown_dict[a_id] < 0:
                self.cooldown_dict[a_id] = 0

        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        # Check if roach is unloaded in dhls
        self.clean_load()

        for al_id, al_unit in self.agents.items():
            if al_unit != None:
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
            if e_unit != None:
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

    def check_structure(self, ally = True):
        """Check if the enemy's GateWay or any agent's structure is destroyed."""
        if ally == False:
            for e in self.enemies.values():
                if e != None and e.unit_type == 62:
                    if e.health > 0:
                        return False
                    return True
        if ally == True:
            for a in self.agents.values():
                if a != None:
                    if (a.unit_type == self.rlunit_ids.get("warpgate") or
                        a.unit_type == self.rlunit_ids.get("warpPrism") or
                        a.unit_type == self.rlunit_ids.get("warpPrismPhasing")):
                        if a.health <= 0:
                            return True
        return False
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == True and self.resource == 0:
            for a in self.agents.values():
                if (a != None and a.unit_type != self.rlunit_ids.get("warpgate") and
                    a.unit_type != self.rlunit_ids.get("pylon") and a.health > 0):
                    return False
            return True

        return False
    
    def get_valid_point(self, unit):
        px, py = unit.pos.x, unit.pos.y
        for i in range(20):
            if unit.unit_type == self.rlunit_ids.get("pylon"):
                dx = random.uniform(-R_PYLON, R_PYLON)
                dy = random.uniform(-R_PYLON, R_PYLON)
            elif unit.unit_type == self.rlunit_ids.get("warpPrismPhasing"):
                dx = random.uniform(-R_PRISM, R_PRISM)
                dy = random.uniform(-R_PRISM, R_PRISM)
            tx = px + dx
            ty = py + dy
            valid_pt = True
            if not 0 <= tx < self.pathing_grid.shape[0] or not 0 <= ty < self.pathing_grid.shape[1]:
                continue
            if not self.pathing_grid[int(tx), int(ty)]:
                continue
            for a in self.agents.values():
                if a != None:
                    a_x = a.pos.x
                    a_y = a.pos.y
                    if a.unit_type == self.rlunit_ids.get("warpgate"):
                        if self.distance(a_x, a_y, tx, ty) < 2:
                            valid_pt = False
                            break
                    elif (a.unit_type == self.rlunit_ids.get("warpPrism") or
                        a.unit_type == self.rlunit_ids.get("warpPrismPhasing")):
                        continue
                    else:
                        if self.distance(a_x, a_y, tx, ty) < 1:
                            valid_pt = False
                            break
            if not valid_pt:
                continue
            for e in self.enemies.values():
                if e != None:
                    e_x = e.pos.x
                    e_y = e.pos.y
                    if self.distance(e_x, e_y, tx, ty) < 1:
                        valid_pt = False
                        break
            if valid_pt:
                return tx, ty
        print(f"cannot find a valid point for unit type {unit.unit_type}")
        return None, None


    def init_cooldown(self):
        for i in self.agents:
            self.cooldown_dict[i] = 0
        self.load = {}

    def clean_load(self):
        """If the self.last_action for WarpPrism is unloading, remove the respective unit from self.load"""
        for a_id, a_unit in self.agents.items():
            if (a_unit.unit_type == self.rlunit_ids.get("warpPrism") or
                a_unit.unit_type == self.rlunit_ids.get("warpPrismPhasing")):
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
    
    def is_warping(self):
        for a in self.agents.values():
            if a != None and a.build_progress != 1.0:
                return True
        return False

    def reward_battle(self):
        """
        [反客为主 - 决战版 v2]
        修正点：
        1. [新增] 潜入成功路标奖励 (+200)：这是打破 Entropy 2.19 的关键。
        2. [修改] 伤害权重：打建筑分极高，打追猎者几乎没分（强制忽视干扰）。
        3. [简化] 移除了复杂的向量计算，减少噪声，让梯度更纯净。
        """
        # --- 1. 初始化 ---
        if not hasattr(self, 'last_dists'): self.last_dists = {}
        if not hasattr(self, 'last_zealot_count'): self.last_zealot_count = 0 
        if not hasattr(self, 'last_near_zealot_count'): self.last_near_zealot_count = 0
        
        # [新增] 潜入成功的一次性标记，防止反复刷分
        if not hasattr(self, 'has_reached_sneak_zone'): self.has_reached_sneak_zone = False
        
        _ = super().reward_battle()
        total_reward = 0
        
        # --- 2. 目标识别 ---
        ENEMY_GATEWAY = 62
        ENEMY_STALKER = 74
        PHOTON_CANNON = 66
        
        target_gateway = None
        stalkers = []
        cannons = []
        
        for e in self.enemies.values():
            if e is not None and e.health > 0:
                if e.unit_type == ENEMY_GATEWAY: target_gateway = e
                elif e.unit_type == ENEMY_STALKER: stalkers.append(e)
                elif e.unit_type == PHOTON_CANNON: cannons.append(e)
        
        # [核心] 胜利大奖
        if target_gateway is None: 
            time_bonus = max(0, (120 - self._episode_steps) * 2.0)
            return 1000.0 + time_bonus
        
        # 基础时间税 (稍微降低一点，给它一点绕后的耐心)
        total_reward -= 0.1

        # =========================================================
        # [A] 差异化伤害 (Target Shaping) - [重点修改]
        # =========================================================
        for e_id, e_unit in self.enemies.items():
            prev_e = self.previous_enemy_units.get(e_id, None)
            if prev_e is None: continue 
            
            if e_unit is None or e_unit.health == 0:
                curr_h, curr_s = 0, 0
            else:
                curr_h, curr_s = e_unit.health, e_unit.shield
            
            d_health = max(0, prev_e.health - curr_h)
            d_shield = max(0, prev_e.shield - curr_s)
            
            # 逻辑：拆迁办模式。只要打建筑，给巨额奖励；打人，给低保。
            if prev_e.unit_type == ENEMY_GATEWAY:
                if d_shield > 0: total_reward += d_shield * 5.0
                if d_health > 0: total_reward += d_health * 20.0 # [暴击] 鼓励疯狂输出基地
            elif prev_e.unit_type == PHOTON_CANNON:
                total_reward += (d_health + d_shield) * 1.0  # [中等] 拆炮台也是好事
            elif prev_e.unit_type == ENEMY_STALKER:
                total_reward += (d_health + d_shield) * 0.01 # [极低] 别理追猎者，别被风筝

        # =========================================================
        # [B] 经济控制
        # =========================================================
        prism_pos = None
        warp_prism_unit = None # 保留棱镜对象用于坐标判断
        for agent in self.agents.values():
            if agent is not None and agent.health > 0:
                u_type = agent.unit_type
                if u_type == self.rlunit_ids.get("warpPrism") or u_type == self.rlunit_ids.get("warpPrismPhasing"):
                    prism_pos = agent.pos
                    warp_prism_unit = agent
                    break
        
        current_zealot_count = 0       
        current_near_zealot_count = 0  
        for agent in self.agents.values():
            if agent is not None and agent.health > 0 and agent.unit_type == self.rlunit_ids.get("zealot"):
                current_zealot_count += 1
                if prism_pos is not None:
                    d_to_prism = self.distance(agent.pos.x, agent.pos.y, prism_pos.x, prism_pos.y)
                    if d_to_prism < 10.0: 
                        current_near_zealot_count += 1

        total_spawned = current_zealot_count - self.last_zealot_count
        prism_spawned = current_near_zealot_count - self.last_near_zealot_count
        base_spawned = max(0, total_spawned - max(0, prism_spawned))

        if base_spawned > 0: total_reward -= 50.0 
        if prism_spawned > 0: total_reward += 50.0 # [提升] 鼓励折跃

        self.last_zealot_count = current_zealot_count
        self.last_near_zealot_count = current_near_zealot_count

        # =========================================================
        # [C] 路标奖励 (Checkpoint Reward) - [新增 & 关键]
        # =========================================================
        # 只要棱镜活着到达了敌方基地附近（距离小于10），直接发奖。
        # 这连接了“出门”和“胜利”之间的巨大鸿沟。
        
        if warp_prism_unit is not None and not self.has_reached_sneak_zone:
            # 计算到敌方基地的距离
            dist_to_gate = self.distance(warp_prism_unit.pos.x, warp_prism_unit.pos.y, target_gateway.pos.x, target_gateway.pos.y)
            
            if dist_to_gate < 10.0:
                total_reward += 200.0 # [巨奖] “你到位置了！”
                self.has_reached_sneak_zone = True
                # 可选：打印调试信息，确认是否触发
                # print(">>> SNEAK BONUS TRIGGERED! <<<")

        # =========================================================
        # [D] 行为势场 (Shaping Reward)
        # =========================================================
        
        shaping_reward = 0
        for agent_id, agent in self.agents.items():
            if agent is None or agent.health <= 0:
                if agent_id in self.last_dists: del self.last_dists[agent_id]
                continue
            
            curr_dist = self.distance(agent.pos.x, agent.pos.y, target_gateway.pos.x, target_gateway.pos.y)
            
            # --- 角色 1: 折跃棱镜 ---
            if agent.unit_type == self.rlunit_ids.get("warpPrism") or agent.unit_type == self.rlunit_ids.get("warpPrismPhasing"):
                
                # 1. 威胁感知 (简化版)
                nearest_threat_dist = 999.0
                threats = stalkers + cannons
                for t in threats:
                    d = self.distance(agent.pos.x, agent.pos.y, t.pos.x, t.pos.y)
                    if d < nearest_threat_dist: nearest_threat_dist = d
                
                # 稍微离远点，保持风筝
                if nearest_threat_dist < 7.0: 
                     shaping_reward -= (7.0 - nearest_threat_dist) * 2.0

                # 2. 引导去潜入点 (仅当未到达时)
                if not self.has_reached_sneak_zone:
                     # 持续给正向奖励，拉它过去
                     prev_dist = self.last_dists.get(agent_id, curr_dist)
                     shaping_reward += (prev_dist - curr_dist) * 5.0
                     self.last_dists[agent_id] = curr_dist
                else:
                     # 到达后，奖励它展开相位模式
                     if agent.unit_type == self.rlunit_ids.get("warpPrismPhasing"):
                         shaping_reward += 0.1

            # --- 角色 2: 狂热者 ---
            elif agent.unit_type == self.rlunit_ids.get("zealot"):
                # 死士无脑冲
                if curr_dist < 3.0: 
                    shaping_reward += 1.0 # 砍到塔了
                else:
                    shaping_reward += 0.1 # 冲锋中

        return total_reward + shaping_reward