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
    "BurrowDown": 1390,
    "BurrowUp": 1388,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsJCTQEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        map_params = get_map_params(self.map_name)
        self.n_agents_alive = map_params["n_agents_alive"]
        self.n_actions += 1
        self.n_actions_no_attack += 1
        print("----------------------")
        print("You create a JCTQ env!")
        print("----------------------")
    
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

        elif action == 6:
            # burrow or unburrow
            if unit.unit_type == self.rlunit_ids.get("roach"):
                # burrowDown
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["BurrowDown"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                if self.debug:
                    logging.debug("Agent {}: burrowDown".format(a_id))
            elif unit.unit_type == self.rlunit_ids.get("roachBurrowed"):
                # burrowUp
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["BurrowUp"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                if self.debug:
                    logging.debug("Agent {}: burrowUp".format(a_id))
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
    
    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 110:
                type_id = 0
            elif unit.unit_type == 118:
                type_id = 0
        else:  # use default SC2 unit types
            if unit.unit_type == 74:
                type_id = 0 # stalker
            elif unit.unit_type == 77:
                type_id = 1 # sentry
            elif unit.unit_type == 82:
                type_id = 2 # observer
        return type_id
    
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

            # unit can always choose burrow or unburrow
            avail_actions[6] = 1
            if unit.unit_type == self.rlunit_ids.get("roachBurrowed"):
                return avail_actions

            # Can attack only alive units that are alive in the shooting range
            # shoot_range = self.unit_shoot_range(agent_id)

            # target_items = self.enemies.items()

            # for t_id, t_unit in target_items:
            #     if t_unit.health > 0:
            #         dist = self.distance(
            #             unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
            #         )
            #         # cannot attack observer (flying unit)
            #         if dist <= shoot_range and t_unit.unit_type != 82:
            #             avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)
    
    def _kill_all_units(self):
        """Kill all units and force field on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit != None and unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit != None and unit.health > 0] + [
            unit.tag for unit in self._obs.observation.raw_data.units if unit != None and unit.owner == 16
        ]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit != None and al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated and al_unit != None:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit != None and e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated and e_unit != None:  # dead
                e_unit.health = 0

        if (
            n_ally_alive < self.n_agents_alive # The target is to make sure every unit is alive
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

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def check_unit_killed(self, ally = True):
        """The target is to make sure every unit is alive"""
        if ally == False:
            if self.check_time():
                return True
            for e in self.enemies.values():
                if e.health > 0:
                    return False
            return True
        if ally == True:
            n_alive = 0
            for a in self.agents.values():
                if a.health > 0:
                    n_alive += 1
            if n_alive < self.n_agents_alive:
                return True
        return False

    def check_time(self):
        if self._episode_steps == self.episode_limit - 1:
            return True
        return False

    def check_end_code(self, ally=True):
        return self.check_unit_killed(ally)
    
    def reward_battle(self):
        """
        [金蝉脱壳 - 最终修正版]
        基于你的三点核心建议进行修正：
        1. [防刷分] 逃跑奖励改为完全对称（远离+，接近-）。
        2. [去噪声] 移除无意义的钻地奖励，仅在穿墙时奖励钻地。
        3. [路标] 新增“贴近力场”和“成功逃脱”的各种一次性大奖。
        """
        # --- 1. 初始化一次性奖励标记 (Lazy Initialization) ---
        # 记录哪些蟑螂已经拿过“贴近力场奖励”
        if not hasattr(self, 'agents_reached_ff'): self.agents_reached_ff = set()
        # 记录哪些蟑螂已经“成功逃脱”
        if not hasattr(self, 'agents_escaped'): self.agents_escaped = set()
        
        # 记录上一帧的距离用于计算差分
        if not hasattr(self, 'last_min_stalker_dists'): self.last_min_stalker_dists = {}

        # 每次 reset 时清理标记 (通过 episode_steps 判断是否是新的一局)
        if self._episode_steps == 0:
            self.agents_reached_ff = set()
            self.agents_escaped = set()
            self.last_min_stalker_dists = {}

        # 继承父类基础逻辑
        _ = super().reward_battle()
        total_reward = 0
        
        # --- 2. 获取关键单位 ---
        # 追猎者 (威胁)
        stalkers = [e for e in self.enemies.values() if e is not None and e.health > 0 and e.unit_type == 74]
        
        # 力场 (障碍) - Force Field ID: 135
        force_fields = []
        for u in self._obs.observation.raw_data.units:
            if u.unit_type == 135: 
                force_fields.append(u)

        # =========================================================
        # [A] 生存奖励 (Survival)
        # =========================================================
        live_agents = 0
        for agent in self.agents.values():
            if agent is not None and agent.health > 0:
                live_agents += 1
                total_reward += 0.05 

        if live_agents == 0:
            total_reward -= 10.0
        
        # =========================================================
        # [B] 智能体循环：核心逻辑
        # =========================================================
        
        step_escape_reward = 0
        
        for a_id, agent in self.agents.items():
            if agent is None or agent.health <= 0:
                if a_id in self.last_min_stalker_dists: del self.last_min_stalker_dists[a_id]
                continue
            
            # --- 1. 计算与最近追猎者的距离 ---
            min_stalker_dist = 9999.0
            for s in stalkers:
                d = self.distance(agent.pos.x, agent.pos.y, s.pos.x, s.pos.y)
                if d < min_stalker_dist: min_stalker_dist = d
            
            # 计算差分
            prev_dist = self.last_min_stalker_dists.get(a_id, min_stalker_dist)
            delta_dist = min_stalker_dist - prev_dist
            self.last_min_stalker_dists[a_id] = min_stalker_dist
            
            # --- [修正点 1] 对称的逃跑奖励 (防止刷分) ---
            # 远离给正分，接近给负分，总和趋近于零（减去时间税），杜绝反复横跳
            # 系数稍微调低一点，避免掩盖了一次性大奖
            step_escape_reward += delta_dist * 2.0 

            # --- 2. 计算与最近力场的距离 ---
            min_ff_dist = 9999.0
            for ff in force_fields:
                d_ff = self.distance(agent.pos.x, agent.pos.y, ff.pos.x, ff.pos.y)
                if d_ff < min_ff_dist: min_ff_dist = d_ff
            
            # --- [修正点 3] 一次性大奖机制 ---
            
            # <奖励一>：接触力场大奖 (Touch the Wall)
            # 鼓励它们先冲到墙边上再说
            FF_CHECKPOINT_DIST = 2.0
            if a_id not in self.agents_reached_ff and min_ff_dist < FF_CHECKPOINT_DIST:
                total_reward += 50.0  # 一次性大奖
                self.agents_reached_ff.add(a_id)
                # print(f"Agent {a_id} Reached Force Field!")

            # <奖励二>：成功逃脱大奖 (The Great Escape)
            # 逻辑：如果距离追猎者非常远（超过了射程和视野），说明已经穿过了力场并逃之夭夭
            # 追猎者射程约 6，视野约 10-12。如果距离 > 12，基本安全。
            SAFE_DISTANCE = 12.0
            if a_id not in self.agents_escaped and min_stalker_dist > SAFE_DISTANCE:
                total_reward += 100.0 # 巨额大奖
                self.agents_escaped.add(a_id)
                # print(f"Agent {a_id} Escaped Successfully!")

            # --- [修正点 2] 严格的钻地逻辑 ---
            # 只有在非常靠近力场时，钻地才有意义 (穿墙)
            # 其他任何时候钻地都是活靶子 (因为有 Observer)
            
            ROACH_BURROWED_ID = 118
            is_burrowed = (agent.unit_type == ROACH_BURROWED_ID)
            
            # 力场交互区 (略大于 Checkpoint，给一点反应时间)
            FF_INTERACTION_DIST = 2.5
            
            if min_ff_dist < FF_INTERACTION_DIST:
                # 只有在这里，钻地才是被允许和鼓励的
                if is_burrowed:
                    total_reward += 0.5 # 持续给分，鼓励保持钻地状态穿墙
                    
                    # 如果钻地且正在远离敌人（正在穿墙中）
                    if delta_dist > 0:
                        total_reward += 1.0 
                else:
                    # 贴墙了还不钻地，必须重罚
                    total_reward -= 0.5

        if live_agents > 0:
            total_reward += step_escape_reward / live_agents

        return total_reward