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
import time

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
    "Hallucination": 158,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsWZSYEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        map_params = get_map_params(self.map_name)
        self.n_actions += 1
        self.n_actions_no_attack += 1
        self.n_agents_max = map_params["n_agents_max"]
        self.last_action = np.zeros((self.n_agents_max, self.n_actions))
        self.death_tracker_ally = np.zeros(self.n_agents_max)
        self.n_agents_hal_temp = 0
        print("----------------------")
        print("You create a WZSY env!")
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

        elif action == 6:
            if self.n_agents <= self.n_agents_max - 2 - self.n_agents_hal_temp:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["Hallucination"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                self.n_agents_hal_temp += 2
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

        # 【修改】增加特征维度
        # Ally: +1 用于 Energy
        # Enemy: +1 用于 Is_Structure (高价值目标标识)
        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes()+3
        nf_en = self.get_enemy_num_attributes()+1

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
                
                # 【新增】能量特征 (归一化, Sentry Max Energy = 200)
                # 放在特征向量的最后一位，帮助智能体决策何时释放幻象
                ally_state[al_id, -3] = al_unit.energy / 200.0

                # 【新增 2】最近敌方【战斗单位】距离 (用于决策是否放幻象/逃跑)
                # [倒数第2位] 最近战斗单位距离
                min_combat_dist = 1.0
                # [倒数第1位] 【新增】最近建筑距离 (Target Distance)
                min_struct_dist = 1.0

                for e_unit in self.enemies.values():
                    if e_unit.health > 0:
                        d = self.distance(x, y, e_unit.pos.x, e_unit.pos.y)
                        nd = d / self.max_distance_x # 归一化
                        
                        if e_unit.unit_type in [59, 60]:
                            # 是建筑
                            if nd < min_struct_dist:
                                min_struct_dist = nd
                        else:
                            # 是战斗单位
                            if nd < min_combat_dist:
                                min_combat_dist = nd
                
                ally_state[al_id, -2] = min_combat_dist
                ally_state[al_id, -1] = min_struct_dist

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

                # 【新增】建筑标识 (Is_Structure)
                # 如果是水晶(60)或基地(59)，置为 1，明确告知这是高价值目标
                is_struct = 1.0 if e_unit.unit_type in [59, 60] else 0.0
                enemy_state[e_id, -1] = is_struct

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

        """【修改】返回 Ally 特征数量 (+1 Energy)"""
        return self.n_agents_max - 1, nf_al+3
    
    def get_obs_enemy_feats_size(self):
        """【新增】返回 Enemy 特征数量 (+1 Structure Flag)"""
        # 必须显式定义，否则父类会返回错误的维度导致报错
        nf_en = 3 + self.unit_type_bits
        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy
        
        return self.n_enemies, nf_en + 1 # +1 for Is_Structure
    
    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        # 修改重新计算 feature 数量
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits+3  # +3 for Energy, Min Combat Distance, Min Structure Distance
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits+1  # +1 for Is_Structure

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
            if unit.unit_type == 77:
                type_id = 0
            elif unit.unit_type == 74 and not unit.is_hallucination:
                type_id = 1
            elif unit.unit_type == 74 and unit.is_hallucination:
                type_id = 2
            else:
                type_id = 99
        else:  # use default SC2 unit types
            if unit.unit_type == 83:
                type_id = 0
            elif unit.unit_type == 73:
                type_id = 1
            elif unit.unit_type == 59:
                type_id = 2
            elif unit.unit_type == 60:
                type_id = 3
        return type_id
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit != None and unit.health > 0:
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

            if (unit.unit_type == self.rlunit_ids.get("sentry") and
                unit.energy >= 100 and
                self.n_agents <= self.n_agents_max - 2):
                avail_actions[6] = 1
            else:
                avail_actions[6] = 0

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
    
    def update_units(self):
        self.update_hallucination()
        return super().update_units()

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def check_structure(self, ally = True):
        """Check if the enemy's Nexus unit is killed."""
        if ally == False:
            for e in self.enemies.values():
                if (e.unit_type == 59 or e.unit_type == 60) and e.health <= 0:
                    return True
                    
        return False
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 59 and e.unit_type != 60 and e.health > 0:
                    return False
            return True
        if ally == True:
            for a in self.agents.values():
                if a != None and a.health > 0:
                    return False
            return True

        return False
    
    def update_hallucination(self):
        self.n_agents_hal_temp = 0
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
        return

    def step(self, actions):
        """Execute actions and return reward/obs."""


        # 1. 【新增】计算移动前的状态 (用于引力奖励)
        # 记录每个 agent 到最近建筑的距离
        dists_before = {}
        target_structures = [u for u in self.enemies.values() if u.unit_type in [59, 60] and u.health > 0]
        
        # 如果还有建筑活着，才计算距离
        if target_structures:
            for a_id, unit in self.agents.items():
                if unit is not None and unit.health > 0:
                    min_d = min([self.distance(unit.pos.x, unit.pos.y, t.pos.x, t.pos.y) for t in target_structures])
                    dists_before[a_id] = min_d
        # 1. 记录动作执行前的活体敌人，用于计算距离
        pre_enemies = {e_id: unit for e_id, unit in self.enemies.items() if unit.health > 0}

        # B. 【新增】记录抱团程度 (平均分散距离)
        spread_before = 0.0
        alive_agents_before = [u for u in self.agents.values() if u is not None and u.health > 0]
        if len(alive_agents_before) > 1:
            # 获取所有存活单位坐标矩阵
            coords = np.array([[u.pos.x, u.pos.y] for u in alive_agents_before])
            # 计算重心
            centroid = np.mean(coords, axis=0)
            # 计算每个单位到重心的距离
            dists = np.linalg.norm(coords - centroid, axis=1)
            # 计算平均分散度
            spread_before = np.mean(dists)
        
        # 2. 执行动作 (父类 step)
        out = super().step(actions)
        
        # 解包返回值 (兼容不同版本的 SMAC)
        if len(out) == 3:
            base_reward, terminated, info = out
        else:
            base_reward, terminated, info = out[1], out[2], out[3]

        extra_reward = 0

        # ==================== 奖励计算开始 ====================

        # A. 【新增】引力奖励 (Gravity Reward)
        # 再次获取活着的建筑 (防止本帧被打爆了)
        target_structures_now = [u for u in self.enemies.values() if u.unit_type in [59, 60] and u.health > 0]
        
        if target_structures_now and dists_before:
            for a_id, unit in self.agents.items():
                if unit is not None and unit.health > 0 and a_id in dists_before:
                    # 计算现在的新距离
                    min_d_now = min([self.distance(unit.pos.x, unit.pos.y, t.pos.x, t.pos.y) for t in target_structures_now])
                    
                    # 距离差值：正数代表靠近了，负数代表远离了
                    diff = dists_before[a_id] - min_d_now
                    
                    # 放大系数：移动一步大概 0.5~1.0 距离，给予适当的奖励
                    # 比如靠近 1 格，给 +0.1 分
                    extra_reward += diff * 0.01

        # B. 【新增】抱团奖励 (Cohesion Reward - 靠近队友)
        alive_agents_now = [u for u in self.agents.values() if u is not None and u.health > 0]
        # 只有当依然有2个以上单位存活时才计算，否则抱团无意义
        if len(alive_agents_now) > 1:
            coords = np.array([[u.pos.x, u.pos.y] for u in alive_agents_now])
            centroid = np.mean(coords, axis=0)
            dists = np.linalg.norm(coords - centroid, axis=1)
            spread_now = np.mean(dists)
            
            # 如果这一步有单位死亡，spread_before 和 spread_now 的比较可能会有噪音
            # 我们加一个简单的长度判断，只在存活数量没变时计算，或者忽略噪音
            if len(alive_agents_before) == len(alive_agents_now):
                # 距离变小(正) -> 奖励; 距离变大(负) -> 惩罚
                # 系数 0.2：比引力奖励大一点，因为保持阵型很重要
                diff_spread = spread_before - spread_now
                extra_reward += diff_spread * 0.1
        
        # 3. 拆迁奖励 (Structure Destruction Reward)
        # 由于我们在 get_avail_agent_actions 里屏蔽了攻击人
        # 所以 base_reward 里的伤害分几乎全是对建筑的伤害。
        # 我们给予 2.0 倍率，让智能体觉得"拆塔比杀人爽多了"
        if isinstance(base_reward, (float, int)):
            extra_reward += base_reward * 2.0
        elif isinstance(base_reward, (list, tuple)):
            # 如果是 tuple (reward, stats...)，取第一个元素
            extra_reward += float(base_reward[0]) * 2.0

        # 4. 【核心修改】动作意图奖励 (Action Intent Reward)
        for agent_id, action in enumerate(actions):
            if agent_id >= len(self.agents) or self.agents[agent_id] is None:
                continue
            unit = self.agents[agent_id]
            if unit.health <= 0:
                continue

            # A. 攻击动作判定
            if action >= self.n_actions_no_attack:
                target_id = action - self.n_actions_no_attack
                target_unit = self.enemies.get(target_id)
                
                if target_unit:
                    if target_unit.unit_type in [59, 60]:
                        # 攻击水晶/基地 -> 奖励 (Good!)
                        extra_reward += 10.0
                    else:
                        # 攻击战斗部队 -> 惩罚 (Bad! Don't engage units!)
                        extra_reward -= 0.1

            # B. 幻象战术判定 (Hallucination Logic)
            elif action == 6:
                # 寻找最近的敌方【战斗单位】
                min_dist = 999.0
                combat_enemy_present = False
                for e_unit in pre_enemies.values():
                    if e_unit.unit_type not in [59, 60]:
                        dist = self.distance(unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            combat_enemy_present = True
                
                if combat_enemy_present and min_dist < 8.0:
                    extra_reward += 5.0 # 完美吓退
                else:
                    pass

        # 5. 合并奖励 (修复 Bug：保留元组结构)
        if isinstance(base_reward, (float, int)):
            # 如果父类只返回了一个数值，直接相加
            final_reward = base_reward + extra_reward
        else:
            # 如果父类返回的是元组 (reward, delta_enemy, delta_deaths, delta_ally)
            # 我们只修改第一个元素(数值奖励)，保留后面的统计信息
            new_scalar_reward = float(base_reward[0]) + extra_reward
            
            # 重新打包成元组返回
            # base_reward[1:] 包含了 delta_enemy, delta_deaths 等统计数据
            final_reward = (new_scalar_reward,) + tuple(base_reward[1:])

        time.sleep(0.5)
        return final_reward, terminated, info