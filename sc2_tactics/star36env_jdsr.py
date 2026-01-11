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
    "NeuralParasite": 249,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsJDSREnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        map_params = get_map_params(self.map_name)
        self.n_agents_max = map_params["n_agents_max"]
        assert(self.n_agents_max == self.n_agents + self.n_enemies)
        self.last_action = np.zeros((self.n_agents_max, self.n_actions))
        self.death_tracker_ally = np.zeros(self.n_agents_max)
        print("----------------------")
        print("You create a JDSR env!")
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
        if unit == None or unit.owner == 2:
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

        else:
            # attack units that are in range
            target_id = action - self.n_actions_no_attack
            
            # [新增] 目标寻址逻辑 (Target Resolution)
            target_unit = None
            
            # 情况 A: 目标是普通敌人 (在 enemies 列表里)
            if target_id in self.enemies:
                target_unit = self.enemies[target_id]
            
            # 情况 B: 目标是巨像 (可能变成盟友了，不在 enemies 里)
            if target_unit is None and target_id == self.colossus_id:
                # 去全局列表里找这个巨像
                for u in self._obs.observation.raw_data.units:
                    if u.unit_type == 4: # Colossus
                        target_unit = u
                        break
            
            # 如果还是没找到，说明出错了或单位已死，直接返回
            if target_unit is None:
                return None

            if unit.unit_type == self.rlunit_ids.get("infestor"):
                action_name = "NeuralParasite"
            else:
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
            if al_unit != None and al_unit.health > 0 and al_unit.owner == 1:
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
            if e_unit.health > 0 and e_unit.owner == 2:
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
            if unit.unit_type == 110:
                type_id = 0
            elif unit.unit_type == 111:
                type_id = 1
            elif unit.unit_type == 74:
                type_id = 2
            elif unit.unit_type == 4:
                type_id = 3
            else:
                type_id = -1
        else:  # use default SC2 unit types
            if unit.unit_type == 74:
                type_id = 0
            elif unit.unit_type == 4:
                type_id = 1
            else:
                type_id = -1
        return type_id
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit is None or unit.health <= 0 or unit.owner != 1:
            return [1] + [0] * (self.n_actions - 1)

        avail_actions = [0] * self.n_actions
        avail_actions[1] = 1 # Stop always allowed
        
        # --- [全局态势感知] ---
        stalkers_exist = False
        for e in self.enemies.values():
            if e.health > 0 and e.unit_type == 74 and e.owner == 2:
                stalkers_exist = True
                break

        # =========================================================
        # 1. 移动逻辑 (全员解锁，为了冲锋建立阵型)
        # =========================================================
        if self.can_move(unit, Direction.NORTH): avail_actions[2] = 1
        if self.can_move(unit, Direction.SOUTH): avail_actions[3] = 1
        if self.can_move(unit, Direction.EAST):  avail_actions[4] = 1
        if self.can_move(unit, Direction.WEST):  avail_actions[5] = 1

        # =========================================================
        # 2. 技能与攻击逻辑 (核心战术区)
        # =========================================================
        
        # --- A. 感染虫 (Infestor) ---
        if unit.unit_type == self.rlunit_ids.get("infestor"):
            has_energy = (unit.energy >= 75)
            shoot_range = self.unit_control_range(agent_id)
            target_items = self.enemies.items()

            if has_energy:
                # 有蓝：只准放技能控巨像
                for t_id, t_unit in target_items:
                    if t_unit.health > 0 and t_unit.owner == 2:
                        dist = self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                        if dist <= shoot_range and t_unit.unit_type == 4: # 4=Colossus
                            avail_actions[t_id + self.n_actions_no_attack] = 1
                # 感染虫有蓝时禁止平A，防止走位失误
                return avail_actions
            else:
                # 没蓝：允许平A (虽然没伤害，但可以吸引火力)
                pass 

        # --- B. 蟑螂 (Roach) & 没蓝的感染虫 ---
        
        # 射程判定
        shoot_range = self.unit_shoot_range(agent_id)
        target_items = self.enemies.items()

        # 遍历所有敌人生成攻击动作
        for t_id, t_unit in target_items:
            if t_unit.health > 0 and t_unit.owner == 2:
                dist = self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                if dist <= shoot_range:
                    # [战术核心] 目标筛选
                    if stalkers_exist:
                        # 阶段 1: 还有追猎者
                        # 严禁攻击巨像 (ID 4)，只准打追猎者 (ID 74)
                        if t_unit.unit_type != 4: 
                            avail_actions[t_id + self.n_actions_no_attack] = 1
                    else:
                        # 阶段 2: 追猎者死光 -> 自由攻击 (打剩下的敌人)
                        avail_actions[t_id + self.n_actions_no_attack] = 1

        # --- [特殊逻辑] 阶段 2: 强制攻击盟友巨像 (卸磨杀驴) ---
        if not stalkers_exist and self.colossus_id != -1:
             # 检查巨像是否还活着 (无论敌我)
            for u in self._obs.observation.raw_data.units:
                if u.unit_type == 4 and u.health > 0:
                    dist = self.distance(unit.pos.x, unit.pos.y, u.pos.x, u.pos.y)
                    # 如果在射程内，强制开启针对该巨像的攻击动作
                    if dist <= self.unit_shoot_range(agent_id):
                        # 注意：这里我们复用初始记录的 colossus_id
                        # 前提是 target_id 在 get_agent_action 里会被特殊处理映射到这个 Tag
                        avail_actions[self.n_actions_no_attack + self.colossus_id] = 1
                    break

        return avail_actions
        
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

            # [新增] 锁定巨像的原始 Enemy ID
            self.colossus_id = -1
            for e_id, unit in self.enemies.items():
                if unit.unit_type == 4:  # 4 = Colossus
                    self.colossus_id = e_id  # ✅ 补全这句
                    break  # 找到后跳出，养成好习惯

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

    def unit_control_range(self, agent_id):
        """Returns the neural parasite range for an infestor."""
        return 8
    
    def update_units(self):
        for t_unit in self._obs.observation.raw_data.units:
            if t_unit.owner == 1:
                find_same = False
                for a_unit in self.agents.values():
                    if a_unit != None and a_unit.tag == t_unit.tag:
                        find_same = True
                        break
                if not find_same:
                    for e_id, e_unit in self.enemies.items():
                        if e_unit.tag == t_unit.tag:
                            self.agents[self.n_agents + e_id] = e_unit
                            break
        return super().update_units()

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)
    
    def check_unit_killed(self, ally=True):
        """
        [修正版] 胜负判定逻辑
        不仅要检查 enemies 列表，还要强制检查全图是否还有 ID=4 的巨像存活。
        """
        # 1. 检查我方是否全灭 (标准逻辑)
        if ally == True:
            for a in self.agents.values():
                if a is not None and a.health > 0 and a.owner == 1:
                    return False
            return True

        # 2. 检查敌方是否全灭 (修正逻辑)
        if ally == False:
            # A. 先检查普通敌人 (追猎者) 是否死光
            for e in self.enemies.values():
                if e.health > 0:
                    return False
            
            # B. [核心修正] 额外检查：全图是否还有活着的巨像？
            # 即使巨像被控制变成了 owner=1，只要它还活着，游戏就不能结束！
            colossus_alive = False
            for u in self._obs.observation.raw_data.units:
                if u.unit_type == 4 and u.health > 0: # 4 = Colossus
                    colossus_alive = True
                    break
            
            # 如果巨像还活着，视为“没赢”，继续打！
            if colossus_alive:
                return False

            return True
        
    def get_obs_agent(self, agent_id):
        """
        [借刀杀人 - 融合版观测]
        1. Base Obs: 继承父类的通用观测 (包含友方/敌方列表、地形等)
        2. Extra Obs: 追加定制特征 (VIP巨像状态、战场计数)
        """
        # 1. 获取基础通用观测 (包含丰富的所有单位详情)
        base_obs = super().get_obs_agent(agent_id)
        
        unit = self.get_unit_by_id(agent_id)
        if unit is None:
            # 如果单位挂了，补齐 0 向量
            return np.concatenate((base_obs, np.zeros(7, dtype=np.float32)))

        # 2. 计算额外特征 (共 7 维)
        extra_obs = np.zeros(7, dtype=np.float32)

        # [Extra 0]: 自身类型 (1.0=感染虫, 0.0=蟑螂)
        extra_obs[0] = 1.0 if unit.unit_type == self.rlunit_ids.get("infestor") else 0.0

        # 寻找巨像 (VIP)
        colossus = None
        for u in self._obs.observation.raw_data.units:
            if u.unit_type == 4: # Colossus ID
                colossus = u
                break
        
        if colossus:
            # [Extra 1-4]: 巨像的相对位置与状态
            extra_obs[1] = (colossus.pos.x - unit.pos.x) / self.map_x # dx
            extra_obs[2] = (colossus.pos.y - unit.pos.y) / self.map_y # dy
            extra_obs[3] = colossus.health / colossus.health_max      # health
            # [Extra 5]: 控制权感知 (1=我方控制, 0=敌方)
            extra_obs[4] = 1.0 if colossus.owner == 1 else 0.0
        
        # [Extra 6]: 战场局势 - 敌方存活追猎者计数
        enemy_stalker_count = 0
        for u in self._obs.observation.raw_data.units:
            if u.unit_type == 74 and u.owner == 2 and u.health > 0:
                enemy_stalker_count += 1
        extra_obs[5] = enemy_stalker_count / 5.0 # 归一化

        # [Extra 7]: 巨像距离 (方便判定技能范围)
        if colossus:
            dist = self.distance(unit.pos.x, unit.pos.y, colossus.pos.x, colossus.pos.y)
            extra_obs[6] = dist / self.map_x
        
        # 3. 拼接
        return np.concatenate((base_obs, extra_obs))

    def get_obs_size(self):
        """
        返回总维度 = 父类基础维度 + 新增维度(7)
        注意：必须动态调用 super().get_obs_size() 以防父类配置变动
        """
        return super().get_obs_size() + 7
    
    def reward_battle(self):
        """
        [借刀杀人 - 最终修正版 (反逃跑/高风险高回报)]
        
        核心逻辑变动：
        1. [修正] 感染虫死亡惩罚降低 (-50 -> -20)，鼓励冒险。
        2. [新增] 逃兵惩罚：感染虫如果离巨像太远 (>9.0)，每步重罚，逼迫回头。
        3. [增强] 夺刀大奖翻倍 (+200 -> +500)，让冒险物超所值。
        4. [新增] 护卫奖励：鼓励蟑螂站在感染虫前面挡枪。
        5. [保留] 变脸逻辑：阶段2反水攻击巨像给予高额奖励 (+2.0系数)。
        """
        
        # --- 1. 初始化 ---
        if not hasattr(self, 'has_controlled_colossus'): self.has_controlled_colossus = False
        if not hasattr(self, 'stalkers_cleared'): self.stalkers_cleared = False
        if not hasattr(self, 'colossus_killed'): self.colossus_killed = False
        if not hasattr(self, 'last_colossus_health'): self.last_colossus_health = -1
        
        if self._episode_steps == 0:
            self.has_controlled_colossus = False
            self.stalkers_cleared = False
            self.colossus_killed = False
            self.last_colossus_health = -1
            
        _ = super().reward_battle()
        total_reward = 0
        
        # --- 2. 战场态势 ---
        infestor = None
        roaches = []
        for u in self.agents.values():
            if u is not None and u.health > 0:
                if u.unit_type == self.rlunit_ids.get("infestor"):
                    infestor = u
                else:
                    roaches.append(u)
        
        colossus = None
        # 全局搜索巨像
        for u in self._obs.observation.raw_data.units:
            if u.unit_type == 4: 
                colossus = u
                break
        
        alive_stalkers = []
        for u in self._obs.observation.raw_data.units:
            if u.unit_type == 74 and u.owner == 2 and u.health > 0:
                alive_stalkers.append(u)
        
        n_stalkers_alive = len(alive_stalkers)

        # =========================================================
        # [A] 基础生存与逃兵惩罚 (修正项)
        # =========================================================
        if infestor is None:
            # 1. 降低死亡惩罚 (-50 -> -20)
            # 只要能换掉巨像，死得其所。降低对死亡的恐惧。
            total_reward -= 20.0 
        else:
            # 2. 逃兵惩罚 (Coward Penalty)
            # 如果巨像还是敌人，感染虫必须靠近！
            if colossus and (colossus.owner == 2):
                dist_to_colossus = self.distance(infestor.pos.x, infestor.pos.y, colossus.pos.x, colossus.pos.y)
                
                # 施法距离是 8.0。设定 9.0 为容忍线。
                if dist_to_colossus > 9.0:
                    # 距离越远，罚得越狠！(例如距离 14，每步扣 1.0 分，比死还痛)
                    total_reward -= (dist_to_colossus - 9.0) * 0.2
                else:
                    # 进入战斗距离，给一点勇气奖
                    total_reward += 0.1

        # =========================================================
        # [B] 巨像状态与里程碑 (增强项)
        # =========================================================
        is_colossus_ours = False
        current_col_health = 0
        
        if colossus:
            current_col_health = colossus.health + colossus.shield
            is_colossus_ours = (colossus.owner == 1)
            
            # 里程碑 1: 夺刀大奖 (增强至 +500)
            if is_colossus_ours and not self.has_controlled_colossus:
                total_reward += 500.0  # 重赏之下必有勇夫
                self.has_controlled_colossus = True
        else:
            # 里程碑 3: 斩首 (胜利)
            if self.stalkers_cleared and not self.colossus_killed:
                total_reward += 200.0
                self.colossus_killed = True
            self.last_colossus_health = -1

        # =========================================================
        # [C] 动态变脸逻辑
        # =========================================================
        
        # ---------------------------------------------------------
        # 阶段 1: 借刀杀人 (还有追猎者)
        # ---------------------------------------------------------
        if n_stalkers_alive > 0:
            
            # 1. 护主惩罚
            if colossus and self.last_colossus_health > 0:
                damage_taken = self.last_colossus_health - current_col_health
                if damage_taken > 0:
                    total_reward -= damage_taken * 0.5 

            # 2. 借刀伤害奖励
            if is_colossus_ours:
                stalker_damage_dealt = 0
                for e_id, e_unit in self.enemies.items():
                    if e_unit is not None and e_unit.unit_type == 74:
                        prev_e = self.previous_enemy_units.get(e_id, None)
                        if prev_e:
                            loss = (prev_e.health + prev_e.shield) - (e_unit.health + e_unit.shield)
                            if loss > 0: stalker_damage_dealt += loss
                if stalker_damage_dealt > 0:
                    total_reward += stalker_damage_dealt * 1.0

            # 3. 蟑螂战术站位 (甜点区 2.5 - 4.5)
            for r in roaches:
                min_dist = 999.0
                for s in alive_stalkers:
                    d = self.distance(r.pos.x, r.pos.y, s.pos.x, s.pos.y)
                    if d < min_dist: min_dist = d
                if min_dist <= 4.5: 
                    total_reward += 0.05
                    if min_dist < 2.5: total_reward -= 0.05 
            
            # 4. [新增] 蟑螂护卫奖励 (Bodyguard Reward)
            # 鼓励蟑螂处于感染虫和巨像之间，充当掩体
            if infestor and colossus and not is_colossus_ours:
                dist_inf_col = self.distance(infestor.pos.x, infestor.pos.y, colossus.pos.x, colossus.pos.y)
                for r in roaches:
                    dist_roach_col = self.distance(r.pos.x, r.pos.y, colossus.pos.x, colossus.pos.y)
                    # 如果蟑螂比感染虫更靠近巨像，说明它在前面扛着
                    if dist_roach_col < dist_inf_col:
                        total_reward += 0.05 

        # ---------------------------------------------------------
        # 阶段 2: 卸磨杀驴 (追猎者全灭)
        # ---------------------------------------------------------
        else:
            # 里程碑 2: 清场
            if not self.stalkers_cleared:
                self.stalkers_cleared = True
                total_reward += 200.0
            
            if colossus and colossus.health > 0:
                # 1. 围杀奖励
                for r in roaches:
                    d = self.distance(r.pos.x, r.pos.y, colossus.pos.x, colossus.pos.y)
                    if d < 5.0: total_reward += 0.1
                
                # 2. 反水伤害奖励 (背刺大奖)
                # 即使是友军，打掉血也给 2.0 倍奖励
                if self.last_colossus_health > 0:
                    colossus_damage_taken = self.last_colossus_health - current_col_health
                    if colossus_damage_taken > 0:
                        total_reward += colossus_damage_taken * 2.0

        # --- 更新血量记录 ---
        if colossus:
            self.last_colossus_health = current_col_health
        else:
            self.last_colossus_health = -1

        return total_reward