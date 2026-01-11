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
    "heal": 386,  # Unit
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsSDJXEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        map_params = get_map_params(self.map_name)
        if self.n_agents - map_params["n_madivac"] > self.n_enemies:
            self.n_actions = self.n_actions_no_attack + self.n_agents - map_params["n_madivac"]
        print("----------------------")
        print("You create a SDJX env!")
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
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if unit.unit_type == self.rlunit_ids.get("medivac"):
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
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

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if unit.unit_type == self.rlunit_ids.get("medivac"):
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.rlunit_ids.get("medivac")
                ]

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
        
    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        # 【修复关键点】必须与 get_state_dict 中的特征增加量保持一致
        # 我们在 get_state_dict 中增加了: 
        #   +1 (Is Decoy 角色编码) 
        #   +2 (Enemy Center X, Y 敌军重心) 
        #   -----------------------
        #   总共 +3
        nf_al = self.get_ally_num_attributes()+3 
        nf_en = self.get_enemy_num_attributes()

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - enemies: numpy array containing enemies and their attributes
        - last_action: numpy array of previous actions for each agent
        - timestep: current no. of steps divided by total no. of steps

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes()+3   # 新增+1 Role, +2 EnemyCenter
        nf_en = self.get_enemy_num_attributes()

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        # 计算敌军战斗单位重心 (排除建筑)
        combat_enemy_pos = []
        for e_unit in self.enemies.values():
            if e_unit.health > 0 and e_unit.unit_type not in [59, 60, 61]:
                combat_enemy_pos.append([e_unit.pos.x, e_unit.pos.y])
        if combat_enemy_pos:
            mean_pos = np.mean(combat_enemy_pos, axis=0)
            enemy_center_rel_x = (mean_pos[0] - center_x) / self.max_distance_x
            enemy_center_rel_y = (mean_pos[1] - center_y) / self.max_distance_y
        else:
            # 如果没有敌军战斗单位，重心设为地图中心或 0
            enemy_center_rel_x = 0
            enemy_center_rel_y = 0

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.cooldown_map.get(al_unit.unit_type, 15)
                
                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if al_unit.unit_type == self.rlunit_ids.get("medivac"):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
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

                # --- 【新增特征】 ---
                # [倒数第3位] 角色编码 (Is Decoy?)
                is_decoy = 1.0 if al_id in self.decoy_ids else 0.0
                ally_state[al_id, -3] = is_decoy

                # [倒数第2, 1位] 敌军重心相对位置 (所有 agent 共享这个全局信息)
                ally_state[al_id, -2] = enemy_center_rel_x
                ally_state[al_id, -1] = enemy_center_rel_y

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
    
    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 48:
                    type_id = 0
            elif unit.unit_type == 54:
                type_id = 1
        else:
            if unit.unit_type == 4:
                    type_id = 0
            elif unit.unit_type == 74:
                type_id = 1
            elif unit.unit_type == 73:
                type_id = 2
            elif unit.unit_type == 59:
                type_id = 3
            elif unit.unit_type == 60:
                type_id = 4
            elif unit.unit_type == 61:
                type_id = 5
        return type_id
    
    def _kill_vespene(self):
        """Kill all vespene gas left by dead Assimilators on the map."""
        vespene_alive = [
            unit.tag for unit in self._controller.observe().observation.raw_data.units if unit.owner == 16
        ]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=vespene_alive))
        ]
        self._controller.debug(debug_command)
        self._controller.step(2)

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
            self._kill_vespene()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.rlunit_ids.get("medivac"))
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.rlunit_ids.get("medivac"))
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False
        
    def check_structure(self, ally = True):
        """Check if the enemy's Nexus unit is killed."""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type == 59 and e.health <= 0:
                    return True
        return False
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 59 and e.unit_type != 133 and e.unit_type != 61 and e.unit_type != 60 and e.health > 0:
                    return False
            return True
        if ally == True and self.only_medivac_left(ally):
            return True
        return False
    

    def reset(self):
        """Reset the environment and identify strategic targets."""
        self._episode_steps = 0
        if self._episode_count == 0:
            self._launch()
        else:
            self._restart()

        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        
        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
            
            # =========================================================
            # 【新增】战略目标识别 (Strategy Identification)
            # =========================================================
            self.decoy_ids = []      # 诱饵飞机 ID 列表
            self.main_force_ids = [] # 主力部队 ID 列表
            self.left_nexus = None   # 左侧基地 (真目标)
            self.right_nexus = None  # 右侧基地 (假目标/诱饵区)
            
            # 1. 识别我方角色
            # 根据坐标判断：初始在右边(x > 50)的是诱饵，左边的是主力
            for a_id, unit in self.agents.items():
                if unit.unit_type == self.rlunit_ids.get("medivac") and unit.pos.x > 50:
                    self.decoy_ids.append(a_id)
                else:
                    self.main_force_ids.append(a_id)
            
            # 2. 识别敌方建筑
            for e_id, unit in self.enemies.items():
                if unit.unit_type == 59: # Nexus
                    if unit.pos.x < 40:
                        self.left_nexus = unit # (约 24.5, 47.5)
                    else:
                        self.right_nexus = unit # (约 54.5, 47.5)

            # 3. 初始化奖励计算所需的上一帧状态
            self.medivac_last_health = {}
            for a_id in self.decoy_ids:
                if self.agents[a_id]:
                    self.medivac_last_health[a_id] = self.agents[a_id].health
            
            # 记录上一帧距离 (用于计算引力奖励 delta)
            self.last_dist_to_objective = {} 
            # debug
            # print(f"Decoys: {self.decoy_ids}, LeftNexus: {self.left_nexus.pos.x if self.left_nexus else 'None'}")

        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}".format(self._episode_count))

        return self.get_obs(), self.get_state()
    
    def step(self, actions):
        """Execute actions with Sound East Strike West rewards."""
        
        # 1. 记录动作前的状态 (用于计算引力)
        # --------------------------------------------------
        # 更新左侧基地的位置 (如果还活着)
        left_nexus_pos = None
        right_nexus_pos = None
        
        # 重新扫描基地位置，防止引用失效
        for unit in self.enemies.values():
            if unit.unit_type == 59 and unit.health > 0:
                if unit.pos.x < 40: left_nexus_pos = np.array([unit.pos.x, unit.pos.y])
                else: right_nexus_pos = np.array([unit.pos.x, unit.pos.y])

        # 记录每个单位到其战术目标的距离
        dists_before = {}
        for a_id, unit in self.agents.items():
            if unit.health > 0:
                if a_id in self.decoy_ids and right_nexus_pos is not None:
                    dists_before[a_id] = self.distance(unit.pos.x, unit.pos.y, right_nexus_pos[0], right_nexus_pos[1])
                elif a_id in self.main_force_ids and left_nexus_pos is not None:
                    dists_before[a_id] = self.distance(unit.pos.x, unit.pos.y, left_nexus_pos[0], left_nexus_pos[1])
        
        # 2. 执行动作
        # --------------------------------------------------
        out = super().step(actions)
        if len(out) == 3:
            base_reward, terminated, info = out
        else:
            base_reward, terminated, info = out[1], out[2], out[3]

        extra_reward = 0

        # 3. 计算战术指标
        # --------------------------------------------------
        # 计算敌军战斗重心 (判断是否调虎离山成功)
        combat_enemy_pos = [np.array([u.pos.x, u.pos.y]) for u in self.enemies.values() 
                            if u.health > 0 and u.unit_type not in [59, 60, 61]]
        
        is_enemy_distracted = False
        if combat_enemy_pos and left_nexus_pos is not None:
            enemy_center = np.mean(combat_enemy_pos, axis=0)
            dist_center_to_left = np.linalg.norm(enemy_center - left_nexus_pos)
            # 如果敌军重心距离左侧基地超过 20 (说明被拉到右边去了)
            if dist_center_to_left > 15.0:
                is_enemy_distracted = True

        # 4. 应用奖励逻辑
        # --------------------------------------------------
        
        # === A. 诱饵组奖励 (Decoy Team) ===
        # 目标：靠近右侧基地，吸引火力，承伤
        for a_id in self.decoy_ids:
            unit = self.agents.get(a_id)
            if unit and unit.health > 0:
                # 1. 承伤奖励 (Tanking Reward)
                last_hp = self.medivac_last_health.get(a_id, unit.health)
                if unit.health < last_hp:
                    damage_taken = last_hp - unit.health
                    # 诱饵挨打是好事，给奖励
                    extra_reward += damage_taken * 0.1
                self.medivac_last_health[a_id] = unit.health

                # 2. 引力奖励 (Gravity to Right Base)
                if right_nexus_pos is not None and a_id in dists_before:
                    dist_now = self.distance(unit.pos.x, unit.pos.y, right_nexus_pos[0], right_nexus_pos[1])
                    diff = dists_before[a_id] - dist_now
                    extra_reward += diff * 0.2  # 鼓励靠近右侧
                    
                    # 3. 吸引奖励 (Attraction Reward)
                    # 如果成功把敌军主力拉近了 (例如距离诱饵 < 15)
                    if combat_enemy_pos:
                        min_enemy_dist = min([np.linalg.norm(np.array([unit.pos.x, unit.pos.y]) - e_pos) for e_pos in combat_enemy_pos])
                        if min_enemy_dist < 10.0:
                             extra_reward += 0.0005 # 持续奖励：你成功吸引了敌人注意

        # === B. 主力组奖励 (Main Force) ===
        # 目标：在敌人被调离时突击左侧，敌人未调离时保持距离
        for a_id in self.main_force_ids:
            unit = self.agents.get(a_id)
            if unit and unit.health > 0:
                # 获取当前到左侧基地的距离
                if left_nexus_pos is not None:
                    dist_now = self.distance(unit.pos.x, unit.pos.y, left_nexus_pos[0], left_nexus_pos[1])
                    
                    if is_enemy_distracted:
                        # 场景 1: 调虎离山成功 -> 猛攻！
                        # 高引力奖励
                        if a_id in dists_before:
                            diff = dists_before[a_id] - dist_now
                            extra_reward += diff * 0.5 # 系数较高，鼓励冲锋
                        
                    else:
                        # 场景 2: 敌人还在家 -> 猥琐发育
                        # 如果靠太近 (<= 15)，给予惩罚 (防止送死)
                        if dist_now < 15.0:
                             # 距离越近罚得越重，除非正在攻击
                             extra_reward -= 0.05
        
        # === C. 造成伤害奖励 (Damage Reward) ===
        # 特别奖励对左侧基地的伤害，防止打右侧基地造成局部最优
        # 我们通过检查 action 意图来估算 (因为 base_reward 是总伤害)
        if isinstance(base_reward, (float, int)) and base_reward > 0:
            # 这里的逻辑比较粗糙，假设产生 base_reward 就是因为有人在攻击
            # 我们检查主力部队是否在攻击左侧基地
            hitting_left_nexus = False
            for a_id in self.main_force_ids:
                action = actions[a_id] if a_id < len(actions) else 0 # 获取上一帧动作需要你在 step 里记录 actions
                # 由于 step 参数 actions 是这一帧的动作，我们可以直接用
                if a_id < len(actions):
                    act = actions[a_id]
                    if act >= self.n_actions_no_attack:
                        target_id = act - self.n_actions_no_attack
                        target_unit = self.enemies.get(target_id)
                        if target_unit and target_unit.unit_type == 59 and target_unit.pos.x < 40:
                            hitting_left_nexus = True
                            break
            
            if hitting_left_nexus:
                # 超级大奖：主力正在打左侧水晶
                extra_reward += base_reward * 5.0 
            # else:
            #     # 普通伤害
            #     extra_reward += base_reward * 1.0

        # 5. 合并奖励
        if isinstance(base_reward, (float, int)):
            final_reward = base_reward + extra_reward
        else:
            new_scalar_reward = float(base_reward[0]) + extra_reward
            final_reward = (new_scalar_reward,) + tuple(base_reward[1:])
            
        return final_reward, terminated, info
    

# ==============================
# === 地图坐标探测 (Episode 0) ===
# 地图尺寸: 80 x 80
# --- 我方单位 (Allies) ---
# ID: 0 | Type: 48 (Marine (陆战队员)) | Pos: (24.46, 38.58)
# ID: 1 | Type: 48 (Marine (陆战队员)) | Pos: (25.21, 37.83)
# ID: 2 | Type: 48 (Marine (陆战队员)) | Pos: (25.21, 39.33)
# ID: 3 | Type: 48 (Marine (陆战队员)) | Pos: (25.27, 38.58)
# ID: 4 | Type: 48 (Marine (陆战队员)) | Pos: (25.96, 37.08)
# ID: 5 | Type: 48 (Marine (陆战队员)) | Pos: (25.96, 37.89)
# ID: 6 | Type: 48 (Marine (陆战队员)) | Pos: (25.96, 38.58)
# ID: 7 | Type: 48 (Marine (陆战队员)) | Pos: (25.96, 39.27)
# ID: 8 | Type: 48 (Marine (陆战队员)) | Pos: (25.96, 40.08)
# ID: 9 | Type: 48 (Marine (陆战队员)) | Pos: (26.65, 38.58)
# ID: 10 | Type: 48 (Marine (陆战队员)) | Pos: (26.71, 37.08)
# ID: 11 | Type: 48 (Marine (陆战队员)) | Pos: (26.71, 37.83)
# ID: 12 | Type: 48 (Marine (陆战队员)) | Pos: (26.71, 39.33)
# ID: 13 | Type: 48 (Marine (陆战队员)) | Pos: (27.46, 38.58)
# ID: 14 | Type: 54 (Medivac (医疗运输机)) | Pos: (25.96, 37.14)
# ID: 15 | Type: 54 (Medivac (医疗运输机)) | Pos: (25.96, 38.58)
# ID: 16 | Type: 54 (Medivac (医疗运输机)) | Pos: (63.11, 21.03)
# ID: 17 | Type: 54 (Medivac (医疗运输机)) | Pos: (63.11, 22.47)

# --- 敌方单位/建筑 (Enemies) ---
# ID: 0 | Type: 59 (Nexus (【关键】基地/星灵枢纽)) | Pos: (54.50, 47.50)
# ID: 1 | Type: 60 (Pylon (水晶塔)) | Pos: (62.00, 38.00)
# ID: 2 | Type: 4 (Colossus (巨像)) | Pos: (38.67, 50.63)
# ID: 3 | Type: 74 (Stalker (追猎者)) | Pos: (36.83, 49.80)
# ID: 4 | Type: 74 (Stalker (追猎者)) | Pos: (37.61, 48.90)
# ID: 5 | Type: 73 (Zealot (狂热者)) | Pos: (35.04, 48.51)
# ID: 6 | Type: 73 (Zealot (狂热者)) | Pos: (36.97, 47.52)
# ID: 7 | Type: 73 (Zealot (狂热者)) | Pos: (38.76, 50.73)
# ID: 8 | Type: 73 (Zealot (狂热者)) | Pos: (36.76, 51.73)
# ID: 9 | Type: 73 (Zealot (狂热者)) | Pos: (38.76, 49.73)
# ID: 10 | Type: 74 (Stalker (追猎者)) | Pos: (35.99, 50.64)
# ID: 11 | Type: 74 (Stalker (追猎者)) | Pos: (37.67, 50.64)
# ID: 12 | Type: 4 (Colossus (巨像)) | Pos: (38.08, 48.33)
# ID: 13 | Type: 4 (Colossus (巨像)) | Pos: (36.88, 49.85)
# ID: 14 | Type: 59 (Nexus (【关键】基地/星灵枢纽)) | Pos: (24.50, 47.50)
# ID: 15 | Type: 61 (Gateway (折跃门)) | Pos: (46.50, 54.50)
# ID: 16 | Type: 61 (Gateway (折跃门)) | Pos: (62.50, 47.50)
# ==============================