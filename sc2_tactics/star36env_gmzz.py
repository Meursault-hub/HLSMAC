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

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "DepotLower": 556,
    "DepotRaise": 558,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsGMZZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("----------------------")
        print("You create a GMZZ env!")
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
        elif unit.unit_type == self.rlunit_ids.get("Depot") and action == 6:
            # lower the supply depot
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["DepotLower"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Lower Supply Depot".format(a_id))
        elif unit.unit_type == self.rlunit_ids.get("DepotLowered") and action == 6:
            # Raise the supply depot
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["DepotRaise"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Raise Supply Depot".format(a_id))
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

            if (unit.unit_type == self.rlunit_ids.get("Depot") or 
                unit.unit_type == self.rlunit_ids.get("DepotLowered")):
                if agent_id != self.active_door_id:
                    # 如果不是活动门，强制禁止动作 6 (Raise/Lower)
                    avail_actions[6] = 0
                    return avail_actions

                # 如果是活动门，允许动作 6
                avail_actions[6] = 1

                # 场景 1: 诱敌阶段 (Luring) -> 必须开门
                # 如果门已经开了(DepotLowered)，禁止关门(Raise)
                if not self.last_phase_is_trapping: 
                    if unit.unit_type == self.rlunit_ids.get("DepotLowered"):
                        avail_actions[6] = 0 # 禁止 Raise
                
                # 场景 2: 围歼阶段 (Trapping) -> 必须关门
                # 如果门已经关了(Depot)，禁止开门(Lower)
                else:
                    if unit.unit_type == self.rlunit_ids.get("Depot"):
                        avail_actions[6] = 0 # 禁止 Lower

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
            if unit.unit_type == 48:
                type_id = 0
            elif unit.unit_type == 19:
                type_id = 1
            elif unit.unit_type == 47:
                type_id = 2
            else:
                type_id = 99
                if self.debug:
                    logging.debug("Agent has unknown type: {}".format(unit.unit_type))
        else:
            if unit.unit_type == 105:
                type_id = 0
            elif unit.unit_type == 98:
                type_id = 1
            else:
                type_id = 99
                if self.debug:
                    logging.debug("Enemy has unknown type: {}".format(unit.unit_type))
        return type_id

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 98 and e.health > 0:
                    return False
            return True
        if ally == True:
            for a in self.agents.values():
                if (a.unit_type != self.rlunit_ids.get("Depot") and
                    a.unit_type != self.rlunit_ids.get("DepotLowered")
                    and a.health > 0):
                    return False
            return True
        return False
    
    def reset(self):

        # 【修复关键】先初始化关键变量，防止父类 reset 时报错
        # =========================================================
        # 供 get_avail_agent_actions 使用
        self.last_phase_is_trapping = False 
        self.action_reward_received = False
        
        # 供 get_obs_agent 使用 (防止 AttributeError)
        self.door_pos = None 
        self.fixed_home_pos = None
        self.safe_radius = 15.0 # 默认值
        self.depot_last_health = {}
        self.battle_step_count = 0
        self.depot_next_ready_tick = {}

        self.active_door_id = -1

        """初始化环境，记录门和防守中心的位置"""
        obs, state = super().reset()
        
        # 1. 寻找补给站位置 (作为"门口"的坐标参考)
        self.door_pos = None
        # 存储所有补给站的信息
        depot_list = []
        
        # 获取补给站的单位类型ID (从别名表中获取)
        depot_id = self.rlunit_ids.get("Depot")         # 升起的 (堵路)
        depot_low_id = self.rlunit_ids.get("DepotLowered") # 降下的 (通行)
        
        depot_positions = []
        for a_id, unit in self.agents.items():
            if unit.unit_type == depot_id or unit.unit_type == depot_low_id:
                depot_positions.append(np.array([unit.pos.x, unit.pos.y]))
                depot_list.append({
                    'id': a_id,
                    'pos': unit.pos,
                    'sum_coord': unit.pos.x + unit.pos.y # 用于斜向排序
                })
        
        # 默认值 (防止没有补给站报错)
        self.door_pos = np.array([16.0, 16.0])
        self.door_normal = np.array([1.0, 0.0]) # 默认指向右侧

        if len(depot_positions) > 0:
            # A. 找出中间的一个作为唯一的门
            # 根据坐标排序 (x+y 适合你的斜向路口)
            depot_list.sort(key=lambda d: d['sum_coord'])
            
            self.active_door_id = depot_list[0]['id']
            
            # 此时 door_pos 使用这个活动门的坐标，更加精准
            active_door_pos = depot_list[0]['pos']
            self.door_pos = np.array([active_door_pos.x, active_door_pos.y])
            
            # B. 计算墙壁走向 (Wall Vector)
            if len(depot_positions) >= 2:
                # 简单法：如果有多个补给站，找出距离最远的两个点，它们连线就是墙的方向
                # 这比 PCA 简单且在星际中足够有效
                max_dist = -1.0
                p1_best, p2_best = depot_positions[0], depot_positions[0]
                
                for i in range(len(depot_positions)):
                    for j in range(i + 1, len(depot_positions)):
                        d = np.linalg.norm(depot_positions[i] - depot_positions[j])
                        if d > max_dist:
                            max_dist = d
                            p1_best = depot_positions[i]
                            p2_best = depot_positions[j]
                
                wall_vector = p2_best - p1_best
                # 归一化
                if np.linalg.norm(wall_vector) > 0:
                    wall_vector = wall_vector / np.linalg.norm(wall_vector)
                
                # C. 计算法向量 (旋转 90 度)
                # 2D 向量 (x, y) 旋转 90 度变成 (-y, x)
                normal_vector = np.array([-wall_vector[1], wall_vector[0]])
            else:
                # 如果只有一个补给站，无法连线，暂时随便设一个方向
                normal_vector = np.array([1.0, 0.0])

            # D. 【关键】校正法向量方向
            # 我们需要法向量指向 "家(Home)" 的方向
            # 先找到家 (复用你原来的逻辑)
            marine_initial_positions = [np.array([u.pos.x, u.pos.y]) for u in self.agents.values() if u.unit_type == 48]
            if marine_initial_positions:
                self.fixed_home_pos = np.mean(marine_initial_positions, axis=0)
            else:
                self.fixed_home_pos = np.array([10.0, 10.0])
            
            # 建立一个从门指向家的向量
            door_to_home = self.fixed_home_pos - self.door_pos
            
            # 如果计算出的法向量和 door_to_home 方向相反(点积<0)，就反转它
            if np.dot(normal_vector, door_to_home) < 0:
                normal_vector = -normal_vector
            
            self.door_normal = normal_vector
            
            # DEBUG: 打印出来看看方向对不对
            # print(f"DEBUG: Door Pos: {self.door_pos}, Normal Vector: {self.door_normal}")

        else:
            # 没有补给站时的兜底逻辑
            marine_initial_positions = [np.array([u.pos.x, u.pos.y]) for u in self.agents.values() if u.unit_type == 48]
            if marine_initial_positions:
                self.fixed_home_pos = np.mean(marine_initial_positions, axis=0)
            else:
                self.fixed_home_pos = np.array([10.0, 10.0])


        # 2. 【新增】确定初始防守中心与安全半径 (Fixed Safe Zone)
        # 我们不能用 step 里的动态中心，因为如果人都跑出去了，动态中心也会跑出去

        # 计算安全半径：从家到门的距离
        if self.door_pos is not None:
            # 基础半径
            dist_home_to_door = np.linalg.norm(self.door_pos - self.fixed_home_pos)
            # 【关键】加上 3.0 的缓冲距离
            # 允许陆战队员站在门口射击，但不能超过门口太多
            self.safe_radius = dist_home_to_door + 3.0 #新增
        else:
            self.safe_radius = 15.0


        # 补给站血量记录 (用于计算承伤奖励)
        self.depot_last_health = {}
        depot_types = [self.rlunit_ids.get("Depot"), self.rlunit_ids.get("DepotLowered")]
        
        for a_id, unit in self.agents.items():
            if unit.unit_type in depot_types:
                self.depot_last_health[a_id] = unit.health


        return obs, state
    
    def step(self, actions):
        # 1. 执行动作
        out = super().step(actions)
        if len(out) == 3:
            reward_pkg, terminated, info = out
        elif len(out) == 4: 
            _, reward_pkg, terminated, info = out
        else:
             raise ValueError(f"父类返回值异常: {len(out)}")

        # 解包统计数据
        if isinstance(out[1], (tuple, list)):
            reward_pkg = out[1]
            # reward_pkg 结构通常是 (reward, delta_enemy, delta_deaths, delta_ally)
            # 我们需要 delta_deaths (索引 2)
            enemy_deaths = float(reward_pkg[2]) if len(reward_pkg) > 2 else 0.0
        else:
            enemy_deaths = 0.0
        
        base_reward = float(reward_pkg[0]) if isinstance(reward_pkg, (tuple, list)) else float(reward_pkg)
        extra_reward = 0
        
        # ==========================================
        # 3. 上下文感知 (Context Perception)
        # ==========================================
        
        # A. 确定防守中心 (机枪兵的重心)
        marine_positions = [np.array([u.pos.x, u.pos.y]) for u in self.agents.values() 
                            if u.unit_type == 48 and u.health > 0]
        
        if not marine_positions:
            # print("Warning: No marines alive!")
            # 【修复】即使提前返回，也要保持 Tuple 格式
            if isinstance(reward_pkg, (tuple, list)):
                # 如果父类本身就是包，直接原样返回
                return reward_pkg, terminated, info
            else:
                # 如果父类只是数值，手动补齐 (数值, 0, 0, 0)
                return float(base_reward), terminated, info

        defense_center = np.mean(marine_positions, axis=0)
        
        # B. 判定门的状态 (Door Status)
        # 规则：只要有一个补给站是 Lowered，就算开门；全部是 Depot，才算关门。
        is_door_open = False
        depot_units = [] # 记录所有补给站，用于后续检查动作
        
        for a_id, unit in self.agents.items():
            if unit.unit_type == self.rlunit_ids.get("DepotLowered") and unit.health > 0:
                is_door_open = True # 发现漏洞，门开了
                depot_units.append((a_id, unit))
            elif unit.unit_type == self.rlunit_ids.get("Depot") and unit.health > 0:
                depot_units.append((a_id, unit))
        
        # C. 判定敌人位置 (Inside vs Outside)
        
        # 【修改】加上列表存储，而不仅仅是计数
        enemies_inside_list = []
        n_inside = 0
        n_outside = 0
        
        # 判定缓冲带 (Hysteresis Buffer)
        # 只有深入门内一定距离才算进，只有完全出去一定距离才算出
        # 这能彻底消除"门口反复横跳"的问题
        threshold = -0.5  # 必须越过中线 0.5 格才算进
        
        for e_id, unit in self.enemies.items():
            if unit.health > 0 and unit.unit_type == 105:
                e_pos = np.array([unit.pos.x, unit.pos.y])
                
                # 向量：从门口指向敌人
                vec_door_to_enemy = e_pos - self.door_pos
                
                # 点积投影：计算敌人在法向量方向上的距离
                # > 0 表示在内侧， < 0 表示在外侧
                projection_dist = np.dot(vec_door_to_enemy, self.door_normal)
                
                if projection_dist > threshold:
                    n_inside += 1
                    enemies_inside_list.append(unit)
                else:
                    n_outside += 1

        #【新增】计算补给站本帧承受的伤害 (Tanking Calculation)
        depot_damage_taken = 0.0
        depot_types = [self.rlunit_ids.get("Depot"), self.rlunit_ids.get("DepotLowered")]
        
        for a_id, unit in self.agents.items():
            if unit.unit_type in depot_types:
                # 获取上一帧血量
                last_hp = self.depot_last_health.get(a_id, unit.health)
                current_hp = unit.health
                
                # 如果血量减少了，累加伤害值
                if current_hp < last_hp:
                    depot_damage_taken += (last_hp - current_hp)
                
                # 更新记录
                self.depot_last_health[a_id] = current_hp
        
        # ==========================================
        # 4. 动态策略奖励 (Dynamic Strategy Reward)
        # ==========================================

        # 【新增】越界惩罚 (Out of Bounds Penalty)
        # 遍历所有活着的陆战队员
        for m_id, unit in self.agents.items():
            # 确保是陆战队员 (48)
            if unit.unit_type == 48 and unit.health > 0:
                pos = np.array([unit.pos.x, unit.pos.y])
                
                # 计算离家的距离
                dist_to_home = np.linalg.norm(pos - self.fixed_home_pos)
                
                # 如果超出了安全半径 (跑到了补给站外面)
                if dist_to_home > self.safe_radius:
                    # 给一个显著的负奖励，迫使它退回去
                    # 这个惩罚应该比"攻击敌人的收益"稍微大一点点，防止它贪刀
                    extra_reward -= 0.2
        
        # 设定阈值：比如进来 1/3 的敌人就开始关门，或者固定数量
        # 这里假设只要进来 2 个以上，或者外面没啥人了，就关门打狗
        total_enemies = n_inside + n_outside

        # if self.last_phase_is_trapping:
        #     # 保持围歼: 只要屋里还有敌人，或者外面已经没人了，就继续关着
        #     should_trap = (n_inside > 0) or (n_outside == 0)
        # else:
        #     # 开启围歼: 必须有足够多的敌人进来 (比如 > 2) 才触发
        #     # 或者是残局(total < 3)且有人进来
        if total_enemies >= 7:
            should_trap = (n_outside == 0)# (n_inside >= 4) 
        else:
            should_trap = True # 残局只要进来一个就关

        self.last_phase_is_trapping = should_trap

        # --- 场景 1: 诱敌阶段 (Luring) ---
        # 目标：保持开门，引诱敌人深入
        if not should_trap:
            if is_door_open:

                for a_id, unit in depot_units:
                # 不多开门
                    if unit.unit_type == self.rlunit_ids.get("Depot") and actions[a_id] == 6: # Lower
                        extra_reward -= 2.5 
            else:
                # 错误状态：门关着，把客人挡外面了
                extra_reward -= 0.2
                
                # 引导动作：奖励执行 Lower 操作
                # 【防刷分修改】动作分：只有没领过奖时才给
                for a_id, unit in depot_units:
                # 1. 奖励正确的动作：门是关的(Depot)，执行降下(Lower)
                    if not self.action_reward_received:
                        if unit.unit_type == self.rlunit_ids.get("Depot") and actions[a_id] == 6: # Lower
                            extra_reward += 1.5 
                            self.action_reward_received = True 
                    
                    # 2. 【新增】惩罚错误的动作：门是开的(DepotLowered)，执行升起(Raise)
                    # 敌人还在外面，你关什么门？罚！
                    if unit.unit_type == self.rlunit_ids.get("DepotLowered") and actions[a_id] == 6: # Raise
                        extra_reward -= 1.5

        # --- 场景 2: 围歼阶段 (Trapping) ---
        # 目标：立刻关门，防止逃跑，并集火
        else:
            # 【新增】肉盾奖励 (Tanking Reward)
            # 核心逻辑：门关着 + 正在挨打 = 英雄行为 -> 给分！
            # 只有当门是关着的(not is_door_open)才给这个分
            if depot_damage_taken > 0:
                # 奖励系数 0.2：如果掉了 10 点血，给 +2.0 分
                # 这足以抵消死亡带来的恐惧，让它觉得"挨打真爽"
                extra_reward += depot_damage_taken * 0.005


            if not is_door_open:
                # 正确状态：门关得死死的

                # 【新增】追击势场 (Pursuit Field)
                # ==========================================
                # 只有当门内有猎物时，才激活追击逻辑
                if enemies_inside_list:
                    # 遍历每一个活着的陆战队员
                    for m_id, m_unit in self.agents.items():
                        # 确保是陆战队员(48)
                        if m_unit.unit_type == 48 and m_unit.health > 0:
                            m_pos = np.array([m_unit.pos.x, m_unit.pos.y])
                            
                            # 1. 寻找最近的猎物 (Nearest Neighbor)
                            min_dist_to_prey = 999.0
                            for e_unit in enemies_inside_list:
                                e_pos = np.array([e_unit.pos.x, e_unit.pos.y])
                                d = np.linalg.norm(m_pos - e_pos)
                                if d < min_dist_to_prey:
                                    min_dist_to_prey = d
                            
                            # 2. 施加距离压力
                            # 陆战队员射程约 5.0，我们设定 6.0 为警戒线
                            # 如果距离 > 6.0，说明打不到且离得远，必须惩罚
                            if min_dist_to_prey > 5.0:
                                # 系数 0.1：每远 1.0 距离扣 0.1 分
                                # 这比"越界惩罚"(-0.2)轻，但足以驱动它移动
                                extra_reward -= (min_dist_to_prey - 5.0) * 0.01 #元0.3
                            
                            # 3. 鼓励进入射程
                            # 如果已经进入射程 (<= 5.0)，给一点点保持奖励，防止它来回抖动
                            else:
                                # extra_reward += 0.3
                                pass

                # 关门打狗加成：此时造成的每一处基础伤害，都额外给 50% 的分
                if base_reward > 0:
                    extra_reward += base_reward * 3.5 #原0.5
                if enemy_deaths > 0:
                    extra_reward += enemy_deaths * 20.0

                # 2. 【新增】惩罚错误的动作：门是关的(Depot)，执行降下(Lower)
                # 已经关门打狗了，你开什么门？罚！
                if unit.unit_type == self.rlunit_ids.get("Depot") and actions[a_id] == 6: # Lower
                    extra_reward -= 3.5 #原2.0
                                
            else:
                # 错误状态：居然还开着门！敌人要跑了！
                extra_reward -= 2.0  #原1.5
                
                # 引导动作：重赏执行 Raise 操作
                # 【防刷分修改】动作分：只有没领过奖时才给
                for a_id, unit in depot_units:
                    # 1. 奖励正确的动作：门是开的(DepotLowered)，执行升起(Raise)
                    if unit.unit_type == self.rlunit_ids.get("DepotLowered") and actions[a_id] == 6: # Raise
                        extra_reward += 3.5 #原1.5
                        self.action_reward_received = True 

                    # 2. 【新增】惩罚错误的动作：门是关的(Depot)，执行降下(Lower)
                    # 已经关门打狗了，你开什么门？罚！
                    if unit.unit_type == self.rlunit_ids.get("Depot") and actions[a_id] == 6: # Lower
                        extra_reward -= 3.5 #原2.0

        # 5. 奖励截断
        extra_reward = np.clip(extra_reward, -5.0, 15.0) #元-5, 10.0

        # 6. 打包返回 (修复：保留 tuple 结构)
        if isinstance(reward_pkg, (tuple, list)):
            # 如果父类返回的是包 (reward, delta_enemy, delta_deaths, delta_ally)
            base_reward_scalar = float(reward_pkg[0])
            rest_stats = reward_pkg[1:]  # 提取后面的统计数据
            
            total_reward = base_reward_scalar + extra_reward
            # 重新打包：(新奖励, 统计数据...)
            final_reward = (total_reward, *rest_stats)
        else:
            # 父类只返回了数值
            final_reward = float(reward_pkg) + extra_reward


        # 慢放
        time.sleep(0.5)

        return final_reward, terminated, info