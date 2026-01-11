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
    "NydusCanalLoad": 1437,
    "NydusCanalUnload": 1438,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsDHLSEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load = {}

        self.phase = kwargs.get("phase", 0) 

        # 初始化变量
        self.bait_agents = []
        self.main_force_agents = []
        self.nydus_entry_pos = None
        self.nydus_exit_pos = None
        self.last_dist_info = {}
        self.target_base_pos = np.array([32.0, 32.0])
        self.last_mean_dist_to_base = 200.0
        
        # 记录基地是否已经被摧毁，防止重复给分
        self.base_destroyed = False 

        print("----------------------")
        print("You create a DHLS env!")
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

        elif action == 6 and unit.unit_type == self.rlunit_ids.get("nydusNetwork"):
            # Load all units in the sight range
            target_tag = 0
            for t_id, t_unit in self.agents.items():
                if ((t_unit.unit_type == self.rlunit_ids.get("roach") or
                    t_unit.unit_type == self.rlunit_ids.get("zergling")) and 
                    self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    <= self.unit_sight_range(t_id) and
                    t_id not in self.load):
                    target_tag = t_unit.tag
                    self.load[t_id] = t_unit
                    break
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["NydusCanalLoad"],
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Load Unit {}".format(a_id, t_id))

        elif action == 6 and unit.unit_type == self.rlunit_ids.get("nydusCanal"):
            # Unload all agents
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["NydusCanalUnload"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Unload ALL Units".format(a_id))

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

            if unit.unit_type == self.rlunit_ids.get("hatchery"):
                avail_actions[1] = 1
                return avail_actions

            if (unit.unit_type != self.rlunit_ids.get("roach") and
                unit.unit_type != self.rlunit_ids.get("zergling")):
                # the structures in dhls can only stop
                avail_actions[1] = 1

                if unit.unit_type == self.rlunit_ids.get("nydusNetwork"):
                # check if roach can enter the NydusNetwork
                    for t_id, t_unit in self.agents.items():
                        if ((t_unit.unit_type == self.rlunit_ids.get("roach") or
                            t_unit.unit_type == self.rlunit_ids.get("zergling")) and
                            t_id not in self.load):
                            dist = self.distance(
                                unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                            )
                            sight_range = self.unit_sight_range(agent_id)
                            if dist <= sight_range:
                                avail_actions[6] = 1
                                break
                
                if unit.unit_type == self.rlunit_ids.get("nydusCanal") and self.load != {}:
                    avail_actions[6] = 1
                
                return avail_actions

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

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )

                    target_radius = 0.0
                    if t_unit.unit_type == 18: # 敌方基地
                        target_radius = 2.75 
                    else:
                        target_radius = 0.5

                    if dist <= shoot_range + target_radius + 2.0:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 86:
                type_id = 0
            elif unit.unit_type == 95:
                type_id = 1
            elif unit.unit_type == 142:
                type_id = 2
            elif unit.unit_type == 110:
                type_id = 3
            elif unit.unit_type == 105:
                type_id = 4
        else:
            if unit.unit_type == 48:
                type_id = 0
            elif unit.unit_type == 33:
                type_id = 1
            elif unit.unit_type == 18:
                type_id = 2
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
        ]
        self.load = {}
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def check_structure(self, ally = True):
        """Check if the enemy's Nexus unit is killed."""
        if ally == True:
            for a in self.agents.values():
                if a.unit_type == self.rlunit_ids.get("hatchery") and a.health <= 0:
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
                if ((a.unit_type == self.rlunit_ids.get("roach") or
                     a.unit_type == self.rlunit_ids.get("zergling")) and a.health > 0):
                    return False
            return True
        
    def clean_load(self):
        """If roach is closed enough to Nydus Canal, which means it is just unloaded,
           remove it from self.load"""
        for a_id, a_unit in self.agents.items():
            if a_unit.unit_type == self.rlunit_ids.get("nydusCanal"):
                del_id = []
                for t_id in self.load:
                    t_unit = self.get_unit_by_id(t_id)
                    dist = self.distance(a_unit.pos.x, a_unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist <= 5:
                        del_id.append(t_id)
                for t_id in del_id:
                    del self.load[t_id]
                return
        return
    
    def check_load(self, a_tag, updated):
        """make sure the roach in load is not dead"""
        for t_unit in self.load.values():
            if a_tag == t_unit.tag:
                return True
        return updated
    
    def reset(self):
        """重置环境，初始化角色和基地位置"""
        self.bait_agents = []
        self.main_force_agents = []
        self.nydus_entry_pos = None
        self.nydus_exit_pos = None
        self.base_destroyed = False # 重置基地状态

        obs, state = super().reset()
        
        # 1. 定义角色
        self.bait_agents = []
        self.main_force_agents = []
        zergling_count = 0
        max_zergling_main = 5 
        
        for a_id, unit in self.agents.items():
            if unit.unit_type == self.rlunit_ids.get("zergling") and zergling_count < max_zergling_main:
                self.bait_agents.append(a_id)
                zergling_count += 1
            else:
                self.main_force_agents.append(a_id)
        
        # 2. 寻找坑道虫
        for unit in self.agents.values():
            if unit.unit_type == self.rlunit_ids.get("nydusNetwork"):
                self.nydus_entry_pos = np.array([unit.pos.x, unit.pos.y])
            elif unit.unit_type == self.rlunit_ids.get("nydusCanal"):
                self.nydus_exit_pos = np.array([unit.pos.x, unit.pos.y])

        # 3. 寻找敌方基地
        base_unit_id = 18 
        found_base = False
        for e_id, unit in self.enemies.items():
            if unit.unit_type == base_unit_id:
                self.target_base_pos = np.array([unit.pos.x, unit.pos.y])
                found_base = True
                break
        
        # 初始化上一帧基地血量
        self.last_base_health = 0
        for e in self.enemies.values():
            if e.unit_type == 18: # 找到基地
                self.last_base_health = e.health
                break
        
        if not found_base:
             # 如果没找到，尝试找任意一个活着的敌人作为目标，防止报错
            if len(self.enemies) > 0:
                first_enemy = next(iter(self.enemies.values()))
                self.target_base_pos = np.array([first_enemy.pos.x, first_enemy.pos.y])
            else:
                self.target_base_pos = np.array([32.0, 32.0])

        self.last_mean_dist_to_base = 200.0
        obs = self.get_obs()
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
        
        # 2. 获取基础奖励 (SMAC 默认是伤害值)
        # 此时 reward_pkg 应该已经是数值了，或者 tuple 的第一个是数值
        base_reward_val = float(reward_pkg[0]) if isinstance(reward_pkg, (tuple, list)) else float(reward_pkg)
        
        extra_reward = 0
        
        # 3. 上下文信息
        enemy_positions = [np.array([u.pos.x, u.pos.y]) for u in self.enemies.values() if u.health > 0]

        dist_enemy_to_base = 0.0 # 默认值
        
        if len(enemy_positions) > 0:
            # 计算敌方所有单位的平均坐标 (重心)
            avg_enemy_pos = np.mean(enemy_positions, axis=0)
            
            # 计算重心到敌方基地 (self.target_base_pos) 的距离
            # self.target_base_pos 在 reset() 中已经初始化
            dist_enemy_to_base = np.linalg.norm(avg_enemy_pos - self.target_base_pos)
        else:
            # 如果敌人全死光了，距离设为 0 或者一个不影响逻辑的值
            dist_enemy_to_base = 0.0
        
        # 检查基地是否刚被摧毁
        base_alive = False
        for e in self.enemies.values():
            if e.unit_type == 18 and e.health > 0:
                base_alive = True
                break

        # 计算只针对基地的伤害 (Base Specific Damage)
        current_base_health = 0
        for e in self.enemies.values():
            if e.unit_type == 18: # 找到基地
                current_base_health = e.health
                break
        # 计算本帧造成的基地伤害
        # 注意：如果基地回血了(比如人族修补)，我们不扣分，只算伤害(damage > 0)
        damage_to_base = 0
        if self.last_base_health > current_base_health:
            damage_to_base = self.last_base_health - current_base_health
        # 更新上一帧血量，供下一步使用
        self.last_base_health = current_base_health

        
        # --- 全局奖励：基地摧毁 (Winning Condition) ---
        if not base_alive and not self.base_destroyed:
            extra_reward += 5.0 # 巨额悬赏
            self.base_destroyed = True

        # --- 全局奖励：引力势场 (针对主力部队) ---
        main_force_dists = []
        for a_id in self.main_force_agents:
            unit = self.get_unit_by_id(a_id)
            if unit and unit.health > 0 and a_id not in self.load:
                    d = np.linalg.norm(np.array([unit.pos.x, unit.pos.y]) - self.target_base_pos)
                    main_force_dists.append(d)
        
        ATTACK_TRIGGER_RANGE = 7.0

        if len(main_force_dists) > 0 and dist_enemy_to_base > 15.0:
            current_mean_dist = sum(main_force_dists) / len(main_force_dists)
            damage_to_base = min(main_force_dists)
            dist_diff = self.last_mean_dist_to_base - current_mean_dist
            self.last_mean_dist_to_base = current_mean_dist

            # [策略 A] 远程引力：只有在圈外才给引力，防止圈内刷分
            
            if -5.0 < dist_diff < 5.0:
                # 确保移动一步的收益 (约 0.5 * 0.05 = 0.025) 能被感知到
                extra_reward += dist_diff * 0.005
            
            # [策略 B] 近战强制：圈内必须打出伤害

            if dist_enemy_to_base <= ATTACK_TRIGGER_RANGE:
                # 如果本回合造成了伤害 (Base Reward > 0)
                if damage_to_base > 0:
                    # 伤害倍率奖励：造成的伤害越多，奖励越大
                    # 假设造成 1 点伤害，SMAC 给 1 分，这里额外给 5 分
                    extra_reward += damage_to_base * 0.05
                else:
                    # 没伤害的惩罚
                    # 必须比移动奖励大，逼迫它不要退回到圈外
                    extra_reward -= 0.01

        # --- 个体微观奖励 ---
        for a_id, unit in self.agents.items():
            if unit.health <= 0: continue
            pos = np.array([unit.pos.x, unit.pos.y])

            # === 诱饵组 (Bait) ===
            if a_id in self.bait_agents:
                # 1. 存活奖励
                extra_reward += 0.001
                
                # 2. 距离控制 (Scale Up)
                min_dist_enemy = min([np.linalg.norm(pos - e) for e in enemy_positions]) if enemy_positions else 99

                if 5.0 <= min_dist_enemy <= 10.0:
                    extra_reward += 0.005 # 完美距离，重赏

                # ==========================================
                # 3. 【新增】调虎离山奖励 (Strategic Pull)
                # ==========================================
                    
                SAFE_ZONE_RADIUS = 15.0
                if dist_enemy_to_base > SAFE_ZONE_RADIUS:
                    # 基础牵制分
                    extra_reward += 0.01
                    
                    # 进阶梯度分：拉得越远，分越高
                    # 举例：如果拉到了 25.0 的距离，额外再给 (25-15)*0.1 = 1.0 分
                    extra_reward += (dist_enemy_to_base - SAFE_ZONE_RADIUS) * 0.05

            # === 主力组 (Main Force) ===
            elif a_id in self.main_force_agents:
                # 1. 进洞奖励 (Nydus Entry)
                if a_id not in self.load:
                    if self.nydus_entry_pos is not None:
                        dist_to_entry = np.linalg.norm(pos - self.nydus_entry_pos)
                        # 如果在入口附近且还没过河
                        if dist_to_entry < 5.0 and self.nydus_exit_pos is not None:
                             dist_to_exit = np.linalg.norm(pos - self.nydus_exit_pos)
                             if dist_to_exit > 20.0:
                                 if actions[a_id] == 6: # Load 动作
                                     extra_reward += 2.0 # 一次性给个大的，相当于杀两只怪

        # 4. 奖励截断 (Clipping)
        # 因为现在奖励数值很大 (10~100)，传统的 [-1, 1] 截断已不适用
        # 我们放宽截断范围，或者仅做极端值保护
        # extra_reward = np.clip(extra_reward, -20.0, 20.0) 
        
        # 注意：如果是摧毁基地的 +100 分，可能会被 Clip 掉
        # 可以在 Clip 之后单独把 Winning Reward 加回来，或者把 Clip 设大一点
        if not base_alive and self.base_destroyed:
             # 确保这一帧的 100 分能完整传出去，不受 Clip 影响
             pass 
             # 这里实际上上面 clip 已经执行了，如果想保留 100，建议 Clip 范围设为 [-20, 100]
             # 或者简化处理：
        
        # 修正 Clip 逻辑以适应 Winning Reward
        extra_reward = np.clip(extra_reward, -1.0, 5.0)

        # 5. 打包返回
        if isinstance(reward_pkg, (tuple, list)):
            base_reward = float(reward_pkg[0])
            rest_stats = reward_pkg[1:]
            total_reward = base_reward + extra_reward
            final_reward = (total_reward, *rest_stats)
        else:
            final_reward = float(reward_pkg) + extra_reward

        return final_reward, terminated, info
    
    # 添加 get_obs_agent 方法
    def get_obs_agent(self, agent_id):
        # 获取原本的观测向量
        obs = super().get_obs_agent(agent_id)
        
        # 添加 One-Hot 角色标识 [is_bait, is_main]
        role_feat = np.zeros(2, dtype=np.float32)
        if agent_id in self.bait_agents:
            role_feat[0] = 1
        else:
            role_feat[1] = 1
            
        # 添加坑道虫相对位置信息 (帮助主力找洞)
        nydus_feat = np.zeros(4, dtype=np.float32)
        unit = self.get_unit_by_id(agent_id)
        if unit and self.nydus_entry_pos is not None and self.nydus_exit_pos is not None:
             # 归一化相对坐标
             pos = np.array([unit.pos.x, unit.pos.y])
             nydus_feat[0] = (self.nydus_entry_pos[0] - pos[0]) / self.map_x
             nydus_feat[1] = (self.nydus_entry_pos[1] - pos[1]) / self.map_y
             nydus_feat[2] = (self.nydus_exit_pos[0] - pos[0]) / self.map_x
             nydus_feat[3] = (self.nydus_exit_pos[1] - pos[1]) / self.map_y
             
        # 拼接观测
        new_obs = np.concatenate([obs, role_feat, nydus_feat])
        return new_obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        # 1. 获取父类原本计算的基础观测大小
        base_obs_size = super().get_obs_size()
        
        # 2. 加上新增特征的长度
        # role_feat (2) + nydus_feat (4) = 6
        return base_obs_size + 6