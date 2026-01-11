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


class SC2TacticsYQGZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("----------------------")
        print("You create a YQGZ env!")
        print("----------------------")

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 105:
                type_id = 0 # zergling
            else:
                type_id = 99 # error
        else:  # use default SC2 unit types
            if unit.unit_type == 48:
                type_id = 0 # marine
            elif unit.unit_type == 33:
                type_id = 1 # siegeTank
            elif unit.unit_type == 32:
                type_id = 2 # siegeTankSieged
            else:
                type_id = 99 # error
        return type_id
    
    def get_obs_agent(self, agent_id):
        """
        [欲擒故纵 - 定制观测]
        
        新增特征 (Extra Features, 5维):
        1. 敌方架起坦克数量 (Sieged Tank Count): 红灯信号。存在的越多，越不能冲。
        2. 敌方移动坦克数量 (Unsieged Tank Count): 绿灯信号。存在的越多，越要冲。
        3. 诱饵统计 (Allies in Danger Zone): 此时此刻，有多少队友处于架起坦克的射程(13)覆盖下？
           - 这个数值很小(1-3)说明是在诱敌。
           - 这个数值很大说明是在送死(除非坦克已经没架起来)。
        4. 自身危险状态 (Am I Bait): 我自己是不是在架起坦克的射程里？
        5. 最近架起坦克的距离: 方便智能体微操卡距离。
        """
        # 1. 获取基础观测
        base_obs = super().get_obs_agent(agent_id)
        
        unit = self.get_unit_by_id(agent_id)
        if unit is None:
            # 如果单位死亡，补齐 5 维 0 向量
            return np.concatenate((base_obs, np.zeros(5, dtype=np.float32)))

        # 2. 扫描全局战场信息
        sieged_tanks = []   # 架起的坦克 (ID 32)
        unsieged_tanks = [] # 移动的坦克 (ID 33)
        
        # 遍历 raw_data 寻找坦克
        for u in self._obs.observation.raw_data.units:
            if u.owner == 2: # 敌方
                if u.unit_type == 32: # SiegeTankSieged
                    sieged_tanks.append(u)
                elif u.unit_type == 33: # SiegeTank (Tank Mode)
                    unsieged_tanks.append(u)
        
        # 3. 计算诱饵统计数据
        # 攻城坦克架起模式射程为 13。我们设定 14 为危险警戒线。
        siege_range_threshold = 14.0
        allies_in_danger_count = 0
        am_i_in_danger = 0.0
        min_dist_to_sieged = 999.0
        
        if len(sieged_tanks) > 0:
            # A. 计算最近的架起坦克距离 (用于特征 5)
            for t in sieged_tanks:
                d = self.distance(unit.pos.x, unit.pos.y, t.pos.x, t.pos.y)
                if d < min_dist_to_sieged:
                    min_dist_to_sieged = d
            
            # B. 判定自身是否在危险区 (用于特征 4)
            if min_dist_to_sieged <= siege_range_threshold:
                am_i_in_danger = 1.0
                
            # C. 统计有多少队友在危险区 (用于特征 3)
            # 这一步是为了让智能体形成 "集体意识"，知道现在是不是只有少数人在送
            for a_unit in self.agents.values():
                if a_unit is not None and a_unit.health > 0:
                    # 检查该队友离任意一个架起坦克的距离
                    in_range = False
                    for t in sieged_tanks:
                        d_friend = self.distance(a_unit.pos.x, a_unit.pos.y, t.pos.x, t.pos.y)
                        if d_friend <= siege_range_threshold:
                            in_range = True
                            break
                    if in_range:
                        allies_in_danger_count += 1
        
        # 4. 构建额外特征向量 (5维)
        extra_obs = np.zeros(5, dtype=np.float32)
        
        # [Feature 0]: 架起坦克数量 (归一化, 假设最多2个坦克)
        extra_obs[0] = len(sieged_tanks) / 2.0
        
        # [Feature 1]: 移动坦克数量 (归一化)
        # 这是进攻号角：一旦这个数变大，说明敌人解除架起了
        extra_obs[1] = len(unsieged_tanks) / 2.0
        
        # [Feature 2]: 诱饵浓度 (全局统计)
        # 归一化：除以最大盟友数 (24只跳虫)
        extra_obs[2] = allies_in_danger_count / self.n_agents
        
        # [Feature 3]: 自身是否是诱饵 (Boolean)
        # 1.0 = 我在射程内，0.0 = 我在安全区
        extra_obs[3] = am_i_in_danger
        
        # [Feature 4]: 离最近架起坦克的距离 (归一化)
        # 如果没有架起坦克，置为 1.0 (表示很远/安全)
        if len(sieged_tanks) > 0:
            extra_obs[4] = min_dist_to_sieged / self.map_x
        else:
            extra_obs[4] = 1.0

        return np.concatenate((base_obs, extra_obs))

    def get_obs_size(self):
        """
        返回总维度 = 父类基础维度 + 新增维度(5)
        """
        return super().get_obs_size() + 5
    
    def reward_battle(self):
        """
        [欲擒故纵 - 动态双模式奖励函数]
        
        包含两个截然不同的阶段：
        1. 诱饵阶段 (Lure Phase): 坦克架起。
           - 目标: 保持大部队在圈外，只允许 1-3 只跳虫进圈送死。
           - 惩罚: 进圈人数 > 3 时，触发 "群体送死惩罚"。
        2. 围杀阶段 (Swarm Phase): 坦克收起。
           - 目标: 全军突击，贴脸输出。
           - 奖励: 距离越近分越高，击杀坦克给高倍率。
        
        里程碑:
        - 坦克解除架起: +100.0 (战机出现)
        - 胜利: +300.0
        """
        
        # --- 1. 初始化状态记录 ---
        if not hasattr(self, 'has_triggered_unsiege_bonus'): self.has_triggered_unsiege_bonus = False
        if not hasattr(self, 'enemies_cleared'): self.enemies_cleared = False
        if not hasattr(self, 'previous_sieged_count'): self.previous_sieged_count = 0
        
        if self._episode_steps == 0:
            self.has_triggered_unsiege_bonus = False
            self.enemies_cleared = False
            self.previous_sieged_count = 0
            
        # 获取基础奖励 (击杀/掉血)
        # 注意: 我们会在后面针对诱饵阶段做修正，抵消掉诱饵的掉血惩罚
        _ = super().reward_battle()
        total_reward = 0
        
        # --- 2. 全局态势感知 ---
        sieged_tanks = []   # 架起的坦克 (ID 32)
        unsieged_tanks = [] # 移动的坦克 (ID 33)
        marines = []        # 陆战队员 (ID 48)
        
        for u in self._obs.observation.raw_data.units:
            if u.owner == 2 and u.health > 0:
                if u.unit_type == 32: sieged_tanks.append(u)
                elif u.unit_type == 33: unsieged_tanks.append(u)
                elif u.unit_type == 48: marines.append(u)
        
        zerglings = [u for u in self.agents.values() if u is not None and u.health > 0]
        
        # 坦克架起状态判定 (Sieged Mode)
        is_sieged_mode = (len(sieged_tanks) > 0)
        
        # 危险距离阈值 (坦克射程约为 13)
        DANGER_ZONE = 14.0
        
        # =========================================================
        # [阶段 A] 红灯模式: 诱饵与蛰伏 (Lure Phase)
        # =========================================================
        if is_sieged_mode:
            # 统计在危险区内的己方单位
            allies_in_danger = 0
            for z in zerglings:
                # 只要在任意一个架起坦克的射程内，就算危险
                in_range = False
                for t in sieged_tanks:
                    d = self.distance(z.pos.x, z.pos.y, t.pos.x, t.pos.y)
                    if d < DANGER_ZONE:
                        in_range = True
                        break
                if in_range:
                    allies_in_danger += 1
            
            # --- 策略 1: 大部队禁入 (Mass Suicide Penalty) ---
            # 如果进圈人数超过 3 个 (诱饵上限)，每多一个给重罚
            # 这会像电网一样把大部队挡在外面
            if allies_in_danger > 3:
                # 惩罚系数要大，足以抵消冲上去造成的任何潜在伤害收益
                penalty = (allies_in_danger - 3) * 0.5
                total_reward -= penalty
            
            # --- 策略 2: 诱饵豁免与奖励 (Bait Incentive) ---
            # 如果有 1-3 个单位在里面，说明在执行诱敌任务
            elif allies_in_danger > 0:
                # 给予微量奖励，告诉它 "这里留几个人是对的"
                total_reward += 0.1 
                # 注意：这里我们不显式补偿掉血惩罚，因为 +0.1 每步累积下来
                # 足够覆盖跳虫那点血量的损失 (跳虫血很少，死得快，总扣分有限)
            
            # 如果全是 0，说明都在挂机，给一点点点负反馈逼它们派人去送
            else:
                total_reward -= 0.05

            # 更新计数器，为检测解除架起做准备
            self.previous_sieged_count = len(sieged_tanks)

        # =========================================================
        # [阶段 B] 绿灯模式: 围杀与突击 (Swarm Phase)
        # =========================================================
        else:
            # --- 里程碑 1: 战机大奖 (Opportunity Bonus) ---
            # 如果上一帧还有架起的坦克，这一帧没了，说明敌人上钩了/解除了！
            if self.previous_sieged_count > 0 and len(sieged_tanks) == 0:
                if not self.has_triggered_unsiege_bonus:
                    total_reward += 100.0  # 冲锋号角！
                    self.has_triggered_unsiege_bonus = True
            
            self.previous_sieged_count = 0 # 重置
            
            # 只有当敌人还没死光时才计算冲锋奖励
            if len(unsieged_tanks) + len(marines) > 0:
                # --- 策略 3: 全军突击 (Charge Reward) ---
                # 鼓励所有人贴脸。目标是最近的敌方坦克 (优先) 或 枪兵
                for z in zerglings:
                    min_dist = 999.0
                    target_priority = None
                    
                    # 1. 寻找最近目标 (优先坦克)
                    # 优先找坦克
                    for t in unsieged_tanks:
                        d = self.distance(z.pos.x, z.pos.y, t.pos.x, t.pos.y)
                        if d < min_dist: 
                            min_dist = d
                            target_priority = 'tank'
                    
                    # 没坦克找枪兵
                    if min_dist == 999.0:
                        for m in marines:
                            d = self.distance(z.pos.x, z.pos.y, m.pos.x, m.pos.y)
                            if d < min_dist: min_dist = d
                    
                    # 2. [新增] 远距离引力奖励 (Gravity Reward)
                    # 这是一个连续的引导信号，解决 "不知道往哪走" 的问题。
                    # 设定一个较大的感知半径 (例如 15.0)。
                    # 公式逻辑: (最大距离 - 当前距离) * 系数。离得越近，得分越高。
                    if min_dist < 15.0:
                        # 例如: 距离 15 时得 0 分; 距离 0 时得 0.15 分。
                        # 系数 0.01 足够引导方向，又不会掩盖掉击杀奖励。
                        total_reward += (15.0 - min_dist) * 0.01

                    # 3. 近战爆发奖励 (原有逻辑保持，作为额外的高潮奖励)
                    if min_dist < 3.0:
                        total_reward += 0.05
                        # 如果贴到了坦克脸上，额外加分 (切后排)
                        if target_priority == 'tank' and min_dist < 1.5:
                            total_reward += 0.1

        # =========================================================
        # [通用] 胜利与伤害修正
        # =========================================================
        
        # --- 里程碑 2: 胜利大奖 ---
        # 检查是否全歼敌人
        all_enemies_dead = True
        for u in self._obs.observation.raw_data.units:
            if u.owner == 2 and u.health > 0:
                all_enemies_dead = False
                break
        
        if all_enemies_dead and not self.enemies_cleared:
            total_reward += 300.0
            self.enemies_cleared = True
            
        # --- 伤害修正: 鼓励集火坦克 ---
        # 我们手动计算对坦克的伤害奖励，给予高权重
        # 注意: 这需要维护 previous_units 才能精确计算，为简化代码，
        # 这里我们假设 super().reward_battle() 已经处理了基础伤害。
        # 如果需要更激进，可以依赖 victory reward 倒逼集火。
        
        return total_reward