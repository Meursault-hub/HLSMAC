import mpyq
import numpy as np
import re
import json
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
from smac.env.sc2_tactics.maps import get_map_params
from absl import flags
import atexit
from pysc2 import run_configs, maps
from pysc2.lib import units, remote_controller
from s2clientprotocol import common_pb2, raw_pb2
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import debug_pb2

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

# import matplotlib.pyplot as plt

def process_pathing_grid(bits_per_pixel, pathing_grid_data, map_x, map_y):
    if bits_per_pixel == 1:
        vals = np.array(list(pathing_grid_data)).reshape(
            map_y, int(map_x / 8)
            )
        pathing_grid = np.transpose(
            np.array([ [(b >> i) & 1 for b in row for i in range(7, -1, -1)] for row in vals ], dtype=bool,)
            )
    elif bits_per_pixel == 8:
        pathing_grid = np.invert(
            np.flip(
                np.transpose(
                    np.array(
                        list(pathing_grid_data), dtype=bool
                        ).reshape(map_x, map_y)
                    ),
                axis=1,
                )
            )
    else:
        raise ValueError(f"Unsupported bits_per_pixel value {bits_per_pixel}")
    return pathing_grid

def process_terrain_height(terrain_height_data, map_x, map_y):
    terrain_height = (
        np.transpose(
            np.array(list(terrain_height_data)).reshape(
                map_y, map_x
                )
            ) /255
        )
    return terrain_height

# def process_terrain_height(terrain_height_data, map_x, map_y):
#     terrain_height = (
#         np.flip(
#             np.transpose(
#                 np.array(list(terrain_height_data)).reshape(
#                     map_y, map_x
#                     )
#                 ),
#             1,
#             )
#         / 255
#         )
#     return terrain_height

def calculate_max_reward(n_enemies, reward_death_value, reward_win):
    """
    计算最大奖励。

    Parameters:
    - n_enemies (int): 敌方单位数量。
    - reward_death_value (float): 消灭每个敌方单位的奖励值。
    - reward_win (float): 胜利的额外奖励。

    Returns:
    - float: 计算得到的最大奖励值。
    """
    return n_enemies * reward_death_value + reward_win

# common_utils.py

def initialize_state_attr_names(shield_bits_ally, shield_bits_enemy, unit_type_bits):
    """
    根据参数初始化 ally 和 enemy 的状态属性名称列表。
    
    Parameters:
    - shield_bits_ally (int): 盟军是否有盾的属性位。
    - shield_bits_enemy (int): 敌军是否有盾的属性位。
    - unit_type_bits (int): 单位类型的属性位数量。

    Returns:
    - tuple: 包含 ally 和 enemy 的状态属性名称列表 (ally_state_attr_names, enemy_state_attr_names)。
    """
    # 盟军的状态属性
    ally_state_attr_names = ["health", "energy/cooldown", "rel_x", "rel_y"]
    if shield_bits_ally > 0:
        ally_state_attr_names.append("shield")
    
    # 敌军的状态属性
    enemy_state_attr_names = ["health", "rel_x", "rel_y"]
    if shield_bits_enemy > 0:
        enemy_state_attr_names.append("shield")
    
    # 单位类型的属性
    if unit_type_bits > 0:
        bit_attr_names = ["type_{}".format(bit) for bit in range(unit_type_bits)]
        ally_state_attr_names.extend(bit_attr_names)
        enemy_state_attr_names.extend(bit_attr_names)
    
    return ally_state_attr_names, enemy_state_attr_names

def generate_unit_aliases(map_name, min_unit_type):
    """
    根据地图类型和 min_unit_type 动态初始化单位 ID。
    
    Parameters:
    - map_type (str): 地图类型。
    - min_unit_type (int): 最小单位类型的 ID。
    
    Returns:
    - dict: 包含单位名称及其对应 ID 的字典。
    """
    map_params = get_map_params(map_name)
    if not map_params or "support_info" not in map_params:
        raise ValueError(f"No support_info found for map {map_name}")

    # 提取单位偏移信息
    unit_offsets = map_params["support_info"].get("unit_relative_offsets", {})
    
    # 生成单位别名映射
    rl_unit_types = {unit_name: min_unit_type + offset for unit_name, offset in unit_offsets.items()}
    return rl_unit_types

def generate_unit_aliases_pure(map_name, min_unit_type):
    """
    生成对应非rl单位的字典
    Parameters:
    - map_type (str): 地图类型。
    - min_unit_type (int): 最小单位类型的 ID。
    
    Returns:
    - dict: 包含单位名称及其对应 ID 的字典。
    """
    map_params = get_map_params(map_name)
    if not map_params or "support_info" not in map_params:
        raise ValueError(f"No support_info found for map {map_name}")

    # 提取单位偏移信息
    unit_offsets = map_params["support_info"].get("unit_id_dict", {})
    
    # 生成单位别名映射
    rl_unit_types = {unit_name: unit_id for unit_name, unit_id in unit_offsets.items()}
    return rl_unit_types

def build_cooldown_map(rl_unit_types):
    """
    Builds a cooldown map by combining rl_unit_types and UNIT_COOLDOWNS.
    
    Parameters:
    - rl_unit_types (dict): A dictionary mapping unit names to dynamically assigned IDs.
    
    Returns:
    - dict: A dictionary that maps unit_type IDs to their cooldown values.
    """
    # 冷却时间默认字典
    UNIT_COOLDOWNS = {
        "marine_rl": 15,
        "marauder_rl": 25,
        "medivac_rl": 200,  # max energy
        "stalker_rl": 35,
        "zealot_rl": 22,
        "colossus_rl": 24,
        "hydralisk_rl": 10,
        "zergling_rl": 11,
        "baneling_rl": 1
    }
    
    # 构建冷却时间映射字典
    cooldown_map = {rl_unit_types[unit_name]: cooldown for unit_name, cooldown in UNIT_COOLDOWNS.items() if unit_name in rl_unit_types}
    
    return cooldown_map

def unit_max_shield(unit_type, rl_unit_types=None):
    """
    Returns the maximal shield for a given unit based on unit type, considering both fixed and dynamically assigned unit IDs.

    Parameters:
    - unit_type (int): The ID of the unit type.
    - rl_unit_types (dict, optional): A dictionary mapping unit names to their dynamically assigned IDs (if any).

    Returns:
    - int: The maximal shield for the given unit type.
    """
    # 默认单位护盾值字典
    UNIT_SHIELDS = {
        "stalker_rl": 80,
        "zealot_rl": 50,
        "colossus_rl": 150,
        "nexus_rl": 100,
        "warpgate_rl": 50,
        "warpPrism_rl": 40,
        "hydralisk_rl": 1,   # Protoss Hydralisk doesn't have a shield
        "pylon_rl": 200,
        "assimilator_rl": 450,
        "sentry_rl": 40,
    }

    # 基于固定 ID 的初始 shield_map
    shield_map = {
        74: 80,   # Stalker
        73: 50,   # Zealot
        4: 150,   # Colossus
        59: 1000, # Nexus
        133: 500, # WarpGate
        107: 1,   # Hydralisk doesn't have a shield
        60: 200,  # Pylon
        61: 450,  # Assimilator
        66: 100,  # Photon Cannon
        84: 20,   # Probe
        77: 40,   # Sentry
        81: 40,   # Warp Prism
        136: 40,  # Warp Prism Phasing mode
        82: 20,   # Observer
        62: 500,  # GateWay
        83: 100,  # Immortal
    }

    # 如果提供了 rl_unit_types，将其单位映射添加到 shield_map
    if rl_unit_types:
        dynamic_shield_map = {rl_unit_types[unit_name]: shield for unit_name, shield in UNIT_SHIELDS.items() if unit_name in rl_unit_types}
        shield_map.update(dynamic_shield_map)

    # 返回对应 unit_type 的护盾值，默认为 0（表示无护盾）
    return shield_map.get(unit_type, 0)

def load_and_export_map_params_mpyq(map_name, map_type):
    """
    Loads and parses an SC2Map file via mpyq lib to extract map parameters, then exports them to a JSON-compatible format.

    Parameters:
    - map_name (str): Name of the SC2 map file to load and parse.
    - map_type (str): Type or category of the map, used to determine specific parsing logic.

    Returns:
    - dict: A dictionary containing parsed map parameters (e.g., terrain details, objectives), 
            ready for JSON export.
    """
    starcraft2_path = os.getenv('SC2PATH')
    if starcraft2_path is None:
        raise EnvironmentError("SC2PATH 环境变量未设置")
    
    archive = mpyq.MPQArchive(os.path.join(starcraft2_path, "Maps", "Tactics_Maps", f"{map_name}.SC2Map"))
    script_data = archive.read_file('MapScript.galaxy')
    if isinstance(script_data, bytes):
        script_data = script_data.decode('utf-8', errors='replace')
    
    # Updated regular expression to account for the extra parameter between unit_name and player_id
    unit_pattern = re.compile(r'libNtve_gf_CreateUnitsWithDefaultFacing\((\d+),\s*"(\w+)",\s*\d+,\s*(\d+)')
    
    # Dictionary to store units by faction
    map_info = {
        "ally": {},
        "enemy": {}
    }
    
    # Map player IDs to faction names
    faction_map = {1: "ally", 2: "enemy"}
    
    # Parse units from script data and categorize them by player ID
    for count, unit_name, player_id in unit_pattern.findall(script_data):
        count = int(count)
        player_id = int(player_id)
        faction = faction_map.get(player_id, "unknown")
        
        if faction != "unknown":
            if unit_name not in map_info[faction]:
                map_info[faction][unit_name] = 0
            map_info[faction][unit_name] += count
    
    # 从 mpyq 提取并解析的 XML 数据（假设已正确解码为字符串）
    UnitDataXML = archive.read_file('Base.SC2Data\\GameData\\UnitData.xml')
    root = ET.fromstring(UnitDataXML.decode('utf-8'))

    # 提取单位的 abilities
    unit_abilities = {}
    for unit in root.findall('.//CUnit'):
        unit_id = unit.get('id')  # 获取单位的ID
        abilities = [abil.get('Link') for abil in unit.findall('.//AbilArray')]  # 提取 abilities
        
        # 只存储在地图信息中存在的单位
        if unit_id in map_info['ally'] or unit_id in map_info['enemy']:
            unit_abilities[unit_id] = abilities

    # 将能力信息整合到地图信息结构中
    for unit_type, unit_data in map_info.items():
        for unit_name, count in unit_data.items():
            # 如果该单位存在能力信息，则添加 abilities 字段
            if unit_name in unit_abilities:
                map_info[unit_type][unit_name] = {
                    'count': count,
                    'abilities': unit_abilities[unit_name]
                }
    
    # return json.dumps(map_info, indent=4)
    return map_info

# print(load_and_export_map_params_mpyq("wzsy", "wzsy"))

def load_and_export_map_params_pysc2(map_name, game_version, agent_race, bot_race, difficulty, window_size=(1024, 768)):
    """
    Launch the specified StarCraft II map, retrieve raw observation data, and export it as JSON.

    Parameters:
    - map_name (str): The name of the map to initialize.
    - game_version (str): StarCraft II game version.
    - agent_race (str): Race of the agent player.
    - bot_race (str): Race of the bot player.
    - difficulty (str): Difficulty level for the bot.
    - window_size (tuple): Window dimensions for the StarCraft II game.

    Returns:
    - dict: JSON serializable dictionary of raw data used for initializing SC2TacticsMap.
    """
    try:
        # Configure and start StarCraft II
        run_config = run_configs.get(version=game_version)
        _map = maps.get(map_name)
        
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        sc2_proc = run_config.start(window_size=window_size, want_rgb=False)
        controller = sc2_proc.controller

        atexit.register(lambda: controller.quit() if controller else None)
        atexit.register(lambda: sc2_proc.close() if sc2_proc else None)
        
        # Create game request
        create_game_request = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=run_config.map_data(_map.path)
            ),
            realtime=False,
            random_seed=None  # Adjust seed if required
        )
        create_game_request.player_setup.add(type=sc_pb.Participant)
        create_game_request.player_setup.add(
            type=sc_pb.Computer,
            race=bot_race,
            difficulty=difficulty
        )

        controller.create_game(create_game_request)

        # Join game as agent
        join_game_request = sc_pb.RequestJoinGame(
            race=agent_race,
            options=interface_options
        )
        controller.join_game(join_game_request)

        # Retrieve game info and map details
        game_info = controller.game_info()
        map_info = game_info.start_raw
        playable_area = map_info.playable_area
        map_play_area_min = playable_area.p0
        map_play_area_max = playable_area.p1

        # Extract raw data from observation
        observation = controller.observe()
        raw_data = observation.observation.raw_data

        # Convert raw data to JSON-compatible dictionary
        raw_data_dict = {
            "map_name": map_name,
            "playable_area": {
                "min_x": map_play_area_min.x,
                "min_y": map_play_area_min.y,
                "max_x": map_play_area_max.x,
                "max_y": map_play_area_max.y
            },
            "map_size": {
                "width": map_info.map_size.x,
                "height": map_info.map_size.y
            },
            "raw_data": {
                "units": [
                    {
                        "unit_type": unit.unit_type,
                        "owner": unit.owner,
                        "pos_x": unit.pos.x,
                        "pos_y": unit.pos.y,
                        "health": unit.health
                    } for unit in raw_data.units
                ]
            }
        }

        # Export raw data to JSON file
        with open(f"{map_name}_raw_data.json", "w") as json_file:
            json.dump(raw_data_dict, json_file, indent=4)

        print(f"Raw data for {map_name} exported to {map_name}_raw_data.json")
    except:
        raise RuntimeError("Failed to load and export map parameters")
    
    return json.dumps(raw_data_dict)
    
# print(load_and_export_map_params_pysc2("wzsy", "4.10.0", sc_common.Protoss, sc_common.Protoss, sc_pb.VeryHard))

def load_and_export_map_description(map_name, map_type):
    """
    Loads and parses an SC2Map file to extract detailed map and task descriptions, then exports them to a JSON-compatible format.

    Parameters:
    - map_name (str): Name of the SC2 map file to load and parse.
    - map_type (str): Type or category of the map, used to determine specific parsing logic.

    Returns:
    - dict: A dictionary containing detailed map descriptions (e.g., mission objectives, special conditions, terrain features), 
            ready for JSON export.
    """
    return None

def parse_attributes_and_abilities():
    pass

def get_map_type_base(map_name):
    map_type_dict = {
        "3m_te" : "marines",
        "8m_te" : "marines",
        "25m_te" : "marines",
        "5m_vs_6m_te" : "marines",
        "8m_vs_9m_te" : "marines",
        "10m_vs_11m_te" : "marines",
        "27m_vs_30m_te" : "marines",
        "1s1z_te" : "stalkers_and_zealots",
        "2s3z_te" : "stalkers_and_zealots",
        "3s5z_te" : "stalkers_and_zealots",
        "3s5z_vs_3s6z_te" : "stalkers_and_zealots",
        "3s_vs_3z_te" : "stalkers",
        "3s_vs_4z_te" : "stalkers",
        "3s_vs_5z_te" : "stalkers",
        "1c3s5z_te" : "colossi_stalkers_zealo",
        "2m_vs_1z_te" : "marines",
        "corridor_te" : "zealots",
        "6h_vs_8z_te" : "hydralisks",
        "2s_vs_1sc_te" : "stalkers",
        "so_many_baneling_te" : "zealots",
        "bane_vs_bane_te" : "bane",
        "2c_vs_64zg_te" : "colossus",
    }
    return map_type_dict[map_name]

def get_n_counts(map_name, map_type):
    map_info_mpyq = load_and_export_map_params_mpyq(map_name, map_type)
    n_agents = 0
    n_enemies = 0
    for a in map_info_mpyq["ally"].values():
        n_agents += a["count"]
    for e in map_info_mpyq["enemy"].values():
        n_enemies += e
    return n_agents, n_enemies

def print_map(grid, name="Placement"):
    import matplotlib.pyplot as plt
    plt.imshow(grid.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    plt.title(f'{name} Grid')
    plt.show()
