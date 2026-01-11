from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib

class SC2TacticsMap(lib.Map):
    directory = "Tactics_Maps"
    download = "https://github.com/xxxxhong/star36/tree/main/maps/Tactics_Maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0

map_param_registry = {
    "2s3z_te": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 10,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
            },
            "unit_relative_offsets": {
                "stalker_rl": 0,
                "zealot_rl": 1
            }
        },
    },

    "3s5zvs3s6z_te": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 10,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
            },
            "unit_relative_offsets": {
                "stalker_rl": 0,
                "zealot_rl": 1
            }
        },
    },

    # 本地图用于测试计策1：瞒天过海 - Crossing the Sea Under Camouflage
    "mtgh": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "mtgh",
    },

    # 本地图用于测试计策2：围魏救赵 - Relieving the State of Zhao by Besieging the State of Wei
    "wwjz_te": {
        "n_agents": 8,
        "n_enemies": 15,
        "limit": 200,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "wwjz",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 10,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
            },
            "unit_relative_offsets": {
                "nexus_rl": 0,
                "stalker_rl": 1,
                "warpPrism_rl": 2,
                "zealot_rl": 3,
            },
            "unit_id_dict": {
                "nexus": 59,
                "zealot": 73,
            },
        },
    },

    # 本地图用于测试计策3：借刀杀人 - Killing Someone with a Borrowed Knife
    "jdsr_te": {  
        "n_agents": 5,
        "n_enemies": 4,
        "n_agents_max": 9, # controlled
        "limit": 200,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 4,
        "map_type": "jdsr",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 10,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "NeuralParasite": 249,
            },
            "unit_relative_offsets": {
            },
            "unit_id_dict": {
                "roach": 110,
                "infestor": 111,
                "stalker": 74,
                "colossus": 4,
            },
        },
    },

    "jdsr_test_te": {  
        "n_agents": 7,
        "n_enemies": 5,
        "n_agents_max": 12, # controlled, agents + enemies
        "limit": 200,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 4,
        "map_type": "jdsr",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 11,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "NeuralParasite": 249,
            },
            "unit_relative_offsets": {
            },
            "unit_id_dict": {
                "roach": 110,
                "infestor": 111,
                "stalker": 74,
                "colossus": 4,
            },
        },
    },

    # 本地图用于测试计策4：以逸待劳 - Waiting at One’s Ease for the Exhausted Enemy
    "yydl": {  
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 110,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "yydl",
    },

    # 本地图用于测试计策5：趁火打劫 - Plundering a Burning House
    "chdj": {  
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "chdj",
    },

    # 本地图用于测试计策6：声东击西 - Making a Feint to the East and Attacking in the West
    # 局部拷贝自地图 antiga_shipyard
    # 灵感来自战报 http://example.com
    # 在创建完地图之后可以用接口打印出这样的信息，便于核对作静态信息检验
    "sdjx_te": {  
        "n_agents": 18,
        "n_enemies": 17,
        "limit": 150,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 6,
        "map_type": "sdjx",
        "n_madivac": 4,
        "support_info": {
            "num_medivac": 4,
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            # "n_actions": 10,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "heal": 386
            },
            "unit_relative_offsets": {
                "marine_rl": 0,
                "medivac_rl": 1
            },
            "unit_id_dict": {
                "marine": 48,
                "medivac": 54,
            },
        },
    },

    # 本地图用于测试计策7：无中生有 - Creating Something Out of Nothing
    "wzsy_te": {  
        "n_agents": 6,
        "n_enemies": 10,
        "n_agents_max": 12,
        "limit": 300,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 4,
        "map_type": "wzsy",
        "support_info": {
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "Hallucination": 158,
            },
            "unit_relative_offsets": {
                "sentry_rl": 0,
                "stalker_rl": 1
            },
            "unit_id_dict": {
                "stalker": 74,
                "sentry": 77,
            },
        },
    },

    # 本地图用于测试计策8：暗度陈仓 - Advancing Secretly by an Unknown Path
    "adcc_pure_te": {  
        "n_agents": 4,
        "n_enemies": 6,
        "n_agents_max": 10,
        "limit": 200,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 4,
        "map_type": "adcc",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 12,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "BurrowDown": 0,
                "BurrowUp": 0,
                "egg": 0,
            },
            "unit_relative_offsets": {
                "hatchery": 0,
                "infestor": 1,
                "egg": 2,
                "infestedTerran": 3,
            },
            "unit_id_dict": {
                "hatchery": 86,
                "infestor": 111,
                "infestorBurrowed": 127,
                "infestedEgg": 150,
                "infestedTerran": 7,
            },
        },
    },

    "adcc_te": {  
        "n_agents": 17,
        "n_enemies": 5,
        "limit": 200,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "adcc",
        "support_info": {
            "n_actions_no_attack": 7,
            "n_actions_move": 4,
            "n_actions": 12,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "BurrowDown": 1390,
                "BurrowUp": 1392,
            },
            "unit_id_dict": {
                "hatchery": 86,
                "zergling": 105,
                "zerglingBurrowed": 119,
            },
        },
    },

    # 本地图用于测试计策9：隔岸观火 - Watching a Fire from the Other Side of the River
    "gagh": {  
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 130,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "gagh",
    },

    # 本地图用于测试计策10：笑里藏刀 - Covering the Dagger with a Smile
    "xlcd": {  
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 3,
        "map_type": "xlcd",
    },

    # 本地图用于测试计策11：李代桃僵 - Palming Off Substitute for the Real Thing
    "ldtj": {  
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 110,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "ldtj",
    },

    # 本地图用于测试计策12：顺手牵羊 - Picking Up Something in Passing
    "sshq": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 1,
        "map_type": "sshq",
    },

    # 本地图用于测试计策13：打草惊蛇 - Beating the Grass to Frighten the Snake
    "dcjs": {  
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 110,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "dcjs",
    },

    # 本地图用于测试计策14：借尸还魂 - Resurrecting a Dead Soul by Borrowing a Corpse
    "jshh": {  
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 120,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "jshh",
    },

    # 本地图用于测试计策15：调虎离山 - Luring the Tiger Out of His Den
    # 局部拷贝自地图 Tal_darim_Altar
    # 灵感来自战报 http://example.com
    "dhls_te": {  
        "n_agents": 16,
        "n_enemies": 11,
        "limit": 200,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 5,
        "map_type": "dhls",
        # "action_configs": {
        "support_info": {
            "n_actions_no_attack": 5,
            "n_actions_move": 4,
            # "n_actions": 10,
            "actions": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "NydusCanalLoad": 1437,
                "NydusCanalUnload": 1438
            },
            "unit_relative_offsets": {
                "hatchery_rl": 0,
                "nydusNetwork_rl": 1,
                "nydusCanal_rl": 2,
                "roach_rl": 3
            },
            "unit_id_dict": {
                "hatchery": 86,
                "nydusNetwork": 95,
                "nydusCanal": 142,
                "roach": 110,
                "zergling": 105,
            }
        }
    },

    # 本地图用于测试计策16：欲擒故纵 - Letting the Enemy Off in Order to Catch Him
    "yqgz_te": {  
        "n_agents": 24,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "yqgz",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 14,
            "actions": {
                "move": 16,
                "attack": 23,
                "stop": 4,
            },
            "unit_id_dict": {
                "zergling": 105,
            }
        },
    },

    # 本地图用于测试计策17：抛砖引玉 - Giving the Enemy Something to Induce Him to Lose More Valuable Things
    "pzyy": {  
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "pzyy",
    },

    # 本地图用于测试计策18：擒贼擒王 - Capturing the Ringleader First in Order to Capture All the Followers
    "qzqw": {  
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 110,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "qzqw",
    },

    # 本地图用于测试计策19：釜底抽薪 - Extracting the Firewood from Under the Cauldron
    "fdcx": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 1,
        "map_type": "fdcx",
    },

    # 本地图用于测试计策20：混水摸鱼 - Muddling the Water to Catch the Fish; Fishing in Troubled Waters
    "hsmq": {  
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 110,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "hsmq",
    },

    # 本地图用于测试计策21：金蝉脱壳 - Slipping Away by Casting Off a Cloak; Getting Away Like the Cicada Sloughing Its Skin
    "jctq_te": {  
        "n_agents": 4,
        "n_enemies": 9,      # stalker, sentry, observer
        "n_agents_alive": 1, # 至少要活这么多个友方单位
        "limit": 45,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "jctq",
        "support_info": {
            "n_actions_no_attack": 7,
            "n_actions_move": 4,
            "n_actions": 16,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "BurrowDown": 1390,
                "BurrowUp": 1388,
            },
            "unit_id_dict": {
                "roach": 110,
                "roachBurrowed": 118,
            }
        },
    },

    # 本地图用于测试计策22：关门捉贼 - Catching the Thief by Closing/Blocking His Escape Route
    "gmzz_te": {  
        "n_agents": 8,
        "n_enemies": 11,
        "limit": 200,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 3,
        "map_type": "gmzz",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 12,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "DepotLower": 556,
                "DepotRaise": 558,
            },
            "unit_relative_offsets": {
            },
            "unit_id_dict": {
                "marine": 48,
                "Depot": 19,
                "DepotLowered": 47,
            },
        },
    },

    # 本地图用于测试计策23：远交近攻 - Befriending the Distant Enemy While Attacking a Nearby Enemy
    "yjjg": {  
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 130,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "yjjg",
    },

    # 本地图用于测试计策24：假道伐虢 - Attacking the Enemy by Passing Through a Common Neighbor
    "jdfg": {  
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "jdfg",
    },

    # 本地图用于测试计策25：偷梁换柱 - Stealing the Beams and Pillars and Replacing Them with Rotten Timbers
    "tlhz_te": {  
        "n_agents": 4,
        "n_enemies": 2,
        "n_agents_max": 16, # 1 drone/H/E, 3 (larva -> egg), 6 zergling, 6 broodling
        "limit": 300,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 7,
        "map_type": "tlhz",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 9,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "BuildHatchery": 1152,
                "BuildChamber": 1156,
                "CancelBuilding": 314,
                "TrainZerg": 1343,
            },
            "unit_relative_offsets": {
            },
            "unit_id_dict": {
                "hatchery": 86,
                "evolutionChamber": 90,
                "drone": 104,
                "zergling": 105,
                "broodling": 289,
                "larva": 151,
                "egg": 103,
            },
        },
    },

    "tlhz_test_te": {  
        "n_agents": 1,
        "n_enemies": 2,
        "n_agents_max": 13, # 1 drone/H/E, 6 zergling, 6 broodling
        "n_enemies_max": 3,
        "limit": 300,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 5,
        "map_type": "tlhz",
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 9,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "BuildHatchery": 1152,
                "BuildChamber": 1156,
                "CancelBuilding": 1182,
            },
            "unit_relative_offsets": {
            },
            "unit_id_dict": {
                "hatchery": 86,
                "evolutionChamber": 90,
                "drone": 104,
                "zergling": 105,
                "broodling": 289,
                "larva": 151,
            },
        },
    },

    # 本地图用于测试计策26：指桑骂槐 - Reviling/Abusing the Locust Tree While Pointing to the Mulberry
    "zsmh": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 1,
        "map_type": "zsmh",
    },

    # 本地图用于测试计策27：假痴不癫 - Feigning Madness Without Becoming Insane
    "jcbd": {  
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 110,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "jcbd",
    },
    
    # 本地图用于测试计策28：上屋抽梯 - Removing the Ladder After the Enemy Has Climbed Up the Roof
    "swct_te": { 
        "n_agents": 5,
        "n_enemies": 11,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "swct",
        "support_info": {
            "n_actions_no_attack": 7,
            "n_actions_move": 4,
            "n_actions": 18,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "WarpPrismLoad": 911,
                "WarpPrismUnload": 913,
                "ForceField": 1526,
            },
            "unit_id_dict": {
                "sentry": 77,
                "warpPrism": 81,
            }
        },
    },

    # 本地图用于测试计策29：树上开花 - Putting Artificial Flowers on Trees
    "sskh": {  
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "sskh",
    },

    # 本地图用于测试计策30：反客为主 - Turning from the Guest into the Host
    "fkwz_te": {  
        "n_agents": 5,
        "n_enemies": 7,
        "n_agents_max": 9, # 2 pylon, 2 warpgate, 4 zealot, 1 warpPrism
        "n_enemies_max": 13, # 1 gateway, 1 stalker + 6 stalker, 3 cannon, 2 pylon
        "limit": 300,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 5,
        "map_type": "fkwz",
        "resource_start": 400,
        "resource_ssh": 000,
        "unit_killed_ssh": 4,
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 13,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "warpgateTrain": 1413,
                "WarpPrismLoad": 911,
                "WarpPrismUnload": 913,
                "PhasingMode": 1528,
                "TransportMode": 1530,
            },
            "unit_relative_offsets": {
            },
            "unit_id_dict": {
                "warpgate": 133,
                "pylon": 60,
                "zealot": 73,
                "warpPrism": 81,
                "warpPrismPhasing": 136,
            },
        },
    },

    # 本地图用于测试计策31：美人计 - Using Seductive Women to Corrupt the Enemy
    "mrj": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 3,
        "map_type": "mrj",
    },

    # 本地图用于测试计策32：空城计 - Presenting a Bold Front to Conceal Unpreparedness
    "kcj": {  
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 110,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "kcj",
    },

    # 本地图用于测试计策33：反间计 - Sowing Discord Among the Enemy
    "fjj": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 1,
        "map_type": "fjj",
    },

    # 本地图用于测试计策34：苦肉计 - Deceiving the Enemy by Torturing One’s Own Man
    "krj": {  
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 110,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "krj",
    },

    # 本地图用于测试计策35：连环计 - Coordinating One Stratagem with Another
    "lhj": {  
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 120,
        "a_race": "P",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "lhj",
    },

    # 本地图用于测试计策36：走为上计 - Decamping Being the Best; Running Away as the Best Choice
    "zwsj": {  
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "zwsj",
    },
}

def get_tactics_map_registry():
    return map_param_registry

for name in map_param_registry.keys():
    globals()[name] = type(name, (SC2TacticsMap,), dict(filename=name))
    
"""
{
    "sdjx": {
        "n_agents": 18,
        "n_enemies": 17,
        "limit": 150,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 6,
        "map_type": "sdjx",
        "unit_summary": {
            "Marine_RL": {"quantity": 14, "player_id": 1, "position_ids": [336436468]},
            "Medivac_RL": {"quantity": 4, "player_id": 1, "position_ids": [336436468, 1017891732]},
            "Assimilator": {"quantity": 2, "player_id": 2, "position_ids": [1977438158, 250759362]},
            // Add other units as parsed
        },
        "support_info": {
            "num_medivac": 4,
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 10,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "heal": 386
            },
            "unit_relative_offsets": {
                "marine_id": 0,
                "medivac_id": 1,
                // other units as parsed
            }
        },
        "position_map": {
            "336436468": {"x": 12, "y": 34, "z": 0},  // Example of position mappings
            "1017891732": {"x": 13, "y": 35, "z": 0},
            // Add other position mappings
        }
    }
}
"""
"""
{
  "units": {
    "marine_id": {"cooldown": 15, "shield": 0},
    "marauder_id": {"cooldown": 25, "shield": 0},
    "medivac_id": {"cooldown": 200, "shield": 0},
    "stalker_id": {"cooldown": 35, "shield": 80},
    "zealot_id": {"cooldown": 22, "shield": 50},
    "colossus_id": {"cooldown": 24, "shield": 150},
    "hydralisk_id": {"cooldown": 10, "shield": 1},
    "zergling_id": {"cooldown": 11, "shield": 0},
    "baneling_id": {"cooldown": 1, "shield": 0}
  },
  "static_units": {
    "74": {"cooldown": 35, "shield": 80},
    "73": {"cooldown": 22, "shield": 50},
    "4": {"cooldown": 24, "shield": 150},
    "59": {"cooldown": 0, "shield": 100},
    "133": {"cooldown": 0, "shield": 50},
    "107": {"cooldown": 10, "shield": 1},
    "60": {"cooldown": 0, "shield": 200},
    "61": {"cooldown": 0, "shield": 450}
  }
}
"""
