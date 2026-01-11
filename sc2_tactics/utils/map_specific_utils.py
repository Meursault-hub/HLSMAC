from abc import ABC, abstractmethod
from smac.env.sc2_tactics.utils import common_utils  # 导入通用功能模块
from smac.env.sc2_tactics.maps import get_map_params


# 地图工具的抽象基类
class MapSpecificUtils(ABC):
    def __init__(self, env):
        self.env = env
        
    @abstractmethod
    def set_action_params(self):
        pass
    
class SDJXMapUtils(MapSpecificUtils):
    def set_action_params(self):
        print(get_map_params(self.env.map_name))
        return 

class DHLSMapUtils(MapSpecificUtils):
    def set_action_params(self):
        print(self.env)
        return
    
class MapSpecificUtilsRegistry:
    _map_utils_classes = {
        ("sdjx","sdjx"): SDJXMapUtils,
        ("dhls","dhls"): DHLSMapUtils,
    }

    @staticmethod
    def create_map_utils(map_type, map_name, env):
        map_utils_class = MapSpecificUtilsRegistry._map_utils_classes.get((map_type, map_name))
        if map_utils_class:
            return map_utils_class(env)
        else:
            raise ValueError(f"No MapUtils found for map_type {map_type}")
