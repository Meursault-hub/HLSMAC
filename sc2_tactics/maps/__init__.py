from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.sc2_tactics.maps import sc2_tactics_maps

def get_map_params(map_name):
    map_param_registry = sc2_tactics_maps.get_tactics_map_registry()
    map_param = map_param_registry.get(map_name, {}).copy()
    if not map_param:
        raise ValueError(f"Map parameters for '{map_name}' not found in map_param_registry.")
    return map_param

#map_param_registry = sc2_thirty_six_tactics_maps.get_tactics_map_registry()
#return map_param_registry[map_name]