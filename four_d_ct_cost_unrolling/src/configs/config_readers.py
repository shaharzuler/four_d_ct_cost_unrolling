import json
import pkg_resources
import os

def _get_config(json_path:str) -> dict:
    package_name = __name__.split('.')[0]
    file_path = pkg_resources.resource_filename(package_name, json_path)
    with open(file_path) as file:
        args = json.load(file)
    return args

def get_default_backbone_config() -> dict:
    json_path = os.path.join('src','configs','backbone_default_configs.json')
    args = _get_config(json_path)
    return args

def get_default_w_constraints_config() -> dict:
    json_path = os.path.join('src','configs','w_constraints_default_configs.json')
    args = _get_config(json_path)
    return args



