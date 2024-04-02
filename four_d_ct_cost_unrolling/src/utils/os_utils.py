import os
import pkg_resources
import json


def get_checkpoints_path(model_dir):
    from ..string_table import checkpoints_dir_name, best_model_name
    return os.path.join(model_dir, checkpoints_dir_name, best_model_name)

def get_default_checkpoints_path():
    package_name = __name__.split('.')[0]
    file_path = pkg_resources.resource_filename(package_name, get_checkpoints_path('src'))
    return file_path


def write_dict_to_json(path, content_dict):
    with open(path, 'w') as f:
        f.write(json.dumps(content_dict, indent=4) )