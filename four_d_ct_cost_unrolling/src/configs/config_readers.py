import json
import pkg_resources

def get_default_backbone_config() -> dict:
    package_name = __name__.split('.')[0]
    resource_path = '/'.join(('resources', 'four_d_ct_cost_unrolling','four_d_ct_cost_unrolling','src','configs','backbone_default_configs.json'))
    file_path = pkg_resources.resource_filename(package_name, resource_path)
    
    with open(file_path) as file: #"four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/configs/backbone_default_configs.json") as file:
        args = json.load(file)
    return args

def get_default_w_constraints_config() -> dict:
    package_name = __name__.split('.')[0]
    resource_path = '/'.join(('resources', 'four_d_ct_cost_unrolling','four_d_ct_cost_unrolling','src','configs','w_constraints_default_configs.json'))
    file_path = pkg_resources.resource_filename(package_name, resource_path)

    with open(file_path) as file: #open("four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/configs/w_constraints_default_configs.json") as file:
        args = json.load(file)
    return args

