import json

def get_default_backbone_config():
    with open("four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/configs/backbone_default_configs.json") as file:
        args = json.load(file)
    return args

def get_default_w_constraints_config():
    with open("four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/configs/w_constraints_default_configs.json") as file:
        args = json.load(file)
    return args

