import os



def get_checkpoints_path(model_dir):
    from ..string_table import checkpoints_dir_name, best_model_name
    return os.path.join(model_dir, checkpoints_dir_name, best_model_name)