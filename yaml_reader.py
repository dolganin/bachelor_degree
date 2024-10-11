import yaml
from typing import List


class YAMLParser():
    def __init__(self, path_base: str ="configs/", config: str = None) -> None:
        if config:
            ext = ".yaml"
            self.basename = path_base + config + ext
        else:
            raise Exception("Path is None")

        self.parse_config()
    
    def parse_config(self) -> dict:
         # Open the configuration file in read mode
        with open(self.basename, 'r') as f:
            # Read the data from the file and convert it from YAML format to Python objects
            config = yaml.safe_load(f)
        # Save the configuration to an object attribute
        return config

def constants(yml: dict = None) -> List:
    if yml:
        learning_rate = yml["learning_parameters"]["learning_rate"]
        batch_size = yml["learning_parameters"]["batch_size"]
        replay_memory_size = yml["learning_parameters"]["replay_memory_size"]
        discount_factor = yml["learning_parameters"]["discount_factor"]
        train_epochs = yml["learning_parameters"]["train_epochs"]
        frame_repeat = yml["learning_parameters"]["frame_repeat"]
        learning_steps_per_epoch = yml["learning_parameters"]["learning_steps_per_epoch"]
        weight_decay = yml["learning_parameters"]["weight_decay"]
        test_episodes_per_epoch = yml["learning_parameters"]["test_episodes_per_epoch"]
        lambda_intrinsic = yml["learning_parameters"]["lambda_intrinsic"]
        entropy_coef = yml["learning_parameters"]["entropy_coef"]
        clip_epsiolon = yml["learning_parameters"]["clip_epsiolon"]
        hidden_dim = yml["learning_parameters"]["hidden_dim"]


        save_model = yml["meta_parameters"]["save_model"]
        load_model = yml["meta_parameters"]["load_model"]
        out_video = yml["meta_parameters"]["out_video_file"]
        

        resolution = yml["env_parameters"]["resolution"]
        
        cfg_path = yml["doom_cfg_path"]

    else:
        raise Exception("Path to .yaml is None")
    
    return learning_rate, batch_size, replay_memory_size,discount_factor, train_epochs, \
    frame_repeat, learning_steps_per_epoch, cfg_path, resolution, test_episodes_per_epoch, \
        save_model, weight_decay, load_model, out_video, lambda_intrinsic, entropy_coef, clip_epsilon, hidden_dim