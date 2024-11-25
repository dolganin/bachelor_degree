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
        learning_rate_forward = yml["learning_parameters"]["learning_rate_forward"]
        learning_rate_policy = yml["learning_parameters"]["learning_rate_policy"]
        learning_rate_value = yml["learning_parameters"]["learning_rate_value"]
        batch_size = yml["learning_parameters"]["batch_size"]
        replay_memory_size = yml["learning_parameters"]["replay_memory_size"]
        discount_factor = yml["learning_parameters"]["discount_factor"]
        train_epochs = yml["learning_parameters"]["train_epochs"]
        frame_repeat = yml["learning_parameters"]["frame_repeat"]
        learning_steps_per_epoch = yml["learning_parameters"]["learning_steps_per_epoch"]
        weight_decay = yml["learning_parameters"]["weight_decay"]
        test_episodes_per_epoch = yml["learning_parameters"]["test_episodes_per_epoch"]
        hidden_dim = yml["learning_parameters"]["hidden_dim"]
        channels = yml["learning_parameters"]["channels"]
        window_size = yml["learning_parameters"]["window_size"]
        evaluate_every = yml["learning_parameters"]["evaluate_every"]
        
        embedding_dim = yml["ppo_parameters"]["embedding_dim"]
        num_heads = yml["ppo_parameters"]["num_heads"]
        num_layers = yml["ppo_parameters"]["num_layers"]
        mlp_dim = yml["ppo_parameters"]["mlp_dim"]
        patch_size = yml["ppo_parameters"]["patch_size"]
        dropout_rate = yml["ppo_parameters"]["dropout_rate"]
        ex_loss = yml["ppo_parameters"]["ex_loss"]
        lambda_intrinsic = yml["ppo_parameters"]["lambda_intrinsic"]
        entropy_coef = yml["ppo_parameters"]["entropy_coef"]
        clip_epsilon = yml["ppo_parameters"]["clip_epsiolon"]


        resolution = yml["env_parameters"]["resolution"]
        fps = yml["env_parameters"]["fps"]

        cfg_path = yml["doom_cfg_path"]

    else:
        raise Exception("Path to .yaml is None")

    return  (learning_rate_forward, learning_rate_policy, learning_rate_value, batch_size, replay_memory_size, discount_factor, train_epochs,
            frame_repeat, learning_steps_per_epoch, cfg_path, resolution, test_episodes_per_epoch,
            weight_decay, lambda_intrinsic, entropy_coef, clip_epsilon, hidden_dim, channels, 
            patch_size, dropout_rate, embedding_dim, num_heads, num_layers, mlp_dim, ex_loss, 
            window_size, evaluate_every, fps)
