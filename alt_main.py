from create_game import create_simple_game
from q_learning.qagent import DQNAgent
from ppo_with_curiosity.ppo_agent import PPOAgent
from ppo_with_curiosity.ppo_trainer import PPOTrainer
from yaml_reader import YAMLParser
from itertools import product
from q_learning.qtrainer import QTrainer

from argparse import ArgumentParser
from torch.cuda import is_available
from yaml_reader import constants
from torch.utils.tensorboard import SummaryWriter


import warnings
import os
import logging
from video_logger import VideoLogger


def main() -> None:
    DEVICE = "cuda:0" if is_available() else "cpu"

    parser = ArgumentParser(description='Bachelor Degree Script')

    parser.add_argument('-y', '--yaml', type=str, help='A path to yaml file', default="base_config")
    parser.add_argument('-r', '--runname', type=str, help='A path to name of folder for run', default="runs/run_0")
    parser.add_argument('-w', '--weights', type=str, help='A path to weights of model', default=None)
    parser.add_argument('-d', '--debug', type=bool, help='A flag to debug mode', default=False)
    parser.add_argument('-t', '--test', type=bool, help='A flag to test mode', default=False)
    
    args = parser.parse_args()
    yaml = args.yaml
    runname = args.runname
    weights = args.weights
    debug = args.debug
    test = args.test

    if not debug:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Только ошибки
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    writter = SummaryWriter(log_dir=runname)

    config = YAMLParser(config=yaml).parse_config()

    learning_rate, batch_size, replay_memory_size,discount_factor, train_epochs, \
    frame_repeat, learning_steps_per_epoch, cfg_path, resolution, test_episodes_per_epoch, \
        save_model, weight_decay, load_model, out_video_file, lambda_intrinsic, entropy_coef,\
              clip_epsilon, hidden_dim = constants(config)

    # Initialize game and actions
    game = create_simple_game(config_file_path=cfg_path)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
        model_savefile=weights,
        device=DEVICE, 
        weight_decay=weight_decay,
    )
            # Инициализация агента
    agent = PPOAgent(
            action_size=n,
            memory_size=replay_memory_size,
            batch_size=batch_size,
            discount_factor=discount_factor,
            lr=learning_rate,
            device=DEVICE,
            model_savefile=weights,
            lambda_intrinsic=lambda_intrinsic,
            entropy_coef=entropy_coef,
            clip_epsilon=clip_epsilon,
            hidden_dim=hidden_dim
        )
        

    vlogger = VideoLogger(filepath=out_video_file)

    # trainer = QTrainer(agent=agent, 
    #                    env=game, 
    #                    tensor_logger=writter, 
    #                    device=DEVICE,
    #                    steps_per_epoch=learning_steps_per_epoch,
    #                    resolution=resolution,
    #                    frame_repeat=frame_repeat,
    #                    actions=actions,
    #                    test_episodes_per_epoch=test_episodes_per_epoch,
    #                    video_logger=vlogger
    #                    )
    trainer = PPOTrainer(agent=agent,env=game, 
                         tensor_logger=writter, 
                         device=DEVICE, 
                         steps_per_epoch=learning_steps_per_epoch, 
                         resolution=resolution, 
                         frame_repeat=frame_repeat, 
                         actions=actions, 
                         test_episodes_per_epoch=test_episodes_per_epoch, 
                         video_logger=vlogger)
    
    trainer.run(epochs=train_epochs)

    print("======================================")
    print("Training finished!")


if __name__ == "__main__":
   main()