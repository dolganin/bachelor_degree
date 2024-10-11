from create_game import create_simple_game
from q_learning.qagent import DQNAgent
from yaml_reader import YAMLParser
from itertools import product
from q_learning.train import train

import vizdoom as vzd
from time import sleep
from argparse import ArgumentParser
from torch.cuda import is_available
from yaml_reader import constants
from torch.utils.tensorboard import SummaryWriter

import warnings
import os
import logging
from preprocessing import preprocess


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

    learning_rate, batch_size, replay_memory_size,discount_factor, skip_learning, train_epochs, \
    frame_repeat, learning_steps_per_epoch, load_model, episodes_to_watch, \
    cfg_path, resolution, test_episodes_per_epoch, save_model, model_savefile, weight_decay = constants(config)

    if weights:
        model_savefile = weights

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
        model_savefile=model_savefile,
        device=DEVICE, 
        weight_decay=weight_decay,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = train(
            game,
            writter,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
            resolution=resolution,
            save_model=save_model,
            test_episodes_per_epoch=test_episodes_per_epoch,
            model_savefile=model_savefile,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    if test:
        game.close()
        game.set_window_visible(True)
        game.set_mode(vzd.Mode.ASYNC_PLAYER)
        game.init()
        for _ in range(episodes_to_watch):
            game.new_episode()
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer, resolution=resolution)
                best_action_index = agent.get_action(state)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                game.set_action(actions[best_action_index])
                for _ in range(frame_repeat):
                    game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = game.get_total_reward()
            print("Total score: ", score)

    


if __name__ == "__main__":
   main()