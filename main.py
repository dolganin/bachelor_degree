from create_game import create_simple_game
from qagent import DQNAgent
from yaml_reader import YAMLParser
import itertools as it
from train_test import run, preprocess
import vizdoom as vzd
from time import sleep
import argparse
from torch.cuda import is_available
from yaml_reader import constants


def main() -> None:
    DEVICE = "cuda:0" if is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Bachelor Degree Script')
    parser.add_argument('-y', '--yaml', type=str, help='A path to yaml file', default="base_config")
    args = parser.parse_args()
    yaml = args.yaml

    config = YAMLParser(config=yaml).parse_config()

    learning_rate, batch_size, replay_memory_size,discount_factor, skip_learning, train_epochs, \
    frame_repeat, learning_steps_per_epoch, load_model, episodes_to_watch, \
    cfg_path, resolution, test_episodes_per_epoch, save_model, model_savefile = constants(config)

    # Initialize game and actions
    game = create_simple_game(config_file_path=cfg_path)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
        model_savefile=model_savefile,
        device=DEVICE
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
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