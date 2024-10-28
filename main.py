from utilities.create_game import create_simple_game
from ppo_with_curiosity.ppo_agent import PPOAgent
from ppo_with_curiosity.ppo_trainer import PPOTrainer
from base.agent_evaluator import AgentEvaluator
from utilities.video_logger import VideoLogger
from utilities.yaml_reader import YAMLParser, constants

import warnings
import os
from argparse import ArgumentParser
from itertools import product

from torch.cuda import is_available
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    # Выбор устройства
    #DEVICE = 'cuda:0' if is_available() else 'cpu'
    DEVICE = 'cpu'
    print(f"Device selected for training: {DEVICE}")

    parser = ArgumentParser(description='Bachelor Degree Script')

    parser.add_argument('-y', '--yaml', type=str, help='A path to yaml file', default="ppobase_config")
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

    print("Starting with the following parameters:")
    print(f"YAML Config Path: {yaml}")
    print(f"Run Name: {runname}")
    print(f"Weights Path: {weights}")
    print(f"Debug Mode: {debug}")
    print(f"Test Mode: {test}")

    if not debug:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Только ошибки
        print("Debug mode is off. Only errors will be displayed.")

    writter = SummaryWriter(log_dir=runname)
    print(f"TensorBoard writer initialized at: {runname}")

    # Загрузка конфигурации из YAML
    try:
        config = YAMLParser(config=yaml).parse_config()
        print("Config successfully parsed.")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Распаковка параметров из конфигурации
    try:
        (learning_rate, batch_size, replay_memory_size, discount_factor, train_epochs, 
        frame_repeat, learning_steps_per_epoch, cfg_path, resolution, test_episodes_per_epoch, 
        save_model, weight_decay, load_model, out_video_file, lambda_intrinsic, entropy_coef, 
        clip_epsilon, hidden_dim) = constants(config)
        print("Parameters extracted from config:")
        print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Memory Size: {replay_memory_size}")
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return

    # Инициализация игры и действий
    try:
        game = create_simple_game(config_file_path=cfg_path)
        n = game.get_available_buttons_size()
        actions = [list(a) for a in product([0, 1], repeat=n)]
        print(f"Game initialized with {n} available actions.")
    except Exception as e:
        print(f"Error initializing game: {e}")
        return

    # Инициализация агента
    try:
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
        print("Agent successfully initialized.")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    # Инициализация вспомогательных объектов
    try:
        vlogger = VideoLogger(filepath=out_video_file)
        evaluator = AgentEvaluator(window_size=100)
        print("Video Logger and Agent Evaluator initialized.")
    except Exception as e:
        print(f"Error initializing logger/evaluator: {e}")
        return

    # Инициализация тренера
    try:
        trainer = PPOTrainer(
            agent=agent,
            env=game,
            tensor_logger=writter,
            device=DEVICE,
            steps_per_epoch=learning_steps_per_epoch,
            resolution=resolution,
            frame_repeat=frame_repeat,
            actions=actions,
            test_episodes_per_epoch=test_episodes_per_epoch,
            video_logger=vlogger,
            agent_evaluator=evaluator
        )
        print("Trainer successfully initialized.")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return

    # Запуск обучения
    try:
        print("Starting training...")
        trainer.run(epochs=train_epochs, evaluate_every=learning_steps_per_epoch)
        print("Training finished!")
    except Exception as e:
        print(f"Error during training: {e}")

    print("======================================")
    print("Script finished execution.")


if __name__ == "__main__":
    main()
