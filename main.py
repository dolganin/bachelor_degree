from termcolor import colored
from colorama import init
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

# Инициализация colorama
init(autoreset=True)

def print_debug_message(message, color="cyan"):
    line = "=" * 50
    print(f"\n{colored(line.center(80), color)}")
    print(colored(message.center(80), color))
    print(f"{colored(line.center(80), color)}\n")

def print_parameters_table(parameters: dict):
    """Форматированный вывод параметров в виде таблицы"""
    print("\n" + colored("=" * 60, "yellow"))
    print(colored("Parameters".center(60), "yellow"))
    print(colored("=" * 60, "yellow"))
    for key, value in parameters.items():
        print(f"{key:<20}: {value}")
    print(colored("=" * 60, "yellow"))

def main() -> None:
    # Устройство
    #DEVICE = 'cuda:0' if is_available() else 'cpu'
    DEVICE = 'cpu'
    print_debug_message(f"Device selected for training: {DEVICE}", "green")

    parser = ArgumentParser(description='Bachelor Degree Script')
    parser.add_argument('-y', '--yaml', type=str, help='Path to yaml file', default="ppobase_config")
    parser.add_argument('-r', '--runname', type=str, help='Folder name for run', default="runs/run_0")
    parser.add_argument('-w', '--weights', type=str, help='Path to model weights', default=None)
    parser.add_argument('-d', '--debug', type=bool, help='Debug mode flag', default=False)
    parser.add_argument('-t', '--test', type=bool, help='Test mode flag', default=False)
    
    args = parser.parse_args()
    yaml, runname, weights, debug, test = args.yaml, args.runname, args.weights, args.debug, args.test
    
    # Параметры для вывода в таблице
    parameters = {
        "YAML Config Path": yaml,
        "Run Name": runname,
        "Weights Path": weights,
        "Debug Mode": debug,
        "Test Mode": test
    }
    
    print_debug_message("Starting with the following parameters:", "yellow")
    print_parameters_table(parameters)

    if not debug:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        print_debug_message("Debug mode is off. Only errors will be displayed.", "red")

    writter = SummaryWriter(log_dir=runname)
    print_debug_message(f"TensorBoard writer initialized at: {runname}", "green")

    # Загрузка конфигурации из YAML
    try:
        config = YAMLParser(config=yaml).parse_config()
        print_debug_message("Config successfully parsed.", "green")
    except Exception as e:
        print_debug_message(f"Error loading config: {e}", "red")
        return

    # Распаковка параметров из конфигурации
    try:
        (learning_rate, batch_size, replay_memory_size, discount_factor, train_epochs, 
        frame_repeat, learning_steps_per_epoch, cfg_path, resolution, test_episodes_per_epoch, 
        save_model, weight_decay, load_model, out_video_file, lambda_intrinsic, entropy_coef, 
        clip_epsilon, hidden_dim) = constants(config)
        
        # Параметры конфигурации
        config_parameters = {
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Memory Size": replay_memory_size,
            "Discount Factor": discount_factor,
            "Train Epochs": train_epochs,
            "Frame Repeat": frame_repeat,
            "Steps per Epoch": learning_steps_per_epoch,
            "Resolution": resolution,
            "Episodes per Epoch": test_episodes_per_epoch
        }
        
        print_debug_message("Parameters extracted from config:", "yellow")
        print_parameters_table(config_parameters)
    except Exception as e:
        print_debug_message(f"Error extracting parameters: {e}", "red")
        return

    # Инициализация игры и действий
    try:
        game = create_simple_game(config_file_path=cfg_path)
        n = game.get_available_buttons_size()
        actions = [list(a) for a in product([0, 1], repeat=n)]
        print_debug_message(f"Game initialized with {n} available actions.", "green")
    except Exception as e:
        print_debug_message(f"Error initializing game: {e}", "red")
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
        print_debug_message("Agent successfully initialized.", "green")
    except Exception as e:
        print_debug_message(f"Error initializing agent: {e}", "red")
        return

    # Инициализация вспомогательных объектов
    try:
        vlogger = VideoLogger(filepath=out_video_file)
        evaluator = AgentEvaluator(window_size=100)
        print_debug_message("Video Logger and Agent Evaluator initialized.", "green")
    except Exception as e:
        print_debug_message(f"Error initializing logger/evaluator: {e}", "red")
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
        print_debug_message("Trainer successfully initialized.", "green")
    except Exception as e:
        print_debug_message(f"Error initializing trainer: {e}", "red")
        return

    # Запуск обучения
    print_debug_message("Starting training...", "yellow")
    trainer.run(epochs=train_epochs, evaluate_every=learning_steps_per_epoch)
    print_debug_message("Training finished!", "green")

    print_debug_message("Script finished execution.", "blue")


if __name__ == "__main__":
    main()
