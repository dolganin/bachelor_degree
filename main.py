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
from datetime import datetime
import signal
from time import sleep

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

def timeout_handler(signum, frame):
    raise TimeoutError

# Функция для получения названия видео с таймаутом
def get_video_filename(time):
    if os.getenv('IN_DOCKER'):
        print(colored("Running in Docker, using default filename.", "yellow"))
        date_str = datetime.now().strftime("%d_%m")
        name = f"ppo_c_{date_str}"
        return f"server_consumer/static/gameplay/{name}.webm"
    else:
        print(colored(f"Enter the name for the video file (you have {time} seconds):", "yellow"))
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time)  # Set timer for 30 seconds
        try:
            name = input("Filename: ")
            signal.alarm(0)  # Reset timer if input is successful
        except TimeoutError:
            print(colored("Time is up! Generating default filename.", "red"))
            date_str = datetime.now().strftime("%d_%m")
            name = f"ppo_c_{date_str}"
        return f"server_consumer/static/gameplay/{name}.webm"

def get_default_name(prefix):
    date_str = datetime.now().strftime("%d_%m")
    return f"{prefix}_{date_str}"



def get_latest_model_path(directory="weights"):
    """Возвращает путь к самой последней модели, соответствующей шаблону 'ppo_dd_mm.pth'."""
    
    # Получаем все файлы, которые начинаются с `ppo_` и заканчиваются на `.pth`
    model_files = [
        f for f in os.listdir(directory)
        if f.startswith("ppo_") and f.endswith(".pth")
    ]

    if not model_files:
        print_debug_message("No model files found matching pattern 'ppo_dd_mm.pth'.", "red")
        return None

    # Список для хранения файлов и их дат
    valid_models = []
    for f in model_files:
        try:
            # Извлекаем дату, предполагая, что она находится сразу после `ppo_`
            date_part = "_".join(f.rsplit("_", 2)[1:]).split(".")[0]
            model_date = datetime.strptime(date_part, "%d_%m")
            valid_models.append((model_date, f))
        except (IndexError, ValueError):
            # Пропускаем файлы с неверным форматом имени или даты
            print_debug_message(f"Skipping file with invalid date format: {f}", "red")
            continue

    if not valid_models:
        print_debug_message("No valid model files found with date format 'dd_mm'.", "red")
        return None

    # Сортируем по дате и выбираем последний по дате файл
    latest_model_file = max(valid_models, key=lambda x: x[0])[1]
    return os.path.join(directory, latest_model_file)




def main() -> None:
    # Устройство
    DEVICE = 'cuda:0' if is_available() else 'cpu'
    #DEVICE = 'cpu'
    print_debug_message(f"Device selected for training: {DEVICE}", "green")

    parser = ArgumentParser(description='Bachelor Degree Script')
    parser.add_argument('-y', '--yaml', type=str, help='Path to yaml file', default="ppobase_config")
    parser.add_argument('-r', '--runname', type=str, help='Folder name for run', default=None)
    parser.add_argument('-w', '--weights', nargs='?', const=True, help='Path to model weights (optional)')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode flag')
    parser.add_argument('-t', '--test', action='store_true', help='Test mode flag')
    parser.add_argument('--timer', type=int, help="Time for timer in awaiting", default=5)
    
    args = parser.parse_args()


    yaml, runname, weights, debug, test, timer = (args.yaml, args.runname, args.weights, args.debug, 
                                                  args.test, args.timer)

    # Устанавливаем имена по умолчанию, если они не заданы
    runname = f"runs/{args.runname}" if args.runname else f"runs/{get_default_name('run')}"
     # Логика определения пути к весам модели
    if weights is True:
        # Флаг задан, но путь не указан: загружаем последнюю модель
        weights = get_latest_model_path()
        if weights:
            print_debug_message(f"Loading the latest model: {weights}", "green")
        else:
            print_debug_message("No saved models found. Starting new training.", "green")
            weights = f"weights/{get_default_name('ppo')}"
    elif isinstance(weights, str):
        # Задан конкретный путь
        weights = f"weights/{weights}"
    else:
        # Флаг не задан: создаем новую модель
        weights = f"weights/{get_default_name('model')}"

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

    from torch.utils.tensorboard import SummaryWriter
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
            weight_decay, lambda_intrinsic, entropy_coef, clip_epsilon, hidden_dim, channels, 
            patch_size, dropout_rate, embedding_dim, num_heads, num_layers, mlp_dim, 
            ex_loss, window_size, evaluate_every, fps) = constants(config)
        
        # Параметры конфигурации
# Параметры конфигурации
        config_parameters = {
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Memory Size": replay_memory_size,
            "Discount Factor": discount_factor,
            "Train Epochs": train_epochs,
            "Frame Repeat": frame_repeat,
            "Learning Steps per Epoch": learning_steps_per_epoch,
            "Configuration Path": cfg_path,
            "Resolution": resolution,
            "Test Episodes per Epoch": test_episodes_per_epoch,
            "Weight Decay": weight_decay,
            "Lambda Intrinsic": lambda_intrinsic,
            "Entropy Coefficient": entropy_coef,
            "Clip Epsilon": clip_epsilon,
            "Hidden Dimension": hidden_dim,
            "Channels": channels,
            "Patch Size": patch_size,
            "Dropout Rate": dropout_rate,
            "Embedding Dimension": embedding_dim,
            "Number of Heads": num_heads,
            "Number of Layers": num_layers,
            "MLP Dimension": mlp_dim,
            "Extra Loss": ex_loss,
            "Window Size": window_size
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
            lambda_intrinsic=lambda_intrinsic,
            entropy_coef=entropy_coef,
            clip_epsilon=clip_epsilon,
            hidden_dim=hidden_dim,
            screen_resolution=resolution,       # Передаем разрешение экрана
            channels=channels,                         # RGB-каналы
            patch_size=patch_size,                      # Размер патча, можно изменить по необходимости
            dropout_rate=dropout_rate,                   # Дефолтное значение, можно изменить по конфигурации
            embedding_dim=embedding_dim,                  # Задает размерность эмбеддинга
            num_heads=num_heads,                        # Задает количество голов в multi-head attention
            num_layers=num_layers,                       # Задает количество слоев в трансформере
            mlp_dim=mlp_dim,                        # Размерность MLP в трансформере
            ex_loss=ex_loss                         # Задает коэффициент внешней потери
)
        print_debug_message("Agent successfully initialized.", "green")
    except Exception as e:
        print_debug_message(f"Error initializing agent: {e}", "red")
        return

    # Инициализация вспомогательных объектов
    try:
        out_video_file = get_video_filename(timer)  # Используем новую функцию для получения имени файла
        vlogger = VideoLogger(filepath=out_video_file, fps=fps)
        evaluator = AgentEvaluator(window_size=window_size)
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
            agent_evaluator=evaluator,
            model_savefile=weights
        )
        print_debug_message("Trainer successfully initialized.", "green")

        if os.path.isfile(weights):
            trainer.load_model(weights)
            print_debug_message(f"Model weights loaded from {weights}", "green")
        else:
            print_debug_message("Starting new training without preloaded weights.", "green")
    except Exception as e:
        print_debug_message(f"Error initializing trainer: {e}", "red")
        return

    

    # Запуск обучения
    print_debug_message("Starting training...", "yellow")
    if not debug:
        try:
            trainer.run(epochs=train_epochs, evaluate_every=evaluate_every)
        except Exception as e:
            trainer.save_model(weights)
            print_debug_message(f"Error training the agent: {e}", "red")
        print_debug_message("Training finished!", "green")
    if debug:
        trainer.run(epochs=train_epochs, evaluate_every=evaluate_every)

    print_debug_message("Script finished execution.", "blue")


if __name__ == "__main__":
    main()
