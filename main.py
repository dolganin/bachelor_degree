from termcolor import colored
from colorama import init
import torch
from ppo.ppo_agent import PPOAgent
from ppo.ppo_trainer import PPOTrainer
from base.agent_evaluator import AgentEvaluator
from utilities.video_logger import VideoLogger
from utilities.yaml_reader import YAMLParser, constants
from utilities.create_game import create_connquest_env
from utilities.vec_env import SubprocVecEnv
import sys

import warnings
import os
from argparse import ArgumentParser
from itertools import product
from torch.cuda import is_available
from datetime import datetime
import signal
import wandb

init(autoreset=True)

def print_debug_message(message, color="cyan"):
    line = "=" * 50
    print(f"\n{colored(line.center(80), color)}")
    print(colored(message.center(80), color))
    print(f"{colored(line.center(80), color)}\n")



def print_parameters_table(parameters: dict):
    print("\n" + colored("=" * 60, "yellow"))
    print(colored("Параметры запуска".center(60), "yellow"))
    print(colored("=" * 60, "yellow"))
    for key, value in parameters.items():
        print(f"{key:<25}: {value}")
    print(colored("=" * 60, "yellow"))

def timeout_handler(signum, frame):
    raise TimeoutError

def get_video_filename(timeout):
    if os.getenv('IN_DOCKER'):
        print(colored("Работа в Docker, используется имя по умолчанию.", "yellow"))
        date_str = datetime.now().strftime("%d_%m")
        name = f"ppo_{date_str}"
        return f"server_consumer/static/gameplay/{name}.webm"
    else:
        print(colored(f"Введите имя файла видео (у вас {timeout} сек):", "yellow"))
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            name = input("Имя файла: ")
            signal.alarm(0)
        except TimeoutError:
            print(colored("Время вышло! Сгенерировано имя по умолчанию.", "red"))
            date_str = datetime.now().strftime("%d_%m")
            name = f"ppo_{date_str}"
        return f"server_consumer/static/gameplay/{name}.webm"

def get_default_name(prefix):
    date_str = datetime.now().strftime("%d_%m")
    return f"{prefix}_{date_str}"

def get_latest_model_path(directory="weights"):
    model_files = [f for f in os.listdir(directory) if f.startswith("ppo_") and f.endswith(".pth")]
    if not model_files:
        print_debug_message("Нет подходящих моделей по шаблону 'ppo_dd_mm.pth'.", "red")
        return None
    valid_models = []
    for f in model_files:
        try:
            date_part = "_".join(f.rsplit("_", 2)[1:]).split(".")[0]
            model_date = datetime.strptime(date_part, "%d_%m")
            valid_models.append((model_date, f))
        except (IndexError, ValueError):
            print_debug_message(f"Пропущен файл с некорректной датой: {f}", "red")
            continue
    if not valid_models:
        print_debug_message("Нет корректных моделей с датой 'dd_mm'.", "red")
        return None
    latest_model_file = max(valid_models, key=lambda x: x[0])[1]
    return os.path.join(directory, latest_model_file)

def make_env():
    return create_connquest_env("coNNquest/configs/conquest.yaml")


def main():
    DEVICE = 'cuda:0' if is_available() else 'cpu'
    print_debug_message(f"Устройство: {DEVICE}", "green")

    parser = ArgumentParser(description='Скрипт дипломного проекта')
    parser.add_argument('-y', '--yaml', type=str, help='Путь к YAML-конфигу', default="ppobase_config")
    parser.add_argument('-r', '--runname', type=str, help='Имя запуска (папка)', default=None)
    parser.add_argument('-w', '--weights', nargs='?', const=True, help='Путь к весам модели (необязательно)')
    parser.add_argument('-d', '--debug', action='store_true', help='Режим отладки')
    parser.add_argument('-t', '--test', action='store_true', help='Режим тестирования')
    parser.add_argument('--timer', type=int, help="Таймер для ввода имени видео", default=5)
    args = parser.parse_args()

    yaml, runname, weights, debug, test, timer = args.yaml, args.runname, args.weights, args.debug, args.test, args.timer
    runname = f"runs/{runname}" if runname else f"runs/{get_default_name('run')}"
    if weights is True:
        weights = get_latest_model_path()
        if weights:
            print_debug_message(f"Загружается последняя модель: {weights}", "green")
        else:
            print_debug_message("Сохранённых моделей не найдено. Новое обучение.", "green")
            weights = f"weights/{get_default_name('ppo')}"
    elif isinstance(weights, str):
        weights = f"weights/{weights}"
    else:
        weights = f"weights/{get_default_name('model')}"

    parameters = {
        "Путь к YAML": yaml,
        "Имя запуска": runname,
        "Файл весов": weights,
        "Режим отладки": debug,
        "Режим теста": test
    }
    print_debug_message("Начальные параметры:", "yellow")
    print_parameters_table(parameters)

    if not debug:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        print_debug_message("Режим отладки выключен. Будут отображаться только ошибки.", "red")

    try:
        config = YAMLParser(config=yaml).parse_config()
        print_debug_message("Конфигурация успешно загружена.", "green")
    except Exception as e:
        print_debug_message(f"Ошибка загрузки конфигурации: {e}", "red")
        return

    try:
        (learning_rate_policy, learning_rate_value, batch_size, discount_factor,
            frame_repeat, learning_steps, resolution, test_episodes_per_epoch,
            weight_decay, lambda_intrinsic, entropy_coef, clip_epsilon, hidden_dim, channels, 
            patch_size, dropout_rate, embedding_dim, num_heads, num_layers, mlp_dim, ex_loss, 
            window_size, evaluate_every, fps, validate_fold, epoch, num_envs) = constants(config)
        config_parameters = {
            "Learning Rate Policy": learning_rate_policy,
            "Learning Rate Value": learning_rate_value,
            "Batch Size": batch_size,
            "Learning Steps": learning_steps,
            "Discount Factor": discount_factor,
            "PPO epoch": epoch,
            "Frame Repeat": frame_repeat,
            "Validate Fold": validate_fold,
            "Resolution": resolution,
            "Entropy Coef": entropy_coef,
            "Clip Epsilon": clip_epsilon,
            "Embedding Dim": embedding_dim,
            "Num Heads": num_heads,
            "Num Layers": num_layers,
            "MLP Dim": mlp_dim,
        }
        print_debug_message("Параметры из YAML:", "yellow")
        print_parameters_table(config_parameters)
    except Exception as e:
        print_debug_message(f"Ошибка чтения параметров: {e}", "red")
        return

    wandb.init(project="DITH", entity="dolganin", config=config_parameters, name=runname)
    print_debug_message(f"WandB инициализирован: {runname}", "green")

    try:
        env = create_connquest_env("coNNquest/configs/conquest.yaml")
        n = env.game.get_available_buttons_size()
        env.close()
        print_debug_message(f"Среда ConNquest загружена. Кнопок: {n}. Пробую создать векторную среду", "green")
    except Exception as e:
        print_debug_message(f"Ошибка инициализации среды: {e}", "red")
        return
    try:
        env_fns = [make_env for _ in range(num_envs)]
        # 5. Создаём векторную среду
        envs = SubprocVecEnv(env_fns)
        print_debug_message(f"Вектор сред загружен в размере: {num_envs}")
    except Exception as e:
        print_debug_message(f"Ошибка инициализации вектора: {e}")
        return

    try:
        agent = PPOAgent(
            action_size=n,
            batch_size=batch_size,
            discount_factor=discount_factor,
            lr_value=learning_rate_value,
            lr_policy=learning_rate_policy,
            device=DEVICE,
            entropy_coef=entropy_coef,
            clip_epsilon=clip_epsilon,
            screen_resolution=resolution,
            channels=channels,
            patch_size=patch_size,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim
        )
        print_debug_message("Агент успешно инициализирован.", "green")
    except Exception as e:
        print_debug_message(f"Ошибка инициализации агента: {e}", "red")
        return

    try:
        out_video_file = get_video_filename(timer)
        vlogger = VideoLogger(filepath=out_video_file, fps=fps)
        evaluator = AgentEvaluator(window_size=window_size)
        print_debug_message("Логгер и оценщик готовы.", "green")
    except Exception as e:
        print_debug_message(f"Ошибка инициализации логгера/оценщика: {e}", "red")
        return

    try:
        trainer = PPOTrainer(
            agent=agent,
            env=envs,
            wandb_logger=wandb,
            device=DEVICE,
            resolution=resolution,
            frame_repeat=frame_repeat,
            video_logger=vlogger,
            agent_evaluator=evaluator,
            model_savefile=weights,
            ppo_epochs=epoch,
            n_envs=num_envs
        )
        print_debug_message("Trainer успешно инициализирован.", "green")
        if os.path.isfile(weights):
            trainer.load_model(weights)
            print_debug_message(f"Загружены веса из: {weights}", "green")
        else:
            print_debug_message("Обучение начнётся с нуля.", "green")
    except Exception as e:
        print_debug_message(f"Ошибка создания Trainer: {e}", "red")
        return
    def signal_handler(sig, frame):
        print("\n[MAIN] Получен сигнал прерывания. Завершаю работу...")
        try:
            trainer.env.close()  # закрыть векторную среду
            torch.cuda.empty_cache()  # очистить CUDA
            print("[MAIN] Среда и память GPU очищены.")
        except Exception:
            pass
        sys.exit(0)
    print_debug_message("Старт обучения...", "yellow")
    signal.signal(signal.SIGINT, signal_handler)
    try:
        trainer.run(total_steps=learning_steps, validate_every_split=validate_fold, batch_size=batch_size)
        print_debug_message("Обучение завершено.", "green")
    except Exception as e:
        trainer.save_model(weights)
        print_debug_message(f"Ошибка обучения: {e}", "red")
    finally:
        # Гарантируем очистку
        try:
            trainer.env.close()
        except Exception:
            pass
        torch.cuda.empty_cache()
    return  

if __name__ == "__main__":
    main()
