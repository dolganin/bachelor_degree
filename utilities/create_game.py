import vizdoom as vzd
import os
from termcolor import colored
from colorama import init

import os
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Инициализация colorama для поддержки цветного вывода
init(autoreset=True)

def print_debug_message(message, color="cyan"):
    line = "=" * 50
    print(f"\n{colored(line.center(80), color)}")
    print(colored(message.center(80), color))
    print(f"{colored(line.center(80), color)}\n")

def create_simple_game(config_file_path):
    print_debug_message("Initializing Doom...", color="green")
    
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    
    print_debug_message("Doom initialized.", color="green")

    return game
