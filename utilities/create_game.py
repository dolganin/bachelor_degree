import vizdoom as vzd
import os

def center_text(text):
    """Возвращает текст, центрированный относительно ширины терминала."""
    terminal_width = os.get_terminal_size().columns
    # Считаем отступ, чтобы текст был в центре
    padding = max((terminal_width - len(text)) // 2, 0)
    return ' ' * padding + text

def create_simple_game(config_file_path):
    print(center_text("Initializing doom..."))
    
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    
    print(center_text("=" * 60))
    print(center_text("Doom initialized."))
    print(center_text("=" * 60))

    return game
