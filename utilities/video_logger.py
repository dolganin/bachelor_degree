import cv2
import numpy as np

class VideoLogger:
    def __init__(self, filepath, fps=10):
        """
        Инициализация видеозаписи.
        
        Args:
            filepath: Путь для сохранения видео.
            fps: Частота кадров.
        """
        self.filepath = filepath
        self.fps = fps
        self.frames = []  # Список для хранения кадров

    def add_frame(self, frame):
        """Добавляет кадр в видеопоток."""
        # Проверяем, что кадр в правильном формате
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)  # Преобразование к uint8, если необходимо
        # Кадры подаются в формате BGR, поэтому не нужно делать преобразование
        self.frames.append(frame)

    def save(self):
        """Сохраняет видео на диск."""
        if not self.frames:
            print("Нет кадров для сохранения!")
            return

        height, width, _ = self.frames[0].shape
        
        # Используем mp4v кодек для формата mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.filepath, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(frame)
        
        out.release()
        print(f"Gameplay of agent is saved to {self.filepath}")

    def clear(self):
        """Очищает сохраненные кадры."""
        self.frames.clear()
