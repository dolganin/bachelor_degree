import cv2
import numpy as np

class VideoLogger:
    def __init__(self, filepath, fps=30):
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
        if frame.shape[2] == 3:  # Если кадр в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Преобразуем RGB в BGR
        self.frames.append(frame)

    def save(self):
        """Сохраняет видео на диск."""
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.filepath, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(frame)
        
        out.release()

    def clear(self):
        """Очищает сохраненные кадры."""
        self.frames.clear()


