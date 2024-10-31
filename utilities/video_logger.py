import cv2
import numpy as np

class VideoLogger:
    def __init__(self, filepath, fps=5):
        """
        Initializes video logging.

        Args:
            filepath: Path for saving the video.
            fps: Frames per second.
        """
        self.filepath = filepath
        self.fps = fps
        self.frames = []  # List to store frames

    def add_frame(self, frame):
        """Adds a frame to the video stream."""
        # Check if the frame is in the correct format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8 if necessary
        # Frames are expected in BGR format, no need to convert
        self.frames.append(frame)

    def save(self):
        """Saves the video to disk."""
        if not self.frames:
            print("No frames to save!")
            return

        height, width, _ = self.frames[0].shape
        
        # Using XVID codec for AVI format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.filepath, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(frame)
        
        out.release()
        print(f"Gameplay of agent is saved to {self.filepath}")

    def clear(self):
        """Clears saved frames."""
        self.frames.clear()
