import cv2
import numpy as np
import imageio

class VideoLogger:
    def __init__(self, filepath, fps=5):
        """  
        Initializes video logging.

        Args:
            filepath: Path for saving the video.
            fps: Frames per second.
        """
        self.filepath = filepath
        self.bestfile_path = ('/'.join(filepath.split("/")[0:-1])) + "/best_video.webm"
        self.fps = fps
        self.frames = []  # List to store frames

    def add_frame(self, frame):
        """Adds a frame to the video stream."""
        # Convert frame to uint8 and RGB format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.frames.append(frame_rgb)

    def save(self):
        """Saves the video to disk."""
        if not self.frames:
            print("No frames to save!")
            return

        # Use imageio to write frames as WebM video
        imageio.mimsave(self.filepath, self.frames, fps=self.fps, codec='vp8')
        imageio.mimsave(self.bestfile_path, self.frames, fps=self.fps, codec='vp8')
        
        print(f"Gameplay of agent is saved to {self.filepath} and the best video was updated")
        self.frames.clear()

    def clear(self):
        """Clears saved frames."""
        self.frames.clear()
