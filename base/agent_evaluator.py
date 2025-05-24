from collections import deque

class AgentEvaluator:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.best_score = -float('inf')
        self.best_agent_weights = None
        self.mean_rewards = deque(maxlen=window_size)
        self.std_rewards = deque(maxlen=window_size)

    def evaluate_and_save(self, trainer, mean_reward, std_reward):
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)

        if mean_reward > self.best_score:
            self.best_score = mean_reward
            trainer.save_model(trainer.model_savefile)
            print(f"Сохранена новая лучшая модель — Средняя награда: {mean_reward:.2f}, Отклонение: {std_reward:.2f}")
            if trainer.video_logger:
                trainer.video_logger.save()
                print("Видео сохранено.")
        else:
            if trainer.video_logger:
                trainer.video_logger.clear()
                print("Видео очищено.")
