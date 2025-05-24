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

        # Критерий: просто максимальная средняя награда
        if mean_reward > self.best_score:
            self.best_score = mean_reward
            trainer.save_model(trainer.model_savefile)
            print(f"New best model saved — Mean Reward: {mean_reward:.2f}, Std: {std_reward:.2f}")
            trainer.video_logger.save()
        else:
            trainer.video_logger.clear()
