class AgentEvaluator:
    def __init__(self, window_size=100):
        self.window_size = window_size  # Длина скользящего окна для минимальных и максимальных значений
        self.best_score = -float('inf')
        self.best_agent_weights = None

        # Дек для хранения истории значений, используемой для нахождения min и max
        self.mean_rewards = deque(maxlen=window_size)
        self.std_rewards = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
    
    def update_min_max(self, metric_values):
        # Возвращает минимум и максимум в пределах текущего окна
        return min(metric_values), max(metric_values) if len(metric_values) > 0 else (None, None)

    def normalize(self, value, metric_values):
        min_val, max_val = self.update_min_max(metric_values)
        # Избегаем деления на 0, если min == max, нормализуем в диапазоне [0, 1]
        if min_val is None or max_val is None or max_val == min_val:
            return 0.5  # Среднее значение по умолчанию, если неизвестен диапазон
        return (value - min_val) / (max_val - min_val)
    
    def evaluate_and_save(self, agent, mean_reward, std_reward, loss):
        # Обновляем деки для отслеживания скользящих min и max
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)
        self.losses.append(loss)

        # Нормализация метрик по текущим диапазонам
        mean_reward_norm = self.normalize(mean_reward, self.mean_rewards)
        std_reward_norm = self.normalize(std_reward, self.std_rewards)
        loss_norm = self.normalize(loss, self.losses)

        # Расчет интегрального показателя
        score = (mean_reward_norm - std_reward_norm - loss_norm) / 3

        # Сравнение с текущим лучшим показателем и сохранение лучшего агента
        if score > self.best_score:
            self.best_score = score
            self.best_agent_weights = agent.get_weights()  # Сохраняем веса агента
            print("Новый лучший агент сохранён с показателями:",
                  f"Mean Reward: {mean_reward}, Std Reward: {std_reward}, Loss: {loss}")

