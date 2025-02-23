from collections import deque

class AgentEvaluator:
    def __init__(self, window_size=100):
        """
        Инициализация класса AgentEvaluator.

        Args:
            window_size (int): Размер скользящего окна для отслеживания минимальных и максимальных значений
                               средних вознаграждений, стандартного отклонения и потерь.
        """
        self.window_size = window_size  # Длина скользящего окна для минимальных и максимальных значений
        self.best_score = -float('inf')  # Инициализация лучшего показателя
        self.best_agent_weights = None  # Хранение лучших весов агента

        # Деки для хранения истории значений в пределах текущего окна
        self.mean_rewards = deque(maxlen=window_size)
        self.std_rewards = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
    
    def update_min_max(self, metric_values):
        """
        Вычисляет минимальное и максимальное значения метрики в пределах текущего скользящего окна.

        Args:
            metric_values (deque): Дек с отслеживаемыми значениями метрики.

        Returns:
            (float, float): Минимальное и максимальное значения или (None, None), если дек пуст.
        """
        return min(metric_values), max(metric_values) if len(metric_values) > 0 else (None, None)

    def normalize(self, value, metric_values):
        """
        Нормализует значение метрики в диапазоне [0, 1] в зависимости от текущего минимума и максимума
        в пределах скользящего окна.

        Args:
            value (float): Значение метрики для нормализации.
            metric_values (deque): Дек с историей значений метрики для вычисления min и max.

        Returns:
            float: Нормализованное значение метрики от 0 до 1.
        """
        min_val, max_val = self.update_min_max(metric_values)
        # Избегаем деления на 0, если min == max, нормализуем в диапазоне [0, 1]
        if min_val is None or max_val is None or max_val == min_val:
            return 0.5  # Среднее значение по умолчанию, если диапазон неизвестен
        return (value - min_val) / (max_val - min_val)
    
    def evaluate_and_save(self, trainer, mean_reward, std_reward):
        """
        Оценивает и сохраняет лучшие веса агента на основе заданных метрик.

        Args:
            trainer (object): Объект тренера для управления обучением и сохранением модели.
            mean_reward (float): Среднее вознаграждение.
            std_reward (float): Стандартное отклонение вознаграждения.
            loss (float): Потеря агента.

        Метод добавляет метрики в историю, нормализует их, рассчитывает интегральный показатель,
        и сохраняет лучшие веса агента, если интегральный показатель улучшился.
        """
        # Обновляем деки для отслеживания скользящих min и max
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)

        # Нормализация метрик по текущим диапазонам
        mean_reward_norm = self.normalize(mean_reward, self.mean_rewards)
        std_reward_norm = self.normalize(std_reward, self.std_rewards)

        # Расчет интегрального показателя
        score = (mean_reward_norm - std_reward_norm) / (2 * std_reward_norm)

        # Сравнение с текущим лучшим показателем и сохранение лучшего агента
        if score > self.best_score:
            self.best_score = score
            self.best_agent_weights = trainer.save_model(path=trainer.model_savefile)  # Сохраняем веса агента
            print("New best ensemble saved with parameters:",
                  f"Mean Reward: {round(mean_reward, 2)}, Std Reward: {round(std_reward, 2)}")
            trainer.video_logger.save()  # Сохраняем видео сессии
        else:
            trainer.video_logger.clear()  # Очистка, если агент не улучшил показатель
