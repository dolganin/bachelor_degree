from abc import ABC, abstractmethod

class TrainerRL(ABC):
    def __init__(self, env, agent, config):
        """
        Инициализация тренера.

        Args:
            env: Среда для обучения агента.
            agent: Объект агента, реализующий логику действий и обновления.
            config: Словарь или объект с конфигурациями для тренера.
        """
        self.env = env  # Среда, с которой агент взаимодействует
        self.agent = agent  # Агент, выполняющий действия и обучающийся
        self.config = config  # Конфигурация тренировки (гиперпараметры)
        self.current_step = 0  # Шаг обучения
        self.total_rewards = []  # Для хранения суммарных наград по эпизодам

    @abstractmethod
    def train(self, num_episodes: int):
        """
        Основной цикл обучения агента в среде.

        Args:
            num_episodes: Количество эпизодов для обучения.
        """
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int):
        """
        Оценка агента без обновления весов.

        Args:
            num_episodes: Количество эпизодов для оценки.
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        """
        Сохранение текущей модели агента на диск.

        Args:
            filepath: Путь для сохранения модели.
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """
        Загрузка модели агента с диска.

        Args:
            filepath: Путь для загрузки модели.
        """
        pass

    def log_metrics(self, episode: int, reward: float, loss: float = None):
        """
        Логгирование метрик обучения, таких как награды и потери.

        Args:
            episode: Текущий номер эпизода.
            reward: Суммарная награда за эпизод.
            loss: Потери модели (если есть).
        """
        print(f"Episode {episode}: Reward = {reward}, Loss = {loss}")

    def run(self, num_episodes: int, evaluate_every: int = 10):
        """
        Полный процесс обучения с периодической оценкой.

        Args:
            num_episodes: Количество эпизодов для обучения.
            evaluate_every: Частота оценок после определенного количества эпизодов.
        """
        for episode in range(1, num_episodes + 1):
            # Запуск тренировки на одном эпизоде
            reward = self.train(episode)
            self.total_rewards.append(reward)
            
            # Периодическая оценка
            if episode % evaluate_every == 0:
                self.evaluate(1)
            
            # Логгирование результатов
            self.log_metrics(episode, reward)

