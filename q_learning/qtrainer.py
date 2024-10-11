from base.trainer_base import TrainerRL
from tqdm import trange
from torch import tensor
from preprocessing import preprocess
import numpy as np
from server_consumer.broker_kafka import publish_data
import torch


class QTrainer(TrainerRL):
    
    def train(self, episode: int, steps_per_epoch: int = 1000):
        """
        Основной цикл обучения агента для одного эпизода.
        
        Args:
            episode (int): Номер текущего эпизода.
            actions (list, optional): Список возможных действий. По умолчанию None.
            frame_repeat (int, optional): Количество повторов действия в среде. По умолчанию 20.
            resolution (tuple, optional): Разрешение изображений. По умолчанию (30, 45).
            steps_per_epoch (int, optional): Количество шагов за эпизод. По умолчанию 2000.
            DEVICE (str, optional): Устройство для вычислений (CPU или GPU). По умолчанию "cuda:0".
        
        Returns:
            float: Суммарная награда за эпизод.
        """
        loss_lst = []
        self.env.new_episode()
        train_scores = []
        global_step = 0
        loss = tensor(0.0).to(self.device)
        total_reward = 0.0

        for _ in trange(self.steps_per_epoch, leave=False):
            state = preprocess(self.env.get_state().screen_buffer, resolution=self.resolution)
            
            temporal_state = np.array(self.env.get_state().screen_buffer, dtype=np.uint8)
            new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)

            action = self.agent.get_action(state)
            if self.actions is None:
                # Если actions не переданы, используем действие как есть
                selected_action = action
            else:
                selected_action = self.actions[action]
            
            reward = self.env.make_action(selected_action, self.frame_repeat)
            done = self.env.is_episode_finished()

            total_reward += reward

            if not done:
                next_state = preprocess(self.env.get_state().screen_buffer, resolution=self.resolution)
            else:
                next_state = np.zeros((1, self.resolution[0], self.resolution[1])).astype(np.float32)

            self.agent.append_memory(state, action, reward, next_state, done)

            if global_step > self.agent.batch_size:
                loss = self.agent.train()
                loss_lst.append(loss.cpu().detach().numpy())

            if done:
                train_scores.append(self.env.get_total_reward())
                self.env.new_episode()

            publish_data(
                array=new_state, 
                epoch=episode, 
                loss=loss.cpu().detach().numpy(), 
                mean_reward=np.array(train_scores).mean() if train_scores else 0.0, 
                mode="Train"
            )

            global_step += 1

        self.agent.update_target_net()
        train_scores = np.array(train_scores)
        
        return total_reward, np.array(loss_lst)  # Возвращаем суммарную награду для логирования

    def save_model(self, model_savefile: str):
        """
        Сохранение модели агента на диск.
        
        Args:
            model_savefile (str): Путь для сохранения модели.
        """
        torch.save(self.agent.q_net.state_dict(), model_savefile)
    
    def load_model(self, model_savefile: str):
         checkpoint = torch.load(model_savefile, location=self.device)
         self.agent.q_net.load_state_dict(checkpoint["model_state_dict"])
