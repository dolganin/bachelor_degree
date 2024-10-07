from time import time
from torch.nn import Module
from torch import save
from torch import tensor
from itertools import product
from broker_kafka import publish_numpy_array

import numpy as np
import torchvision.transforms as transforms
from tqdm import trange
from vizdoom import DoomGame
from typing import Tuple



def preprocess(img: list, resolution: tuple) -> tensor:
    """Down samples image to resolution"""

    transformer = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Resize(tuple(resolution))
            ])
    img = transformer(img)
    return img


def test(game, writter, epoch, agent: Module, test_episodes_per_epoch, frame_repeat, n: int = 5, resolution: tuple = (30, 45)) -> None:
    actions = [list(a) for a in product([0, 1], repeat=n)]

    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for _ in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution=resolution)
            temporal_state = np.array(game.get_state().screen_buffer, dtype=np.uint8)
            new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)

            publish_numpy_array(new_state)

            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )
    writter.add_scalar('Test score minimum', test_scores.min(), epoch)
    writter.add_scalar('Test score maximum', test_scores.max(), epoch)
    writter.add_scalar('Test score mean', test_scores.mean(), epoch)
    writter.add_scalar('Test score std', test_scores.std(), epoch)


def run(game, writter, agent: Module, actions: list, num_epochs: int = 10, frame_repeat: int = 20,\
         resolution: tuple = [30, 45], save_model: bool = False, test_episodes_per_epoch: int = 1000,\
              model_savefile: str = None, steps_per_epoch=2000) -> Tuple[DoomGame, Module]:
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer, resolution=resolution)

            state = preprocess(game.get_state().screen_buffer, resolution=resolution)
            temporal_state = np.array(game.get_state().screen_buffer, dtype=np.uint8)
            new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)

            publish_numpy_array(new_state)

            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer, resolution=resolution)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                loss = agent.train()
                writter.add_scalar('Loss', loss.cpu().detach().numpy(), epoch)

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1



        agent.update_target_net()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )
        writter.add_scalar('Train score minimum', train_scores.min(), epoch)
        writter.add_scalar('Train score maximum', train_scores.max(), epoch)
        writter.add_scalar('Train score mean', train_scores.mean(), epoch)
        writter.add_scalar('Train score std', train_scores.std(), epoch)
                

        test(game, writter, epoch, agent, test_episodes_per_epoch=test_episodes_per_epoch, frame_repeat=frame_repeat, resolution=resolution)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game