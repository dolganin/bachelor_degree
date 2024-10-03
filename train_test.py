#!/usr/bin/env python3

from time import time
from torch.nn import Module
from itertools import product

import numpy as np
import torchvision.transforms as transforms
import torch
from tqdm import trange


# Uses GPU if available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def preprocess(img: list, resolution: tuple) -> torch.tensor:
    """Down samples image to resolution"""

    transformer = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Resize(tuple(resolution))
            ])
    img = transformer(img)
    return img


def test(game, agent: Module, test_episodes_per_epoch, frame_repeat, n: int = 5) -> None:
    actions = [list(a) for a in product([0, 1], repeat=n)]

    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
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


def run(game, agent: Module, actions: list, num_epochs: int = 10, frame_repeat: int = 20,\
         resolution: tuple = [30, 45], save_model: bool = False, test_episodes_per_epoch: int = 1000, model_savefile: str = None, steps_per_epoch=2000) -> Module:
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
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer, resolution=resolution)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

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

        test(game, agent, test_episodes_per_epoch=test_episodes_per_epoch, frame_repeat=frame_repeat)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

