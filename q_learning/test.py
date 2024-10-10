from torch.nn import Module
from server_consumer.broker_kafka import publish_data

import numpy as np
from tqdm import trange
from preprocessing import preprocess


def test(game, writter, epoch, agent: Module, test_episodes_per_epoch, frame_repeat, resolution: tuple = (30, 45), actions: list = None) -> None:

    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for _ in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution=resolution)

            temporal_state = np.array(game.get_state().screen_buffer, dtype=np.uint8)
            new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)


            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)

            publish_data(array=new_state, epoch=epoch, loss=float("NaN"), mean_reward=np.array(test_scores).mean(), mode="Test")
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