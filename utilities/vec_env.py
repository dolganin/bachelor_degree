import multiprocessing as mp
import cloudpickle
from typing import Callable, List

def _worker(remote, parent_remote, env_fn_pickle):
    parent_remote.close()
    env_fn: Callable = cloudpickle.loads(env_fn_pickle)
    env = env_fn()
    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break

            if cmd == 'step':
                o, r, d, info = env.step(data)
                if d:
                    o = env.reset()
                remote.send((o, r, d, info))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'spawn_wave':
                env.spawn_wave()
                remote.send(None)
            elif cmd == 'reset_stats':
                env.reset_stats()
                remote.send(None)
            elif cmd == 'reset_waves':
                env.reset_waves()
                remote.send(None)
            elif cmd == 'close':
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class SubprocVecEnv:
    """
    Параллельная обёртка для списка функций, создающих ConNquestEnv.
    Каждый env запускается в своём процессе.
    """
    def __init__(self, env_fns: List[Callable]):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_fns])
        self.processes = []
        for work_remote, remote, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(target=_worker, args=(work_remote, remote, cloudpickle.dumps(fn)))
            p.daemon = False
            p.start()
            work_remote.close()
            self.processes.append(p)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def spawn_wave(self):
        for remote in self.remotes:
            remote.send(('spawn_wave', None))
        for remote in self.remotes:
            remote.recv()

    def reset_stats(self):
        for remote in self.remotes:
            remote.send(('reset_stats', None))
        for remote in self.remotes:
            remote.recv()

    def reset_waves(self):
        for remote in self.remotes:
            remote.send(('reset_waves', None))
        for remote in self.remotes:
            remote.recv()

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception:
                pass
        for p in self.processes:
            p.join()
