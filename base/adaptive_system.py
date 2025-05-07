import numpy as np
import onnxruntime as ort
from abc import ABC

class AdaptiveEnsembleAgent(ABC):
    def __init__(self, stormtrooper_model_path, pacifist_model_path, balanced_model_path,
                 d_min=0.0, d_max=10.0, v_min=0.0, v_max=5.0, p_min=0.0, p_max=50.0):
        self.stormtrooper_sess = ort.InferenceSession(stormtrooper_model_path)
        self.pacifist_sess = ort.InferenceSession(pacifist_model_path)
        self.balanced_sess = ort.InferenceSession(balanced_model_path)

        self.stormtrooper_input = self.stormtrooper_sess.get_inputs()[0].name
        self.pacifist_input = self.pacifist_sess.get_inputs()[0].name
        self.balanced_input = self.balanced_sess.get_inputs()[0].name

        self.deaths = 0
        self.total_distance = 0.0
        self.total_damage = 0.0
        self.total_time = 0.0
        self.episodes = 0

        self.d_min, self.d_max = d_min, d_max
        self.v_min, self.v_max = v_min, v_max
        self.p_min, self.p_max = p_min, p_max

    def log_step(self, moved_distance, damage, alive=True):
        self.total_distance += moved_distance
        self.total_damage += damage
        if alive:
            self.total_time += 1

    def log_death(self):
        self.deaths += 1
        self.episodes += 1

    def _normalize(self, x, x_min, x_max):
        return (x - x_min) / (x_max - x_min + 1e-8)

    def _compute_weights(self):
        d = self.deaths / max(1, self.episodes)
        v = self.total_distance / max(1, self.total_time)
        p = self.total_damage / max(1, self.total_time)

        d_tilde = self._normalize(d, self.d_min, self.d_max)
        v_tilde = self._normalize(v, self.v_min, self.v_max)
        p_tilde = self._normalize(p, self.p_min, self.p_max)

        S_p = d_tilde + (1 - v_tilde) + (1 - p_tilde)
        S_sr = (1 - d_tilde) + v_tilde + p_tilde
        S_s = 1 - abs(S_sr - S_p) / 3

        scores = np.array([S_p, S_s, S_sr])
        exp_scores = np.exp(scores)
        weights = exp_scores / np.sum(exp_scores)
        return weights  # w_p, w_s, w_sr

    def _onnx_inference(self, session, input_name, state):
        inputs = {input_name: state.astype(np.float32)}
        outputs = session.run(None, inputs)
        return outputs[0]

    def get_action(self, state: np.ndarray):
        w_p, w_s, w_sr = self._compute_weights()

        prob_p = self._onnx_inference(self.pacifist_sess, self.pacifist_input, state)
        prob_s = self._onnx_inference(self.balanced_sess, self.balanced_input, state)
        prob_sr = self._onnx_inference(self.stormtrooper_sess, self.stormtrooper_input, state)

        final_probs = w_p * np.array(prob_p) + w_s * np.array(prob_s) + w_sr * np.array(prob_sr)
        selected_action_idx = int(np.argmax(final_probs))
        return selected_action_idx
