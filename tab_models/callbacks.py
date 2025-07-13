import os

from .model_wrapper import ModelCallback, ModelWrapper
import logging


class CheckpointCallback(ModelCallback):
    def __init__(self, path_to_checkpoints, n_iterations, base_name=""):
        self.path_to_checkpoints = path_to_checkpoints
        self.n_iterations = n_iterations
        self.base_name = base_name

        if not os.path.exists(self.path_to_checkpoints):
            os.mkdir(self.path_to_checkpoints)

    def after_iteration(self, iter_num: int, model: ModelWrapper):
        if (iter_num + 1) % self.n_iterations == 0:
            fpath = os.path.join(
                self.path_to_checkpoints, f"{self.base_name}_{iter_num}.bin"
            )
            logging.debug(f"Iteration {iter_num}, saving model to {fpath}")
            model.save(fpath)
