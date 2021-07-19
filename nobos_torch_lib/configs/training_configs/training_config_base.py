import os

from nobos_commons.utils.file_helper import get_create_path

from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_base import LearningRateSchedulerBase
from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_dummy import LearningRateSchedulerDummy


class TrainingConfigBase(object):
    def __init__(self, model_name: str, model_dir: str):
        self.model_name: str = model_name
        self.model_dir: str = get_create_path(model_dir)

        self.num_epochs = 151
        self.checkpoint_epoch: int = 20
        self.batch_size = 128 # sadece comment kisminda gösterebilmek ve tensorboardda görsellik acisindan yazdim

        # Optimizer
        self.learning_rate: float = 0.01
        self.momentum: float = 0.9
        self.weight_decay: float = 5e-4

        # LR Scheduler
        self.learning_rate_scheduler: LearningRateSchedulerBase = LearningRateSchedulerDummy()

    def get_output_path(self, epoch: int,epoch_checkpoint :int):
        return os.path.join(self.model_dir, "{}_cp{}_checkpoint{}.pth".format(self.model_name, str(epoch).zfill(4),str(epoch_checkpoint).zfill(4)))
