# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .controlnet_marigold_trainer import ControlNetMarigoldTrainer


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "ControlNetMarigoldTrainer": ControlNetMarigoldTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
