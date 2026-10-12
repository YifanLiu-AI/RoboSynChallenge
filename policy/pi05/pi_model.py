#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
from pathlib import Path

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, pytorch_device="cuda"):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pytorch_device = pytorch_device

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        norm_files = list(Path(specified_path).rglob("norm_stats.json"))
        if len(norm_files) != 1:
            raise ValueError(f"Expected one checkpoint norm_stats.json, found {len(norm_files)}")
        assets_id = str(norm_files[0].parent.relative_to(specified_path))
        norm_stats = _checkpoints.load_norm_stats(specified_path, assets_id)

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            norm_stats=norm_stats,
            pytorch_device=pytorch_device,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"\nsuccessfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        self.observation_window = {
            "observation/state": state,
            "observation/image": img_front,
            "observation/left_wrist_image": img_left,
            "observation/right_wrist_image": img_right,
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
