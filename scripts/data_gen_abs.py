#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2026 Huawei Technologies Co., Ltd
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    shape = [8, 128]

    rng = np.random.default_rng(seed=42)
    input_x = rng.uniform(-100, 100, shape).astype(np.float16)
    Path("./input").mkdir(parents=True, exist_ok=True)
    input_x.tofile("./input/input_x.bin")
