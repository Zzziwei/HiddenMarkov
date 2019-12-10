#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from tqdm import tqdm

from helpers import *
from part2 import get_emission_parameters
from part3 import get_transition_parameters, viterbi, predict_and_save


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.3f}"})
    DATA_ROOT = Path("data")
    DATASETS = ["SG", "CN", "EN", "AL"]
    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        predict_and_save(
            f"data/{dataset}/train",
            f"data/{dataset}/dev.in",
            f"data/{dataset}/dev.p4.out",
            k=7,
        )
