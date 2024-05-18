# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import sys
from collections import defaultdict
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile

from pesq import pesq
from pystoi import stoi


def extract_id(filename):
    return int(filename.split("_")[-1][:-4])


def evaluate_dns(path_clean, path_target):
    result = defaultdict(int)
    posix_list_clean = list(path_clean.rglob("*.wav"))
    posix_list_target = list(path_target.rglob("*.wav"))

    assert len(posix_list_clean) == len(posix_list_target)

    data_clean = sorted(map(str, posix_list_clean), key=extract_id)
    data_target = sorted(map(str, posix_list_target), key=extract_id)

    for i in tqdm(range(len(data_target))):
        rate, clean = wavfile.read(data_clean[i])
        rate, target_wav = wavfile.read(data_target[i])

        length = target_wav.shape[-1]

        result["pesq_wb"] += pesq(16000, clean, target_wav, "wb") * length  # wide band
        result["pesq_nb"] += (
            pesq(16000, clean, target_wav, "nb") * length
        )  # narrow band
        result["stoi"] += stoi(clean, target_wav, rate) * length
        result["count"] += 1 * length

        # print(result["pesq_wb"])

    return result


if __name__ == "__main__":
    from pathlib import Path

    CUR_DIR_PATH = Path(__file__).parent

    path_clean = (
        CUR_DIR_PATH
        / "Data"
        / "testsets"
        / "test_set"
        / "synthetic"
        / "no_reverb"
        / "clean"
    )
    ENHANCED_DIR = CUR_DIR_PATH / "Data" / "Enhanced"

    result = evaluate_dns(path_clean=path_clean, path_target=ENHANCED_DIR)

    # logging
    for key in result:
        if key != "count":
            print("{} = {:.3f}".format(key, result[key] / result["count"]), end=", ")
