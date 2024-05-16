# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import argparse
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter

import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread

from dataset import DatasetCleanNoisy
from util import rescale, find_max_epoch, print_size, sampling
from pathlib import Path
from torch.utils.data import DataLoader


# def denoise(output_directory, ckpt_iter, subset, dump=False):
def denoise(output_directory, dataset, net, device, dump=False):
    """
    Denoise audio

    Parameters:
    output_directory (str):         save generated speeches to this path
    dataset:
    net:
    device:                         cpu or gpu
    dump (bool):                    whether save enhanced (denoised) audio
    """

    dataloader = DataLoader(dataset, batch_size=1)

    net.eval()

    output_directory = Path(output_directory)
    Path.mkdir(output_directory, parents=True, exist_ok=True)

    # inference
    all_generated_audio = []
    all_clean_audio = []

    for clean_audio, noisy_audio, clean_file_path in tqdm(dataloader):

        fileid = clean_file_path[0].split()[-1][:-4].split("_")[-1]

        noisy_audio = noisy_audio.to(device)
        LENGTH = len(noisy_audio[0].squeeze())
        generated_audio = net(noisy_audio)

        if dump:
            filename = output_directory / f"enhanced_fileid_{fileid}.wav"
            wavwrite(
                str(filename),
                dataset.sample_rate,
                generated_audio[0].squeeze().detach().cpu().numpy(),
            )
        else:
            all_clean_audio.append(clean_audio[0].squeeze().detach().cpu().numpy())
            all_generated_audio.append(
                generated_audio[0].squeeze().detach().cpu().numpy()
            )

    return all_clean_audio, all_generated_audio


if __name__ == "__main__":

    # #  PARSER TO BE ADDED IN THE FUTURE
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--path_clean', type=str)
    # parser.add_argument('-n', '--path_noisy', type=str)
    # parser.add_argument('-f', '--model_config', type=str)
    # parser.add_argument('-m', '--model_path', type=str)
    # parser.add_argument('-o', '--output_path', type=str)
    # args = parser.parse_args()

    # DATASET
    CUR_DIR_PATH = Path(__file__).parent

    path_clean = (
        CUR_DIR_PATH
        / "Data"
        / "testsets"
        / "test_set"
        / "synthetic"
        / "with_reverb"
        / "clean"
    )
    path_noisy = (
        CUR_DIR_PATH
        / "Data"
        / "testsets"
        / "test_set"
        / "synthetic"
        / "with_reverb"
        / "noisy"
    )

    dataset = DatasetCleanNoisy(
        path_clean=path_clean, path_noisy=path_noisy, subset="testing"
    )

    # OUTPUT DIR
    OUTPUT_DIR = CUR_DIR_PATH / "Data" / "Enhanced"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MODEL
    model_path = CUR_DIR_PATH / "Saved_models" / "large_full_pretrained.pkl"
    model_config = CUR_DIR_PATH / "Saved_models" / "DNS-large-full.json"

    # predefine model
    with open(str(model_config)) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config["network_config"]
    net = CleanUNet(**network_config)
    print_size(net)

    # load it
    checkpoint = torch.load(model_path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # DENOISE
    all_clean_audio, all_generated_audio = denoise(
        OUTPUT_DIR, dataset, net, device, True
    )
