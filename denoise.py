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
from torch.utils.data import DataLoader, random_split
from unet import UNet1D

torch.manual_seed(0)


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

    dataset.sample_rate = 16000
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

    # DATASET
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[0]

    path_clean = ROOT / "Data" / "training_set" / "clean"
    path_noisy = ROOT / "Data" / "training_set" / "noisy"
    dataset = DatasetCleanNoisy(
        path_clean=path_clean,
        path_noisy=path_noisy,
        subset="training",
        crop_length_sec=10,
    )

    # split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # OUTPUT DIR
    OUTPUT_DIR = ROOT / "Data" / "Enhanced"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MODEL
    model_path = ROOT / "Saved_models" / "best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = [16, 32, 64, 128, 256]
    model = UNet1D(in_channels=1, out_channels=1, features=features)

    # load trained model
    model_state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    model.to(device)

    # DENOISE
    all_clean_audio, all_generated_audio = denoise(
        OUTPUT_DIR, val_dataset, model, device, True
    )
