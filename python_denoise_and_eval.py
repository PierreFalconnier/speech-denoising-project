from collections import defaultdict
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")
from pesq import pesq
from pystoi import stoi
from scipy.io.wavfile import write as wavwrite
import numpy as np
import torch
import random

random.seed(0)
np.random.seed(0)
from dataset import DatasetCleanNoisy
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from unet import UNet1D
from network import CleanUNet

torch.manual_seed(0)

# configs for networks
features = [16, 32, 64, 128, 256]

network_config = {
    "channels_input": 1,
    "channels_output": 1,
    "channels_H": 64,
    "max_H": 768,
    "encoder_n_layers": 6,
    "kernel_size": 4,
    "stride": 2,
    "tsfm_n_layers": 3,
    "tsfm_n_head": 5,
    "tsfm_d_model": 256,
    "tsfm_d_inner": 1024,
}

pretrained_network_config = {
    "channels_input": 1,
    "channels_output": 1,
    "channels_H": 64,
    "max_H": 768,
    "encoder_n_layers": 8,
    "kernel_size": 4,
    "stride": 2,
    "tsfm_n_layers": 5,
    "tsfm_n_head": 8,
    "tsfm_d_model": 512,
    "tsfm_d_inner": 2048,
}


def denoise_and_eval(output_directory, dataset, net, device, save=False):
    result = defaultdict(int)

    dataset.sample_rate = 16000
    rate = 16000
    dataloader = DataLoader(dataset, batch_size=1, num_workers=6, pin_memory=True)

    net.eval()

    output_directory = Path(output_directory)
    Path.mkdir(output_directory, parents=True, exist_ok=True)

    for clean_audio, noisy_audio, clean_file_path in tqdm(dataloader):

        # INFERENCE
        noisy_audio = noisy_audio.to(device)
        generated_audio = net(noisy_audio)

        # SAVE TO WAV
        if save:
            fileid = clean_file_path[0].split()[-1][:-4].split("_")[-1]
            filename = output_directory / f"enhanced_fileid_{fileid}.wav"
            wavwrite(
                str(filename),
                dataset.sample_rate,
                generated_audio[0].squeeze().detach().cpu().numpy(),
            )

        # EVALUATION
        length = generated_audio.shape[-1]

        target_wav = generated_audio[-1].detach().cpu().numpy()[-1]
        clean = clean_audio[-1].detach().cpu().numpy()[-1]

        result["pesq_wb"] += pesq(16000, clean, target_wav, "wb") * length
        result["pesq_nb"] += pesq(16000, clean, target_wav, "nb") * length
        result["stoi"] += stoi(clean, target_wav, rate) * length
        result["count"] += 1 * length

    return result


if __name__ == "__main__":

    # PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()
    print(args)

    # DATASET
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[0]

    path_clean = Path(args.dataset_path) / "clean"
    path_noisy = Path(args.dataset_path) / "noisy"
    # if using dns testset
    subset = "testing" if "testsets" in str(path_clean) else "training"
    crop_length_sec = 0 if "testsets" in str(path_clean) else 10

    dataset = DatasetCleanNoisy(
        path_clean=path_clean,
        path_noisy=path_noisy,
        subset=subset,
        crop_length_sec=crop_length_sec,
    )

    if subset == "training":
        # split to get test set
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.10 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    elif subset == "testing":
        test_dataset = dataset
    else:
        raise NotImplementedError

    # MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.network == "unet":
        model = UNet1D(in_channels=1, out_channels=1, features=features)
    elif args.network == "cleanunet":
        model = CleanUNet(**network_config)
    elif args.network == "pretrained_cleanunet":
        model = CleanUNet(**pretrained_network_config)
    else:
        raise NotImplementedError

    # load
    if args.network != "pretrained_cleanunet":
        model_state_dict = torch.load(args.model_path, map_location="cpu")
    else:
        if "fine" in args.model_path:
            model_state_dict = torch.load(args.model_path, map_location="cpu")
        else:
            model_state_dict = torch.load(args.model_path, map_location="cpu")[
                "model_state_dict"
            ]

    model.load_state_dict(model_state_dict)
    model.to(device)

    # OUTPUT DIR
    name = (
        "Enhanced_"
        + args.model_path.split("/")[-2]
        + "_"
        + args.dataset_path.split("/")[-1].split("_")[0]
    )
    OUTPUT_DIR = ROOT / "Data" / name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # DENOISE
    result = denoise_and_eval(OUTPUT_DIR, test_dataset, model, device, True)

    # logging
    with open("results.txt", "a") as file:

        print(args, file=file)

        for key in result:
            if key != "count":
                print(
                    "{} = {:.3f}".format(key, result[key] / result["count"]), file=file
                )
        print("\n", file=file)
