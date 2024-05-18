from tqdm import tqdm
import torch
from scipy.io.wavfile import write as wavwrite
from dataset import DatasetCleanNoisy
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from unet import UNet1D
from network import CleanUNet

torch.manual_seed(0)


def denoise(output_directory, dataset, net, device):

    dataset.sample_rate = 16000
    dataloader = DataLoader(dataset, batch_size=1)
    net.eval()
    output_directory = Path(output_directory)
    Path.mkdir(output_directory, parents=True, exist_ok=True)

    for clean_audio, noisy_audio, clean_file_path in tqdm(dataloader):

        fileid = clean_file_path[0].split()[-1][:-4].split("_")[-1]

        noisy_audio = noisy_audio.to(device)
        generated_audio = net(noisy_audio)

        filename = output_directory / f"enhanced_fileid_{fileid}.wav"
        wavwrite(
            str(filename),
            dataset.sample_rate,
            generated_audio[0].squeeze().detach().cpu().numpy(),
        )


if __name__ == "__main__":

    # DATASET
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[0]
    path_clean = ROOT / "Data" / "noisy_training_set" / "clean"
    path_noisy = ROOT / "Data" / "noisy_training_set" / "noisy"
    dataset = DatasetCleanNoisy(
        path_clean=path_clean,
        path_noisy=path_noisy,
        crop_length_sec=10,
    )

    # split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # OUTPUT DIR
    OUTPUT_DIR = ROOT / "Data" / "Enhanced_cleanunet_pretrained_noisy3"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # for unet
    # model_path = (
    #     ROOT
    #     / "Saved_models"
    #     / "train_unet_15000_6_0002_16-32-64-128-256_reverb_training_set"
    #     / "best_model.pt"
    # )
    # features = [16, 32, 64, 128, 256]
    # model = UNet1D(in_channels=1, out_channels=1, features=features)

    # for cleanunet
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
    model_path = (
        ROOT
        / "Saved_models"
        / "train_cleanunet_25000_3_0002_noisy_training_set"
        / "best_model.pt"
    )
    model = CleanUNet(**network_config)
    model_state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    model.to(device)

    # DENOISE
    all_clean_audio, all_generated_audio = denoise(
        OUTPUT_DIR, val_dataset, model, device
    )
