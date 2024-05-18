from pathlib import Path
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import json
from stft_loss import MultiResolutionSTFTLoss
from util import loss_fn, LinearWarmupCosineDecay
import torch
from unet import UNet1D

torch.autograd.set_detect_anomaly(True)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(0)

loss_config = {
    "ell_p": 1,
    "ell_p_lambda": 1,
    "stft_lambda": 1,
    "stft_config": {
        "sc_lambda": 0.5,
        "mag_lambda": 0.5,
        "band": "full",
        "hop_sizes": [50, 120, 240],
        "win_lengths": [240, 600, 1200],
        "fft_sizes": [512, 1024, 2048],
    },
}

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

# network_config = {
#     "channels_input": 1,
#     "channels_output": 1,
#     "channels_H": 64,
#     "max_H": 768,
#     "encoder_n_layers": 8,
#     "kernel_size": 4,
#     "stride": 2,
#     "tsfm_n_layers": 3,
#     "tsfm_n_head": 8,
#     "tsfm_d_model": 512,
#     "tsfm_d_inner": 2048,
# }

from network import CleanUNet


if __name__ == "__main__":

    # PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--nitermax", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log-interval-train", type=int, default=10)
    parser.add_argument("--log-interval-val", type=int, default=500)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    print(args)

    # IMPORTATIONS & DEVICE
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[0]

    from dataset import DatasetCleanNoisy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET

    path_clean = ROOT / "Data" / args.dataset / "clean"
    path_noisy = ROOT / "Data" / args.dataset / "noisy"
    dataset = DatasetCleanNoisy(
        path_clean=path_clean,
        path_noisy=path_noisy,
        subset="training",
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

    # DATALOADERS
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6
    )
    # MODEL
    model = CleanUNet(**network_config).to(device)
    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # LOSS, OPTIMIZER, SCHEDULER

    step = 0

    criterion = MultiResolutionSTFTLoss(**loss_config["stft_config"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=args.lr,
        n_iter=args.nitermax,
        iteration=step,
        divider=25,
        warmup_proportion=0.05,
        phase=("linear", "cosine"),
    )

    # LOGGER
    # features_str = "-".join(map(str, features))
    name = Path(__file__).name[:-3]
    name += (
        "_"
        + str(args.nitermax)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.lr)[2:]
        + "_"
        + args.dataset
    )
    LOG_DIR = ROOT / "Logs" / name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"LOG_DIR: {LOG_DIR}")
    writer = SummaryWriter(log_dir=LOG_DIR)

    # TRAINING

    best_val_loss = float("inf")
    best_state_dict = model.state_dict()
    SAVED_MODEL_DIR = ROOT / "Saved_models" / name
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    early_stopping_tol = 10
    prev_val_loss = float("inf")
    counter = 0

    hparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    step = 0
    with tqdm(total=args.nitermax) as pbar:

        while step < args.nitermax:

            for clean_audio, noisy_audio, _ in train_dataloader:

                clean_audio = clean_audio.to(device)
                noisy_audio = noisy_audio.to(device)

                optimizer.zero_grad()
                X = (clean_audio, noisy_audio)

                loss, loss_dic = loss_fn(model, X, **loss_config, mrstftloss=criterion)

                loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e9)

                optimizer.step()
                scheduler.step()

                if step % args.log_interval_train == 0:
                    writer.add_scalar(
                        "train_loss_step",
                        loss.item(),
                        step,
                    )
                    writer.add_scalar(
                        "Learning-rate", optimizer.param_groups[0]["lr"], step
                    )

                # VALIDATION
                if step % args.log_interval_val == 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for clean_audio, noisy_audio, _ in val_dataloader:
                            clean_audio = clean_audio.to(device)
                            noisy_audio = noisy_audio.to(device)
                            X = (clean_audio, noisy_audio)
                            loss, loss_dic = loss_fn(
                                model, X, **loss_config, mrstftloss=criterion
                            )
                            val_loss += loss.item()
                    val_loss /= len(val_dataloader)
                    writer.add_scalar(
                        "val_loss_step",
                        val_loss,
                        step,
                    )

                    # KEEP TRACK OF IMPROVMENT
                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
                        best_state_dict = model.state_dict().copy()
                        torch.save(
                            model.state_dict(), SAVED_MODEL_DIR / "best_model.pt"
                        )
                        counter = 0
                    else:
                        counter += 1

                    prev_val_loss = val_loss

                # # EARLY STOPPING
                # if counter == early_stopping_tol:
                #     print("stopped early")
                #     steps = args.nitermax  # to end while loop
                #     break

                step += 1
                pbar.update(1)
                if step == args.nitermax:
                    break

    torch.save(model.state_dict(), SAVED_MODEL_DIR / "last_model.pt")

    # Log hyperparameters and metrics
    writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={
            "hparam/best_val_loss": best_val_loss,
        },
    )
