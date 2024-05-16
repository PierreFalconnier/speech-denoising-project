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
from util import loss_fn
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
if __name__ == "__main__":

    # PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    # IMPORTATIONS & DEVICE
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[0]

    from dataset import DatasetCleanNoisy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET

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

    # DATALOADERS
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # LOGGER
    name = Path(__file__).name[:-3]
    name += "_" + str(args.epochs) + "_" + str(args.batch_size) + "_" + str(args.lr)[2:]
    LOG_DIR = ROOT / "Logs" / name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"LOG_DIR: {LOG_DIR}")
    writer = SummaryWriter(log_dir=LOG_DIR)

    # MODEL
    model = UNet1D().to(device)
    print(model)
    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # LOSS, OPTIMIZER

    criterion = MultiResolutionSTFTLoss(**loss_config["stft_config"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TRAINING

    best_val_loss = float("inf")
    best_state_dict = model.state_dict()
    SAVED_MODEL_DIR = ROOT / "Saved_models"
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    early_stopping_tol = 5  # num of epochs
    prev_val_loss = float("inf")
    counter = 0

    hparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0

        for step, (clean_audio, noisy_audio, _) in enumerate(train_dataloader):

            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)

            # back-propagation
            optimizer.zero_grad()
            X = (clean_audio, noisy_audio)

            loss, loss_dic = loss_fn(model, X, **loss_config, mrstftloss=criterion)

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e9)

            optimizer.step()

            train_loss += loss.item()

            if step % args.log_interval == 0:
                writer.add_scalar(
                    "train_loss_step",
                    loss.item(),
                    epoch * len(train_dataloader) + step,
                )

        # VALIDATION
        model.eval()

        with torch.no_grad():
            val_loss = 0.0

            for clean_audio, noisy_audio, _ in val_dataloader:

                clean_audio = clean_audio.to(device)
                noisy_audio = noisy_audio.to(device)

                X = (clean_audio, noisy_audio)

                loss, loss_dic = loss_fn(model, X, **loss_config, mrstftloss=criterion)

                val_loss += loss.item()

        # LOGGING
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)

        # EARLY STOPPING
        # stop if no amelioration for early_stopping_tol epochs
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter == early_stopping_tol:
            print("stopped early")
            break

        prev_val_loss = val_loss

    # TRAINING OVER
    best_state_dict = model.state_dict().copy()
    torch.save(model.state_dict(), SAVED_MODEL_DIR / "best_model.pt")

    # Log hyperparameters and metrics
    writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={
            "hparam/best_val_loss": best_val_loss,
        },
    )
