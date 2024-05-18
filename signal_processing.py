import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import correlate, lfilter
import numpy as np


def remove_echo(y):
    # autocorrelation
    c = correlate(y.numpy(), y.numpy(), mode="full")[0]
    N = len(c)

    n0 = np.argmax(c[N // 2 + 10 :]) + 10 - 1
    Rxx_n0 = np.max(c[N // 2 + 10 :])
    Rxx_0 = np.max(c)

    # N = len(c)
    # p = np.linspace(-N // 2, N // 2, N)

    # plt.figure()
    # plt.plot(p, c)
    # plt.title("Autocorrelation of the signal", fontsize=15)
    # plt.xlabel("p", fontsize=13)
    # plt.ylabel("Autocorrelation", fontsize=13)
    # plt.show()

    Alpha1 = (Rxx_0 + np.sqrt(Rxx_0**2 - 4 * Rxx_n0**2)) / (2 * Rxx_n0)
    Alpha2 = (Rxx_0 - np.sqrt(Rxx_0**2 - 4 * Rxx_n0**2)) / (2 * Rxx_n0)

    Alpha = None
    if Alpha1 < 1:
        Alpha = Alpha1
    if Alpha2 < 1:
        Alpha = Alpha2

    if Alpha is None:
        Alpha = 0.6525  # Default value if both Alpha1 and Alpha2 are not suitable

    # Filter to remove the echo
    Num = [1]
    Den = np.zeros(n0 + 1)
    Den[0] = 1
    Den[n0] = Alpha

    y_filtre = lfilter(Num, Den, y.numpy())

    # # Verify the elimination of the echo
    # c_filtre = correlate(y_filtre, y_filtre, mode="full")
    # N = len(c_filtre)
    # p = np.linspace(-N // 2, N // 2, N)

    # plt.figure()
    # plt.plot(p, c_filtre)
    # plt.title("Autocorrelation after echo removal", fontsize=15)
    # plt.xlabel("p", fontsize=13)
    # plt.ylabel("Autocorrelation", fontsize=13)
    # plt.show()

    return torch.tensor(y_filtre)


from pathlib import Path

CUR_DIR_PATH = Path(__file__)
ROOT = CUR_DIR_PATH.parents[0]
from dataset import DatasetCleanNoisy

# DATASET
path_clean = ROOT / "Data" / "reverb_training_set" / "clean"
path_noisy = ROOT / "Data" / "reverb_training_set" / "noisy"
dataset = DatasetCleanNoisy(
    path_clean=path_clean,
    path_noisy=path_noisy,
    subset="training",
    crop_length_sec=10,
)

echo_signal = dataset[0][1]

y = remove_echo(echo_signal)[0]


sf.write("Pa11_corrige.wav", y.numpy(), 16000)


exit()
