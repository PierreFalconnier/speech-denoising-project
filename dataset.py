from typing import Any, Dict, List, Tuple, Type, Callable, Optional, Union
import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
import torchaudio
import torch


def extract_id(filename):
    return int(filename.split("_")[-1][:-4])


class DatasetCleanNoisy(Dataset):

    def __init__(
        self, path_clean, path_noisy, subset="training", crop_length_sec=0
    ) -> None:
        """
        Args:
            transform (list):
        """
        super(Dataset, self).__init__()
        self.subset = subset
        self.crop_length_sec = crop_length_sec
        posix_list_clean = list(path_clean.rglob("*.wav"))
        posix_list_noisy = list(path_noisy.rglob("*.wav"))

        assert len(posix_list_clean) == len(posix_list_noisy)
        assert subset is None or subset in ["training", "testing"]

        self.data_clean = sorted(map(str, posix_list_clean), key=extract_id)
        self.data_noisy = sorted(map(str, posix_list_noisy), key=extract_id)

        # get the sample rate
        _, sample_rate = torchaudio.load(self.data_clean[0])
        self.sample_rate = sample_rate

    def __getitem__(self, index: int) -> dict:

        clean_audio, sample_rate = torchaudio.load(self.data_clean[index])
        noisy_audio, sample_rate = torchaudio.load(self.data_noisy[index])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)

        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)

        # random crop
        if self.subset != "test" and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start : (start + crop_length)]
            noisy_audio = noisy_audio[start : (start + crop_length)]

        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)

        return clean_audio, noisy_audio, self.data_clean[index]

    def __len__(self) -> int:
        return len(self.data_clean)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    CUR_DIR_PATH = Path(__file__).parent
    print(CUR_DIR_PATH)
    path_clean = CUR_DIR_PATH / "Data" / "training_set" / "clean"
    path_noisy = CUR_DIR_PATH / "Data" / "training_set" / "noisy"

    dataset = DatasetCleanNoisy(
        path_clean=path_clean, path_noisy=path_noisy, subset="training"
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(len(dataloader))
