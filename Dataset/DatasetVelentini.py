from typing import Any, Dict, List, Tuple, Type, Callable, Optional, Union
import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
import torchaudio
import torch


class MyDataSet(Dataset):

    def __init__(self, path, set="training", crop_length_sec=0) -> None:
        """
        Args:
            transform (list):
        """
        super(Dataset, self).__init__()
        self.set = set
        self.crop_length_sec = crop_length_sec
        if set == "training":
            posix_list_clean = list(path.rglob("clean_trainset*/**/*.wav"))
            posix_list_noisy = list(path.rglob("noisy_trainset*/**/*.wav"))
        elif set == "test":
            posix_list_clean = list(path.rglob("clean_testset*/**/*.wav"))
            posix_list_noisy = list(path.rglob("noisy_testset*/**/*.wav"))
        else:
            raise NotImplementedError

        assert len(posix_list_clean) == len(posix_list_noisy)

        self.data_clean = sorted(map(str, posix_list_clean))
        self.data_noisy = sorted(map(str, posix_list_noisy))

    def __getitem__(self, index: int) -> dict:

        clean_audio, sample_rate = torchaudio.load(self.data_clean[index])
        noisy_audio, sample_rate = torchaudio.load(self.data_noisy[index])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)

        # random crop
        if self.set != "test" and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start : (start + crop_length)]
            noisy_audio = noisy_audio[start : (start + crop_length)]

        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)

        return clean_audio, noisy_audio

    def __len__(self) -> int:
        return len(self.data_clean)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    CUR_DIR_PATH = Path(__file__).parent
    print(CUR_DIR_PATH)

    dataset = MyDataSet(path=CUR_DIR_PATH / "Data" / "valentini", crop_length_sec=1)
    print(len(dataset))

    print(dataset.data_clean[0])
    print(dataset.data_noisy[0])

    print(dataset[0][0])
    print(dataset[0][1])
    print(dataset[0][1].shape)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(next(iter(dataloader)))
