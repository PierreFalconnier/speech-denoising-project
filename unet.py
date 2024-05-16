import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        x = self.relu(self.batchnorm(self.conv2(x)))
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet1D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from dataset import DatasetCleanNoisy
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

    x = dataset[0][0].unsqueeze(0)
    print(x.shape)

    # Create a random input tensor with shape (batch_size, channels, length)
    # x = torch.randn((8, 1, 2048))  # Example input

    model = UNet1D()

    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    preds = model(x)
    print(preds.shape)  # Should be (batch_size, out_channels, length)
