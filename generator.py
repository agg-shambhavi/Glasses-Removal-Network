import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=1,
            )
            # nn.Conv2d(
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=7,
            #     stride=1,
            #     padding=3,
            # )
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()

        downConvBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=6,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )

        upConvBlock = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=1, kernel_size=6, stride=2, padding=2
            ),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn(2, img_channels, img_size, img_size)
    gen = Generator(img_channels)
    # print(gen)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
