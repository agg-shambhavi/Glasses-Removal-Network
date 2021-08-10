import torch
import torch.nn as nn
import selfAttentionBlock


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # I am not sure which res block to take
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
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

        self.downConvBlock = nn.Sequential(
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
                stride=2,
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

        self.convtrans1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.active1 = nn.Sequential(nn.InstanceNorm2d(128), nn.ReLU())

        self.convtrans2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1
        )

        self.convtrans3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=1, kernel_size=6, stride=2, padding=2
        )

        self.selfAttentionlayer = selfAttentionBlock.SelfAttentionBlock(in_dim=256)
        self.resBlock1 = ResidualBlock(channels=256)
        self.resBlock2 = ResidualBlock(channels=256)

    def forward(self, x):
        batch_size, ch, h, w = x.shape
        out = self.downConvBlock(x)
        out = self.resBlock1(out)
        out, _ = self.selfAttentionlayer(out)
        out = self.resBlock2(out)
        out = self.convtrans1(out, output_size=(batch_size, 128, 32, 32))
        out = self.active1(out)
        out = self.convtrans2(out, output_size=(batch_size, 64, 64, 64))
        out = self.convtrans3(out, output_size=(batch_size, 1, 128, 128))
        out = self.lastlayer(out)
        return out


def test():
    img_channels = 1
    img_size = 128
    x = torch.randn(2, img_channels, img_size, img_size)
    gen = Generator(img_channels)
    print(gen)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
