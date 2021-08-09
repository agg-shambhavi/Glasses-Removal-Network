import torch
import torch.nn as nn
import sys

# # test functionality of view
# class testClass(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.in_channel = in_dim

#         self.query_conv = nn.Conv2d(
#             in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
#         )
#         self.key_conv = nn.Conv2d(
#             in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
#         )
#         self.value_conv = nn.Conv2d(
#             in_channels=in_dim, out_channels=in_dim, kernel_size=1
#         )
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, c, h, w = x.shape
#         # proj_query  = self.query_conv(x).view(batch_size,-1,w*h)
#         proj_query = self.query_conv(x).view(batch_size, -1, w * h).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, w * h)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)

#         proj_value = self.value_conv(x).view(batch_size, -1, w * h)
#         out = torch.bmm(proj_value,attention.permute(0,2,1) )
#         out = out.view(batch_size, c, w , h)
#         return out


# def testFunc():
#     x = torch.randn(3, 256, 32, 32)
#     simpleModel = testClass(in_dim=256)
#     print(simpleModel(x).shape)


# testFunc()


# sys.exit()


class SelfAttentionBlock(nn.Module):
    """Self Attention Layer/Block"""

    def __init__(self, in_dim):
        super().__init__()
        self.in_channel = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B X C X W X H)
        returns:
            out : self attention value + input features
            attention: B x N X N (N is Width*Height)
        """

        batch_size, c, h, w = x.shape

        proj_query = self.query_conv(x).view(batch_size, -1, w * h).permute(0, 2, 1)

        # proj_query with view is of shape (batch_size, c, w*h)
        # proj_query with permute becomes batch_size, w*h, c), ie transpose of proj_query for matrix multiplication
        proj_key = self.key_conv(x).view(batch_size, -1, w * h)  # (batch_size, c, w*h)

        energy = torch.bmm(proj_query, proj_key)  # (batch_size, w*h, w*h)

        attention = self.softmax(energy)  # (batch_size, w*h, w*h)

        proj_value = self.value_conv(x).view(
            batch_size, -1, w * h
        )  # (batch_size, c*8, w*h) c*8 = C

        out = torch.bmm(
            proj_value, attention.permute(0, 2, 1)
        )  # (batch_size, c*8, w*h)

        # attention is transposed for matrix mul
        out = out.view(batch_size, c, w, h)

        out = self.gamma * out + x
        return out, attention


def testFunc():
    x = torch.randn(3, 256, 16, 16)
    simpleModel = SelfAttentionBlock(in_dim=256)
    print(simpleModel(x)[0].shape)


testFunc()
