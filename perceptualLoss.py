import torch
import torchvision.models.vgg as models


# def getVggFeatures(input, i):
#     layer_dict = {1: 5, 2: 10, 3: 17, 4: 24, 5: 30}
#     vgg16 = models.vgg16(pretrained=True)
#     output = vgg16.features[: layer_dict[i]](input)
#     return output


# input = torch.rand(1, 3, 128, 128)
# for i in range(1,6):
#     print(getVggFeatures(input, i).shape, i)

# print(models.vgg16(pretrained=True))

# input_vect = torch.rand(1, 1, 3, 3)
# print(input_vect.repeat(1,2,1,1), input_vect.repeat(1,2,1,1).shape)
