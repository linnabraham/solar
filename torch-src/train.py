import torch.nn as nn
import torchvision.models

def modify_alexnet(model):
    """
    This function modifies the base architecture of AlexNet to conform as far as
    posssible with the architecture that used in tensorflow
    """

    model.features[0] = nn.Conv2d(
        in_channels=7,
        out_channels=96,
        kernel_size=(5, 5),
        stride=(2, 2),
    )

    model.features[3] = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding='same')
    model.features[6] = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding='same')
    model.features[8] = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding='same')
    model.features[10] = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding='same')

    #TODO: find out why the following code doesn't work
    # model.classifier[6].out_features = 2

    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # Append sigmoid to convert logits to probability (e.g., for binary classification)
    model = nn.Sequential(model, nn.Sigmoid())

    return model

if __name__=="__main__":
    alexnet = torchvision.models.alexnet()
    model = modify_alexnet(alexnet)
