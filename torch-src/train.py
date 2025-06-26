import torch.nn as nn
import torchvision.models

def modify_alexnet(model):
    """
    This function modifies the base architecture of AlexNet to conform as far as
    posssible with the architecture that used in tensorflow
    """

    model.features[0].out_channels = 96
    model.features[0].kernel_size = (5,5)
    #model.features[0].padding = 'same'
    model.features[0].stride = (2,2)

    model.features[3].in_channels = 96
    model.features[3].out_channels = 256
    model.features[3].kernel_size = (5,5)
    model.features[3].padding = 'same'
    #model.features[3].stride = (2,2)

    model.features[6].in_channels = 256
    model.features[6].out_channels = 384
    model.features[6].kernel_size = (3,3)
    model.features[6].padding = 'same'

    model.features[8].in_channels = 384
    model.features[8].out_channels = 384
    model.features[8].padding = 'same'

    model.features[10].in_channels = 384
    model.features[10].out_channels = 256
    model.features[10].padding = 'same'

    #TODO: find out why the following code doesn't work
    # model.classifier[6].out_features = 2
    model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
    model = nn.Sequential(model, nn.Sigmoid())

    return model

if __name__=="__main__":
    alexnet = torchvision.models.alexnet()
    model = modify_alexnet(alexnet)
    print(f"Model Architecture: \n {model}")
