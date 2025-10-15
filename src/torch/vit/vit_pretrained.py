import pytorch_lightning as pl
import torchvision
import torch

def make_model():
    model = torchvision.models.vit_l_16(weights="IMAGENET1K_V1")

    old_conv = model.conv_proj

    # Create a new one with 7 input channels instead of 3
    new_conv = torch.nn.Conv2d(
        in_channels=7,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    model.conv_proj = new_conv

    # Random initialization for new input channel weights
    torch.nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if new_conv.bias is not None:
        torch.nn.init.zeros_(new_conv.bias)

    # Replace the classification head (1000 → 2 classes)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
    return model

if __name__=="__main__":
    model = modify_model(model)
    x = torch.randn(1, 7, 224, 224)
    out = model(x)
    print(out.shape)
