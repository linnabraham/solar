import torchvision
from torchvision.transforms import v2
from aarp_ml.torch.model import BaseModel
import torch.nn as nn

class VIT_Pretrained(BaseModel):

    def __init__(self, d_input, d_output, eve_norm, root_dir, lr_linear = 1e-2, cnn_dp = 0.75, cnn_lambda = 0.5,
                 lambda_mom0 = 0.1, lambda_mom1 = 0.1, lambda_mom2 = 0.1,  scale_learn = True, bias = False,loss_func = nn.HuberLoss(),
                 lr_cnn=0.0001, ln_params=None,  dropout_prob = 0.3, height = 512, patch_size = 32, linear_learn = False,
                 wavelength_array_path = "wavelength_grid.npz", ion_data_path = "ion_data.csv"):

        super(VIT_Pretrained, self).__init__(d_input, d_output,root_dir, loss_func, lr_linear, lr_cnn, cnn_dp, ln_params, cnn_lambda,
                 lambda_mom0, lambda_mom1, lambda_mom2, dropout_prob, height, patch_size, wavelength_array_path, ion_data_path)
        self.eve_norm = eve_norm
        self.n_channels = d_input
        self.outSize = d_output
        self.ln_params = ln_params
        self.lr_cnn = lr_cnn
        self.lambda_mom0 = lambda_mom0
        self.lambda_mom1 = lambda_mom1
        self.lambda_mom2 = lambda_mom2
        self.root_dir = root_dir
        self.resize = torchvision.transforms.Resize(224)
        self.model = torchvision.models.vit_l_16(weights='IMAGENET1K_V1')
        conv1_out = self.model.conv_proj.out_channels
        self.model.conv_proj = nn.Conv2d(
            d_input,
            conv1_out,
            kernel_size=(16, 16),
            stride=(16, 16)
        )

        lin_in = self.model.heads.head.in_features
        regression = nn.Sequential(
            nn.Dropout(p=dropout_prob, inplace=True),
            nn.Linear(in_features=lin_in, out_features=d_output, bias=True)
        )
        self.model.heads = regression

        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = dropout_prob

        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = dropout_prob

        self.transforms = v2.Compose([v2.RandomHorizontalFlip(p = 0.7),
                                     v2.RandomVerticalFlip(p = 0.7)])

    def forward(self, x):
        return self.model(self.resize(self.transforms(x)))  #aiadata.permute(0,3,1,2)

if __name__=="__main__":
    model = VIT_Pretrained(d_input=7, d_output=2, root_dir=".", eve_norm=True).model
    print(model)
