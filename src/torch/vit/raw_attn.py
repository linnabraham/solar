import vit_pytorch
import torch
from torch import nn
from einops import rearrange, repeat
from aarp_ml.torch.model import DeepFlare_ViT
from dataclasses import dataclass
# imports for dataloader
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import math
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from src.torch.vit.my_agcam import AGCAM

def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create a sampler that handles class imbalance."""
    # Count instances of each class
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    # Compute class weights (inverse of frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[label] for label in dataset.labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

class PatchedAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        # hooks for explainability
        self.forward_hook_before_softmax = nn.Identity()
        self.backward_hook_after_softmax = nn.Identity()

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = x.detach()

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = self.forward_hook_before_softmax(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn = self.backward_hook_after_softmax(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

@dataclass
class Config:
    trained_model_path: str = None
    batch_size = 32
    json_path = "solar_dataset.json"
    stats_file = "stats.pkl"
    n_passbands = 7
    n_classes = 2
    height = 512

config = Config()
config.trained_model_path = "outputs/glad-shape-197/trained_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vit_pytorch.vit.Attention = PatchedAttention
model = DeepFlare_ViT(
  height=config.height,
  n_classes=config.n_classes,
  n_passbands=config.n_passbands
).model

#print(model)

# Load statistics
with open(config.stats_file, 'rb') as f:
    stats = pickle.load(f)
    means = [stats['mean'][f'channel_{i}'] for i in range(config.n_passbands)]
    stds = [stats['std'][f'channel_{i}'] for i in range(config.n_passbands)]

train_dataset = aia_euv(
    config.json_path,
    subset='training',
    transform=v2.Compose([
        AIALogTransform(means, stds),
        #v2.Resize((224, 224)),          # <--- add this line
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=get_weighted_sampler(train_dataset)
)

train_iter = iter(train_loader)

checkpoint = torch.load(config.trained_model_path, map_location=device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', -1) + 1
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()
ours_method = AGCAM(model)
x, y = next(train_iter)
x, y = next(train_iter)
x, y = next(train_iter)

image = x[0].unsqueeze(0)
image = image.to(device)

with torch.enable_grad():
    # Generate heatmap of our method
    prediction, ours_heatmap = ours_method.generate(image)
    #ours_heatmap = transforms.Resize((224, 224))(ours_heatmap[0])
    ours_heatmap = ours_heatmap[0]
    ours_heatmap = (ours_heatmap - ours_heatmap.min())/(ours_heatmap.max()-ours_heatmap.min())
    ours_heatmap = ours_heatmap.detach().cpu().numpy()
    ours_heatmap = np.transpose(ours_heatmap, (1, 2, 0))


fig, axs = plt.subplots(1,2 )
axs[0].set_title('Original')
axs[0].imshow(image[0,0].cpu().detach().numpy())
axs[0].axis('off')

axs[1].set_title('Ours')
axs[1].imshow(ours_heatmap, cmap='jet', alpha=0.5)
axs[1].axis('off')

fig.savefig("comparison.png", bbox_inches='tight')
