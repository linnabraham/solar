import torch
from captum.attr import IntegratedGradients
from aarp_ml.torch.model import DeepFlare_ViT

class trained_model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7).model
        self.model.load_state_dict(torch.load(self.model_path))

    def get_ig_attribution(images, labels):
        baseline_zero = torch.zeros_like(images)
        ig = IntegratedGradients(model)
        ig_b0, _ = ig.attribute(images, baseline_zero, target=labels, n_steps=100,
                                internal_batch_size=1,
                                            return_convergence_delta=True)
        ig_b0 = ig_b0.squeeze().detach().cpu().numpy()
        return ig_b0

