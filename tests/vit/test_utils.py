import math
import unittest
import matplotlib.pyplot as plt
import torch
from vit.scripts.train import TrainingConfig
from vit.scripts.ig import single_aarp
from vit.scripts.predictions_analyze import dfs_from_metadata
from vit.utils import get_metadata
from aarp_ml.dataset import all_wavelengths
from astro_utils.aia import plot_aia_image_grid
from astro_utils.visualization import plot_image_grid
from vit.utils import get_attribution_for_image, get_model_and_transform

class TestSingleImageBaseline(unittest.TestCase):
    def setUp(self):
        config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
        metadata = get_metadata(config)
        training_df, val_df, test_df = dfs_from_metadata(metadata)
        config.trained_model_path = "output/glad-shape-197/trained_model.pth"
        model, self.transform, self.device = get_model_and_transform(config)
        self.model = model.to(self.device)
        print(f"Trained model loaded from {config.trained_model_path} to {self.device}")
        aarp_id = 1449
        aarp_id_df =test_df.query(f"aarp_id=={aarp_id}")
        self.s_aarp = single_aarp(aarp_id, aarp_id_df)
        s_images = self.s_aarp.get_images()
        print(f"Loaded all images for {aarp_id}")
        t_idx = 15
        self.image = s_images[t_idx]
        print(f"Using image from the {t_idx} th timestep")

    def test_zero_baseline(self):
        tensor_image = torch.from_numpy(self.image).to(torch.float32)
        tensor_image = self.transform(tensor_image)
        pred = self.model(tensor_image.unsqueeze(0).to(device="cuda"))
        self.assertEqual(pred.argmax(dim=1).item(), 1)

        fig = plot_aia_image_grid(self.image, passbands=all_wavelengths, vmax_percentile=99.9)
        print(f"Saving image to disk...")
        fig.savefig("tests_outputs/test_image_grid.png", bbox_inches="tight")

        print(f"Getting IG attribution for image")
        ig_out = get_attribution_for_image(self.image, self.s_aarp.label, transform=self.transform, model=self.model, device=self.device)
        fig = plot_image_grid(ig_out, origin="lower")
        print(f"Saving attribution image to disk..")
        fig.savefig("tests_outputs/ig_out_test.png", bbox_inches="tight")

if __name__ == "__main__":
    unittest.main()
