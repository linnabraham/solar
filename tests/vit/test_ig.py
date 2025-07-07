import unittest
from vit.scripts.train import TrainingConfig
from vit.scripts.ig import single_aarp
from vit.scripts.predictions_analyze import dfs_from_metadata
from vit.utils import get_metadata

class TestSingleAARP(unittest.TestCase):
    def read_data(self):
        config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
        metadata = get_metadata(config)
        training_df, val_df, test_df = dfs_from_metadata(metadata)
        aarp_id = 4920
        aarp_id_df =test_df.query(f"aarp_id=={aarp_id}")
        s_aarp = single_aarp(aarp_id, aarp_id_df)
        try:
            s_images = s_aarp.get_images()
        except Exception as e:
            print("Caught exception:", repr(e))
            # Optionally, assert something about it
            self.assertTrue(isinstance(e, Exception))
        else:
            self.fail("Expected some exception, but none was raised.")
