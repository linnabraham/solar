"""Fine-tune torchvision vit_l_16 (ImageNet-pretrained) on AIA imagery.

Reuses train.py's train()/_run_epoch()/validate_model() loop unmodified via the
build_model_fn/extra_transform hooks -- no training-loop logic is duplicated here.
See train.py for the shared loop implementation, and aarp_ml/torch/model.py for the
shared build_pretrained_vit() architecture builder (also used by test_pretrained.py
and evaluate_aarp.py, so all three stay checkpoint-compatible with each other).

Note: pretrained=True downloads ImageNet1K_V1 weights via torch hub on first use --
requires network access (or a warm torch hub cache) on whatever node this runs on.

Usage:
    python -m src.torch.vit.train_pretrained \\
        --json-path cv_folds/fold_0.json --stats-file cv_folds/fold_0_stats.pkl \\
        --epochs 50 --lr 0.0001 --run-name my-pretrained-run
"""

from torchvision.transforms import v2

from aarp_ml.torch.model import build_pretrained_vit
from aarp_ml.dataset import all_wavelengths
from src.torch.vit.config import TrainingConfig
from src.torch.vit.train import train, add_common_training_args

# vit_l_16 (~304M params) vs DeepFlare_ViT (~24M) -- confirmed against treasured-blaze-221's
# wandb-logged memory_threshold=8000, itself traceable to origin/vit-pt commit a004741
# ("Fix: Increase memory requirement for pre-trained model to run").
DEFAULT_MEMORY_THRESHOLD = 8000

RESIZE_SIZE = 224  # vit_l_16's patch embedding / position embedding are sized for 224x224


def parse_args() -> TrainingConfig:
    """Parse command line arguments and create config."""
    import argparse
    parser = argparse.ArgumentParser()
    add_common_training_args(parser, default_memory_threshold=DEFAULT_MEMORY_THRESHOLD)

    args = parser.parse_args()

    # Validate that --retrain is not given without --trained-model-path
    if args.retrain and not args.trained_model_path:
        parser.error("--retrain requires --trained-model-path to be specified.")

    channel_indices = None
    if args.channels is not None:
        unknown = [c for c in args.channels if c not in all_wavelengths]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {all_wavelengths}")
        channel_indices = [all_wavelengths.index(c) for c in args.channels]

    return TrainingConfig(
        json_path=args.json_path,
        stats_file=args.stats_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        scheduler_type=args.scheduler,
        retrain=args.retrain,
        trained_model_path=args.trained_model_path,
        use_l1=args.use_l1,
        l1_lambda=args.l1_lambda,
        channel_indices=channel_indices,
        memory_threshold=args.memory_threshold,
        run_name=args.run_name,
        save_all_epochs=args.save_all_epochs,
        seed=args.seed,
        model_type="vit_pretrained",
        image_height=RESIZE_SIZE,  # cosmetic -- wandb should log the true model input size
    )


if __name__ == "__main__":
    config = parse_args()
    train(
        config,
        build_model_fn=lambda cfg: build_pretrained_vit(
            n_channels=cfg.n_channels, n_classes=cfg.n_classes, pretrained=True
        ),
        extra_transform=v2.Resize(RESIZE_SIZE),
    )
