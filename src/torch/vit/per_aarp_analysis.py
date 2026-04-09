"""Per-AARP level analysis and visualization.

Generates predictions, integrated gradients attributions, and comparison plots
for a single AARP region. Reuses existing analysis functions to maintain
consistency with full-dataset pipeline.

Typical usage:
    python src/torch/vit/per_aarp_analysis.py --aarp-id 12345
    python src/torch/vit/per_aarp_analysis.py --aarp-id 12345 --subset test
"""

import argparse
import logging
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import torch
import pandas as pd
import numpy as np

from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model, dfs_from_metadata, get_metadata_from_json
from src.torch.vit.ig import single_aarp
from aarp_ml.dataset import all_wavelengths


# ==================== MODULE-LEVEL CONSTANTS ====================

DEFAULT_JSON_PATH = "solar_dataset.json"
DEFAULT_STATS_FILE = "stats.pkl"
DEFAULT_TRAINED_MODEL_PATH = "outputs/glad-shape-197/trained_model.pth"
DEFAULT_OUTPUT_HOME = "plots/per_aarp_analysis"
DEFAULT_LEARNING_RATE = 0.001

# Logging configuration
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


# ==================== LOGGING SETUP ====================

def setup_logging(aarp_id: int) -> logging.Logger:
    """Set up logging for per-AARP analysis session.
    
    Args:
        aarp_id: AARP identifier for log naming
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"per_aarp_analysis_{aarp_id}")
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    return logger


# ==================== CONFIGURATION & SETUP ====================

def build_config(
    json_path: str = DEFAULT_JSON_PATH,
    stats_file: str = DEFAULT_STATS_FILE,
    trained_model_path: str = DEFAULT_TRAINED_MODEL_PATH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str = "cuda:0"
) -> TrainingConfig:
    """Build TrainingConfig from paths and parameters.
    
    Args:
        json_path: Path to solar_dataset.json
        stats_file: Path to stats.pkl
        trained_model_path: Path to trained model checkpoint
        learning_rate: Learning rate for optimizer
        device: Device specification (e.g., "cuda:0")
        
    Returns:
        TrainingConfig instance
    """
    config = TrainingConfig(
        json_path=json_path,
        stats_file=stats_file,
        trained_model_path=trained_model_path,
        learning_rate=learning_rate,
        device=device,
        image_height=512,
        n_classes=2,
        n_channels=7
    )
    return config


# ==================== DATA LOADING ====================

def load_data_and_model(config: TrainingConfig, logger: logging.Logger) -> Tuple[Dict, Any, Any, Any]:
    """Load metadata, model, transform, and device.
    
    Args:
        config: TrainingConfig instance
        logger: Logger instance
        
    Returns:
        Tuple of (metadata_dict, model, transform, device)
    """
    logger.info("Loading metadata from config.json_path...")
    metadata, model, transform, device = get_data_model(config)
    logger.info(f"✓ Metadata loaded (splits: {list(metadata.keys())})")
    logger.info(f"✓ Model loaded on device: {device}")
    
    return metadata, model, transform, device


def get_subset_df(
    metadata: Dict,
    subset: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Extract and return dataframe for specified subset.
    
    Args:
        metadata: Dictionary with 'training', 'validation', 'test' keys
        subset: One of 'test', 'validation', 'training'
        logger: Logger instance
        
    Returns:
        DataFrame for the subset
    """
    valid_subsets = ['test', 'validation', 'training']
    if subset not in valid_subsets:
        logger.error(f"Invalid subset '{subset}'. Must be one of: {valid_subsets}")
        raise ValueError(f"Invalid subset: {subset}")
    
    logger.info(f"Extracting '{subset}' subset...")
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    subset_df_map = {
        'test': test_df,
        'validation': val_df,
        'training': training_df
    }
    subset_df = subset_df_map[subset]
    logger.info(f"✓ Subset loaded: {len(subset_df)} records")
    
    return subset_df


def validate_and_get_aarp(
    aarp_id: int,
    subset_df: pd.DataFrame,
    metadata_full: pd.DataFrame,
    logger: logging.Logger
) -> single_aarp:
    """Validate AARP exists in subset and return single_aarp instance.
    
    Args:
        aarp_id: Target AARP ID
        subset_df: Subset dataframe to search
        metadata_full: Full metadata dataframe (all subsets combined)
        logger: Logger instance
        
    Returns:
        single_aarp instance
        
    Raises:
        ValueError if AARP not found in subset
    """
    logger.info(f"Validating AARP {aarp_id} in subset...")
    
    aarp_in_subset = aarp_id in subset_df.aarp_id.values
    if not aarp_in_subset:
        available_aarp_ids = sorted(subset_df.aarp_id.unique().tolist())
        logger.error(f"AARP {aarp_id} not found in subset")
        logger.error(f"Available AARPs: {available_aarp_ids[:10]}... (showing first 10 of {len(available_aarp_ids)})")
        raise ValueError(f"AARP {aarp_id} not in subset")
    
    aarp_id_df = metadata_full.query(f'aarp_id == {aarp_id}')
    logger.info(f"✓ AARP {aarp_id} found: {len(aarp_id_df)} timesteps, label={aarp_id_df.label.iloc[0]}")
    
    s_aarp = single_aarp(aarp_id, aarp_id_df)
    return s_aarp


# ==================== OUTPUT DIRECTORY SETUP ====================

def create_output_structure(aarp_id: int, output_home: str = DEFAULT_OUTPUT_HOME) -> Dict[str, Path]:
    """Create output directory structure for per-AARP analysis.
    
    Structure:
        output_home/{aarp_id}/
            ├── predictions/
            ├── attributions/
            └── summary.txt
    
    Args:
        aarp_id: AARP identifier
        output_home: Root output directory
        
    Returns:
        Dictionary with keys: 'root', 'predictions', 'attributions'
    """
    aarp_output = Path(output_home) / str(aarp_id)
    aarp_output.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = aarp_output / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    
    attributions_dir = aarp_output / "attributions"
    attributions_dir.mkdir(exist_ok=True)
    
    output_paths = {
        'root': aarp_output,
        'predictions': predictions_dir,
        'attributions': attributions_dir,
        'summary': aarp_output / "summary.txt"
    }
    
    return output_paths


# ==================== ANALYSIS ORCHESTRATION ====================

def run_per_aarp_analysis(
    aarp_id: int,
    subset: str = 'test',
    config: TrainingConfig = None,
    output_home: str = DEFAULT_OUTPUT_HOME,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """Orchestrate full per-AARP analysis pipeline.
    
    This is the main entry point for analyzing a single AARP. It:
    1. Loads model, statistics, and metadata
    2. Validates AARP exists in specified subset
    3. Creates output directory structure
    4. Runs analysis stages (predictions, attributions, plotting)
    5. Saves results and generates summary
    
    Args:
        aarp_id: AARP identifier to analyze
        subset: Data subset ('test', 'validation', 'training')
        config: TrainingConfig (created if None)
        output_home: Root output directory path
        logger: Logger instance (created if None)
        
    Returns:
        Dictionary with execution metadata:
            {
                'aarp_id': int,
                'subset': str,
                'success': bool,
                'start_time': datetime,
                'end_time': datetime,
                'duration_seconds': float,
                'output_paths': dict,
                'error': str (if success=False)
            }
    """
    if logger is None:
        logger = setup_logging(aarp_id)
    
    result = {
        'aarp_id': aarp_id,
        'subset': subset,
        'success': False,
        'start_time': datetime.now(),
        'end_time': None,
        'duration_seconds': None,
        'output_paths': None,
        'error': None
    }
    
    try:
        logger.info("=" * 70)
        logger.info(f"Per-AARP Analysis: AARP {aarp_id} (subset: {subset})")
        logger.info("=" * 70)
        
        # ========== STEP 1: BUILD CONFIGURATION ==========
        if config is None:
            logger.info("Building configuration...")
            config = build_config()
            logger.info(f"✓ Config built:")
            logger.info(f"  - Model: {config.trained_model_path}")
            logger.info(f"  - Stats: {config.stats_file}")
            logger.info(f"  - Data: {config.json_path}")
        
        # ========== STEP 2: LOAD DATA & MODEL ==========
        logger.info("\nLoading data and model...")
        metadata, model, transform, device = load_data_and_model(config, logger)
        
        # Create full metadata dataframe for querying
        training_df, val_df, test_df = dfs_from_metadata(metadata)
        metadata_full = pd.concat([training_df, val_df, test_df], ignore_index=True)
        
        # Get subset-specific dataframe
        subset_df = get_subset_df(metadata, subset, logger)
        
        # ========== STEP 3: VALIDATE & LOAD AARP ==========
        logger.info("\nValidating and loading AARP data...")
        s_aarp = validate_and_get_aarp(aarp_id, subset_df, metadata_full, logger)
        
        # ========== STEP 4: CREATE OUTPUT STRUCTURE ==========
        logger.info(f"\nCreating output directory structure...")
        output_paths = create_output_structure(aarp_id, output_home)
        logger.info(f"✓ Output directories created:")
        logger.info(f"  - Root: {output_paths['root']}")
        logger.info(f"  - Predictions: {output_paths['predictions']}")
        logger.info(f"  - Attributions: {output_paths['attributions']}")
        
        # ========== STEP 5: RUN ANALYSIS STAGES ==========
        # NOTE: Analysis functions will be called here in Phase 1, Step 4 (rest of implementation)
        logger.info("\nReady to run analysis stages:")
        logger.info("  [ ] Run predictions & integrated gradients")
        logger.info("  [ ] Generate prediction plots")
        logger.info("  [ ] Generate attribution plots")
        logger.info("  [ ] Generate summary statistics")
        
        # Placeholder for analysis function calls:
        # - run_pred_and_ig(aarp_id, metadata_full, transform, model, device)
        # - make_prediction_plot(aarp_id, metadata_full, ...)
        # - create_plots(aarp_id, metadata_full, ...)
        
        # ========== SUCCESS ==========
        result['success'] = True
        result['output_paths'] = {k: str(v) for k, v in output_paths.items()}
        logger.info("\n✓ Analysis pipeline setup complete (ready for analysis stages)")
        
    except Exception as e:
        logger.exception(f"✗ Error during analysis: {str(e)}")
        result['error'] = str(e)
        result['success'] = False
    
    finally:
        result['end_time'] = datetime.now()
        result['duration_seconds'] = (result['end_time'] - result['start_time']).total_seconds()
        
        if result['success']:
            logger.info(f"\nCompleted in {result['duration_seconds']:.2f} seconds")
        logger.info("=" * 70)
    
    return result


# ==================== CLI & MAIN ====================

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Per-AARP level analysis and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze AARP 12345 from test set
  python src/torch/vit/per_aarp_analysis.py --aarp-id 12345
  
  # Analyze AARP 12345 from validation set
  python src/torch/vit/per_aarp_analysis.py --aarp-id 12345 --subset validation
  
  # Analyze with custom output location
  python src/torch/vit/per_aarp_analysis.py --aarp-id 12345 --output-home my_plots/
        """
    )
    
    parser.add_argument(
        '--aarp-id',
        type=int,
        required=True,
        help='AARP region identifier to analyze'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        default='test',
        choices=['test', 'validation', 'training'],
        help='Data subset to analyze (default: test)'
    )
    
    parser.add_argument(
        '--output-home',
        type=str,
        default=DEFAULT_OUTPUT_HOME,
        help=f'Root output directory (default: {DEFAULT_OUTPUT_HOME})'
    )
    
    parser.add_argument(
        '--json-path',
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f'Path to solar_dataset.json (default: {DEFAULT_JSON_PATH})'
    )
    
    parser.add_argument(
        '--stats-file',
        type=str,
        default=DEFAULT_STATS_FILE,
        help=f'Path to stats.pkl (default: {DEFAULT_STATS_FILE})'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_TRAINED_MODEL_PATH,
        help=f'Path to trained model (default: {DEFAULT_TRAINED_MODEL_PATH})'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.aarp_id)
    
    # Build config with CLI arguments
    config = build_config(
        json_path=args.json_path,
        stats_file=args.stats_file,
        trained_model_path=args.model_path
    )
    
    # Run analysis
    result = run_per_aarp_analysis(
        aarp_id=args.aarp_id,
        subset=args.subset,
        config=config,
        output_home=args.output_home,
        logger=logger
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
