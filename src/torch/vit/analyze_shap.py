import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ==================== Module Constants ====================
DEFAULT_INPUT_JSON = "shap_stats.json"
DEFAULT_OUTPUT_DIR = "plots/kshap/results"
DEFAULT_FIGSIZE = (12, 7)
DEFAULT_DPI = 300
DEFAULT_ALPHA = 0.3
DEFAULT_POINT_SIZE = 4
DEFAULT_LINE_WIDTH = 1.5
AIA_CHANNEL_PREFIX = 'AIA_'
SIGNIFICANCE_THRESHOLD = 0.05

# ==================== Configuration Class ====================
@dataclass
class SHAPAnalysisConfig:
    """Configuration for SHAP statistical analysis and visualization.
    
    Attributes:
        input_json (str): Path to JSON file with pre-computed SHAP statistics
            (output from kshap.py). Defaults to "shap_stats.json".
        output_dir (str): Directory to save analysis plots. Defaults to "plots/kshap/results".
        figsize (tuple): Figure size (width, height) in inches. Defaults to (12, 7).
        dpi (int): Resolution in dots per inch. Defaults to 300.
        alpha (float): Transparency for histogram bars [0-1]. Defaults to 0.3.
        point_size (int): Size of scatter points in plots. Defaults to 4.
        line_width (float): Line width for axes and reference lines. Defaults to 1.5.
        significance_threshold (float): P-value threshold for significance. Defaults to 0.05.
    """
    input_json: str = DEFAULT_INPUT_JSON
    output_dir: str = DEFAULT_OUTPUT_DIR
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE
    dpi: int = DEFAULT_DPI
    alpha: float = DEFAULT_ALPHA
    point_size: int = DEFAULT_POINT_SIZE
    line_width: float = DEFAULT_LINE_WIDTH
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    
    def __post_init__(self):
        """Validate configuration after initialization.
        
        Raises:
            FileNotFoundError: If input JSON file does not exist.
            ValueError: If dpi, alpha, or point_size are invalid.
        """
        if not Path(self.input_json).exists():
            raise FileNotFoundError(f"Input JSON not found: {self.input_json}")
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.point_size <= 0:
            raise ValueError(f"point_size must be positive, got {self.point_size}")

# ==================== Helper Functions ====================
def get_aia_channels(df: pd.DataFrame) -> List[str]:
    """Extract AIA channel columns from DataFrame.
    
    Identifies columns that represent AIA wavelength channels by checking
    for the AIA_* naming convention.
    
    Args:
        df (pd.DataFrame): DataFrame with column names starting with 'AIA_'.
    
    Returns:
        List[str]: List of AIA channel column names (e.g., ['AIA_94', 'AIA_131', ...]).
    
    Example:
        >>> df = pd.DataFrame({'AIA_94': [1, 2], 'AIA_131': [3, 4], 'label': [0, 1]})
        >>> channels = get_aia_channels(df)
        >>> channels
        ['AIA_94', 'AIA_131']
    """
    return [col for col in df.columns if col.startswith(AIA_CHANNEL_PREFIX)]

def load_and_process(json_path: str) -> pd.DataFrame:
    """Load and process KernelSHAP statistics from JSON file.
    
    Reads a JSON file containing SHAP attribution results (output from kshap.py),
    flattens the nested structure, and returns as a pandas DataFrame for analysis.
    
    Expected JSON structure:
    [
        {
            "round": int,
            "label": int (0 or 1),
            "importance": {
                "AIA_94": float,
                "AIA_131": float,
                ...
            }
        },
        ...
    ]
    
    Args:
        json_path (str): Path to the SHAP statistics JSON file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['round', 'label', 'AIA_*', ...].
    
    Raises:
        FileNotFoundError: If json_path does not exist.
        json.JSONDecodeError: If JSON is malformed.
        KeyError: If required JSON structure is missing.
    
    Example:
        >>> df = load_and_process('shap_stats.json')
        >>> df.shape
        (50, 9)  # 50 rounds, 7 AIA channels + round + label
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validate structure
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of SHAP records")
    
    # Flatten the nested JSON structure for Pandas
    rows = []
    for entry in data:
        if 'importance' not in entry:
            raise KeyError(f"Record missing 'importance' field: {entry}")
        
        row = {
            'round': entry['round'],
            'label': entry['label']
        }
        # Merge the importance dictionary into the row
        row.update(entry['importance'])
        rows.append(row)

    return pd.DataFrame(rows)

def run_statistics(df: pd.DataFrame, config: SHAPAnalysisConfig) -> pd.DataFrame:
    """Compute statistical significance of SHAP importances by channel.
    
    Performs one-sample t-tests to determine if mean SHAP values are
    significantly different from zero (no importance). Prints formatted
    results table.
    
    Args:
        df (pd.DataFrame): DataFrame with SHAP statistics (output from load_and_process).
        config (SHAPAnalysisConfig): Analysis configuration including threshold.
    
    Returns:
        pd.DataFrame: Summary statistics table with columns:
            - Channel: AIA wavelength channel
            - Mean_SHAP: Mean SHAP attribution value
            - Std_Err: Standard error of the mean
            - P_Value: One-sample t-test p-value
            - Significant: Boolean whether p_value < threshold
    
    Example:
        >>> df = load_and_process('shap_stats.json')
        >>> config = SHAPAnalysisConfig()
        >>> stats_df = run_statistics(df, config)
        >>> stats_df[stats_df['Significant']]  # Get significant channels
    """
    channels = get_aia_channels(df)

    print(f"\n{'='*50}")
    print(f"GLOBAL IMPORTANCE STATISTICS")
    print(f"{'='*50}")

    results = []
    for ch in channels:
        mean_val = df[ch].mean()
        std_dev = df[ch].std()
        stderr = stats.sem(df[ch])

        # One-sample t-test: Is the importance significantly different from 0?
        t_stat, p_val = stats.ttest_1samp(df[ch], 0)

        results.append({
            'Channel': ch,
            'Mean_SHAP': mean_val,
            'Std_Err': stderr,
            'P_Value': p_val,
            'Significant': p_val < config.significance_threshold
        })

    stat_df = pd.DataFrame(results).sort_values(by='Mean_SHAP', ascending=False)
    print(stat_df.to_string(index=False))
    print(f"{'='*50}\n")
    return stat_df

def plot_results(
    df: pd.DataFrame,
    output_dir: str,
    config: SHAPAnalysisConfig
) -> None:
    """Generate and save statistical significance visualization.
    
    Creates a boxplot with overlaid scatter points showing the distribution of
    SHAP values across all AIA channels. Highlights the zero-importance line.
    
    Args:
        df (pd.DataFrame): DataFrame with SHAP statistics (output from load_and_process).
        output_dir (str): Directory to save the output plot.
        config (SHAPAnalysisConfig): Configuration with figure parameters (figsize, dpi, etc.).
    
    Side effects:
        - Creates output_dir if it does not exist
        - Saves plot as "significance_plot.png"
        - Prints confirmation message to stdout
    
    Example:
        >>> df = load_and_process('shap_stats.json')
        >>> config = SHAPAnalysisConfig(output_dir='plots/kshap')
        >>> plot_results(df, config.output_dir, config)
    """
    channels = get_aia_channels(df)
    df_melted = df.melt(
        id_vars=['round', 'label'],
        value_vars=channels,
        var_name='Passband',
        value_name='SHAP_Value'
    )

    plt.figure(figsize=config.figsize, dpi=150)  # DPI for interactive display
    sns.set_style("whitegrid")

    # Boxplot shows distribution and medians
    sns.boxplot(
        data=df_melted,
        x='Passband',
        y='SHAP_Value',
        palette='flare',
        showfliers=False
    )

    # Stripplot overlays individual points to see all samples
    sns.stripplot(
        data=df_melted,
        x='Passband',
        y='SHAP_Value',
        color='black',
        alpha=config.alpha,
        size=config.point_size,
        jitter=True
    )

    # Reference line at zero (no importance)
    plt.axhline(0, color='red', linestyle='--', linewidth=config.line_width)
    plt.title(
        f'Statistical Significance of Passband Importance (N={len(df)})',
        fontsize=14
    )
    plt.ylabel('Mean SHAP Attribution (Importance Score)')
    plt.xlabel('Solar Passband (AIA)')

    plt.tight_layout()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save figure at higher DPI for publication
    save_file = output_path / "significance_plot.png"
    plt.savefig(str(save_file), dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved to {save_file}")

# ==================== Main Block ====================
if __name__ == "__main__":
    try:
        # Load configuration
        config = SHAPAnalysisConfig()
        print(f"Loading SHAP statistics from {config.input_json}...")
        
        # Load and parse data
        dataframe = load_and_process(config.input_json)
        print(f"✓ Loaded {len(dataframe)} SHAP records")
        
        # Run statistical analysis
        print("Running statistical significance tests...")
        run_statistics(dataframe, config)
        
        # Generate visualization
        print(f"Generating plots and saving to {config.output_dir}...")
        plot_results(dataframe, config.output_dir, config)
        
        print("✓ SHAP analysis complete")
        
    except FileNotFoundError as e:
        print(f"✗ File error: {e}")
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
    except (ValueError, KeyError) as e:
        print(f"✗ Data validation error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
