"""
Exploratory Data Analysis and Visualization module for Iris Flower Classification.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif

from src.data_loader import FEATURE_NAMES, TARGET_NAME, load_iris_dataframe


def set_plotting_style():
    """Configure modern, aesthetic styling for plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300
    })


def generate_pairplot(df: pd.DataFrame, output_path: str = "reports/figures/pairplot.png"):
    """
    Generate and save a pairwise scatter and KDE distribution plot by species.
    """
    set_plotting_style()
    plot_df = df[FEATURE_NAMES + [TARGET_NAME]].copy()
    
    g = sns.pairplot(
        plot_df, 
        hue=TARGET_NAME, 
        markers=["o", "s", "D"],
        palette="viridis",
        diag_kind="kde",
        plot_kws={"alpha": 0.8, "s": 40},
        diag_kws={"fill": True, "alpha": 0.4}
    )
    g.fig.suptitle("Pairwise Feature Distributions Across Iris Species", y=1.02)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved pairplot to {output_path}")


def generate_boxplots(df: pd.DataFrame, output_path: str = "reports/figures/boxplots.png"):
    """
    Generate box plots showing the distribution of each feature grouped by species.
    """
    set_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("Set2", 3)
    
    for i, feature in enumerate(FEATURE_NAMES):
        sns.boxplot(
            x=TARGET_NAME, 
            y=feature, 
            data=df, 
            ax=axes[i], 
            palette=colors,
            boxprops=dict(alpha=0.8),
            fliersize=4
        )
        # Overlay jitter points for data density visibility
        sns.stripplot(
            x=TARGET_NAME,
            y=feature,
            data=df,
            ax=axes[i],
            color="black",
            alpha=0.3,
            jitter=0.2,
            size=4
        )
        axes[i].set_title(f"Distribution of {feature.replace('_', ' ').title()}", fontweight="bold")
        axes[i].set_xlabel("Species")
        axes[i].set_ylabel(f"{feature.replace('_', ' ').title()} (cm)")
        
    plt.suptitle("Feature Box Plots Grouped by Iris Species", y=0.98, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved box plots to {output_path}")


def generate_correlation_heatmap(df: pd.DataFrame, output_path: str = "reports/figures/correlation_heatmap.png"):
    """
    Compute and plot feature correlation matrix.
    """
    set_plotting_style()
    corr_matrix = df[FEATURE_NAMES].corr()
    
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=1.5,
        cbar_kws={"shrink": 0.8},
        mask=mask
    )
    plt.title("Iris Feature Correlation Heatmap (Lower Triangle)", pad=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")


def compute_feature_discriminative_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ANOVA F-value and p-value for each feature against species.
    Higher F-value signifies greater discriminative capability.
    """
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    f_scores, p_values = f_classif(X, y)
    
    discrim_df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in FEATURE_NAMES],
        "Feature_Key": FEATURE_NAMES,
        "ANOVA_F_Score": f_scores,
        "p_value": p_values
    }).sort_values(by="ANOVA_F_Score", ascending=False).reset_index(drop=True)
    
    return discrim_df


if __name__ == "__main__":
    df = load_iris_dataframe()
    generate_pairplot(df)
    generate_boxplots(df)
    generate_correlation_heatmap(df)
    
    discrim = compute_feature_discriminative_power(df)
    print("\nFeature Discriminative Power Ranking (ANOVA F-Score):")
    print(discrim.to_string(index=False))
