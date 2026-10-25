"""Model explainability and SHAP analysis module for Northstar SaaS Churn Predictor."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def extract_transformed_feature_names(prep_pipeline: Pipeline) -> List[str]:
    """Extract interpretable feature names from fitted ColumnTransformer."""
    col_trans = prep_pipeline.named_steps["preprocessor"]
    num_cols = list(col_trans.transformers_[0][2])
    cat_encoder = col_trans.transformers_[1][1].named_steps["onehot"]
    cat_cols = col_trans.transformers_[1][2]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    return num_cols + cat_feature_names


class ModelExplainer:
    """SHAP explanation engine for the fitted tree-based champion model."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["classifier"]
        self.feature_names = extract_transformed_feature_names(self.prep)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_dataset(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, shap.Explanation]:
        """Compute SHAP values on transformed input features.

        Returns:
            X_trans: Preprocessed feature matrix (numpy array).
            shap_vals: Array of SHAP values matching X_trans dimensions.
            explanation: Full shap.Explanation object.
        """
        X_trans = self.prep.transform(X)
        explanation = self.explainer(X_trans)
        explanation.feature_names = self.feature_names
        shap_vals = explanation.values
        return X_trans, shap_vals, explanation

    def compute_feature_importance(self, shap_vals: np.ndarray) -> pd.DataFrame:
        """Compute mean absolute SHAP values for global ranking."""
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        df_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.round(mean_abs_shap, 4),
            }
        ).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        return df_importance

    def plot_summary_beeswarm(
        self,
        explanation: shap.Explanation,
        max_display: int = 15,
        save_path: Path = None,
    ) -> Path:
        """Generate and save SHAP beeswarm summary plot."""
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.title("SHAP Global Feature Impact (Positive Churn Class)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "08_shap_summary_beeswarm.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")
        return out_path

    def plot_feature_importance_bar(
        self,
        df_importance: pd.DataFrame,
        top_n: int = 15,
        save_path: Path = None,
    ) -> Path:
        """Generate horizontal bar chart of top SHAP influential features."""
        top_df = df_importance.head(top_n).sort_values(by="mean_abs_shap", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        bars = ax.barh(
            top_df["feature"],
            top_df["mean_abs_shap"],
            color="#2b5c8f",
            edgecolor="black",
            height=0.6,
        )
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.set_title(f"Top {top_n} Most Influential Features (Mean |SHAP|)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds)", fontsize=10)
        ax.set_xlim(0, max(top_df["mean_abs_shap"]) * 1.2)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "09_shap_importance_bar.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_individual_waterfalls(
        self,
        explanation: shap.Explanation,
        representative_indices: Dict[str, int],
        save_dir: Path = None,
    ) -> Dict[str, Path]:
        """Generate SHAP waterfall explanations for representative customer archetypes."""
        out_dir = save_dir or FIGURES_DIR
        paths = {}

        for archetype, idx in representative_indices.items():
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
            shap.plots.waterfall(explanation[idx], max_display=10, show=False)
            plt.title(f"SHAP Waterfall: Archetype [{archetype}]", fontsize=12, fontweight="bold", pad=12)
            plt.tight_layout()

            out_path = out_dir / f"10_shap_waterfall_{archetype.lower().replace(' ', '_')}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close("all")
            paths[archetype] = out_path

        return paths

    def plot_dependence_panels(
        self,
        explanation: shap.Explanation,
        top_features: List[str] = ["tenure", "MonthlyCharges", "TotalServices"],
        save_path: Path = None,
    ) -> Path:
        """Generate SHAP dependence scatter plots for key continuous/discrete drivers."""
        fig, axes = plt.subplots(1, len(top_features), figsize=(15, 4.5), dpi=300)

        for i, feat in enumerate(top_features):
            if feat in self.feature_names:
                feat_idx = self.feature_names.index(feat)
                feat_vals = explanation.data[:, feat_idx]
                shap_vals_feat = explanation.values[:, feat_idx]

                axes[i].scatter(
                    feat_vals,
                    shap_vals_feat,
                    alpha=0.4,
                    color="#d95f02",
                    edgecolor="none",
                    s=20,
                )
                axes[i].axhline(0, color="grey", linestyle="--", alpha=0.7)
                axes[i].set_title(f"Dependence: {feat}", fontsize=11, fontweight="bold")
                axes[i].set_xlabel(feat, fontsize=10)
                axes[i].set_ylabel("SHAP Value", fontsize=10)
                axes[i].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("SHAP Dependence: Marginal Association with Churn Risk", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "11_shap_dependence_panels.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
