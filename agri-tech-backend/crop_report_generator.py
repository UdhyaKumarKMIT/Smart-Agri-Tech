import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import datetime
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR  = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-models")
DATA_DIR    = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-processed_data")
REPORTS_DIR = os.path.join(BASE_DIR, "report_crop")

data_path = os.path.join(DATA_DIR, "preprocessed_data.pkl")


# ============================================================
# HELPERS
# ============================================================

def _extract_shap_for_class(shap_values, pred_idx, sample_idx=0):
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[sample_idx, :, pred_idx]
        elif shap_values.ndim == 2:
            return shap_values[sample_idx]
        else:
            return shap_values.flatten()
    elif isinstance(shap_values, list):
        idx = pred_idx if pred_idx < len(shap_values) else 0
        arr = np.array(shap_values[idx])
        if arr.ndim == 2:
            return arr[sample_idx]
        return arr.flatten()
    else:
        return np.array(shap_values).flatten()


def _mean_abs_shap_per_feature(shap_values):
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return np.mean(np.abs(shap_values), axis=(0, 2))
        elif shap_values.ndim == 2:
            return np.mean(np.abs(shap_values), axis=0)
        else:
            return np.abs(shap_values).flatten()
    elif isinstance(shap_values, list):
        stacked = np.stack([np.abs(np.array(sv)) for sv in shap_values], axis=0)
        return np.mean(stacked, axis=(0, 1))
    else:
        return np.abs(np.array(shap_values)).flatten()


def _shap_for_beeswarm(shap_values):
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[:, :, 0]
        return shap_values
    elif isinstance(shap_values, list):
        return np.array(shap_values[0])
    return np.array(shap_values)


# ============================================================
class CropReportGenerator:
# ============================================================

    def __init__(self, model, label_encoder, feature_names, reports_dir):
        self.model               = model
        self.label_encoder       = label_encoder
        self.feature_names       = feature_names
        self.feature_short_names = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
        self.reports_dir         = REPORTS_DIR

        os.makedirs(REPORTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(REPORTS_DIR, "data"),   exist_ok=True)
        os.makedirs(os.path.join(REPORTS_DIR, "images"), exist_ok=True)

        self.X_train = None
        self.y_train = None
        self.X_test  = None
        self.y_test  = None
        self.load_training_data()

        self.shap_explainer = None
        self.lime_explainer = None

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------

    def load_training_data(self):
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_test  = data['X_test']
            self.y_test  = data['y_test']

            n = len(self.feature_names)
            if self.X_train.shape[1] != n:
                print(f"⚠️ Feature count mismatch: {self.X_train.shape[1]} vs {n}")
                if self.X_train.shape[1] > n:
                    self.X_train = self.X_train[:, :n]
                if self.X_test.shape[1] > n:
                    self.X_test  = self.X_test[:, :n]

            print(f"✅ Training data loaded: {self.X_train.shape}")

        except Exception as e:
            print(f"⚠️ Could not load training data: {e}")
            np.random.seed(42)
            n = len(self.feature_names)
            self.X_train = np.random.rand(100, n)
            self.y_train = np.random.randint(0, len(self.label_encoder.classes_), 100)
            self.X_test  = np.random.rand(30,  n)
            self.y_test  = np.random.randint(0, len(self.label_encoder.classes_), 30)
            print("⚠️ Using dummy training data")

    # ----------------------------------------------------------
    # EXPLAINER INITIALISATION
    # ----------------------------------------------------------

    def init_shap_explainer(self):
        if self.shap_explainer is not None:
            return
        try:
            bg_size = min(50, len(self.X_train))
            X_bg    = self.X_train[:bg_size]
            if X_bg.shape[1] > len(self.feature_names):
                X_bg = X_bg[:, :len(self.feature_names)]
            print(f"Initializing SHAP explainer with background shape: {X_bg.shape}")
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP explainer initialized successfully")
        except Exception as e:
            print(f"⚠️ SHAP explainer initialization failed: {e}")

    def init_lime_explainer(self):
        if self.lime_explainer is not None:
            return
        try:
            X_tr = self.X_train
            if X_tr.shape[1] > len(self.feature_names):
                X_tr = X_tr[:, :len(self.feature_names)]
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data         = X_tr,
                feature_names         = self.feature_names,
                class_names           = self.label_encoder.classes_,
                mode                  = 'classification',
                discretize_continuous = True,
                random_state          = 42
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ LIME explainer initialization failed: {e}")

    # ----------------------------------------------------------
    # INTERNAL HELPER
    # ----------------------------------------------------------

    def _safe_features(self, features):
        features = np.array(features, dtype=float)
        if features.ndim > 1:
            features = features.flatten()
        n = len(self.feature_names)
        if len(features) > n:
            features = features[:n]
        elif len(features) < n:
            features = np.pad(features, (0, n - len(features)), 'constant')
        return features

    # ==========================================================
    # GLOBAL EXPLAINABILITY
    # ==========================================================

    def generate_global_feature_importance(self, timestamp):
        print("\n📊 Generating Global Feature Importance Plot...")
        try:
            fi = self.model.feature_importances_
            n  = len(self.feature_names)

            if len(fi) > n:
                fi = fi[:n]
            elif len(fi) < n:
                fi = np.pad(fi, (0, n - len(fi)), 'constant')

            importance_df = pd.DataFrame({
                'Feature'    : self.feature_names,
                'Short_Name' : self.feature_short_names,
                'Importance' : fi
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))
            colors  = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars    = ax.barh(importance_df['Feature'],
                              importance_df['Importance'], color=colors)
            ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features',         fontsize=14, fontweight='bold')
            ax.set_title(
                'Feature Importance',
                fontsize=16, fontweight='bold', pad=20
            )
            ax.invert_yaxis()

            for bar, val in zip(bars, importance_df['Importance']):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"feature_importance_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()

            csv_path = os.path.join(REPORTS_DIR, "data",
                                    f"feature_importance_{timestamp}.csv")
            importance_df.to_csv(csv_path, index=False)

            print("✅ Feature importance plot generated")
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating feature importance: {e}")
            import traceback; traceback.print_exc()
            return None

    # ----------------------------------------------------------

    def generate_shap_summary_plots(self, timestamp):
        print("\n📊 Generating SHAP Summary Plots...")
        try:
            self.init_shap_explainer()
            if self.shap_explainer is None:
                print("⚠️ SHAP explainer not initialized")
                return None, None

            sample_size   = min(100, len(self.X_test))
            X_test_sample = self.X_test[:sample_size]

            shap_values = self.shap_explainer.shap_values(X_test_sample)

            # mean |SHAP| per feature — 1-D, length = n_features
            mean_shap = _mean_abs_shap_per_feature(shap_values)
            mean_shap = mean_shap[:len(self.feature_names)]

            importance_df = pd.DataFrame({
                "Feature"    : self.feature_names,
                "Importance" : mean_shap
            }).sort_values("Importance", ascending=True)

            # ── Plot 1: Bar ──────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_colors = [
                "#E8593C" if v == importance_df["Importance"].max()
                else "#185FA5"
                for v in importance_df["Importance"]
            ]
            ax.barh(importance_df["Feature"],
                    importance_df["Importance"],
                    color=bar_colors)
            ax.set_title("SHAP Feature Importance",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel("Mean |SHAP Value| (averaged across all classes & samples)")
            ax.set_ylabel("Feature")

            for i, val in enumerate(importance_df["Importance"]):
                ax.text(val + 0.0005, i, f"{val:.4f}", va='center', fontsize=9)

            plt.tight_layout()
            bar_path = os.path.join(REPORTS_DIR, "images",
                                    f"shap_importance_{timestamp}.png")
            plt.savefig(bar_path, dpi=100, bbox_inches='tight')
            plt.close()

            # ── Plot 2: Beeswarm ─────────────────────────────────────
            shap_2d = _shap_for_beeswarm(shap_values)
            shap_2d = shap_2d[:, :len(self.feature_names)]

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_2d,
                X_test_sample,
                feature_names = self.feature_names,
                show          = False
            )
            plt.title("SHAP Summary (Beeswarm)",
                      fontsize=14, fontweight='bold')
            dot_path = os.path.join(REPORTS_DIR, "images",
                                    f"shap_dot_summary_{timestamp}.png")
            plt.savefig(dot_path, dpi=100, bbox_inches='tight')
            plt.close()

            print("✅ SHAP summary plots generated")
            return bar_path, dot_path

        except Exception as e:
            print(f"⚠️ Error generating SHAP plots: {e}")
            import traceback; traceback.print_exc()
            return None, None

    # ----------------------------------------------------------

    def generate_partial_dependence_plots(self, timestamp, predicted_crop):
        """
        PDP — exactly mirrors the working Colab notebook code.
        Uses RF feature_importances_ to pick top 4 features,
        resolves target class index from predicted_crop name.
        """
        print("\n📊 Generating Partial Dependency Plots...")
        try:
            # ── Top 4 features by RF importance (same as notebook) ──
            fi_series = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)

            top_features = fi_series.head(4).index.tolist()
            top_indices  = [list(self.feature_names).index(f) for f in top_features]

            # ── Resolve target class index from crop name ──
            classes_list     = list(self.label_encoder.classes_)
            target_class_idx = (classes_list.index(predicted_crop)
                                if predicted_crop in classes_list else 0)
            target_class_name = classes_list[target_class_idx]

            print(f"Generating PDP for class: {target_class_name}")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            PartialDependenceDisplay.from_estimator(
                self.model,
                self.X_train,
                features      = top_indices,
                feature_names = self.feature_names,
                target        = target_class_idx,
                kind          = "average",
                ax            = axes,
                grid_resolution = 50,
                random_state  = 42
            )

            fig.suptitle(
                f"Partial Dependency Plots — Top 4 Features\n"
                f"(Target crop: {target_class_name})",
                fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"partial_dependence_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()

            print("✅ Partial Dependency Plot generated")
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating PDP: {e}")
            import traceback; traceback.print_exc()
            return None

    # ----------------------------------------------------------

    def generate_crop_distribution(self, timestamp):
        try:
            if self.y_train is None:
                return None

            unique_labels           = np.unique(self.y_train)
            crop_counts, crop_names = [], []

            for label in unique_labels:
                crop_counts.append(np.sum(self.y_train == label))
                if label < len(self.label_encoder.classes_):
                    crop_names.append(
                        self.label_encoder.inverse_transform([label])[0])
                else:
                    crop_names.append(f"Crop_{label}")

            fig, ax = plt.subplots(figsize=(12, 6))
            colors  = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
            bars    = ax.bar(range(len(crop_counts)), crop_counts, color=colors)
            ax.set_title('Distribution of Crop Types in Training Data',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Crop Type')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(crop_counts)))
            ax.set_xticklabels(crop_names, rotation=45, ha='right')

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{int(h)}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"crop_distribution_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating crop distribution: {e}")
            return None

    # ----------------------------------------------------------

    def generate_input_distribution(self, timestamp):
        try:
            if self.X_train is None:
                return None

            n_features = len(self.feature_names)
            n_rows     = (n_features + 3) // 4

            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
            axes      = axes.flatten()

            for i in range(4 * n_rows):
                if i < n_features:
                    axes[i].hist(self.X_train[:, i], bins=20,
                                 color='skyblue', edgecolor='black', alpha=0.7)
                    axes[i].set_title(
                        f'Distribution of {self.feature_short_names[i]}',
                        fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(self.feature_names[i])
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].set_visible(False)

            plt.suptitle('Distribution of Input Features',
                         fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"input_distribution_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating input distribution: {e}")
            return None

    # ----------------------------------------------------------

    def generate_correlation_heatmap(self, timestamp):
        try:
            if self.X_train is None:
                return None

            cols = self.feature_short_names[:self.X_train.shape[1]]
            df   = pd.DataFrame(self.X_train, columns=cols)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0,
                        square=True, ax=ax, fmt='.2f',
                        cbar_kws={"shrink": 0.8})
            ax.set_title('Feature Correlation Heatmap',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"correlation_heatmap_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating correlation heatmap: {e}")
            return None

    # ----------------------------------------------------------

    def generate_decision_tree_surrogate(self, timestamp):
        try:
            from sklearn.tree import DecisionTreeClassifier, plot_tree

            if self.X_train is None or self.y_train is None:
                return None

            n    = len(self.feature_names)
            X_tr = self.X_train[:, :n]
            X_te = self.X_test[:, :n]

            surrogate = DecisionTreeClassifier(max_depth=4, random_state=42)
            surrogate.fit(X_tr, self.y_train)
            acc = surrogate.score(X_te, self.y_test)

            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(surrogate,
                      feature_names = self.feature_names,
                      class_names   = self.label_encoder.classes_,
                      filled=True, rounded=True,
                      fontsize=10, max_depth=3, ax=ax)
            plt.title(f'Decision Tree Surrogate  (Accuracy: {acc:.2f})',
                      fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()

            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"decision_tree_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating decision tree: {e}")
            return None

    # ==========================================================
    # LOCAL EXPLAINABILITY
    # ==========================================================

    def generate_shap_plot(self, features, predicted_crop, timestamp):
        try:
            self.init_shap_explainer()
            if self.shap_explainer is None:
                return None

            features  = self._safe_features(features)
            pred_idx  = list(self.label_encoder.classes_).index(predicted_crop)

            shap_values       = self.shap_explainer.shap_values(
                                    features.reshape(1, -1))
            shap_values_class = _extract_shap_for_class(
                                    shap_values, pred_idx, sample_idx=0)
            shap_values_class = shap_values_class[:len(self.feature_names)]

            order        = np.argsort(np.abs(shap_values_class))[::-1]
            shap_sorted  = shap_values_class[order]
            names_sorted = [self.feature_names[i] for i in order]
            vals_sorted  = features[order]

            colors = ['#E8593C' if v > 0 else '#185FA5' for v in shap_sorted]

            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos   = np.arange(len(self.feature_names))
            bars    = ax.barh(y_pos, shap_sorted, color=colors,
                              alpha=0.85, edgecolor='white', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [f"{n}  (val={v:.2f})"
                 for n, v in zip(names_sorted, vals_sorted)],
                fontsize=11
            )
            ax.set_xlabel(
                'SHAP Value  →  positive = supports prediction, '
                'negative = opposes prediction',
                fontsize=11, fontweight='bold'
            )
            ax.set_title(f'Local SHAP — Why predicted: {predicted_crop}',
                         fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, shap_sorted):
                offset = 0.002 if val >= 0 else -0.002
                ha     = 'left' if val >= 0 else 'right'
                ax.text(val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:+.4f}', va='center', ha=ha,
                        fontweight='bold', fontsize=9)

            pos_p = mpatches.Patch(color='#E8593C', label='Supports prediction')
            neg_p = mpatches.Patch(color='#185FA5', label='Opposes prediction')
            ax.legend(handles=[pos_p, neg_p], fontsize=10)

            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"shap_local_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating SHAP plot: {e}")
            import traceback; traceback.print_exc()
            return None

    # ----------------------------------------------------------

    def generate_lime_explanation(self, features, predicted_crop, timestamp):
        try:
            self.init_lime_explainer()
            if self.lime_explainer is None:
                return None

            features = self._safe_features(features)

            lime_exp = self.lime_explainer.explain_instance(
                data_row     = features,
                predict_fn   = self.model.predict_proba,
                num_features = len(self.feature_names),
                top_labels   = 1
            )

            top_label_idx  = lime_exp.top_labels[0]
            top_label_name = self.label_encoder.classes_[top_label_idx]
            lime_vals      = lime_exp.as_list(label=top_label_idx)

            feat_labels  = [x[0] for x in lime_vals]
            feat_weights = [x[1] for x in lime_vals]
            colors       = ['#E8593C' if v > 0 else '#185FA5'
                            for v in feat_weights]

            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.barh(feat_labels, feat_weights, color=colors,
                           edgecolor='white', height=0.55, alpha=0.85)
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xlabel('LIME Weight  →  contribution to prediction',
                          fontsize=11, fontweight='bold')
            ax.set_title(
                f'LIME Explanation — Predicted: {predicted_crop}\n'
                f'(Explaining class: {top_label_name})',
                fontsize=13, fontweight='bold'
            )

            for bar, val in zip(bars, feat_weights):
                offset = 0.001 if val >= 0 else -0.001
                ha     = 'left' if val >= 0 else 'right'
                ax.text(val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:+.4f}', va='center', ha=ha, fontsize=9)

            pos_p = mpatches.Patch(color='#E8593C', label='Supports prediction')
            neg_p = mpatches.Patch(color='#185FA5', label='Opposes prediction')
            ax.legend(handles=[pos_p, neg_p], fontsize=10)

            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"lime_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating LIME explanation: {e}")
            import traceback; traceback.print_exc()
            return None

    # ----------------------------------------------------------

    def generate_individual_breakdown(self, features, predicted_crop, timestamp):
        try:
            self.init_shap_explainer()
            if self.shap_explainer is None:
                return None

            features  = self._safe_features(features)
            pred_idx  = list(self.label_encoder.classes_).index(predicted_crop)

            shap_values   = self.shap_explainer.shap_values(
                                features.reshape(1, -1))
            shap_for_pred = _extract_shap_for_class(
                                shap_values, pred_idx, sample_idx=0)
            shap_for_pred = shap_for_pred[:len(self.feature_names)]

            breakdown_df = pd.DataFrame({
                "Feature"          : self.feature_names,
                "Feature Value"    : features,
                "SHAP Contribution": shap_for_pred
            }).sort_values("SHAP Contribution", ascending=True)

            colors = ['#E8593C' if v > 0 else '#185FA5'
                      for v in breakdown_df["SHAP Contribution"]]

            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.barh(breakdown_df["Feature"],
                           breakdown_df["SHAP Contribution"],
                           color=colors, edgecolor='white',
                           height=0.55, alpha=0.85)

            for bar, (_, row) in zip(bars, breakdown_df.iterrows()):
                val    = row["SHAP Contribution"]
                label  = (f"  val={row['Feature Value']:.2f}"
                          f"   SHAP={val:+.5f}")
                offset = 0.0002 if val >= 0 else -0.0002
                ha     = 'left'  if val >= 0 else 'right'
                ax.text(val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        label, va='center', ha=ha, fontsize=9)

            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel(
                'SHAP Value  →  contribution to predicted probability',
                fontsize=11, fontweight='bold'
            )
            ax.set_title(
                f'Individual Feature Breakdown — Predicted: {predicted_crop}',
                fontsize=13, fontweight='bold'
            )

            pos_p = mpatches.Patch(color='#E8593C', label='Pushes toward prediction')
            neg_p = mpatches.Patch(color='#185FA5', label='Pushes away from prediction')
            ax.legend(handles=[pos_p, neg_p], fontsize=10)

            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"individual_breakdown_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating individual breakdown: {e}")
            import traceback; traceback.print_exc()
            return None

    # ----------------------------------------------------------

    def generate_fallback_explanation(self, features, predicted_crop, timestamp):
        try:
            features    = self._safe_features(features)
            features_2d = features.reshape(1, -1)
            base_pred   = self.model.predict_proba(features_2d)[0]
            pred_idx    = list(self.label_encoder.classes_).index(predicted_crop)
            n_features  = len(self.feature_names)
            impacts     = []

            for i in range(n_features):
                fp    = features.copy()
                fp[i] *= 1.1
                new_p = self.model.predict_proba(fp.reshape(1, -1))[0]
                impacts.append(new_p[pred_idx] - base_pred[pred_idx])

            order           = np.argsort(np.abs(impacts))[::-1]
            sorted_features = [self.feature_names[i] for i in order]
            sorted_impacts  = [impacts[i]            for i in order]
            colors          = ['#1D9E75' if w > 0 else '#E8593C'
                               for w in sorted_impacts]

            fig, ax = plt.subplots(figsize=(14, 8))
            y_pos   = np.arange(len(sorted_impacts))
            bars    = ax.barh(y_pos, sorted_impacts, color=colors,
                              alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features, fontsize=11)
            ax.set_xlabel('Impact on Prediction (Feature +10%)',
                          fontsize=12, fontweight='bold')
            ax.set_title(
                f'Local Feature Impact Analysis for {predicted_crop} (Fallback)',
                fontsize=14, fontweight='bold'
            )
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, sorted_impacts):
                offset = 0.005 if val >= 0 else -0.01
                ha     = 'left' if val >= 0 else 'right'
                ax.text(val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:+.4f}', va='center', ha=ha,
                        fontweight='bold', fontsize=10)

            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "images",
                                     f"fallback_explanation_{timestamp}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            return plot_path

        except Exception as e:
            print(f"⚠️ Error generating fallback explanation: {e}")
            return None

    # ==========================================================
    # ORCHESTRATION
    # ==========================================================

    def generate_all_plots(self, timestamp, predicted_crop=""):
        plots = {}
        print("\n📊 Generating all plots for crop report...")

        print("   1/6 Generating Feature Importance Plot...")
        plots['feature_importance'] = self.generate_global_feature_importance(timestamp)

        print("   2/6 Generating Crop Distribution Plot...")
        plots['crop_distribution']  = self.generate_crop_distribution(timestamp)

        print("   3/6 Generating Input Distribution Plot...")
        plots['input_distribution'] = self.generate_input_distribution(timestamp)

        print("   4/6 Generating Correlation Heatmap...")
        plots['correlation']        = self.generate_correlation_heatmap(timestamp)

        print("   5/6 Generating SHAP Summary Plots...")
        shap_imp, shap_dot          = self.generate_shap_summary_plots(timestamp)
        plots['shap_importance']    = shap_imp
        plots['shap_dot']           = shap_dot

        print("   6/6 Generating Decision Tree Surrogate...")
        plots['decision_tree']      = self.generate_decision_tree_surrogate(timestamp)

        return plots

    # ----------------------------------------------------------

    def generate_prediction_report(self, input_data, prediction_result):
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id  = f"report_{timestamp}"

        features = np.array([
            input_data["nitrogen"],
            input_data["phosphorus"],
            input_data["potassium"],
            input_data["temperature"],
            input_data["humidity"],
            input_data["ph"],
            input_data["rainfall"]
        ], dtype=float)

        predicted_crop = prediction_result["crop"]
        confidence     = prediction_result.get("confidence", 0.95)

        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_crops   = self.label_encoder.inverse_transform(top_5_indices)
        top_5_probs   = probabilities[top_5_indices]

        # ── Global plots ──────────────────────────────────────
        plots = self.generate_all_plots(timestamp, predicted_crop)

        # ── Local XAI plots ───────────────────────────────────
        print("\n📊 Generating local XAI plots...")

        print("   [1/4] SHAP local waterfall...")
        plots['shap_local'] = self.generate_shap_plot(
            features, predicted_crop, timestamp)

        print("   [2/4] LIME explanation...")
        plots['lime'] = self.generate_lime_explanation(
            features, predicted_crop, timestamp)

        print("   [3/4] Individual feature breakdown...")
        plots['individual_breakdown'] = self.generate_individual_breakdown(
            features, predicted_crop, timestamp)

        print("   [4/4] Partial Dependency Plot...")
        plots['partial_dependence'] = self.generate_partial_dependence_plots(
            timestamp, predicted_crop)

        # ── HTML report ───────────────────────────────────────
        html_content = self.create_html_report(
            report_id      = report_id,
            timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_data     = input_data,
            predicted_crop = predicted_crop,
            confidence     = confidence,
            top_5_crops    = top_5_crops,
            top_5_probs    = top_5_probs,
            plots          = plots
        )

        report_path = os.path.join(REPORTS_DIR, f"{report_id}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        latest_path = os.path.join(REPORTS_DIR, "latest_report.html")
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        data_out = os.path.join(REPORTS_DIR, "data", f"{report_id}.json")
        with open(data_out, 'w') as f:
            json.dump({
                "input"      : input_data,
                "prediction" : prediction_result,
                "timestamp"  : timestamp
            }, f, indent=2)

        print(f"\n✅ Report generated: {report_path}")
        return report_path

    # ----------------------------------------------------------

    def create_html_report(self, report_id, timestamp, input_data,
                           predicted_crop, confidence,
                           top_5_crops, top_5_probs, plots):

        feature_table = ""
        for name, value in [
            ("Nitrogen (N)",     input_data["nitrogen"]),
            ("Phosphorus (P)",   input_data["phosphorus"]),
            ("Potassium (K)",    input_data["potassium"]),
            ("Temperature (°C)", input_data["temperature"]),
            ("Humidity (%)",     input_data["humidity"]),
            ("Soil pH",          input_data["ph"]),
            ("Rainfall (mm)",    input_data["rainfall"]),
        ]:
            feature_table += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{value:.2f}</td>
            </tr>"""

        # Only show crops with probability > 0%
        prob_table = ""
        for i, (crop, prob) in enumerate(zip(top_5_crops, top_5_probs)):
            if prob > 0.0:
                prob_table += f"""
                <tr{' class="highlight"' if i == 0 else ''}>
                    <td>{i + 1}</td>
                    <td><strong>{crop}</strong></td>
                    <td>{prob:.2%}</td>
                </tr>"""

        def img_path(filename):
            if filename and os.path.exists(filename):
                return f"images/{os.path.basename(filename)}"
            return ""

        successful_plots = sum(1 for v in plots.values() if v is not None)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Prediction Report - {predicted_crop}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto;
            background-color: white; border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .report-meta {{
            background-color: #f8f9fa; padding: 15px 30px;
            border-bottom: 2px solid #e9ecef;
            display: flex; justify-content: space-between;
        }}
        .main-result {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 40px 30px; text-align: center; color: white;
        }}
        .crop-name {{
            font-size: 4em; font-weight: 800; margin: 10px 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        }}
        .content {{ padding: 40px; }}
        .grid-2 {{
            display: grid; grid-template-columns: repeat(2, 1fr);
            gap: 30px; margin-bottom: 30px;
        }}
        .card {{
            background-color: white; border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden; border: 1px solid #e9ecef;
        }}
        .card-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 15px 20px;
            font-size: 1.2em; font-weight: bold;
        }}
        .card-body {{ padding: 20px; }}
        .plot-container {{
            text-align: center;
            padding: 20px;
            background-color: #fafafa;
            min-height: 350px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .plot-container img {{
            width: 100%;
            height: auto;
            max-height: 600px;
            object-fit: contain;
            border-radius: 10px;
            border: 2px solid #e9ecef;
            display: block;
        }}
        .no-plot {{
            padding: 40px; color: #999; font-style: italic;
            text-align: center;
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        table th, table td {{
            padding: 12px; text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        .highlight {{
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            font-weight: bold;
        }}
        .section-title {{
            font-size: 2em; margin: 40px 0 25px 0; color: #2c3e50;
            border-bottom: 4px solid #667eea; padding-bottom: 10px;
        }}
        .footer {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white; padding: 30px; text-align: center;
        }}
        .button {{
            display: inline-block; padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; text-decoration: none; border-radius: 25px;
            margin: 5px; border: none; cursor: pointer;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .crop-name {{ font-size: 2.5em; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>🌾 Crop Recommendation System</h1>
        <p>Complete XAI Analysis Report</p>
    </div>

    <div class="report-meta">
        <span>📋 Report ID: {report_id}</span>
        <span>📅 Generated: {timestamp}</span>
    </div>

    <div class="main-result">
        <h2>✅ Recommended Crop</h2>
        <div class="crop-name">{predicted_crop}</div>
    </div>

    <div class="content">

        <!-- ── Input Summary ── -->
        <h2 class="section-title">📝 Input Summary</h2>
        <div class="grid-2">
            <div class="card">
                <div class="card-header">📊 Input Parameters</div>
                <div class="card-body">
                    <table>{feature_table}</table>
                </div>
            </div>
            <div class="card">
                <div class="card-header">🎯 Top Predictions</div>
                <div class="card-body">
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Crop</th>
                            <th>Probability</th>
                        </tr>
                        {prob_table}
                    </table>
                </div>
            </div>
        </div>

        <!-- ── Data Exploration ── -->
        <h2 class="section-title">📊 Data Exploration</h2>
        <div class="grid-2">
            
            <div class="card">
                <div class="card-header">📈 Feature Distributions</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('input_distribution'))}"
                         alt="Input Distributions"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">🔗 Correlation Heatmap</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('correlation'))}"
                         alt="Correlation Heatmap"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
        </div>

        <!-- ── Global XAI — ordered as requested ── -->
        <h2 class="section-title">🌍 Global Model Interpretation</h2>
        <div class="grid-2">

            <!-- 1. Feature Importance -->
            <div class="card">
                <div class="card-header">📈 Feature Importance</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('feature_importance'))}"
                         alt="Feature Importance"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>

      
            <!-- 3. SHAP Beeswarm -->
            <div class="card">
                <div class="card-header">🔬 SHAP Summary (Beeswarm)</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('shap_dot'))}"
                         alt="SHAP Summary Beeswarm"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>

            <!-- 4. SHAP Bar -->
            <div class="card">
                <div class="card-header">🎯 SHAP Feature Importance</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('shap_importance'))}"
                         alt="SHAP Importance"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>

            <!-- 5. Decision Tree -->
            <div class="card">
                <div class="card-header">🌳 Decision Tree Surrogate</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('decision_tree'))}"
                         alt="Decision Tree Surrogate"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>

        </div>

        <!-- ── Local XAI ── -->
        <h2 class="section-title">🔍 Local Interpretation (This Prediction)</h2>
        <div class="grid-2">
            <div class="card">
                <div class="card-header">⚡ SHAP Local Waterfall</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('shap_local'))}"
                         alt="SHAP Local"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">🔍 LIME Explanation</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('lime'))}"
                         alt="LIME Explanation"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">📊 Individual Feature Breakdown</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('individual_breakdown'))}"
                         alt="Individual Breakdown"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
        </div>

    </div><!-- /content -->

    <div class="footer">
        <p>Report generated by Crop Recommendation System</p>
        <p>Total Plots Generated: {successful_plots}</p>
        <button class="button" onclick="window.print()">🖨️ Print</button>
        <button class="button" onclick="window.close()">❌ Close</button>
    </div>

</div><!-- /container -->
</body>
</html>"""
        return html