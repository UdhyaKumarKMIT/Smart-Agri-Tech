# fertilizer_report_generator.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import json
import pickle
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# Same MODELS_DIR as predict_fertilizer.py
MODELS_DIR  = os.path.join(BASE_DIR, "fertilizer-recommendation")
REPORTS_DIR = os.path.join(BASE_DIR, "report_fertilizer")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "data"), exist_ok=True)


# ==============================================================================
# HELPERS
# ==============================================================================

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


# ==============================================================================
class FertilizerReportGenerator:
# ==============================================================================

    def __init__(self):
        self.model              = None
        self.encoders           = None   # from label_encoders.pkl
        self.processed_data     = None
        self.X_train            = None
        self.y_train            = None
        self.X_test             = None
        self.y_test             = None
        self.df                 = None
        self.feature_names      = None
        self.fertilizer_classes = None
        self.model_accuracy     = None
        self.shap_explainer     = None
        self.lime_explainer     = None
        self.load_model_and_data()

    # --------------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------------

    def load_model_and_data(self):
        try:
            print("🔍 Loading model and data...")

            # ── stacked_ensemble.pkl ──────────────────────────────────────────
            model_path = os.path.join(MODELS_DIR, 'stacked_ensemble.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("   ✅ Model loaded")
            else:
                print(f"   ❌ Model not found at {model_path}")
                return False

            # ── label_encoders.pkl ────────────────────────────────────────────
            # Keys saved by your preprocessing notebook:
            #   label_encoder (fert), soil_encoder, crop_encoder, scaler,
            #   feature_names
            encoder_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                print("   ✅ Encoders loaded")

                # Fertilizer class names
                le_fert = (self.encoders.get('fert_encoder') or
                           self.encoders.get('label_encoder'))
                if le_fert is not None:
                    self.fertilizer_classes = le_fert.classes_.tolist()

                # Feature names list
                self.feature_names = self.encoders.get('feature_names')
            else:
                print(f"   ❌ Encoders not found at {encoder_path}")
                return False

            # ── preprocessed_data.pkl (for X_train, df_original, etc.) ────────
            candidate_paths = [
                os.path.join(MODELS_DIR, 'processed_data.pkl'),
                os.path.join(MODELS_DIR, 'preprocessed_data.pkl'),
                os.path.join(MODELS_DIR, 'processed_data_fertilizer',
                             'preprocessed_data.pkl'),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        self.processed_data = pickle.load(f)
                    print(f"   ✅ Processed data loaded from {p}")
                    break

            if self.processed_data:
                self.X_train = self.processed_data.get('X_train')
                self.y_train = self.processed_data.get('y_train')
                self.X_test  = self.processed_data.get('X_test')
                self.y_test  = self.processed_data.get('y_test')
                self.df      = self.processed_data.get('df_original')

                # Override feature_names if available in processed_data
                if self.processed_data.get('feature_names'):
                    self.feature_names = list(self.processed_data['feature_names'])

                # Override fertilizer classes if available in processed_data
                for key in ('fert_encoder', 'label_encoder', 'le_fert'):
                    le = self.processed_data.get(key)
                    if le is not None:
                        self.fertilizer_classes = le.classes_.tolist()
                        break

                print(f"   ✅ X_train shape : "
                      f"{self.X_train.shape if self.X_train is not None else 'None'}")
            else:
                print("   ⚠️  Processed data not found — EDA plots will be skipped")

            # Normalise feature_names to plain list
            if self.feature_names is not None:
                self.feature_names = list(self.feature_names)

            print(f"   ✅ Feature names  : {self.feature_names}")
            print(f"   ✅ Classes        : "
                  f"{len(self.fertilizer_classes) if self.fertilizer_classes else 0}")
            return True

        except Exception as e:
            print(f"❌ Error in load_model_and_data: {e}")
            import traceback; traceback.print_exc()
            return False

    # --------------------------------------------------------------------------
    # BUILD CORRECT 8-FEATURE VECTOR
    # --------------------------------------------------------------------------

    def build_features_row(self, input_data):
        """
        Convert raw input dict → correctly encoded & scaled 8-dim numpy vector.

        Training column order (from your preprocessing notebook):
            Soil Type (encoded), Crop Type (encoded),
            Nitrogen, Phosphorus, Potassium,
            Temperature, Humidity, soil_moisture

        Encoders come from label_encoders.pkl — the same file already used by
        predict_fertilizer.py — so the transformation is identical.
        No padding warnings will appear after this fix.
        """
        try:
            if self.encoders is None:
                raise ValueError("Encoders not loaded")

            soil_encoder = self.encoders.get('soil_encoder')
            crop_encoder = self.encoders.get('crop_encoder')
            scaler       = self.encoders.get('scaler')

            soil_type = input_data.get('soil_type', '')
            crop_type = input_data.get('crop_type', '')

            soil_encoded = int(soil_encoder.transform([soil_type])[0]) if soil_encoder else 0
            crop_encoded = int(crop_encoder.transform([crop_type])[0]) if crop_encoder else 0

            # Raw values in training feature order
            raw = np.array([[
                soil_encoded,
                crop_encoded,
                float(input_data.get('nitrogen',    0)),
                float(input_data.get('phosphorus',  0)),
                float(input_data.get('potassium',   0)),
                float(input_data.get('temperature', 0)),
                float(input_data.get('humidity',    0)),
                float(input_data.get('soil_moisture',
                      input_data.get('moisture',    0))),
            ]])

            # Apply the same StandardScaler used during training
            if scaler is not None:
                raw = scaler.transform(raw)

            print(f"✅ features_row built: shape={raw.shape}")
            return raw.flatten()

        except Exception as e:
            print(f"⚠️  build_features_row failed ({e}) — will use training fallback")
            return None

    # --------------------------------------------------------------------------
    # EXPLAINER INIT
    # --------------------------------------------------------------------------

    def init_shap_explainer(self):
        if self.shap_explainer is not None:
            return
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️  SHAP explainer init failed: {e}")

    def init_lime_explainer(self):
        if self.lime_explainer is not None:
            return
        try:
            X_tr = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data         = X_tr,
                feature_names         = self.feature_names,
                class_names           = self.fertilizer_classes,
                mode                  = 'classification',
                discretize_continuous = True,
                random_state          = 42
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️  LIME explainer init failed: {e}")

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------

    def save_plot(self, fig, filename):
        try:
            filepath = os.path.join(REPORTS_DIR, "images", filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return filepath
        except Exception as e:
            print(f"❌ Error saving plot {filename}: {e}")
            plt.close()
            return None

    def _safe_row(self, row):
        if hasattr(row, 'values'):
            return row.values.flatten().astype(float)
        return np.array(row, dtype=float).flatten()

    def _align_features(self, features):
        """
        Safety net only — should not trigger after build_features_row is used.
        If dimensions still mismatch, pad/trim with training medians.
        """
        if self.X_train is None:
            return features
        X_tr     = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
        expected = X_tr.shape[1]
        current  = features.shape[0]
        if current == expected:
            return features
        if current < expected:
            print(f"⚠️  _align_features: padding {expected - current} col(s)")
            medians  = np.median(X_tr, axis=0)
            features = np.concatenate([features, medians[current:]])
        else:
            features = features[:expected]
        return features

    # ==========================================================================
    # EDA PLOTS
    # ==========================================================================

    def generate_fertilizer_distribution(self):
        try:
            if self.df is not None:
                col = next((c for c in self.df.columns
                            if 'fertilizer' in c.lower()), None)
                fert_counts = self.df[col].value_counts().head(10) if col else None
            elif self.y_train is not None and self.fertilizer_classes:
                le = (self.encoders.get('fert_encoder') or
                      self.encoders.get('label_encoder'))
                labels = le.inverse_transform(self.y_train) if le else self.y_train
                fert_counts = pd.Series(labels).value_counts().head(10)
            else:
                return None

            if fert_counts is None:
                return None

            fig, ax = plt.subplots(figsize=(12, 6))
            colors  = plt.cm.Set3(np.linspace(0, 1, len(fert_counts)))
            bars    = ax.bar(range(len(fert_counts)), fert_counts.values, color=colors)
            ax.set_title('Top 10 Fertilizer Types', fontsize=14, fontweight='bold')
            ax.set_xlabel('Fertilizer Type')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(fert_counts)))
            ax.set_xticklabels(fert_counts.index, rotation=45, ha='right')
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{int(h)}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            return self.save_plot(fig, '01_fertilizer_distribution.png')
        except Exception as e:
            print(f"❌ Error in fertilizer_distribution: {e}")
            return None

    def generate_npk_distribution(self):
        try:
            if self.df is None:
                return None
            col_map = {}
            for c in self.df.columns:
                cl = c.lower()
                if 'nitrogen'  in cl: col_map['N'] = c
                if 'phospho'   in cl: col_map['P'] = c
                if 'potassiu'  in cl: col_map['K'] = c
            if len(col_map) < 3:
                return None
            fig, ax = plt.subplots(figsize=(12, 6))
            data    = [self.df[col_map['N']], self.df[col_map['P']], self.df[col_map['K']]]
            bp      = ax.boxplot(data, patch_artist=True,
                                 labels=['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'])
            for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
                patch.set_facecolor(color)
            ax.set_title('Distribution of NPK Values', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return self.save_plot(fig, '02_npk_distribution.png')
        except Exception as e:
            print(f"❌ Error in npk_distribution: {e}")
            return None

    def generate_correlation_heatmap(self):
        try:
            if self.X_train is None or self.feature_names is None:
                return None
            X_tr = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            corr = pd.DataFrame(X_tr, columns=self.feature_names).corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                        square=True, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            return self.save_plot(fig, '03_correlation_heatmap.png')
        except Exception as e:
            print(f"❌ Error in correlation_heatmap: {e}")
            return None

    # ==========================================================================
    # XAI PLOTS — Global
    # ==========================================================================

    def generate_feature_importance(self):
        try:
            if self.model is None or self.feature_names is None:
                return None
            if not hasattr(self.model, 'feature_importances_'):
                return None
            importance_df = pd.DataFrame({
                'feature'   : self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            fig, ax = plt.subplots(figsize=(12, 8))
            colors  = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars    = ax.barh(importance_df['feature'],
                              importance_df['importance'], color=colors)
            ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features',         fontsize=14, fontweight='bold')
            ax.set_title('Feature Importance', fontsize=16, fontweight='bold', pad=20)
            for bar, val in zip(bars, importance_df['importance']):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
            plt.tight_layout()
            return self.save_plot(fig, '04_feature_importance.png')
        except Exception as e:
            print(f"❌ Error in feature_importance: {e}")
            return None

    def generate_partial_dependence(self, predicted_fertilizer=""):
        try:
            if self.model is None or self.X_train is None:
                return None
            X_tr = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            fi_series    = pd.Series(self.model.feature_importances_,
                                     index=self.feature_names).sort_values(ascending=False)
            top_features = fi_series.head(4).index.tolist()
            top_indices  = [list(self.feature_names).index(f) for f in top_features]
            classes_list     = list(self.fertilizer_classes) if self.fertilizer_classes else []
            target_class_idx = (classes_list.index(predicted_fertilizer)
                                if predicted_fertilizer in classes_list else 0)
            target_class_name = classes_list[target_class_idx] if classes_list else "Unknown"
            print(f"   PDP target class: {target_class_name}")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            PartialDependenceDisplay.from_estimator(
                self.model, X_tr,
                features        = top_indices,
                feature_names   = self.feature_names,
                target          = target_class_idx,
                kind            = "average",
                ax              = axes.flatten(),
                grid_resolution = 50,
                random_state    = 42
            )
            fig.suptitle(
                f"Partial Dependency Plots — Top 4 Features\n"
                f"(Target fertilizer: {target_class_name})",
                fontsize=14, fontweight="bold")
            plt.tight_layout()
            return self.save_plot(fig, '05_partial_dependence.png')
        except Exception as e:
            print(f"❌ Error in partial_dependence: {e}")
            import traceback; traceback.print_exc()
            return None

    def generate_shap_beeswarm(self):
        try:
            if self.model is None or self.X_train is None:
                return None
            self.init_shap_explainer()
            if self.shap_explainer is None:
                return None
            X_tr   = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            sample = X_tr[:min(100, len(X_tr))]
            shap_values = self.shap_explainer.shap_values(sample, check_additivity=False)
            shap_2d     = _shap_for_beeswarm(shap_values)[:, :len(self.feature_names)]
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_2d, sample,
                              feature_names=self.feature_names, show=False)
            plt.title("SHAP Summary (Beeswarm)", fontsize=14, fontweight='bold')
            return self.save_plot(plt.gcf(), '06_shap_beeswarm.png')
        except Exception as e:
            print(f"❌ Error in shap_beeswarm: {e}")
            return None

    def generate_shap_importance_bar(self):
        try:
            if self.model is None or self.X_train is None:
                return None
            self.init_shap_explainer()
            if self.shap_explainer is None:
                return None
            X_tr   = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            sample = X_tr[:min(100, len(X_tr))]
            shap_values = self.shap_explainer.shap_values(sample, check_additivity=False)
            mean_shap   = _mean_abs_shap_per_feature(shap_values)[:len(self.feature_names)]
            importance_df = pd.DataFrame({
                "Feature"   : self.feature_names,
                "Importance": mean_shap
            }).sort_values("Importance", ascending=True)
            fig, ax    = plt.subplots(figsize=(10, 6))
            bar_colors = ["#E8593C" if v == importance_df["Importance"].max()
                          else "#185FA5" for v in importance_df["Importance"]]
            ax.barh(importance_df["Feature"], importance_df["Importance"],
                    color=bar_colors)
            ax.set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")
            ax.set_xlabel("Mean |SHAP Value| (averaged across all classes & samples)")
            ax.set_ylabel("Feature")
            for i, val in enumerate(importance_df["Importance"]):
                ax.text(val + 0.0005, i, f"{val:.4f}", va='center', fontsize=9)
            plt.tight_layout()
            return self.save_plot(fig, '07_shap_importance_bar.png')
        except Exception as e:
            print(f"❌ Error in shap_importance_bar: {e}")
            return None

    # ==========================================================================
    # XAI PLOTS — Local
    # ==========================================================================

    def generate_shap_local(self, features_row, predicted_fertilizer, timestamp):
        try:
            self.init_shap_explainer()
            if self.shap_explainer is None:
                return None
            features     = self._align_features(self._safe_row(features_row))
            classes_list = list(self.fertilizer_classes) if self.fertilizer_classes else []
            pred_idx     = (classes_list.index(predicted_fertilizer)
                            if predicted_fertilizer in classes_list else 0)
            shap_values       = self.shap_explainer.shap_values(
                features.reshape(1, -1), check_additivity=False)
            shap_values_class = _extract_shap_for_class(
                shap_values, pred_idx, sample_idx=0)[:len(self.feature_names)]
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
                [f"{n}  (val={v:.2f})" for n, v in zip(names_sorted, vals_sorted)],
                fontsize=11)
            ax.set_xlabel(
                'SHAP Value  →  positive = supports prediction, '
                'negative = opposes prediction',
                fontsize=11, fontweight='bold')
            ax.set_title(f'Local SHAP — Why predicted: {predicted_fertilizer}',
                         fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            for bar, val in zip(bars, shap_sorted):
                offset = 0.002 if val >= 0 else -0.002
                ha     = 'left'  if val >= 0 else 'right'
                ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                        f'{val:+.4f}', va='center', ha=ha,
                        fontweight='bold', fontsize=9)
            pos_p = mpatches.Patch(color='#E8593C', label='Supports prediction')
            neg_p = mpatches.Patch(color='#185FA5', label='Opposes prediction')
            ax.legend(handles=[pos_p, neg_p], fontsize=10)
            plt.tight_layout()
            return self.save_plot(fig, f'08_shap_local_{timestamp}.png')
        except Exception as e:
            print(f"❌ Error in shap_local: {e}")
            import traceback; traceback.print_exc()
            return None

    def generate_lime_local(self, features_row, predicted_fertilizer, timestamp):
        try:
            self.init_lime_explainer()
            if self.lime_explainer is None:
                return None
            # features_row is already the correct 8-dim vector
            features = self._align_features(self._safe_row(features_row))
            lime_exp = self.lime_explainer.explain_instance(
                data_row     = features,
                predict_fn   = self.model.predict_proba,
                num_features = len(self.feature_names),
                top_labels   = 1
            )
            top_label_idx  = lime_exp.top_labels[0]
            top_label_name = self.fertilizer_classes[top_label_idx]
            lime_vals      = lime_exp.as_list(label=top_label_idx)
            feat_labels    = [x[0] for x in lime_vals]
            feat_weights   = [x[1] for x in lime_vals]
            colors         = ['#E8593C' if v > 0 else '#185FA5' for v in feat_weights]
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.barh(feat_labels, feat_weights, color=colors,
                           edgecolor='white', height=0.55, alpha=0.85)
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xlabel('LIME Weight  →  contribution to prediction',
                          fontsize=11, fontweight='bold')
            ax.set_title(
                f'LIME Explanation — Predicted: {predicted_fertilizer}\n'
                f'(Explaining class: {top_label_name})',
                fontsize=13, fontweight='bold')
            for bar, val in zip(bars, feat_weights):
                offset = 0.001 if val >= 0 else -0.001
                ha     = 'left'  if val >= 0 else 'right'
                ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                        f'{val:+.4f}', va='center', ha=ha, fontsize=9)
            pos_p = mpatches.Patch(color='#E8593C', label='Supports prediction')
            neg_p = mpatches.Patch(color='#185FA5', label='Opposes prediction')
            ax.legend(handles=[pos_p, neg_p], fontsize=10)
            plt.tight_layout()
            return self.save_plot(fig, f'09_lime_{timestamp}.png')
        except Exception as e:
            print(f"❌ Error in lime_local: {e}")
            import traceback; traceback.print_exc()
            return None

    def generate_individual_breakdown(self, features_row, predicted_fertilizer, timestamp):
        try:
            self.init_shap_explainer()
            if self.shap_explainer is None:
                return None
            features     = self._align_features(self._safe_row(features_row))
            classes_list = list(self.fertilizer_classes) if self.fertilizer_classes else []
            pred_idx     = (classes_list.index(predicted_fertilizer)
                            if predicted_fertilizer in classes_list else 0)
            shap_values   = self.shap_explainer.shap_values(
                features.reshape(1, -1), check_additivity=False)
            shap_for_pred = _extract_shap_for_class(
                shap_values, pred_idx, sample_idx=0)[:len(self.feature_names)]
            breakdown_df = pd.DataFrame({
                "Feature"          : self.feature_names,
                "Feature Value"    : features[:len(self.feature_names)],
                "SHAP Contribution": shap_for_pred
            }).sort_values("SHAP Contribution", ascending=True)
            colors = ['#E8593C' if v > 0 else '#185FA5'
                      for v in breakdown_df["SHAP Contribution"]]
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.barh(breakdown_df["Feature"],
                           breakdown_df["SHAP Contribution"],
                           color=colors, edgecolor='white', height=0.55, alpha=0.85)
            for bar, (_, row) in zip(bars, breakdown_df.iterrows()):
                val   = row["SHAP Contribution"]
                label = f"  val={row['Feature Value']:.2f}   SHAP={val:+.5f}"
                offset = 0.0002 if val >= 0 else -0.0002
                ha     = 'left'  if val >= 0 else 'right'
                ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                        label, va='center', ha=ha, fontsize=9)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('SHAP Value  →  contribution to predicted probability',
                          fontsize=11, fontweight='bold')
            ax.set_title(
                f'Individual Feature Breakdown — Predicted: {predicted_fertilizer}',
                fontsize=13, fontweight='bold')
            pos_p = mpatches.Patch(color='#E8593C', label='Pushes toward prediction')
            neg_p = mpatches.Patch(color='#185FA5', label='Pushes away from prediction')
            ax.legend(handles=[pos_p, neg_p], fontsize=10)
            plt.tight_layout()
            return self.save_plot(fig, f'10_individual_breakdown_{timestamp}.png')
        except Exception as e:
            print(f"❌ Error in individual_breakdown: {e}")
            import traceback; traceback.print_exc()
            return None

    # ==========================================================================
    # MAIN REPORT ENTRY POINT
    # ==========================================================================

    def generate_html_report(self, input_data=None, prediction_result=None):
        """
        Generate complete HTML report.

        input_data keys (from predict_fertilizer.py):
            soil_type, crop_type, nitrogen, phosphorus, potassium,
            temperature, humidity, soil_moisture

        prediction_result keys:
            fertilizer  ← recommended fertilizer name (string)
        """
        try:
            print("\n🚀 Starting fertilizer report generation...")
            timestamp            = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            predicted_fertilizer = (prediction_result.get('fertilizer', '')
                                    if prediction_result else '')

            # ── Build the correct 8-dim encoded feature vector ────────────────
            # Uses label_encoders.pkl (soil_encoder, crop_encoder, scaler)
            # No padding warnings — all 8 features are now built correctly.
            features_row = None
            if input_data is not None:
                features_row = self.build_features_row(input_data)

            # Fallback: use first row of training data
            if features_row is None and self.X_train is not None:
                X_tr         = (self.X_train.values
                                if hasattr(self.X_train, 'values') else self.X_train)
                features_row = X_tr[0]
                print("⚠️  Using fallback training row for local XAI plots")

            plots = {}

            print("   [1] Fertilizer distribution...")
            plots['fertilizer_distribution'] = self.generate_fertilizer_distribution()

            print("   [2] NPK distribution...")
            plots['npk_distribution'] = self.generate_npk_distribution()

            print("   [3] Correlation heatmap...")
            plots['correlation'] = self.generate_correlation_heatmap()

            print("   [4] Feature importance...")
            plots['feature_importance'] = self.generate_feature_importance()

            print("   [5] Partial dependency plots...")
            plots['pdp'] = self.generate_partial_dependence(predicted_fertilizer)

            print("   [6] SHAP beeswarm...")
            plots['shap_beeswarm'] = self.generate_shap_beeswarm()

            print("   [7] SHAP importance bar...")
            plots['shap_importance'] = self.generate_shap_importance_bar()

            print("   [8] SHAP local waterfall...")
            plots['shap_local'] = self.generate_shap_local(
                features_row, predicted_fertilizer, timestamp)

            print("   [9] LIME explanation...")
            plots['lime'] = self.generate_lime_local(
                features_row, predicted_fertilizer, timestamp)

            print("   [10] Individual breakdown...")
            plots['individual_breakdown'] = self.generate_individual_breakdown(
                features_row, predicted_fertilizer, timestamp)

            
            successful = sum(1 for v in plots.values() if v is not None)
            print(f"\n✅ {successful}/10 plots generated")

            # ── Top-predictions probability table ─────────────────────────────
            prob_table_html = ""
            if features_row is not None:
                try:
                    aligned = self._align_features(self._safe_row(features_row))
                    probs   = self.model.predict_proba(aligned.reshape(1, -1))[0]
                    classes = list(self.fertilizer_classes)
                    rank    = 1
                    for i in np.argsort(probs)[::-1][:10]:
                        p = probs[i]
                        if p > 0.0:
                            hl = ' class="highlight"' if rank == 1 else ''
                            prob_table_html += f"""
                            <tr{hl}>
                                <td>{rank}</td>
                                <td><strong>{classes[i]}</strong></td>
                                <td>{p:.2%}</td>
                            </tr>"""
                            rank += 1
                except Exception as e:
                    print(f"⚠️  Could not build prob table: {e}")

            # ── Input summary table ────────────────────────────────────────────
            input_table_html = ""
            if input_data:
                display_labels = [
                    ('soil_type',    'Soil Type'),
                    ('crop_type',    'Crop Type'),
                    ('temperature',  'Temperature (°C)'),
                    ('humidity',     'Humidity (%)'),
                    ('soil_moisture','Soil Moisture (%)'),
                    ('moisture',     'Soil Moisture (%)'),
                    ('nitrogen',     'Nitrogen (N)'),
                    ('phosphorus',   'Phosphorus (P)'),
                    ('potassium',    'Potassium (K)'),
                ]
                seen_labels = set()
                for key, label in display_labels:
                    if label in seen_labels or key not in input_data:
                        continue
                    seen_labels.add(label)
                    val = input_data[key]
                    try:
                        val = f"{float(val):.2f}"
                    except (ValueError, TypeError):
                        pass
                    input_table_html += f"""
                    <tr>
                        <td><strong>{label}</strong></td>
                        <td>{val}</td>
                    </tr>"""

            def img_path(filepath):
                if filepath and os.path.exists(filepath):
                    return f"images/{os.path.basename(filepath)}"
                return ""

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fertilizer Recommendation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto;
            background-color: white; border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .report-meta {{
            background-color: #f8f9fa; padding: 15px 30px;
            border-bottom: 2px solid #e9ecef;
            display: flex; justify-content: space-between;
        }}
        .main-result {{
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
            padding: 40px 30px; text-align: center; color: white;
        }}
        .fert-name {{
            font-size: 3.5em; font-weight: 800; margin: 10px 0;
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
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white; padding: 15px 20px;
            font-size: 1.2em; font-weight: bold;
        }}
        .card-body {{ padding: 20px; }}
        .plot-container {{
            text-align: center; padding: 20px;
            background-color: #fafafa; min-height: 350px;
            display: flex; align-items: center; justify-content: center;
        }}
        .plot-container img {{
            width: 100%; height: auto; max-height: 600px;
            object-fit: contain; border-radius: 10px;
            border: 2px solid #e9ecef; display: block;
        }}
        .no-plot {{
            padding: 40px; color: #999;
            font-style: italic; text-align: center;
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
            border-bottom: 4px solid #11998e; padding-bottom: 10px;
        }}
        .footer {{
            background: linear-gradient(135deg, #2c3e50, #11998e);
            color: white; padding: 30px; text-align: center;
        }}
        .button {{
            display: inline-block; padding: 12px 30px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white; text-decoration: none; border-radius: 25px;
            margin: 5px; border: none; cursor: pointer;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .fert-name {{ font-size: 2.2em; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>🌱 Fertilizer Recommendation System</h1>
        <p>Complete XAI Analysis Report</p>
    </div>

    <div class="report-meta">
        <span>📋 Report ID: report_{timestamp}</span>
        <span>📅 Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
    </div>

    <div class="main-result">
        <h2>✅ Recommended Fertilizer</h2>
        <div class="fert-name">{predicted_fertilizer}</div>
    </div>

    <div class="content">

        <h2 class="section-title">📝 Input Summary</h2>
        <div class="grid-2">
            <div class="card">
                <div class="card-header">📊 Input Parameters</div>
                <div class="card-body">
                    <table>
                        {input_table_html if input_table_html
                          else '<tr><td>No input data available</td></tr>'}
                    </table>
                </div>
            </div>
            <div class="card">
                <div class="card-header">🎯 Top Predictions</div>
                <div class="card-body">
                    <table>
                        <tr><th>Rank</th><th>Fertilizer</th><th>Probability</th></tr>
                        {prob_table_html if prob_table_html
                          else '<tr><td colspan="3">Not available</td></tr>'}
                    </table>
                </div>
            </div>
        </div>

        <h2 class="section-title">📊 Data Exploration</h2>
        <div class="grid-2">
            <div class="card">
                <div class="card-header">🌿 Fertilizer Distribution</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('fertilizer_distribution'))}"
                         alt="Fertilizer Distribution"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">🧪 NPK Distribution</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('npk_distribution'))}"
                         alt="NPK Distribution"
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

        <h2 class="section-title">🌍 Global Model Interpretation</h2>
        <div class="grid-2">
            <div class="card">
                <div class="card-header">📈 Feature Importance</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('feature_importance'))}"
                         alt="Feature Importance"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">📉 Partial Dependency Plot</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('pdp'))}"
                         alt="Partial Dependency Plot"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">🔬 SHAP Summary (Beeswarm)</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('shap_beeswarm'))}"
                         alt="SHAP Beeswarm"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
            <div class="card">
                <div class="card-header">🎯 SHAP Feature Importance</div>
                <div class="plot-container">
                    <img src="{img_path(plots.get('shap_importance'))}"
                         alt="SHAP Importance"
                         onerror="this.parentElement.innerHTML='<p class=no-plot>Plot not available</p>'">
                </div>
            </div>
        </div>

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

    </div>

    <div class="footer">
        <p>Report generated by Fertilizer Recommendation System</p>
        <p>Total Plots Generated: {successful}/11</p>
        <button class="button" onclick="window.print()">🖨️ Print</button>
        <button class="button" onclick="window.close()">❌ Close</button>
    </div>

</div>
</body>
</html>"""

            report_path = os.path.join(REPORTS_DIR, "latest_report.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            timestamped_path = os.path.join(REPORTS_DIR, f"report_{timestamp}.html")
            with open(timestamped_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ Report saved: {report_path}")
            return {
                'latest_report'     : 'report_fertilizer/latest_report.html',
                'timestamped_report': f'report_fertilizer/report_{timestamp}.html',
                'images'            : [v for v in plots.values() if v is not None]
            }

        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback; traceback.print_exc()
            return {'error': str(e)}


# Create global instance
print("🔧 Initializing FertilizerReportGenerator...")
report_generator = FertilizerReportGenerator()