import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directory
MODELS_DIR = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-models")

# Processed data directory
DATA_DIR = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-processed_data")

# Reports directory
REPORTS_DIR = os.path.join(BASE_DIR, "report_crop")

# Load label encoder
data_path = os.path.join(DATA_DIR, "preprocessed_data.pkl")


class CropReportGenerator:
    def __init__(self, model, label_encoder, feature_names, reports_dir):
        """
        Initialize report generator with model and encoders
        
        Args:
            model: Trained stacked ensemble model
            label_encoder: Label encoder for crop names
            feature_names: List of feature names
            reports_dir: Directory to save reports
        """
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.feature_short_names = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
        self.reports_dir = REPORTS_DIR
        
        # Create the folder structure
        os.makedirs(REPORTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(REPORTS_DIR, "data"), exist_ok=True)
        os.makedirs(os.path.join(REPORTS_DIR, "images"), exist_ok=True)
        
        # Load training data for explainers
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.load_training_data()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
    def load_training_data(self):
        """Load training data from pickle file"""
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            
            # Ensure data has correct dimensions
            if self.X_train.shape[1] != len(self.feature_names):
                print(f"⚠️ Training data feature count mismatch: {self.X_train.shape[1]} vs {len(self.feature_names)}")
                # Truncate if necessary
                if self.X_train.shape[1] > len(self.feature_names):
                    self.X_train = self.X_train[:, :len(self.feature_names)]
                if self.X_test.shape[1] > len(self.feature_names):
                    self.X_test = self.X_test[:, :len(self.feature_names)]
            
            print(f"✅ Training data loaded for explainers: {self.X_train.shape}")
            
        except Exception as e:
            print(f"⚠️ Could not load training data: {e}")
            # Create dummy data for demonstration
            np.random.seed(42)
            self.X_train = np.random.rand(100, len(self.feature_names))
            self.y_train = np.random.randint(0, len(self.label_encoder.classes_), 100)
            self.X_test = np.random.rand(30, len(self.feature_names))
            self.y_test = np.random.randint(0, len(self.label_encoder.classes_), 30)
            print("⚠️ Using dummy training data for explainers")
    
    def init_shap_explainer(self):
        """Initialize SHAP explainer"""
        if self.shap_explainer is None and self.X_train is not None:
            try:
                # Use sample of training data for background
                background_size = min(50, len(self.X_train))
                X_background = self.X_train[:background_size]
                
                # Ensure background data has correct shape
                if X_background.shape[1] != len(self.feature_names):
                    print(f"⚠️ Background data shape mismatch: {X_background.shape[1]} vs {len(self.feature_names)}")
                    if X_background.shape[1] > len(self.feature_names):
                        X_background = X_background[:, :len(self.feature_names)]
                
                print(f"Initializing SHAP explainer with background shape: {X_background.shape}")
                self.shap_explainer = shap.TreeExplainer(self.model)
                print("✅ SHAP explainer initialized successfully")
            except Exception as e:
                print(f"⚠️ SHAP explainer initialization failed: {e}")
    
    def init_lime_explainer(self):
        """Initialize LIME explainer"""
        if self.lime_explainer is None and self.X_train is not None:
            try:
                # Ensure training data has correct shape
                X_train_lime = self.X_train
                if X_train_lime.shape[1] != len(self.feature_names):
                    if X_train_lime.shape[1] > len(self.feature_names):
                        X_train_lime = X_train_lime[:, :len(self.feature_names)]
                
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train_lime,
                    feature_names=self.feature_names,
                    class_names=self.label_encoder.classes_,
                    mode='classification',
                    discretize_continuous=True,
                    random_state=42
                )
                print("✅ LIME explainer initialized")
            except Exception as e:
                print(f"⚠️ LIME explainer initialization failed: {e}")
    
    # ============================================
    # GLOBAL EXPLAINABILITY METHODS
    # ============================================
    
    def generate_global_feature_importance(self, timestamp):
        """Generate feature importance plots"""
        print("\n📊 Generating Global Feature Importance Plots...")
        
        try:
            # Get feature importance from Random Forest
            feature_importance = self.model.feature_importances_
            
            # Ensure importance matches feature count
            if len(feature_importance) != len(self.feature_names):
                print(f"⚠️ Feature importance length mismatch: {len(feature_importance)} vs {len(self.feature_names)}")
                if len(feature_importance) > len(self.feature_names):
                    feature_importance = feature_importance[:len(self.feature_names)]
                else:
                    # Pad with zeros if needed
                    feature_importance = np.pad(feature_importance, (0, len(self.feature_names) - len(feature_importance)), 'constant')
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Short_Name': self.feature_short_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Horizontal bar chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
            ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features', fontsize=14, fontweight='bold')
            ax.set_title('🌍 Global: Feature Importance Plot\n(Random Forest Crop Model)',
                        fontsize=16, fontweight='bold', pad=20)
            ax.invert_yaxis()
            
            # Add value labels
            for bar, val in zip(bars, importance_df['Importance']):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"feature_importance_{timestamp}.png"
            plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Save importance to CSV
            csv_filename = f"feature_importance_{timestamp}.csv"
            csv_path = os.path.join(REPORTS_DIR, "data", csv_filename)
            importance_df.to_csv(csv_path, index=False)
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error generating feature importance: {e}")
            return None
    
    def generate_crop_distribution(self, timestamp):
        """Generate crop distribution plot if training data has crop labels"""
        try:
            if self.y_train is None:
                return None
            
            # Get crop names from labels
            unique_labels = np.unique(self.y_train)
            crop_counts = []
            crop_names = []
            
            for label in unique_labels:
                count = np.sum(self.y_train == label)
                crop_counts.append(count)
                if label < len(self.label_encoder.classes_):
                    crop_names.append(self.label_encoder.inverse_transform([label])[0])
                else:
                    crop_names.append(f"Crop_{label}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
            bars = ax.bar(range(len(crop_counts)), crop_counts, color=colors)
            
            ax.set_title('Distribution of Crop Types in Training Data', fontsize=14, fontweight='bold')
            ax.set_xlabel('Crop Type')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(crop_counts)))
            ax.set_xticklabels(crop_names, rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            plot_filename = f"crop_distribution_{timestamp}.png"
            plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error generating crop distribution: {e}")
            return None
    
    def generate_input_distribution(self, timestamp):
        """Generate distribution plots for input features"""
        try:
            if self.X_train is None:
                return None
            
            n_features = len(self.feature_names)
            n_rows = (n_features + 3) // 4  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
            axes = axes.flatten()
            
            for i in range(4 * n_rows):
                if i < n_features:
                    axes[i].hist(self.X_train[:, i], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                    axes[i].set_title(f'Distribution of {self.feature_short_names[i]}', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(self.feature_names[i])
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].set_visible(False)
            
            plt.suptitle('Distribution of Input Features', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            plot_filename = f"input_distribution_{timestamp}.png"
            plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error generating input distribution: {e}")
            return None
    
    def generate_correlation_heatmap(self, timestamp):
        """Generate correlation heatmap of input features"""
        try:
            if self.X_train is None:
                return None
            
            # Create DataFrame with feature names
            df = pd.DataFrame(self.X_train, columns=self.feature_short_names[:self.X_train.shape[1]])
            corr_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            plot_filename = f"correlation_heatmap_{timestamp}.png"
            plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error generating correlation heatmap: {e}")
            return None
def generate_shap_summary_plots(self, timestamp):
    print("\n📊 Generating SHAP Summary Plots...")

    try:
        self.init_shap_explainer()

        if self.shap_explainer is None:
            print("⚠️ SHAP explainer not initialized")
            return None, None

        # Use small sample for speed
        sample_size = min(100, len(self.X_test))
        X_test_sample = self.X_test[:sample_size]

        # Compute SHAP values
        shap_values = self.shap_explainer.shap_values(X_test_sample)

        # Handle multi-class models
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
            shap_values_combined = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values_plot = shap_values
            shap_values_combined = np.abs(shap_values)

        # ============================================
        # SHAP Feature Importance Bar Plot
        # ============================================

        mean_shap = np.mean(shap_values_combined, axis=0)

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": mean_shap
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10,6))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            color="steelblue"
        )

        ax.set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_ylabel("Feature")

        plt.tight_layout()

        plot_filename = f"shap_importance_{timestamp}.png"
        plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)

        plt.savefig(plot_path, dpi=100)
        plt.close()

        # ============================================
        # SHAP Summary Dot Plot
        # ============================================

        plt.figure(figsize=(12,8))

        shap.summary_plot(
            shap_values_plot,
            X_test_sample,
            feature_names=self.feature_names,
            show=False
        )

        plt.title("SHAP Feature Impact Summary", fontsize=14)

        dot_plot_filename = f"shap_dot_summary_{timestamp}.png"
        dot_plot_path = os.path.join(REPORTS_DIR, "images", dot_plot_filename)

        plt.savefig(dot_plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        print("✅ SHAP plots generated")

        return plot_path, dot_plot_path

    except Exception as e:
        print("⚠️ Error generating SHAP plots:", e)
        import traceback
        traceback.print_exc()
        return None, None
    def generate_decision_tree_surrogate(self, timestamp):
        """Generate decision tree surrogate model"""
        try:
            from sklearn.tree import DecisionTreeClassifier, plot_tree
            
            if self.X_train is None or self.y_train is None:
                return None
            
            # Ensure data has correct dimensions
            X_train_tree = self.X_train
            X_test_tree = self.X_test
            
            if X_train_tree.shape[1] != len(self.feature_names):
                if X_train_tree.shape[1] > len(self.feature_names):
                    X_train_tree = X_train_tree[:, :len(self.feature_names)]
            
            if X_test_tree.shape[1] != len(self.feature_names):
                if X_test_tree.shape[1] > len(self.feature_names):
                    X_test_tree = X_test_tree[:, :len(self.feature_names)]
            
            # Train a simple decision tree
            surrogate_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
            surrogate_tree.fit(X_train_tree, self.y_train)
            surrogate_accuracy = surrogate_tree.score(X_test_tree, self.y_test)
            
            fig, ax = plt.subplots(figsize=(20, 12))
            
            plot_tree(
                surrogate_tree,
                feature_names=self.feature_names,
                class_names=self.label_encoder.classes_,
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=3,
                ax=ax
            )
            
            plt.title(f'Decision Tree Surrogate Model\n(Accuracy: {surrogate_accuracy:.2f})',
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            plot_filename = f"decision_tree_{timestamp}.png"
            plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error generating decision tree: {e}")
            return None
    
    # ============================================
    # LOCAL EXPLAINABILITY METHODS
    # ============================================
    
    def generate_shap_plot(self, features, predicted_crop, timestamp):
        """Generate SHAP waterfall plot for local explanation"""
        try:
            self.init_shap_explainer()
            
            if self.shap_explainer is None:
                return None
            
            # Ensure features is 1D and has correct length
            if len(features.shape) > 1:
                features = features.flatten()
            
            if len(features) != len(self.feature_names):
                print(f"⚠️ Features length mismatch: {len(features)} vs {len(self.feature_names)}")
                if len(features) > len(self.feature_names):
                    features = features[:len(self.feature_names)]
                else:
                    # Pad with zeros if needed
                    features = np.pad(features, (0, len(self.feature_names) - len(features)), 'constant')
            
            # Get prediction class index
            pred_idx = list(self.label_encoder.classes_).index(predicted_crop)
            
            # Calculate SHAP values
            features_reshaped = features.reshape(1, -1)
            shap_values = self.shap_explainer.shap_values(features_reshaped)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                if pred_idx < len(shap_values):
                    shap_values_class = shap_values[pred_idx]
                else:
                    shap_values_class = shap_values[0]
                
                shap_values_class = np.array(shap_values_class)
                if shap_values_class.ndim > 1:
                    shap_values_class = shap_values_class.flatten()
                shap_values_class = shap_values_class[:len(self.feature_names)]
            else:
                shap_values_class = shap_values
                if len(shap_values_class.shape) > 1:
                    shap_values_class = shap_values_class[0]
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort features by absolute SHAP value
            feature_order = np.argsort(np.abs(shap_values_class))[::-1]
            
            y_pos = np.arange(len(self.feature_names))
            shap_vals = shap_values_class[feature_order]
            feature_names_plot = [self.feature_names[i] for i in feature_order]
            
            colors = ['#ff6b6b' if val > 0 else '#4dabf7' for val in shap_vals]
            bars = ax.barh(y_pos, shap_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names_plot, fontsize=11)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
            ax.set_title(f'SHAP Feature Impact for {predicted_crop}', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, val in zip(bars, shap_vals):
                if val > 0:
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
                else:
                    ax.text(val - 0.03, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', ha='right', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            
            plot_filename = f"shap_local_{timestamp}.png"
            plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"Error generating SHAP plot: {e}")
            return None

def generate_fallback_explanation(self, features, predicted_crop, timestamp):
    try:
        # Get feature importance from model
        feature_importance = self.model.feature_importances_
        
        # Get prediction probabilities
        features_2d = features.reshape(1, -1)
        base_pred = self.model.predict_proba(features_2d)[0]
        
        # Calculate local feature impacts by perturbing features
        n_features = len(self.feature_names)
        impacts = []
        
        for i in range(n_features):
            # Create perturbed sample
            features_perturbed = features.copy()
            # Increase feature by 10%
            features_perturbed[i] *= 1.1
            
            # Get new prediction
            features_perturbed_2d = features_perturbed.reshape(1, -1)
            new_pred = self.model.predict_proba(features_perturbed_2d)[0]
            
            # Calculate impact on predicted class
            pred_idx = list(self.label_encoder.classes_).index(predicted_crop)
            impact = new_pred[pred_idx] - base_pred[pred_idx]
            impacts.append(impact)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by absolute impact
        impact_order = np.argsort(np.abs(impacts))[::-1]
        sorted_features = [self.feature_names[i] for i in impact_order]
        sorted_impacts = [impacts[i] for i in impact_order]
        
        colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in sorted_impacts]
        
        y_pos = np.arange(len(sorted_impacts))
        bars = ax.barh(y_pos, sorted_impacts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features, fontsize=11)
        ax.set_xlabel('Impact on Prediction (Feature +10%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Local Feature Impact Analysis for {predicted_crop} (Fallback Method)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, sorted_impacts):
            if val > 0:
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                       f'+{val:.4f}', va='center', fontweight='bold', fontsize=10)
            else:
                ax.text(val - 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', ha='right', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        plot_filename = f"fallback_explanation_{timestamp}.png"
        plot_path = os.path.join(REPORTS_DIR, "images", plot_filename)
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error generating fallback explanation: {e}")
        return None
    
    # ============================================
    # MAIN REPORT GENERATION
    # ============================================
    
    def generate_all_plots(self, timestamp):
        """Generate all plots for the report"""
        plots = {}
        
        print("\n📊 Generating all plots for crop report...")
        
        print("   1/12 Generating Feature Importance Plot...")
        plots['feature_importance'] = self.generate_global_feature_importance(timestamp)
        
        print("   2/12 Generating Crop Distribution Plot...")
        plots['crop_distribution'] = self.generate_crop_distribution(timestamp)
        
        print("   3/12 Generating Input Distribution Plot...")
        plots['input_distribution'] = self.generate_input_distribution(timestamp)
        
        print("   4/12 Generating Correlation Heatmap...")
        plots['correlation'] = self.generate_correlation_heatmap(timestamp)
        
        print("   5/12 Generating SHAP Summary Plots...")
        shap_imp, shap_dot = self.generate_shap_summary_plots(timestamp)
        plots['shap_importance'] = shap_imp
        plots['shap_dot'] = shap_dot
        
        print("   6/12 Generating Decision Tree Surrogate...")
        plots['decision_tree'] = self.generate_decision_tree_surrogate(timestamp)
        
        return plots
    
    def generate_prediction_report(self, input_data, prediction_result):
        """
        Generate comprehensive HTML report
        
        Args:
            input_data: Dictionary with input features
            prediction_result: Dictionary with prediction results
        
        Returns:
            str: Path to generated HTML report
        """
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{timestamp}"
        
        # Convert input to numpy array
        features = np.array([
            input_data["nitrogen"],
            input_data["phosphorus"],
            input_data["potassium"],
            input_data["temperature"],
            input_data["humidity"],
            input_data["ph"],
            input_data["rainfall"]
        ])
        
        # Get prediction details
        predicted_crop = prediction_result["crop"]
        confidence = prediction_result.get("confidence", 0.95)
        
        # Get all probabilities
        features_2d = features.reshape(1, -1)
        probabilities = self.model.predict_proba(features_2d)[0]
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_crops = self.label_encoder.inverse_transform(top_5_indices)
        top_5_probs = probabilities[top_5_indices]
        
        # Generate all plots
        plots = self.generate_all_plots(timestamp)
        
        # Generate local explanation plots
        print("\n📊 Generating local explanation plots...")
        plots['shap_local'] = self.generate_shap_plot(features, predicted_crop, timestamp)
        plots['lime'] = self.generate_lime_explanation(features, predicted_crop, timestamp)
        
        # Generate HTML report
        html_content = self.create_html_report(
            report_id=report_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_data=input_data,
            predicted_crop=predicted_crop,
            confidence=confidence,
            top_5_crops=top_5_crops,
            top_5_probs=top_5_probs,
            plots=plots
        )
        
        # Save HTML report
        report_path = os.path.join(REPORTS_DIR, f"{report_id}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save as latest report
        latest_path = os.path.join(REPORTS_DIR, "latest_report.html")
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save input data as JSON
        data_path = os.path.join(REPORTS_DIR, "data", f"{report_id}.json")
        with open(data_path, 'w') as f:
            json.dump({
                "input": input_data,
                "prediction": prediction_result,
                "timestamp": timestamp
            }, f, indent=2)
        
        print(f"✅ Report generated: {report_path}")
        return report_path
    
    def create_html_report(self, report_id, timestamp, input_data, predicted_crop, 
                          confidence, top_5_crops, top_5_probs, plots):
        """Create HTML report content"""
        
        # Create feature value table
        feature_table = ""
        features_display = [
            ("Nitrogen (N)", input_data["nitrogen"]),
            ("Phosphorus (P)", input_data["phosphorus"]),
            ("Potassium (K)", input_data["potassium"]),
            ("Temperature (°C)", input_data["temperature"]),
            ("Humidity (%)", input_data["humidity"]),
            ("Soil pH", input_data["ph"]),
            ("Rainfall (mm)", input_data["rainfall"])
        ]
        
        for name, value in features_display:
            feature_table += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{value:.2f}</td>
            </tr>
            """
        
        # Create top 5 probabilities table
        prob_table = ""
        for i, (crop, prob) in enumerate(zip(top_5_crops, top_5_probs)):
            prob_table += f"""
            <tr{' class="highlight"' if i == 0 else ''}>
                <td>{i+1}</td>
                <td><strong>{crop}</strong></td>
                <td>{prob:.2%}</td>
            </tr>
            """
        
        # Helper function for image paths
        def img_path(filename):
            if filename and os.path.exists(filename):
                return f"images/{os.path.basename(filename)}"
            return ""
        
        # Count successful plots
        successful_plots = sum(1 for v in plots.values() if v is not None)
        
        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Crop Prediction Report - {predicted_crop}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                .report-meta {{
                    background-color: #f8f9fa;
                    padding: 15px 30px;
                    border-bottom: 2px solid #e9ecef;
                    display: flex;
                    justify-content: space-between;
                }}
                .main-result {{
                    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                    padding: 40px 30px;
                    text-align: center;
                    color: white;
                }}
                .crop-name {{
                    font-size: 4em;
                    font-weight: 800;
                    margin: 10px 0;
                    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
                }}
                .content {{
                    padding: 40px;
                }}
                .grid-2 {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 30px;
                    margin-bottom: 30px;
                }}
                .grid-3 {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 30px;
                    margin-bottom: 30px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    overflow: hidden;
                    border: 1px solid #e9ecef;
                }}
                .card-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px 20px;
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .card-body {{
                    padding: 20px;
                }}
                .plot-container {{
                    text-align: center;
                    padding: 20px;
                    background-color: #fafafa;
                }}
                .plot-container img {{
                    max-width: 100%;
                    max-height: 500px;
                    border-radius: 10px;
                    border: 2px solid #e9ecef;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                table th, table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #e9ecef;
                }}
                .highlight {{
                    background: linear-gradient(135deg, #d4edda, #c3e6cb);
                    font-weight: bold;
                }}
                .section-title {{
                    font-size: 2em;
                    margin: 40px 0 25px 0;
                    color: #2c3e50;
                    border-bottom: 4px solid #667eea;
                    padding-bottom: 10px;
                }}
                .footer {{
                    background: linear-gradient(135deg, #2c3e50, #3498db);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .button {{
                    display: inline-block;
                    padding: 12px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    margin: 5px;
                    border: none;
                    cursor: pointer;
                }}
                .button:hover {{
                    transform: scale(1.05);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                }}
                @media (max-width: 768px) {{
                    .grid-2, .grid-3 {{
                        grid-template-columns: 1fr;
                    }}
                    .crop-name {{
                        font-size: 2.5em;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌾 Crop Recommendation System</h1>
                    <p>Complete Analysis Report</p>
                </div>
                
                <div class="report-meta">
                    <span>📋 Report ID: {report_id}</span>
                    <span>📅 Generated: {timestamp}</span>
                </div>
                
                <div class="main-result">
                    <h2>✅ Recommended Crop</h2>
                    <div class="crop-name">{predicted_crop}</div>
                    <p style="font-size: 1.2em;">Confidence: {confidence:.1%}</p>
                </div>
                
                <div class="content">
                    <!-- Input Summary -->
                    <h2 class="section-title">📝 Input Summary</h2>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-header">📊 Input Parameters</div>
                            <div class="card-body">
                                <table>
                                    {feature_table}
                                </table>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">🎯 Top 5 Predictions</div>
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
                    
                    <!-- Data Exploration Plots -->
                    <h2 class="section-title">📊 Data Exploration</h2>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-header">🌽 Crop Distribution</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('crop_distribution'))}" 
                                     alt="Crop Distribution"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">📈 Feature Distributions</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('input_distribution'))}" 
                                     alt="Input Distributions"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">🔗 Correlation Heatmap</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('correlation'))}" 
                                     alt="Correlation Heatmap"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                    </div>
                    
                    <!-- Global Explainability -->
                    <h2 class="section-title">🌍 Global Model Interpretation</h2>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-header">📈 Feature Importance</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('feature_importance'))}" 
                                     alt="Feature Importance"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">🎯 SHAP Importance</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('shap_importance'))}" 
                                     alt="SHAP Importance"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">🔬 SHAP Summary</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('shap_dot'))}" 
                                     alt="SHAP Summary"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">🌳 Decision Tree Surrogate</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('decision_tree'))}" 
                                     alt="Decision Tree"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                    </div>
                    
                    <!-- Local Explainability -->
                    <h2 class="section-title">🔍 Local Interpretation</h2>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-header">⚡ SHAP Local Analysis</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('shap_local'))}" 
                                     alt="SHAP Local"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">🔍 LIME Explanation</div>
                            <div class="plot-container">
                                <img src="{img_path(plots.get('lime'))}" 
                                     alt="LIME Explanation"
                                     onerror="this.parentElement.innerHTML='<p>Plot not available</p>'">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Report generated by Crop Recommendation System</p>
                    <p>Total Plots Generated: {successful_plots}/12</p>
                    <button class="button" onclick="window.print()">🖨️ Print</button>
                    <button class="button" onclick="window.close()">❌ Close</button>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html