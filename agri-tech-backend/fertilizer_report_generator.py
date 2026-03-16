# fertilizer_report_generator.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import json
import pickle
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "fertilizer-recommendation", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "report_fertilizer")

# Create directories
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "data"), exist_ok=True)

class FertilizerReportGenerator:
    def __init__(self):
        self.model = None
        self.processed_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.df = None
        self.df_encoded = None
        self.feature_names = None
        self.le_fert = None
        self.fertilizer_classes = None
        self.model_accuracy = None
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Load model and processed training data"""
        try:
            print("🔍 Loading model and processed data...")
            
            # Load model
            model_path = os.path.join(MODELS_DIR, 'stacked_ensemble.pkl')
            if not os.path.exists(model_path):
                model_path = os.path.join(BASE_DIR, "fertilizer-recommendation", 'stacked_ensemble.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as file:
                    self.model = pickle.load(file)
                print(f"   ✅ Model loaded successfully")
            else:
                print(f"   ❌ Model not found")
                return False
            
            # Load processed data
            processed_path = os.path.join(MODELS_DIR, 'processed_data.pkl')
            if not os.path.exists(processed_path):
                processed_path = os.path.join(BASE_DIR, "fertilizer-recommendation", 'processed_data.pkl')
            
            if os.path.exists(processed_path):
                with open(processed_path, 'rb') as file:
                    self.processed_data = pickle.load(file)
                print(f"   ✅ Processed data loaded successfully")
                
                # Extract data
                self.X_train = self.processed_data.get('X_train')
                self.y_train = self.processed_data.get('y_train')
                self.X_test = self.processed_data.get('X_test')
                self.y_test = self.processed_data.get('y_test')
                self.feature_names = self.processed_data.get('feature_names')
                self.df = self.processed_data.get('df_original')
                self.df_encoded = self.processed_data.get('df_encoded')
                self.le_fert = self.processed_data.get('le_fert')
                
                if self.le_fert is not None:
                    self.fertilizer_classes = self.le_fert.classes_.tolist()
                
                self.model_accuracy = self.processed_data.get('model_accuracy', 0)
                
                print(f"   ✅ X_train shape: {self.X_train.shape if self.X_train is not None else 'None'}")
                print(f"   ✅ df shape: {self.df.shape if self.df is not None else 'None'}")
                print(f"   ✅ Classes: {len(self.fertilizer_classes) if self.fertilizer_classes else 0}")
                
            else:
                print(f"   ❌ Processed data not found")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error in load_model_and_data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_plot(self, plt_figure, filename):
        """Save plot to file"""
        try:
            filepath = os.path.join(REPORTS_DIR, "images", filename)
            plt_figure.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            return f"images/{filename}"
        except Exception as e:
            print(f"❌ Error saving plot {filename}: {e}")
            return None
    
    # ============================================
    # PLOT 1: Fertilizer Distribution (Target Variable)
    # ============================================
    def generate_fertilizer_distribution(self):
        """Plot 1: Top 10 Fertilizer Types"""
        try:
            if self.df is None or 'Fertilizer' not in self.df.columns:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            fert_counts = self.df['Fertilizer'].value_counts().head(10)
            colors = plt.cm.Set3(np.linspace(0, 1, len(fert_counts)))
            bars = ax.bar(range(len(fert_counts)), fert_counts.values, color=colors)
            
            ax.set_title('Top 10 Fertilizer Types', fontsize=14, fontweight='bold')
            ax.set_xlabel('Fertilizer Type')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(fert_counts)))
            ax.set_xticklabels(fert_counts.index, rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            return self.save_plot(fig, '01_fertilizer_distribution.png')
        except Exception as e:
            print(f"❌ Error in fertilizer_distribution: {e}")
            return None
    
    # ============================================
    # PLOT 2: NPK Distribution Box Plot
    # ============================================
    def generate_npk_distribution(self):
        """Plot 2: NPK Distribution Box Plot"""
        try:
            if self.df is None:
                return None
            
            required_cols = ['Nitrogen', 'Phosphorus', 'Potassium']
            if not all(col in self.df.columns for col in required_cols):
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            nutrient_data = [self.df['Nitrogen'], self.df['Phosphorus'], self.df['Potassium']]
            bp = ax.boxplot(nutrient_data, patch_artist=True,
                           labels=['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'])
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title('Distribution of NPK Values', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self.save_plot(fig, '02_npk_distribution.png')
        except Exception as e:
            print(f"❌ Error in npk_distribution: {e}")
            return None
    
    # ============================================
    # PLOT 3: Soil Type Distribution
    # ============================================
    def generate_soil_distribution(self):
        """Plot 3: Soil Type Distribution"""
        try:
            if self.df is None or 'Soil Type' not in self.df.columns:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            soil_counts = self.df['Soil Type'].value_counts()
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(soil_counts)))
            bars = ax.bar(soil_counts.index, soil_counts.values, color=colors)
            
            ax.set_title('Distribution of Soil Types', fontsize=14, fontweight='bold')
            ax.set_xlabel('Soil Type')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            return self.save_plot(fig, '03_soil_distribution.png')
        except Exception as e:
            print(f"❌ Error in soil_distribution: {e}")
            return None
    
    # ============================================
    # PLOT 4: Correlation Heatmap
    # ============================================
    def generate_correlation_heatmap(self):
        """Plot 4: Correlation Heatmap"""
        try:
            if self.df is None:
                return None
            
            numeric_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature',
                           'Humidity', 'Soil Moisture']
            available_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if len(available_cols) < 2:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            corr_matrix = self.df[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            return self.save_plot(fig, '04_correlation_heatmap.png')
        except Exception as e:
            print(f"❌ Error in correlation_heatmap: {e}")
            return None
    
    # ============================================
    # PLOT 5: Temperature vs Humidity Scatter
    # ============================================
    def generate_temp_humidity_scatter(self):
        """Plot 5: Temperature vs Humidity Scatter"""
        try:
            if self.df is None:
                return None
            
            required_cols = ['Temperature', 'Humidity', 'Fertilizer']
            if not all(col in self.df.columns for col in required_cols):
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            top_fertilizers = self.df['Fertilizer'].value_counts().head(5).index
            df_top = self.df[self.df['Fertilizer'].isin(top_fertilizers)]
            
            if len(df_top) == 0:
                return None
            
            scatter = ax.scatter(df_top['Temperature'], df_top['Humidity'],
                               c=pd.Categorical(df_top['Fertilizer']).codes,
                               cmap='viridis', alpha=0.6, s=30)
            
            ax.set_title('Temperature vs Humidity\n(Top 5 Fertilizers)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Humidity (%)')
            ax.grid(True, alpha=0.3)
            
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                         label=fert, markerfacecolor=plt.cm.viridis(i/5),
                                         markersize=8)
                              for i, fert in enumerate(top_fertilizers)]
            ax.legend(handles=legend_elements, title='Fertilizer', bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            return self.save_plot(fig, '05_temp_humidity_scatter.png')
        except Exception as e:
            print(f"❌ Error in temp_humidity_scatter: {e}")
            return None
    
    # ============================================
    # PLOT 6: Feature Importance (Global)
    # ============================================
    def generate_feature_importance(self):
        """Plot 6: Feature Importance Plot"""
        try:
            if self.model is None or self.feature_names is None:
                return None
            
            if not hasattr(self.model, 'feature_importances_'):
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
            bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
            
            ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features', fontsize=14, fontweight='bold')
            ax.set_title('Feature Importance Plot', fontsize=16, fontweight='bold', pad=20)
            
            for i, (bar, val) in enumerate(zip(bars, feature_importance['importance'])):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            return self.save_plot(fig, '06_feature_importance.png')
        except Exception as e:
            print(f"❌ Error in feature_importance: {e}")
            return None
    
    # ============================================
    # PLOT 7: Partial Dependence Plots
    # ============================================
    def generate_partial_dependence(self):
        """Plot 7: Partial Dependence Plots"""
        try:
            if self.model is None or self.X_train is None:
                return None
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil Moisture']
            
            # Find feature indices
            feature_indices = []
            for feature in features:
                for i, name in enumerate(self.feature_names):
                    if feature in name:
                        feature_indices.append(i)
                        break
            
            if len(feature_indices) < 6:
                return None
            
            target_idx = 0
            
            for i, (feature, idx) in enumerate(zip(features, feature_indices)):
                row, col = i // 3, i % 3
                try:
                    PartialDependenceDisplay.from_estimator(
                        self.model,
                        self.X_train,
                        [idx],
                        target=target_idx,
                        ax=axes[row, col],
                        grid_resolution=50,
                        kind='average'
                    )
                    axes[row, col].set_title(f'PDP: {feature}', fontsize=12, fontweight='bold')
                    axes[row, col].grid(True, alpha=0.3)
                except Exception as e:
                    axes[row, col].text(0.5, 0.5, f'Error: {feature}',
                                       ha='center', va='center', transform=axes[row, col].transAxes)
            
            plt.suptitle('Partial Dependence Plots', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            return self.save_plot(fig, '07_partial_dependence.png')
        except Exception as e:
            print(f"❌ Error in partial_dependence: {e}")
            return None
    
    # ============================================
    # PLOT 8: SHAP Summary Plot
    # ============================================
    def generate_shap_summary(self):
        """Plot 8: SHAP Summary Plot"""
        try:
            if self.model is None or self.X_train is None:
                return None
            
            X_sample = self.X_train.sample(n=min(100, len(self.X_train)), random_state=42)
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if isinstance(shap_values, list):
                shap_values_agg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values_agg = np.abs(shap_values)
            
            shap.summary_plot(shap_values_agg, X_sample, show=False, plot_type="bar",
                            feature_names=self.feature_names)
            plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return self.save_plot(fig, '08_shap_summary.png')
        except Exception as e:
            print(f"❌ Error in shap_summary: {e}")
            return None
    
    # ============================================
    # PLOT 9: SHAP Force Plot (Local)
    # ============================================
    def generate_shap_force(self):
        """Plot 9: SHAP Force Plot for a single prediction"""
        try:
            if self.model is None or self.X_train is None:
                return None
            
            # Take a sample from training data
            X_sample = self.X_train.sample(1, random_state=42)
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Create force plot
            if isinstance(shap_values, list):
                # For multi-class, take the first class
                shap.force_plot(
                    explainer.expected_value[0],
                    shap_values[0][0],
                    X_sample.iloc[0],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
            else:
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    X_sample.iloc[0],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
            
            plt.title('SHAP Force Plot - Local Explanation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return self.save_plot(plt.gcf(), '09_shap_force.png')
        except Exception as e:
            print(f"❌ Error in shap_force: {e}")
            return None
    
    # ============================================
    # PLOT 10: Decision Tree Surrogate
    # ============================================
    def generate_decision_tree(self):
        """Plot 10: Decision Tree Surrogate Model"""
        try:
            if self.X_train is None or self.y_train is None or self.X_test is None:
                return None
            
            surrogate_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
            surrogate_tree.fit(self.X_train, self.y_train)
            surrogate_accuracy = surrogate_tree.score(self.X_test, self.y_test)
            
            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(
                surrogate_tree,
                feature_names=self.feature_names,
                class_names=self.fertilizer_classes,
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=3,
                ax=ax
            )
            plt.title(f'Decision Tree Surrogate Model\n(Accuracy: {surrogate_accuracy:.2f})',
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return self.save_plot(fig, '10_decision_tree.png')
        except Exception as e:
            print(f"❌ Error in decision_tree: {e}")
            return None
    
    # ============================================
    # PLOT 11: LIME Explanation (Local)
    # ============================================
    def generate_lime_explanation(self):
        """Plot 11: LIME Explanation for a single prediction"""
        try:
            if self.X_train is None or self.model is None or self.le_fert is None:
                return None
            
            # Take a sample from training data
            single_case = self.X_train.sample(1, random_state=42)
            
            # Create LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=self.fertilizer_classes,
                mode='classification',
                discretize_continuous=True
            )
            
            # Explain the case
            lime_exp = lime_explainer.explain_instance(
                data_row=single_case.values[0],
                predict_fn=self.model.predict_proba,
                num_features=8
            )
            
            # Plot LIME explanation
            fig = plt.figure(figsize=(12, 6))
            lime_exp.as_pyplot_figure()
            
            # Get prediction
            prediction_idx = self.model.predict(single_case)[0]
            prediction = self.fertilizer_classes[prediction_idx]
            
            plt.title(f'LIME Explanation - Local Interpretation\nPrediction: {prediction}',
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return self.save_plot(fig, '11_lime_explanation.png')
        except Exception as e:
            print(f"❌ Error in lime_explanation: {e}")
            return None
    
    # ============================================
    # PLOT 12: Counterfactual Explanations
    # ============================================
    def generate_counterfactual(self):
        """Plot 12: Counterfactual Explanations"""
        try:
            if self.X_train is None or self.model is None:
                return None
            
            # Take a sample from training data
            single_case = self.X_train.sample(1, random_state=42)
            prediction_idx = self.model.predict(single_case)[0]
            prediction = self.fertilizer_classes[prediction_idx]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            features_to_test = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil Moisture']
            
            for i, feature in enumerate(features_to_test):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Find feature index
                feature_idx = None
                for j, name in enumerate(self.feature_names):
                    if feature in name:
                        feature_idx = j
                        break
                
                if feature_idx is None:
                    ax.text(0.5, 0.5, f'Feature {feature} not found',
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Get current value
                current_val = single_case.iloc[0, feature_idx]
                
                # Test range around current value
                test_range = np.linspace(max(0, current_val - 30), current_val + 30, 50)
                prob_predictions = []
                
                for val in test_range:
                    modified = single_case.copy()
                    modified.iloc[0, feature_idx] = val
                    probs = self.model.predict_proba(modified)[0]
                    prob_predictions.append(probs[prediction_idx])
                
                # Plot
                ax.plot(test_range, prob_predictions, 'b-', linewidth=2)
                ax.axvline(x=current_val, color='red', linestyle='--', linewidth=2, label='Current value')
                ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Decision boundary')
                
                ax.set_xlabel(feature, fontsize=12, fontweight='bold')
                ax.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
                ax.set_title(f'Counterfactual: {feature}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=8)
            
            plt.suptitle(f'Counterfactual Explanations\nOriginal Prediction: {prediction}',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            return self.save_plot(fig, '12_counterfactual.png')
        except Exception as e:
            print(f"❌ Error in counterfactual: {e}")
            return None
    
    # ============================================
    # Generate All Plots
    # ============================================
    def generate_all_plots(self):
        """Generate all 12 plots"""
        plots = {}
        
        print("\n📊 Generating 12 plots...")
        
        # Basic EDA Plots
        print("   1/12 Generating Fertilizer Distribution Plot...")
        plots['fertilizer_distribution'] = self.generate_fertilizer_distribution()
        
        print("   2/12 Generating NPK Distribution Plot...")
        plots['npk_distribution'] = self.generate_npk_distribution()
        
        print("   3/12 Generating Soil Type Distribution Plot...")
        plots['soil_distribution'] = self.generate_soil_distribution()
        
        print("   4/12 Generating Correlation Heatmap...")
        plots['correlation'] = self.generate_correlation_heatmap()
        
        print("   5/12 Generating Temperature-Humidity Scatter Plot...")
        plots['scatter'] = self.generate_temp_humidity_scatter()
        
        # Model Interpretation Plots
        print("   6/12 Generating Feature Importance Plot...")
        plots['feature_importance'] = self.generate_feature_importance()
        
        print("   7/12 Generating Partial Dependence Plots...")
        plots['pdp'] = self.generate_partial_dependence()
        
        print("   8/12 Generating SHAP Summary Plot...")
        plots['shap_summary'] = self.generate_shap_summary()
        
        print("   9/12 Generating SHAP Force Plot...")
        plots['shap_force'] = self.generate_shap_force()
        
        print("   10/12 Generating Decision Tree Surrogate...")
        plots['decision_tree'] = self.generate_decision_tree()
        
        print("   11/12 Generating LIME Explanation...")
        plots['lime'] = self.generate_lime_explanation()
        
        print("   12/12 Generating Counterfactual Explanations...")
        plots['counterfactual'] = self.generate_counterfactual()
        
        # Count successful plots
        successful = sum(1 for v in plots.values() if v is not None)
        print(f"\n✅ Successfully generated {successful}/12 plots")
        
        return plots
    
    # ============================================
    # Generate HTML Report
    # ============================================
    def generate_html_report(self):
        """Generate complete HTML report with all plots"""
        try:
            print("\n🚀 Starting comprehensive report generation...")
            
            if self.df is None:
                print("❌ Cannot generate report: df is None")
                return {'error': 'Data not loaded properly'}
            
            plots = self.generate_all_plots()
            
            # Get insights
            top_fertilizers = self.df['Fertilizer'].value_counts().head(3)
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fertilizer Model Complete Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        text-align: center;
                        border-bottom: 3px solid #27ae60;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #34495e;
                        margin-top: 30px;
                        border-left: 5px solid #27ae60;
                        padding-left: 15px;
                    }}
                    .plot-container {{
                        margin: 20px;
                        text-align: center;
                        background-color: #f9f9f9;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        break-inside: avoid;
                    }}
                    .plot-container img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }}
                    .plot-caption {{
                        margin-top: 10px;
                        color: #7f8c8d;
                        font-style: italic;
                    }}
                    .metrics {{
                        background-color: #ecf0f1;
                        padding: 20px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                    .grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                        gap: 20px;
                    }}
                    .insight-card {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 10px 0;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        color: #7f8c8d;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌱 Fertilizer Recommendation Model - Complete Analysis Report</h1>
                    <p style="text-align: center; color: #27ae60;">
                        Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                    
                    <div class="metrics">
                        <h2>📊 Dataset Overview</h2>
                        <ul>
                            <li><strong>Dataset Shape:</strong> {self.df.shape[0]} rows, {self.df.shape[1]} columns</li>
                            <li><strong>Feature Names:</strong> {', '.join(self.feature_names) if self.feature_names else 'N/A'}</li>
                            <li><strong>Number of Fertilizer Classes:</strong> {len(self.fertilizer_classes) if self.fertilizer_classes else 0}</li>
                            <li><strong>Model Accuracy:</strong> {self.model_accuracy:.2%}</li>
                        </ul>
                    </div>
                    
                    <div class="insight-card">
                        <h3>📈 Top 3 Most Common Fertilizers:</h3>
                        <ul>
            """
            
            for i, (fert, count) in enumerate(top_fertilizers.items(), 1):
                html_content += f"<li><strong>{i}. {fert}:</strong> {count} occurrences ({count/len(self.df)*100:.1f}%)</li>"
            
            html_content += f"""
                        </ul>
                    </div>
                    
                    <div class="metrics">
                        <h3>🧪 Average Nutrient Values:</h3>
                        <ul>
                            <li><strong>Nitrogen:</strong> {self.df['Nitrogen'].mean():.1f}</li>
                            <li><strong>Phosphorus:</strong> {self.df['Phosphorus'].mean():.1f}</li>
                            <li><strong>Potassium:</strong> {self.df['Potassium'].mean():.1f}</li>
                        </ul>
                        
                        <h3>🌡️ Environmental Ranges:</h3>
                        <ul>
                            <li><strong>Temperature:</strong> {self.df['Temperature'].min():.1f}°C to {self.df['Temperature'].max():.1f}°C</li>
                            <li><strong>Humidity:</strong> {self.df['Humidity'].min():.1f}% to {self.df['Humidity'].max():.1f}%</li>
                            <li><strong>Soil Moisture:</strong> {self.df['Soil Moisture'].min():.1f}% to {self.df['Soil Moisture'].max():.1f}%</li>
                        </ul>
                    </div>
                    
                    <h2>📊 Data Exploration Plots</h2>
                    <div class="grid">
            """
            
            # Add EDA plots
            eda_plots = ['fertilizer_distribution', 'soil_distribution', 'npk_distribution', 'scatter', 'correlation']
            for plot_name in eda_plots:
                if plots.get(plot_name):
                    html_content += f"""
                        <div class="plot-container">
                            <img src="{plots[plot_name]}" alt="{plot_name}">
                            <div class="plot-caption">{plot_name.replace('_', ' ').title()}</div>
                        </div>
                    """
            
            html_content += """
                    </div>
                    
                    <h2>🎯 Model Interpretation - Global</h2>
                    <div class="grid">
            """
            
            # Add global interpretation plots
            global_plots = ['feature_importance', 'shap_summary', 'pdp', 'decision_tree']
            for plot_name in global_plots:
                if plots.get(plot_name):
                    html_content += f"""
                        <div class="plot-container">
                            <img src="{plots[plot_name]}" alt="{plot_name}">
                            <div class="plot-caption">{plot_name.replace('_', ' ').title()}</div>
                        </div>
                    """
            
            html_content += """
                    </div>
                    
                    <h2>🔍 Local Interpretations</h2>
                    <div class="grid">
            """
            
            # Add local interpretation plots
            local_plots = ['shap_force', 'lime', 'counterfactual']
            for plot_name in local_plots:
                if plots.get(plot_name):
                    html_content += f"""
                        <div class="plot-container">
                            <img src="{plots[plot_name]}" alt="{plot_name}">
                            <div class="plot-caption">{plot_name.replace('_', ' ').title()}</div>
                        </div>
                    """
            
            html_content += f"""
                    </div>
                    
                    <h2>📋 All Fertilizer Classes ({len(self.fertilizer_classes)})</h2>
                    <div class="metrics">
                        <p style="column-count: 4; column-gap: 20px;">
            """
            
            for fert in sorted(self.fertilizer_classes)[:50]:  # Limit to 50
                html_content += f"{fert}<br>"
            
            html_content += f"""
                        </p>
                    </div>
                    
                    <div class="footer">
                        <p>Report generated by Fertilizer Recommendation System</p>
                        <p>Total Plots Generated: {sum(1 for v in plots.values() if v is not None)}/12</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save report
            report_path = os.path.join(REPORTS_DIR, "latest_report.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_path = os.path.join(REPORTS_DIR, f"report_{timestamp}.html")
            with open(timestamped_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\n✅ Report generated successfully!")
            print(f"   📁 Latest report: {report_path}")
            print(f"   🖼️  Images generated: {sum(1 for v in plots.values() if v is not None)}/12")
            
            return {
                'latest_report': 'report_fertilizer/latest_report.html',
                'timestamped_report': f'report_fertilizer/report_{timestamp}.html',
                'images': [v for v in plots.values() if v is not None]
            }
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

# Create global instance
print("🔧 Initializing FertilizerReportGenerator...")
report_generator = FertilizerReportGenerator()
