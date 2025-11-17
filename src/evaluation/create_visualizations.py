"""
Create visualizations for pass success prediction models.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc


def load_data_and_models():
    """Load test data and trained models."""
    print("Loading data and models...")
    
    # Load ML data
    df = pd.read_csv('data/processed/passes_ml_ready.csv')
    
    # Load models
    with open('results/models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('results/models/xgboost.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('results/models/lightgbm.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }
    
    return df, models


def prepare_test_data(df):
    """Prepare test data (same split as training)."""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(['pass_success', 'match_id'], axis=1)
    y = df['pass_success']
    match_ids = df['match_id']
    
    unique_matches = match_ids.unique()
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=0.2, random_state=42
    )
    
    test_mask = match_ids.isin(test_matches)
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    return X_test, y_test


def plot_model_comparison(models, X_test, y_test, output_dir='results/figures'):
    """Create model comparison bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating model comparison plot...")
    
    # Calculate metrics for each model
    metrics_data = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.15
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 2)
        ax.bar(x + offset, metrics_df[metric], width, label=metric, color=color, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Pass Success Prediction - Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_confusion_matrices(models, X_test, y_test, output_dir='results/figures'):
    """Create confusion matrices for all models."""
    print("\nCreating confusion matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Fail', 'Success'], yticklabels=['Fail', 'Success'])
        axes[idx].set_title(f'{name}', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.suptitle('Confusion Matrices - Pass Success Prediction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_roc_curves(models, X_test, y_test, output_dir='results/figures'):
    """Create ROC curves for all models."""
    print("\nCreating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Pass Success Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_feature_importance(models, X_test, output_dir='results/figures'):
    """Plot feature importance for tree-based models."""
    print("\nCreating feature importance plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (name, model) in enumerate(models.items()):
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_test.columns
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:15]  # Top 15
            
            axes[idx].barh(range(15), importances[indices], color='#3498db', alpha=0.8)
            axes[idx].set_yticks(range(15))
            axes[idx].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
            axes[idx].set_xlabel('Importance', fontweight='bold')
            axes[idx].set_title(f'{name}', fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Top 15 Feature Importances - Pass Success Prediction', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    # Load data and models
    df, models = load_data_and_models()
    X_test, y_test = prepare_test_data(df)
    
    # Create visualizations
    plot_model_comparison(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_feature_importance(models, X_test)
    
    print("\nAll visualizations created successfully!")
