"""
Train ML models for pass success prediction.
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb


def load_ml_data(input_file='data/processed/passes_ml_ready.csv'):
    """Load ML-ready data."""
    print(f"Loading ML data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} passes with {df.shape[1]-2} features")
    return df


def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """
    Prepare train/test split.
    Use match_id for stratification to avoid data leakage.
    """
    print("\nPreparing train/test split...")
    
    # Separate features and target
    X = df.drop(['pass_success', 'match_id'], axis=1)
    y = df['pass_success']
    match_ids = df['match_id']
    
    # Split by match to avoid leakage
    unique_matches = match_ids.unique()
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=test_size, random_state=random_state
    )
    
    train_mask = match_ids.isin(train_matches)
    test_mask = match_ids.isin(test_matches)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Training set: {len(X_train)} passes from {len(train_matches)} matches")
    print(f"Test set: {len(X_test)} passes from {len(test_matches)} matches")
    print(f"Train success rate: {y_train.mean():.2%}")
    print(f"Test success rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("\nTraining Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    print("\nTraining XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    print("XGBoost training complete")
    
    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM classifier."""
    print("\nTraining LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print("LightGBM training complete")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance."""
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    results = {
        'model': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    results['cv_accuracy_mean'] = cv_scores.mean()
    results['cv_accuracy_std'] = cv_scores.std()
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  Precision:      {results['precision']:.4f}")
    print(f"  Recall:         {results['recall']:.4f}")
    print(f"  F1-Score:       {results['f1_score']:.4f}")
    print(f"  ROC-AUC:        {results['roc_auc']:.4f}")
    print(f"  CV Accuracy:    {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std']:.4f})")
    
    return results


def save_models(models, output_dir='results/models'):
    """Save trained models."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        model_file = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} to {model_file}")


def save_results(all_results, output_dir='results/reports'):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as DataFrame
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(output_dir, 'model_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Save detailed report
    report_file = os.path.join(output_dir, 'training_report.txt')
    with open(report_file, 'w') as f:
        f.write("Pass Success Prediction - Model Training Report\n")
        f.write("=" * 60 + "\n\n")
        
        for result in all_results:
            f.write(f"{result['model']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Train Accuracy:     {result['train_accuracy']:.4f}\n")
            f.write(f"Test Accuracy:      {result['test_accuracy']:.4f}\n")
            f.write(f"Precision:          {result['precision']:.4f}\n")
            f.write(f"Recall:             {result['recall']:.4f}\n")
            f.write(f"F1-Score:           {result['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:            {result['roc_auc']:.4f}\n")
            f.write(f"CV Accuracy:        {result['cv_accuracy_mean']:.4f} (+/- {result['cv_accuracy_std']:.4f})\n")
            f.write("\n")
        
        # Best model
        best_model = max(all_results, key=lambda x: x['test_accuracy'])
        f.write("\nBest Model (by Test Accuracy):\n")
        f.write(f"  {best_model['model']} - {best_model['test_accuracy']:.4f}\n")
    
    print(f"Training report saved to {report_file}")


if __name__ == "__main__":
    # Load data
    df = load_ml_data()
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)
    
    # Evaluate models
    rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")
    lgb_results = evaluate_model(lgb_model, X_train, X_test, y_train, y_test, "LightGBM")
    
    # Save models
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }
    save_models(models)
    
    # Save results
    all_results = [rf_results, xgb_results, lgb_results]
    save_results(all_results)
    
    print("\nModel training complete!")
