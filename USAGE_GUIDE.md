# Pass Success Prediction - Usage Guide

Complete guide for testing and using the pass success prediction models.

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/mehmetyalc/pass-success-prediction.git
cd pass-success-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Step 1: Collect data from StatsBomb
python src/data_collection/collect_statsbomb_data_v2.py

# Step 2: Engineer features
python src/features/engineer_pass_features.py

# Step 3: Train models
python src/models/train_pass_models.py

# Step 4: Create visualizations
python src/evaluation/create_visualizations.py
```

## Testing the Models

### Option 1: Use Pre-trained Models (Recommended)

The repository includes pre-trained models in `results/models/`. You can load and use them directly:

```python
import pickle
import pandas as pd
import numpy as np

# Load trained model
with open('results/models/lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
df = pd.read_csv('data/processed/passes_ml_ready.csv')
X = df.drop(['pass_success', 'match_id'], axis=1)
y = df['pass_success']

# Make predictions
predictions = model.predict(X[:10])  # First 10 passes
probabilities = model.predict_proba(X[:10])

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

### Option 2: Test with Custom Pass Data

Create your own pass scenario:

```python
import pickle
import pandas as pd

# Load model
with open('results/models/lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a sample pass (all 46 features required)
sample_pass = {
    # Geometric features
    'pass_distance': 25.0,
    'pass_angle': 45.0,
    'distance_from_own_goal': 50.0,
    'distance_to_opponent_goal': 70.0,
    'pass_progression': 15.0,
    'lateral_movement': 10.0,
    
    # Binary features
    'is_forward_pass': 1,
    'is_backward_pass': 0,
    'is_long_pass': 0,
    'is_short_pass': 0,
    'is_diagonal_pass': 1,
    'is_cross_field': 0,
    'pass_into_final_third': 0,
    'pass_into_box': 0,
    'is_under_pressure': 1,  # Under pressure
    'is_counter_attack': 0,
    
    # Body part (one-hot encoded)
    'body_part_DROP_KICK': 0,
    'body_part_HEAD': 0,
    'body_part_KEEPER_ARM': 0,
    'body_part_LEFT_FOOT': 0,
    'body_part_OTHER': 0,
    'body_part_RIGHT_FOOT': 1,  # Right foot pass
    
    # Pass type (one-hot encoded)
    'pass_type_CROSS': 0,
    'pass_type_HAND_PASS': 0,
    'pass_type_HEAD_PASS': 0,
    'pass_type_HIGH_PASS': 0,
    'pass_type_LONG_BALL': 0,
    'pass_type_SHOT_ASSIST': 0,
    'pass_type_THROUGH_BALL': 0,
    
    # Set piece (one-hot encoded)
    'set_piece_CORNER_KICK': 0,
    'set_piece_FREE_KICK': 0,
    'set_piece_GOAL_KICK': 0,
    'set_piece_KICK_OFF': 0,
    'set_piece_THROW_IN': 0,
    
    # Zone start (one-hot encoded)
    'zone_start_defensive': 0,
    'zone_start_middle': 1,  # Starting from middle third
    'zone_start_attacking': 0,
    
    # Zone end (one-hot encoded)
    'zone_end_defensive': 0,
    'zone_end_middle': 0,
    'zone_end_attacking': 1,  # Ending in attacking third
    
    # Lateral start (one-hot encoded)
    'lateral_start_left': 0,
    'lateral_start_center': 1,  # Center
    'lateral_start_right': 0,
    
    # Lateral end (one-hot encoded)
    'lateral_end_left': 0,
    'lateral_end_center': 0,
    'lateral_end_right': 1,  # Right side
}

# Convert to DataFrame
pass_df = pd.DataFrame([sample_pass])

# Predict
prediction = model.predict(pass_df)[0]
probability = model.predict_proba(pass_df)[0]

print(f"Pass Success Prediction: {'SUCCESS' if prediction == 1 else 'FAIL'}")
print(f"Probability of Success: {probability[1]:.2%}")
print(f"Probability of Failure: {probability[0]:.2%}")
```

## Understanding the Features

### Required Features (46 total)

**Numeric Features (16):**
1. `pass_distance` - Euclidean distance (meters)
2. `pass_angle` - Direction in degrees
3. `distance_from_own_goal` - Distance from own goal
4. `distance_to_opponent_goal` - Distance to opponent goal
5. `pass_progression` - Forward/backward movement
6. `lateral_movement` - Side-to-side movement
7. `is_forward_pass` - 1 if forward, 0 otherwise
8. `is_backward_pass` - 1 if backward, 0 otherwise
9. `is_long_pass` - 1 if >30m, 0 otherwise
10. `is_short_pass` - 1 if <10m, 0 otherwise
11. `is_diagonal_pass` - 1 if diagonal angle, 0 otherwise
12. `is_cross_field` - 1 if large lateral movement, 0 otherwise
13. `pass_into_final_third` - 1 if entering attacking third, 0 otherwise
14. `pass_into_box` - 1 if into penalty area, 0 otherwise
15. `is_under_pressure` - 1 if under pressure, 0 otherwise
16. `is_counter_attack` - 1 if counter attack, 0 otherwise

**One-Hot Encoded Features (30):**
- Body part: DROP_KICK, HEAD, KEEPER_ARM, LEFT_FOOT, OTHER, RIGHT_FOOT
- Pass type: CROSS, HAND_PASS, HEAD_PASS, HIGH_PASS, LONG_BALL, SHOT_ASSIST, THROUGH_BALL
- Set piece: CORNER_KICK, FREE_KICK, GOAL_KICK, KICK_OFF, THROW_IN
- Zone start: defensive, middle, attacking
- Zone end: defensive, middle, attacking
- Lateral start: left, center, right
- Lateral end: left, center, right

## Model Comparison

### When to Use Each Model

**LightGBM (Recommended)**
- Best overall accuracy: 80.4%
- Best balance of precision and recall
- Fast inference time
- Use for: Production deployment

**XGBoost**
- Second best: 79.6% accuracy
- Slightly better recall than LightGBM
- Use for: When recall is more important

**Random Forest**
- Highest recall: 92.8%
- Lower precision: 77.7%
- Use for: When you want to catch all successful passes

## Interpreting Results

### Probability Thresholds

Default threshold is 0.5, but you can adjust:

```python
# Get probabilities
proba = model.predict_proba(X_test)[:, 1]

# Custom threshold (e.g., 0.6 for higher confidence)
predictions_custom = (proba >= 0.6).astype(int)

# This increases precision but decreases recall
```

### Feature Importance

Check which features matter most:

```python
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Sort and plot top 15
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 6))
plt.barh(range(15), importances[indices])
plt.yticks(range(15), [feature_names[i] for i in indices])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_custom.png')
```

## Common Use Cases

### 1. Evaluate Pass Difficulty

```python
# Load multiple pass scenarios
passes = pd.read_csv('my_passes.csv')  # Your custom passes

# Predict
probabilities = model.predict_proba(passes)[:, 1]

# Rank by difficulty (lower probability = harder)
passes['success_probability'] = probabilities
passes_sorted = passes.sort_values('success_probability')

print("Hardest passes:")
print(passes_sorted.head(10))
```

### 2. Compare Players

```python
# Filter passes by player
player_a_passes = df[df['player_id'] == 'player_a_id']
player_b_passes = df[df['player_id'] == 'player_b_id']

# Predict success rates
player_a_success = model.predict(player_a_passes.drop(['pass_success', 'match_id'], axis=1)).mean()
player_b_success = model.predict(player_b_passes.drop(['pass_success', 'match_id'], axis=1)).mean()

print(f"Player A success rate: {player_a_success:.2%}")
print(f"Player B success rate: {player_b_success:.2%}")
```

### 3. Analyze Match Performance

```python
# Filter by match
match_passes = df[df['match_id'] == 7525]  # Russia vs Saudi Arabia

# Predict and compare to actual
X_match = match_passes.drop(['pass_success', 'match_id'], axis=1)
y_match = match_passes['pass_success']

predictions = model.predict(X_match)
accuracy = (predictions == y_match).mean()

print(f"Model accuracy for this match: {accuracy:.2%}")
```

## Troubleshooting

### Issue: "Feature mismatch"

Make sure you have exactly 46 features in the correct order. Load the feature list:

```python
with open('data/processed/feature_list.txt', 'r') as f:
    feature_list = f.read()
print(feature_list)
```

### Issue: "Model file not found"

Retrain the models:

```bash
python src/models/train_pass_models.py
```

### Issue: "Data not found"

Recollect and process data:

```bash
python src/data_collection/collect_statsbomb_data_v2.py
python src/features/engineer_pass_features.py
```

## Performance Metrics Explained

- **Accuracy:** Overall correct predictions (80.4%)
- **Precision:** Of predicted successes, how many were actually successful (83.8%)
- **Recall:** Of actual successes, how many did we predict (89.6%)
- **F1-Score:** Harmonic mean of precision and recall (0.866)
- **ROC-AUC:** Area under ROC curve, discriminative ability (0.853)

## Next Steps

1. Try the model on your own pass data
2. Experiment with different thresholds
3. Analyze feature importances for your data
4. Compare model predictions with actual outcomes
5. Use insights to improve team tactics

## Support

For issues or questions:
- GitHub Issues: https://github.com/mehmetyalc/pass-success-prediction/issues
- Email: Contact via GitHub profile

## References

- StatsBomb Open Data: https://github.com/statsbomb/open-data
- Kloppy Documentation: https://kloppy.pysport.org/
- Model Training Details: See `results/reports/training_report.txt`
