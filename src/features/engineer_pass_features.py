"""
Engineer features for pass success prediction with contextual information.
"""
import os
import pandas as pd
import numpy as np
from math import sqrt, atan2, degrees


def load_raw_data(input_file='data/raw/statsbomb_passes_raw.csv'):
    """Load raw pass data."""
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} passes")
    return df


def calculate_pass_distance(df):
    """Calculate Euclidean distance of pass."""
    df['pass_distance'] = np.sqrt(
        (df['end_coordinates_x'] - df['coordinates_x'])**2 +
        (df['end_coordinates_y'] - df['coordinates_y'])**2
    )
    return df


def calculate_pass_angle(df):
    """Calculate pass angle (direction)."""
    df['pass_angle'] = np.degrees(np.arctan2(
        df['end_coordinates_y'] - df['coordinates_y'],
        df['end_coordinates_x'] - df['coordinates_x']
    ))
    return df


def calculate_field_position_features(df):
    """
    Calculate field position features.
    Assuming pitch dimensions: 120x80 (StatsBomb standard).
    """
    # Distance from own goal
    df['distance_from_own_goal'] = np.sqrt(
        df['coordinates_x']**2 + (df['coordinates_y'] - 40)**2
    )
    
    # Distance from opponent goal
    df['distance_to_opponent_goal'] = np.sqrt(
        (120 - df['coordinates_x'])**2 + (df['coordinates_y'] - 40)**2
    )
    
    # Field zones (defensive, middle, attacking third)
    df['zone_start'] = pd.cut(
        df['coordinates_x'],
        bins=[0, 40, 80, 120],
        labels=['defensive', 'middle', 'attacking']
    )
    
    df['zone_end'] = pd.cut(
        df['end_coordinates_x'],
        bins=[0, 40, 80, 120],
        labels=['defensive', 'middle', 'attacking']
    )
    
    # Lateral position (left, center, right)
    df['lateral_start'] = pd.cut(
        df['coordinates_y'],
        bins=[0, 26.67, 53.33, 80],
        labels=['left', 'center', 'right']
    )
    
    df['lateral_end'] = pd.cut(
        df['end_coordinates_y'],
        bins=[0, 26.67, 53.33, 80],
        labels=['left', 'center', 'right']
    )
    
    # Pass progression (forward/backward)
    df['pass_progression'] = df['end_coordinates_x'] - df['coordinates_x']
    df['is_forward_pass'] = (df['pass_progression'] > 0).astype(int)
    df['is_backward_pass'] = (df['pass_progression'] < 0).astype(int)
    
    # Lateral movement
    df['lateral_movement'] = abs(df['end_coordinates_y'] - df['coordinates_y'])
    
    return df


def calculate_pass_difficulty_features(df):
    """Calculate features related to pass difficulty."""
    
    # Long pass (>30 meters)
    df['is_long_pass'] = (df['pass_distance'] > 30).astype(int)
    
    # Short pass (<10 meters)
    df['is_short_pass'] = (df['pass_distance'] < 10).astype(int)
    
    # Diagonal pass (angle between 30-60 or -30 to -60 degrees)
    df['is_diagonal_pass'] = (
        ((df['pass_angle'] > 30) & (df['pass_angle'] < 60)) |
        ((df['pass_angle'] < -30) & (df['pass_angle'] > -60))
    ).astype(int)
    
    # Cross-field pass (large lateral movement)
    df['is_cross_field'] = (df['lateral_movement'] > 20).astype(int)
    
    # Pass into final third
    df['pass_into_final_third'] = (
        (df['coordinates_x'] < 80) & (df['end_coordinates_x'] >= 80)
    ).astype(int)
    
    # Pass into penalty area (approximate: x>102, 18<y<62)
    df['pass_into_box'] = (
        (df['end_coordinates_x'] > 102) &
        (df['end_coordinates_y'] > 18) &
        (df['end_coordinates_y'] < 62)
    ).astype(int)
    
    return df


def encode_categorical_features(df):
    """Encode categorical features."""
    
    # Binary encode existing categorical columns
    if 'is_under_pressure' in df.columns:
        df['is_under_pressure'] = df['is_under_pressure'].fillna(False).astype(int)
    
    if 'is_counter_attack' in df.columns:
        df['is_counter_attack'] = df['is_counter_attack'].fillna(False).astype(int)
    
    # One-hot encode body part
    if 'body_part_type' in df.columns:
        body_part_dummies = pd.get_dummies(df['body_part_type'], prefix='body_part')
        df = pd.concat([df, body_part_dummies], axis=1)
    
    # One-hot encode pass type
    if 'pass_type' in df.columns:
        pass_type_dummies = pd.get_dummies(df['pass_type'], prefix='pass_type')
        df = pd.concat([df, pass_type_dummies], axis=1)
    
    # One-hot encode set piece type
    if 'set_piece_type' in df.columns:
        set_piece_dummies = pd.get_dummies(df['set_piece_type'], prefix='set_piece')
        df = pd.concat([df, set_piece_dummies], axis=1)
    
    # Encode zones
    zone_start_dummies = pd.get_dummies(df['zone_start'], prefix='zone_start')
    zone_end_dummies = pd.get_dummies(df['zone_end'], prefix='zone_end')
    lateral_start_dummies = pd.get_dummies(df['lateral_start'], prefix='lateral_start')
    lateral_end_dummies = pd.get_dummies(df['lateral_end'], prefix='lateral_end')
    
    df = pd.concat([df, zone_start_dummies, zone_end_dummies, 
                    lateral_start_dummies, lateral_end_dummies], axis=1)
    
    return df


def create_target_variable(df):
    """Create binary target variable for pass success."""
    # result column: COMPLETE, INCOMPLETE, OUT, OFFSIDE
    df['pass_success'] = (df['result'] == 'COMPLETE').astype(int)
    
    print(f"\nTarget variable distribution:")
    print(df['pass_success'].value_counts())
    print(f"Success rate: {df['pass_success'].mean():.2%}")
    
    return df
    
def select_features_for_ml(df):
    """
    Select relevant features for ML model.
    """
    
    # Drop original categorical columns that were one-hot encoded
    df = df.drop(['set_piece_type', 'body_part_type'], axis=1, errors='ignore')
    
    # Numeric features
    numeric_features = [
        'pass_distance', 'pass_angle', 'distance_from_own_goal',
        'distance_to_opponent_goal', 'pass_progression', 'lateral_movement',
        'is_forward_pass', 'is_backward_pass', 'is_long_pass', 'is_short_pass',
        'is_diagonal_pass', 'is_cross_field', 'pass_into_final_third',
        'pass_into_box', 'is_under_pressure', 'is_counter_attack'
    ]
    
    # One-hot encoded features
    onehot_features = [col for col in df.columns if any(
        prefix in col for prefix in ['body_part_', 'pass_type_', 'set_piece_',
                                      'zone_start_', 'zone_end_',
                                      'lateral_start_', 'lateral_end_']
    )]
    
    # All ML features
    ml_features = numeric_features + onehot_features
    
    # Filter to only existing columns
    ml_features = [f for f in ml_features if f in df.columns]
    
    print(f"\nSelected {len(ml_features)} features for ML:")
    for i, feat in enumerate(ml_features, 1):
        print(f"  {i}. {feat}")
    
    return df, ml_features


def save_processed_data(df, ml_features, output_dir='data/processed'):
    """Save processed data for ML."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full processed dataset
    full_output = os.path.join(output_dir, 'passes_with_features.csv')
    df.to_csv(full_output, index=False)
    print(f"\nFull processed data saved to: {full_output}")
    
    # Save ML-ready dataset (features + target)
    ml_data = df[ml_features + ['pass_success', 'match_id']].copy()
    ml_output = os.path.join(output_dir, 'passes_ml_ready.csv')
    ml_data.to_csv(ml_output, index=False)
    print(f"ML-ready data saved to: {ml_output}")
    print(f"Shape: {ml_data.shape}")
    
    # Save feature list
    feature_list_file = os.path.join(output_dir, 'feature_list.txt')
    with open(feature_list_file, 'w') as f:
        f.write("ML Features for Pass Success Prediction\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features: {len(ml_features)}\n\n")
        for i, feat in enumerate(ml_features, 1):
            f.write(f"{i}. {feat}\n")
    print(f"Feature list saved to: {feature_list_file}")
    
    return ml_output


if __name__ == "__main__":
    # Load raw data
    df = load_raw_data()
    
    # Calculate pass features
    print("\nCalculating pass features...")
    df = calculate_pass_distance(df)
    df = calculate_pass_angle(df)
    df = calculate_field_position_features(df)
    df = calculate_pass_difficulty_features(df)
    
    # Encode categorical features
    print("Encoding categorical features...")
    df = encode_categorical_features(df)
    
    # Create target variable
    df = create_target_variable(df)
    
    # Select features for ML
    df, ml_features = select_features_for_ml(df)
    
    # Save processed data
    save_processed_data(df, ml_features)
    
    print("\nFeature engineering complete!")
