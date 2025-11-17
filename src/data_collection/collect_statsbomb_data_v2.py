"""
Collect StatsBomb open data using Kloppy for pass success prediction.
Uses match_id based approach.
"""
import os
import json
from kloppy import statsbomb
from kloppy.domain import EventType
import pandas as pd


def get_sample_match_ids():
    """
    Return a sample of StatsBomb open data match IDs.
    These are known to be available in the open dataset.
    """
    # Sample match IDs from different competitions
    match_ids = [
        # FIFA World Cup 2018
        7478, 7512, 7525, 7537, 7543, 7549, 7556, 7567,
        # La Liga
        266897, 266901, 266904, 266909, 266913,
        # Premier League
        3749, 3750, 3751, 3752, 3753,
        # FA Women's Super League
        2275, 2276, 2277, 2278, 2279,
    ]
    return match_ids


def collect_statsbomb_matches(match_ids, max_matches=20):
    """
    Collect StatsBomb open data matches with pass events.
    """
    print(f"Collecting StatsBomb open data for {len(match_ids)} matches...")
    print(f"(Limited to first {max_matches} successful loads)")
    
    all_matches = []
    successful = 0
    
    for match_id in match_ids:
        if successful >= max_matches:
            break
            
        try:
            print(f"\nLoading match {match_id}...")
            
            # Load match data
            dataset = statsbomb.load_open_data(
                match_id=str(match_id),
                event_types=[EventType.PASS]
            )
            
            # Extract match metadata
            match_info = {
                'match_id': match_id,
                'home_team': dataset.metadata.teams[0].name if dataset.metadata.teams else 'unknown',
                'away_team': dataset.metadata.teams[1].name if len(dataset.metadata.teams) > 1 else 'unknown',
                'dataset': dataset
            }
            
            all_matches.append(match_info)
            successful += 1
            print(f"  Success: {match_info['home_team']} vs {match_info['away_team']}")
            
        except Exception as e:
            print(f"  Error loading match {match_id}: {e}")
            continue
    
    print(f"\nTotal matches collected: {len(all_matches)}")
    return all_matches


def extract_pass_events(matches):
    """
    Extract pass events with relevant features from all matches.
    """
    print("\nExtracting pass events...")
    
    all_passes = []
    
    for match in matches:
        dataset = match['dataset']
        
        # Convert to DataFrame
        df = dataset.to_df()
        
        # Filter only pass events
        passes = df[df['event_type'] == 'PASS'].copy()
        
        if len(passes) == 0:
            continue
        
        # Add match context
        passes['match_id'] = match['match_id']
        passes['home_team'] = match['home_team']
        passes['away_team'] = match['away_team']
        
        all_passes.append(passes)
        print(f"  Match {match['match_id']}: {len(passes)} passes")
    
    # Combine all passes
    if all_passes:
        combined_passes = pd.concat(all_passes, ignore_index=True)
        print(f"\nTotal passes extracted: {len(combined_passes)}")
        return combined_passes
    else:
        print("No passes found!")
        return pd.DataFrame()


def save_raw_data(passes_df, output_dir='data/raw'):
    """
    Save raw pass data to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'statsbomb_passes_raw.csv')
    passes_df.to_csv(output_file, index=False)
    print(f"\nRaw data saved to: {output_file}")
    print(f"Shape: {passes_df.shape}")
    
    # Print column info
    print(f"\nAvailable columns ({len(passes_df.columns)}):")
    for i, col in enumerate(passes_df.columns, 1):
        print(f"  {i}. {col}")
    
    # Check for pass outcome/result column
    result_cols = [col for col in passes_df.columns if 'result' in col.lower() or 'outcome' in col.lower()]
    print(f"\nResult/Outcome columns: {result_cols}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'data_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("StatsBomb Pass Data Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total passes: {len(passes_df)}\n")
        f.write(f"Matches: {passes_df['match_id'].nunique()}\n")
        f.write(f"\nColumns ({len(passes_df.columns)}):\n")
        for col in passes_df.columns:
            f.write(f"  - {col}\n")
        
        # Check pass outcomes
        if result_cols:
            f.write(f"\nPass outcomes:\n")
            for col in result_cols:
                f.write(f"\n{col}:\n")
                f.write(str(passes_df[col].value_counts()) + "\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return output_file


if __name__ == "__main__":
    # Get sample match IDs
    match_ids = get_sample_match_ids()
    
    # Collect matches (limit to 20 for now)
    matches = collect_statsbomb_matches(match_ids, max_matches=20)
    
    if matches:
        # Extract pass events
        passes_df = extract_pass_events(matches)
        
        if not passes_df.empty:
            # Save raw data
            save_raw_data(passes_df)
            print("\nData collection complete!")
        else:
            print("\nNo pass data to save.")
    else:
        print("\nNo matches collected.")
