"""
Collect StatsBomb open data using Kloppy for pass success prediction.
"""
import os
import json
from kloppy import statsbomb
from kloppy.domain import EventType
import pandas as pd


def collect_statsbomb_matches():
    """
    Collect StatsBomb open data matches with pass events.
    Focus on competitions with rich data.
    """
    print("Collecting StatsBomb open data...")
    
    # Target competitions (ID, Season ID, Name)
    competitions = [
        (43, 3, "FIFA World Cup 2018"),
        (11, 90, "La Liga 2020/2021"),
        (11, 42, "La Liga 2019/2020"),
        (2, 44, "Premier League 2003/2004"),
    ]
    
    all_matches = []
    
    for comp_id, season_id, comp_name in competitions:
        print(f"\nProcessing {comp_name}...")
        try:
            # Get all matches for this competition/season
            dataset = statsbomb.load_open_data(
                competition_id=comp_id,
                season_id=season_id,
                event_types=[EventType.PASS]
            )
            
            # Extract match metadata
            match_info = {
                'competition_id': comp_id,
                'season_id': season_id,
                'competition_name': comp_name,
                'match_id': dataset.metadata.match_id if hasattr(dataset.metadata, 'match_id') else 'unknown',
                'home_team': dataset.metadata.teams[0].name if dataset.metadata.teams else 'unknown',
                'away_team': dataset.metadata.teams[1].name if len(dataset.metadata.teams) > 1 else 'unknown',
                'dataset': dataset
            }
            
            all_matches.append(match_info)
            print(f"  Loaded match: {match_info['home_team']} vs {match_info['away_team']}")
            
        except Exception as e:
            print(f"  Error loading {comp_name}: {e}")
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
        passes['competition_id'] = match['competition_id']
        passes['season_id'] = match['season_id']
        passes['competition_name'] = match['competition_name']
        passes['match_id'] = match['match_id']
        passes['home_team'] = match['home_team']
        passes['away_team'] = match['away_team']
        
        all_passes.append(passes)
        print(f"  {match['home_team']} vs {match['away_team']}: {len(passes)} passes")
    
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
    print(f"\nColumns: {list(passes_df.columns)}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'data_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("StatsBomb Pass Data Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total passes: {len(passes_df)}\n")
        f.write(f"Competitions: {passes_df['competition_name'].nunique()}\n")
        f.write(f"Matches: {passes_df['match_id'].nunique()}\n")
        f.write(f"\nPass outcomes:\n")
        if 'result' in passes_df.columns:
            f.write(str(passes_df['result'].value_counts()) + "\n")
        f.write(f"\nColumns ({len(passes_df.columns)}):\n")
        for col in passes_df.columns:
            f.write(f"  - {col}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return output_file


if __name__ == "__main__":
    # Collect matches
    matches = collect_statsbomb_matches()
    
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
