import streamlit as st
import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  


st.set_page_config(page_title='IPL Performance Analysis Portal', layout='wide')
st.title('IPL Performance Analysis Portal')
@st.cache_data
def load_data():
    path = "Dataset/ipl_bbb_2021_25.csv"
    df = pd.read_csv(path)
    return df

df = load_data()    
df = df.rename(columns={'innings': 'inning'})
df['is_wicket'] = df['out'].astype(int)
def categorize_phase(over):
              if over <= 6:
                  return 'Powerplay'
              elif 6 < over < 12:
                  return 'Middle 1'
              elif 11 < over < 17:
                  return 'Middle 2'
              else:
                  return 'Death'

@st.cache_data
def round_up_floats(df, decimal_places=2):
    # Select only float columns from the DataFrame
    float_cols = df.select_dtypes(include=['float64', 'float32'])  # Ensure to catch all float types
    
    # Round up the float columns and maintain the same shape
    rounded_floats = np.ceil(float_cols * (10 ** decimal_places)) / (10 ** decimal_places)
    
    # Assign the rounded values back to the original DataFrame
    df[float_cols.columns] = rounded_floats


import pandas as pd
import numpy as np
import math as mt

# -----------------------------
# cumulator: batter summary
# -----------------------------
def cumulator(temp_df):
    """
    Aggregates batter-level cumulative statistics from ball-by-ball DataFrame.

    Required columns in temp_df:
      ['match_id','ball_id','inning' or 'inn_id','batsman','bowler','batsman_runs',
       'player_dismissed','season','is_dot','is_one','is_two','is_three','is_four','is_six', ...]
    Returns:
      summary_df with per-batsman aggregates and derived metrics including BPB and BPD.
    """

    # Defensive copy
    df = temp_df.copy()

    # Standardize possible column names
    if 'inn_id' not in df.columns and 'inning' in df.columns:
        df = df.rename(columns={'inning': 'inn_id'})

    # Remove exact duplicates at ball-level (same match, same ball, same inning, same players)
    print(f"Before removing duplicates based on 'match_id' and 'ball': {df.shape}")
    df = df.drop_duplicates(subset=['match_id', 'ball_id', 'inn_id', 'batsman', 'bowler'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {df.shape}")

    # Ensure numeric types where needed
    df['batsman_runs'] = pd.to_numeric(df['batsman_runs'], errors='coerce').fillna(0).astype(int)
    # is_dot, is_one, is_two... might be booleans or 0/1
    for col in ['is_dot', 'is_one', 'is_two', 'is_three', 'is_four', 'is_six']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            # Derive them if not present
            if col == 'is_dot':
                df['is_dot'] = (df['batsman_runs'] == 0).astype(int)
            elif col == 'is_one':
                df['is_one'] = (df['batsman_runs'] == 1).astype(int)
            elif col == 'is_two':
                df['is_two'] = (df['batsman_runs'] == 2).astype(int)
            elif col == 'is_three':
                df['is_three'] = (df['batsman_runs'] == 3).astype(int)
            elif col == 'is_four':
                df['is_four'] = (df['batsman_runs'] == 4).astype(int)
            elif col == 'is_six':
                df['is_six'] = (df['batsman_runs'] == 6).astype(int)

    # Basic aggregates
    runs = df.groupby('batsman')['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
    balls = df.groupby('batsman')['ball_id'].count().reset_index().rename(columns={'ball_id': 'balls'})
    innings = df.groupby('batsman')['inn_id'].nunique().reset_index().rename(columns={'inn_id': 'innings'})
    matches = df.groupby('batsman')['match_id'].nunique().reset_index().rename(columns={'match_id': 'matches'})
    dismissals = df[df['player_dismissed'].notna() & (df['player_dismissed'] != "")].groupby('batsman')['player_dismissed'].count().reset_index().rename(columns={'player_dismissed': 'dismissals'})

    # Count boundaries and dots etc.
    fours = df.groupby('batsman')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = df.groupby('batsman')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
    dotballs = df.groupby('batsman')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dot_balls'})
    ones = df.groupby('batsman')['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
    twos = df.groupby('batsman')['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
    threes = df.groupby('batsman')['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})

    # Innings runs (per inn_id) for 100s/50s/30s and HS
    inn_runs = df.groupby(['batsman', 'inn_id'])['batsman_runs'].sum().reset_index()
    hundreds = inn_runs[inn_runs['batsman_runs'] >= 100].groupby('batsman').size().reset_index(name='hundreds')
    fifties = inn_runs[(inn_runs['batsman_runs'] >= 50) & (inn_runs['batsman_runs'] < 100)].groupby('batsman').size().reset_index(name='fifties')
    thirties = inn_runs[(inn_runs['batsman_runs'] >= 30) & (inn_runs['batsman_runs'] < 50)].groupby('batsman').size().reset_index(name='thirties')

    highest_scores = inn_runs.groupby('batsman')['batsman_runs'].max().reset_index().rename(columns={'batsman_runs': 'highest_score'})

    # Merge everything
    summary_df = runs.merge(balls, on='batsman', how='left') \
                     .merge(innings, on='batsman', how='left') \
                     .merge(matches, on='batsman', how='left') \
                     .merge(dismissals, on='batsman', how='left') \
                     .merge(hundreds, on='batsman', how='left') \
                     .merge(fifties, on='batsman', how='left') \
                     .merge(thirties, on='batsman', how='left') \
                     .merge(highest_scores, on='batsman', how='left') \
                     .merge(fours, on='batsman', how='left') \
                     .merge(sixes, on='batsman', how='left') \
                     .merge(dotballs, on='batsman', how='left') \
                     .merge(ones, on='batsman', how='left') \
                     .merge(twos, on='batsman', how='left') \
                     .merge(threes, on='batsman', how='left')

    # Fill NaNs
    summary_df[['dismissals', 'hundreds', 'fifties', 'thirties', 'highest_score', 'fours', 'sixes', 'dot_balls', 'ones', 'twos', 'threes']] = \
        summary_df[['dismissals', 'hundreds', 'fifties', 'thirties', 'highest_score', 'fours', 'sixes', 'dot_balls', 'ones', 'twos', 'threes']].fillna(0)

    # Derived metrics
    # Batting average: runs / dismissals; if dismissals == 0, average = runs / innings (safer fallback)
    def compute_avg(row):
        if row['dismissals'] > 0:
            return row['runs'] / row['dismissals']
        elif row['innings'] > 0:
            return row['runs'] / row['innings']
        else:
            return np.nan

    summary_df['AVG'] = summary_df.apply(compute_avg, axis=1)

    # Strike Rate: needs balls > 0
    summary_df['SR'] = summary_df.apply(lambda r: (r['runs'] / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)

    # Boundary% (percentage of runs coming from boundaries = (4s*4 + 6s*6)/runs)
    summary_df['boundary_runs'] = summary_df['fours'] * 4 + summary_df['sixes'] * 6
    summary_df['Boundary%'] = summary_df.apply(lambda r: (r['boundary_runs'] / r['runs'] * 100) if r['runs'] > 0 else 0.0, axis=1)

    # Dot% (percentage of dot balls)
    summary_df['Dot%'] = summary_df.apply(lambda r: (r['dot_balls'] / r['balls'] * 100) if r['balls'] > 0 else 0.0, axis=1)

    # BPB = Balls per Boundary = balls / (fours + sixes) ; handle zero boundaries
    summary_df['boundary_count'] = summary_df['fours'] + summary_df['sixes']
    summary_df['BPB'] = summary_df.apply(lambda r: (r['balls'] / r['boundary_count']) if r['boundary_count'] > 0 else np.nan, axis=1)

    # BPD = Balls per Dot = balls / dot_balls ; handle zero dot balls
    summary_df['BPD'] = summary_df.apply(lambda r: (r['balls'] / r['dot_balls']) if r['dot_balls'] > 0 else np.nan, axis=1)

    # NBDRY_SR and BP6 are domain specific fields you mentioned earlier; if you have formulas, compute here.
    # For now let's create placeholders (NaN) so downstream code can fill them if needed.
    summary_df['NBDRY_SR'] = np.nan
    summary_df['BP6'] = np.nan

    # Debut and final seasons
    if 'season' in df.columns:
        debut_year = df.groupby('batsman')['season'].min().reset_index().rename(columns={'season': 'debut_year'})
        final_year = df.groupby('batsman')['season'].max().reset_index().rename(columns={'season': 'final_year'})
        summary_df = summary_df.merge(debut_year, on='batsman', how='left').merge(final_year, on='batsman', how='left')
    else:
        summary_df['debut_year'] = np.nan
        summary_df['final_year'] = np.nan

    # Clean up columns and ordering
    cols_order = ['batsman', 'matches', 'innings', 'balls', 'runs', 'AVG', 'SR',
                  'fours', 'sixes', 'boundary_runs', 'Boundary%', 'dot_balls', 'Dot%',
                  'BPB', 'BPD', 'hundreds', 'fifties', 'thirties', 'highest_score',
                  'debut_year', 'final_year', 'ones', 'twos', 'threes']
    # Ensure all columns exist
    for c in cols_order:
        if c not in summary_df.columns:
            summary_df[c] = np.nan

    summary_df = summary_df[cols_order]

    # Final cleaning: replace inf with NaN
    summary_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return summary_df


# -----------------------------
# bowlerstat: bowler summary
# -----------------------------
def bowlerstat(df):
    """
    Aggregates bowler-level cumulative statistics and adds BPB (balls per boundary conceded).
    Expects columns including: match_id, ball_id, inn_id or inning, bowler, batsman_runs, is_dot, is_one, is_two,
    is_three, is_four, is_six, total_runs, out (boolean or 0/1)
    Returns: bowl_rec DataFrame with derived metrics including BPB.
    """

    d = df.copy()

    # Standardize inn_id naming
    if 'inn_id' not in d.columns and 'inning' in d.columns:
        d = d.rename(columns={'inning': 'inn_id'})

    print(f"Before removing duplicates based on 'match_id', 'ball_id', 'inn_id', 'batsman', 'bowler': {d.shape}")
    d = d.drop_duplicates(subset=['match_id', 'ball_id', 'inn_id', 'batsman', 'bowler'], keep='first')
    print(f"After removing duplicates: {d.shape}")

    # Ensure numeric conversions
    for col in ['is_dot', 'is_one', 'is_two', 'is_three', 'is_four', 'is_six']:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0).astype(int)
        else:
            # derive from batsman_runs if not present
            if col == 'is_dot':
                d['is_dot'] = (d['batsman_runs'] == 0).astype(int)
            elif col == 'is_one':
                d['is_one'] = (d['batsman_runs'] == 1).astype(int)
            elif col == 'is_two':
                d['is_two'] = (d['batsman_runs'] == 2).astype(int)
            elif col == 'is_three':
                d['is_three'] = (d['batsman_runs'] == 3).astype(int)
            elif col == 'is_four':
                d['is_four'] = (d['batsman_runs'] == 4).astype(int)
            elif col == 'is_six':
                d['is_six'] = (d['batsman_runs'] == 6).astype(int)

    # Create is_wicket from 'out' column if present else try player_dismissed
    if 'out' in d.columns:
        d['is_wicket'] = pd.to_numeric(d['out'], errors='coerce').fillna(0).astype(int)
    else:
        # fallback: if player_dismissed non-empty, mark wicket
        d['is_wicket'] = d['player_dismissed'].notna().astype(int)

    # Aggregates in single groupby
    agg_dict = {
        'runs': ('batsman_runs', 'sum'),
        'innings': ('inn_id', 'nunique'),
        'balls': ('ball_id', 'count'),
        'wkts': ('is_wicket', 'sum'),
        'dots': ('is_dot', 'sum'),
        'ones': ('is_one', 'sum'),
        'twos': ('is_two', 'sum'),
        'threes': ('is_three', 'sum'),
        'fours': ('is_four', 'sum'),
        'sixes': ('is_six', 'sum'),
    }

    # Build named aggregation
    named_aggs = {key: pd.NamedAgg(column=col, aggfunc=agg) for key, (col, agg) in agg_dict.items()}
    bowl_rec = d.groupby('bowler').agg(**named_aggs).reset_index()

    # Dismissals count per match for 10W (rare) and bbi/bbm logic
    dismissals_per_match = d.groupby(['bowler', 'match_id'])['is_wicket'].sum().reset_index(name='wkts_in_match')
    ten_wicket_counts = dismissals_per_match[dismissals_per_match['wkts_in_match'] >= 10].groupby('bowler')['match_id'].nunique().reset_index(name='10W')
    bowl_rec = bowl_rec.merge(ten_wicket_counts, on='bowler', how='left')
    bowl_rec['10W'] = bowl_rec['10W'].fillna(0).astype(int)

    # Best bowling in match (bbm)
    bbm = dismissals_per_match.groupby('bowler')['wkts_in_match'].max().reset_index().rename(columns={'wkts_in_match': 'bbm'})
    bowl_rec = bowl_rec.merge(bbm, on='bowler', how='left')
    bowl_rec['bbm'] = bowl_rec['bbm'].fillna(0).astype(int)

    # Five wicket hauls (per innings)
    dismissals_per_innings = d.groupby(['bowler', 'inn_id'])['is_wicket'].sum().reset_index(name='wkts_in_inn')
    five_wkt = dismissals_per_innings[dismissals_per_innings['wkts_in_inn'] >= 5].groupby('bowler')['inn_id'].nunique().reset_index(name='5W')
    bowl_rec = bowl_rec.merge(five_wkt, on='bowler', how='left')
    bowl_rec['5W'] = bowl_rec['5W'].fillna(0).astype(int)

    # Maiden overs: need over number. If ball_id is like over.ball (e.g., 12.3) convert to integer over
    # You earlier did df['over_num'] = df['ball_id'].astype(float).apply(lambda x: int(x))
    # But ball_id might be integer sequence; so attempt robust conversion:
    try:
        d['over_num'] = d['ball_id'].astype(float).apply(lambda x: int(x))
    except Exception:
        # Fallback: if you have 'over' column
        if 'over' in d.columns:
            d['over_num'] = d['over'].astype(int)
        else:
            # best-effort: floor division of ball_id by 6 (if ball_id is 0-indexed integer)
            d['over_num'] = (d['ball_id'] // 6).astype(int)

    maiden_overs = (d.groupby(['bowler', 'inn_id', 'over_num'])
                    .agg(balls_in_over=('ball_id', 'count'),
                         runs_in_over=('total_runs', 'sum')).reset_index())

    maiden_overs_count = (maiden_overs[(maiden_overs['balls_in_over'] == 6) & (maiden_overs['runs_in_over'] == 0)]
                          .groupby('bowler').size().reset_index(name='Mdns'))

    bowl_rec = bowl_rec.merge(maiden_overs_count, on='bowler', how='left')
    bowl_rec['Mdns'] = bowl_rec['Mdns'].fillna(0).astype(int)

    # Derived metrics: dot%, avg, sr, econ
    bowl_rec['dot%'] = (bowl_rec['dots'] / bowl_rec['balls']) * 100
    # replace zeros in wkts for avg/sr calculation with NaN to avoid divide-by-zero
    bowl_rec['avg'] = bowl_rec.apply(lambda r: (r['runs'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['sr'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['econ'] = bowl_rec.apply(lambda r: (r['runs'] * 6.0 / r['balls']) if r['balls'] > 0 else np.nan, axis=1)

    # Overs formatting: e.g., 12.3 style as number of complete overs + balls
    def format_overs(balls):
        if pd.isna(balls):
            return ""
        complete_overs = int(balls // 6)
        rem = int(balls % 6)
        if rem == 0:
            return f"{complete_overs}.0"
        else:
            return f"{complete_overs}.{rem}"

    bowl_rec['overs'] = bowl_rec['balls'].apply(format_overs)

    # BPB for bowlers: Balls per Boundary conceded = balls / (fours + sixes) ; handle zero
    bowl_rec['boundary_count'] = bowl_rec['fours'] + bowl_rec['sixes']
    bowl_rec['BPB'] = bowl_rec.apply(lambda r: (r['balls'] / r['boundary_count']) if r['boundary_count'] > 0 else np.nan, axis=1)

    # Keep only relevant columns and fill NaNs
    # You can reorder as needed
    # Replace infinite with NaN
    bowl_rec.replace([np.inf, -np.inf], np.nan, inplace=True)

    return bowl_rec

# -------------------------------------------
# Example usage (uncomment to test):
# df = pd.read_csv("Dataset/ipl_bbb_2021_25.csv")
# bat_summary = cumulator(df)
# bowl_summary = bowlerstat(df)
# print(bat_summary.head())
# print(bowl_summary.head())
pdf = load_data()
pdf['match_id']=pdf['p_match']

pdf['inn_id']=pdf['p_match']

pdf['batting_team']=pdf['team_bat']
pdf['bowling_team']=pdf['team_bowl']
pdf['batsman']=pdf['bat']
pdf['bowler']=pdf['bowl']
bpdf = pdf
idf = cumulator(pdf)
bidf = load_bowling_data()
# Define a mapping dictionary to consolidate bowling styles
bowling_style_mapping = {
    'OB': 'Off-break',
    'LFM': 'Left-arm medium fast',
    'RFM': 'Right-arm fast medium',
    'RF': 'Right-arm fast',
    'SLA': 'Slow left-arm orthodox',
    'OB/LBG': 'Off-break and leg-break googly',
    'RMF': 'Right-arm medium fast',
    'LF': 'Left-arm fast',
    'LBG': 'Leg-break googly',
    'RM': 'Right-arm medium',
    'RM/LB': 'Right-arm medium and leg-break',
    'LM': 'Left-arm medium',
    'RM/OB': 'Right-arm medium and off-break',
    'LWS': 'Left-arm wrist spin',
    'LB': 'Leg-break',
    'OB/LB': 'Off-break and leg-break',
    '-': 'Unknown',  # Treat '-' as Unknown
    'LMF': 'Left-arm medium fast',
    'LS': 'Leg-spin',
    'RS': 'Right-arm spin',
    'LFM/SLA': 'Left-arm medium fast and slow left-arm orthodox',
    'OB/SLA': 'Off-break and slow left-arm orthodox',
    'RMF/OB': 'Right-arm medium fast and off-break'
}

# Apply the mapping to the 'bowling_style' column in your DataFrame
pdf['bowling_style'] = pdf['bowling_style'].replace(bowling_style_mapping)

# st.switch_page("Career_Statistics.py"
sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis","Strength vs Weakness","Match by Match Analysis")
)

allowed_countries = ['India', 'England', 'Australia', 'Pakistan', 'Bangladesh', 
                                'West Indies', 'South Africa', 'New Zealand', 'Sri Lanka']

if sidebar_option == "Player Profile":
    st.header("Player Profile")

    # Player search input (selectbox)
    player_name = st.selectbox("Search for a player", idf['batsman'].unique())

    # Filter the data for the selected player
    temp_df = idf[idf['batsman'] == player_name].iloc[0]
    # Tabs for "Overview", "Career Statistics", and "Current Form"
    tab1, tab2 = st.tabs(["Career Statistics", "Current Form"])
    with tab1:
            st.header("Career Statistics")
    
            # Dropdown for Batting or Bowling selection
            option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))
    
            # Show Career Averages based on the dropdown
            st.subheader("Career Performance")
    
            # Display Career Averages based on selection
            if option == "Batting":
                player_stats = idf[idf['batsman'] == player_name].copy()

                # Drop the 'final_year' column from this player's data only
                player_stats = player_stats.drop(columns=['final_year'])

                # Convert column names to uppercase and replace underscores with spaces
                player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]

                # Apply rounding if necessary (assuming `round_up_floats` is a defined function)
                player_stats = round_up_floats(player_stats)

                # Display the player's statistics in a table format with bold headers
                st.markdown("### Batting Statistics")
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

                # Fill NaN values with 0 for specified columns
                player_stats[columns_to_convert] = player_stats[columns_to_convert].fillna(0)

                # Convert specified columns to integer type
                player_stats[columns_to_convert] = player_stats[columns_to_convert].astype(int)

                # Display the data as a styled table in Streamlit
                st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'")) 
                
            
                # Initializing an empty DataFrame for results and a counter
                result_df = pd.DataFrame()
                i = 0     
                for country in allowed_countries:
                    temp_df = pdf[pdf['batsman'] == player_name]  # Filter data for the selected batsman
                    
                    # Filter for the specific country
                    temp_df = temp_df[temp_df['bowling_team'] == country]
                
                    # Apply the cumulative function (bcum)
                    temp_df = cumulator(temp_df)
                
                    # If the DataFrame is empty after applying `bcum`, skip this iteration
                    if temp_df.empty:
                        continue
                
                    # Add the country column with the current country's value
                    temp_df['opponent'] = country.upper()
                
                    # Reorder columns to make 'country' the first column
                    cols = temp_df.columns.tolist()
                    new_order = ['opponent'] + [col for col in cols if col != 'opponent']
                    temp_df = temp_df[new_order]
                    
                
                    # Concatenate results into result_df
                    if i == 0:
                        result_df = temp_df
                        i += 1
                    else:
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                
                # Display the final result_df
                # result_df.rename(columns={'matches_x':'matches'})
                result_df = result_df.drop(columns=['batsman','debut_year','final_year'])
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                columns_to_convert = ['HUNDREDS', 'FIFTIES','THIRTIES', 'RUNS','HIGHEST SCORE']

                #    # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                    
                #    # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                result_df=round_up_floats(result_df)
                cols = result_df.columns.tolist()

                #    # Specify the desired order with 'year' first
                new_order = ['OPPONENT', 'MATCHES'] + [col for col in cols if col not in ['MATCHES', 'OPPONENT']]
                            
                # #    # Reindex the DataFrame with the new column order
                result_df =result_df[new_order]
    
                st.markdown("### Opponentwise Performance")
                st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))      
                
                tdf = pdf[pdf['batsman'] == player_name]                
                # Populate an array of unique seasons
                unique_seasons = tdf['season'].unique()
                
                # Optional: Convert to a sorted list (if needed)
                unique_seasons = sorted(set(unique_seasons))
                # print(unique_seasons)
                tdf=pd.DataFrame(tdf)
                tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
                tdf['total_runs'] = tdf['total_runs'].astype(int)
                # Run a for loop and pass temp_df to a cumulative function
                i=0
                for season in unique_seasons:
                    print(i)
                    temp_df = tdf[(tdf['season'] == season)]
                    print(temp_df.head())
                    temp_df = cumulator(temp_df)
                    if i==0:
                        result_df = temp_df  # Initialize with the first result_df
                        i=1+i
                    else:
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                    result_df = result_df.drop(columns=['batsman','debut_year'])
                    # Convert specific columns to integers
                    # Round off the remaining float columns to 2 decimal places
                    float_cols = result_df.select_dtypes(include=['float']).columns
                    result_df[float_cols] = result_df[float_cols].round(2)
                result_df=result_df.rename(columns={'final_year':'year'})
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                result_df = round_up_floats(result_df)
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

                # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                    
                # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                        
                # Display the results
                st.markdown(f"### **Yearwise Performnce**")
                cols = result_df.columns.tolist()

                # # Specify the desired order with 'year' first
                # new_order = ['YEAR'] + [col for col in cols if col != 'YEAR']
                new_order = ['YEAR', 'MATCHES'] + [col for col in cols if col not in ['MATCHES', 'YEAR']]
                        
                # # Reindex the DataFrame with the new column order
                result_df = result_df[new_order]
                st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
                
                tdf = pdf[pdf['batsman'] == player_name]
                temp_df=tdf[(tdf['inning']==1)]
                temp_df=cumulator(temp_df)
                temp_df['inning']=1
                cols = temp_df.columns.tolist()
                new_order = ['inning'] + [col for col in cols if col != 'inning']          
                # Reindex the DataFrame with the new column order
                temp_df =temp_df[new_order] 
                result_df = temp_df
                temp_df=tdf[(tdf['inning']==2)]
                temp_df=cumulator(temp_df)
                temp_df['inning']=2
                cols = temp_df.columns.tolist()
                new_order = ['inning'] + [col for col in cols if col != 'inning']          
                # Reindex the DataFrame with the new column order
                temp_df =temp_df[new_order] 
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                temp_df=tdf[(tdf['inning']==3)]
                temp_df=cumulator(temp_df)
                temp_df['inning']=3
                cols = temp_df.columns.tolist()
                new_order = ['inning'] + [col for col in cols if col != 'inning']          
                # Reindex the DataFrame with the new column order
                temp_df =temp_df[new_order] 
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                temp_df=tdf[(tdf['inning']==4)]
                temp_df=cumulator(temp_df)
                temp_df['inning']=4
                cols = temp_df.columns.tolist()
                new_order = ['inning'] + [col for col in cols if col != 'inning']          
                # Reindex the DataFrame with the new column order
                temp_df =temp_df[new_order] 
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                result_df = result_df.drop(columns=['batsman','debut_year','final_year'])
                # Convert specific columns to integers
                # Round off the remaining float columns to 2 decimal places
                float_cols = result_df.select_dtypes(include=['float']).columns
                result_df[float_cols] = result_df[float_cols].round(2)
                
                # result_df=result_df.rename(columns={'final_year':'year'})
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

                # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                    
                #    # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                        
                # Display the results
                result_df = result_df.drop(columns=['MATCHES'])
                st.markdown(f"### **Inningwise Performance**")
                st.table(result_df.reset_index(drop=True).style.set_table_attributes("style='font-weight: bold;'"))
                i=0
                for country in allowed_countries:
                    temp_df = pdf[pdf['batsman'] == player_name]
                    # print(temp_df.match_id.unique())
                    # print(temp_df.head(20))
                    temp_df = temp_df[(temp_df['country'] == country)]
                    temp_df = cumulator(temp_df)
                    temp_df['country']=country.upper()
                    cols = temp_df.columns.tolist()
                    new_order = ['country'] + [col for col in cols if col != 'country']
                    # Reindex the DataFrame with the new column order
                    temp_df =temp_df[new_order]
                    # print(temp_df)
                    # If temp_df is empty after applying cumulator, skip to the next iteration
                    if len(temp_df) == 0:
                        temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
                        continue
                    elif i==0:
                        result_df = temp_df
                        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                        i=i+1
                    else:
                        result_df = result_df.reset_index(drop=True)
                        temp_df = temp_df.reset_index(drop=True)
                        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                        result_df = pd.concat([result_df, temp_df],ignore_index=True)
                        
                
                result_df = result_df.drop(columns=['batsman','debut_year','final_year'])
                    # Round off the remaining float columns to 2 decimal places
                    # float_cols = result_df.select_dtypes(include=['float']).columns
                    # result_df[float_cols] = result_df[float_cols].round(2)
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                # result_df = round_up_floats(result_df)
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

                #    # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                    
                #    # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                cols = result_df.columns.tolist()
                if 'COUNTRY' in cols:
                    # new_order = ['COUNTRY'] + [col for col in cols if col != 'COUNTRY']
                    new_order = ['COUNTRY', 'MATCHES'] + [col for col in cols if col not in ['MATCHES', 'COUNTRY']]
                    result_df = result_df[new_order]
                # result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                    # result_df = result_df.drop(columns=['MATCHES'])
                # result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                st.markdown(f"### **In Host Country**")
                st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
            
            elif option == "Bowling":
                # Prepare the DataFrame for displaying player-specific bowling statistics
                temp_df = bidf
                    
                    # Filter for the selected player
                player_stats = temp_df[temp_df['bowler'] == player_name]  # Assuming bidf has bowler data
                if player_stats.empty:
                    st.markdown("No Bowling stats available")
                else:   
                        # Convert column names to uppercase and replace underscores with spaces
                        player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]
                            
                            # Function to round float values if necessary (assuming round_up_floats exists)
                        player_stats = round_up_floats(player_stats)
                        # columns_to_convert = ['RUNS','FIVE WICKET HAULS', 'MAIDEN OVERS']
            
                        #    # Fill NaN values with 0
                        # player_stats[columns_to_convert] =  player_stats[columns_to_convert].fillna(0)
                            
                        #    # Convert the specified columns to integer type
                        # player_stats[columns_to_convert] =  player_stats[columns_to_convert].astype(int)
                            
                            # Display the player's bowling statistics in a table format with bold headers
                        # player_stats = player_stats.drop(columns=['BOWLER'])
                        st.markdown("### Bowling Statistics")
                        st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'")) 
                        
                        # Initializing an empty DataFrame for results and a counter
                        result_df = pd.DataFrame()
                        i = 0
                        for country in allowed_countries:
                                # Iterate over allowed countries for batting analysis
                                temp_df = bpdf[bpdf['bowler'] == player_name]  # Filter data for the selected batsman
                                    
                                # Filter for the specific country
                                temp_df = temp_df[temp_df['batting_team'] == country]
                        
                                # Apply the cumulative function (bcum)
                                temp_df = bowlerstat(temp_df)
                            
                                # If the DataFrame is empty after applying `bcum`, skip this iteration
                                if temp_df.empty:
                                    continue
                            
                                # Add the country column with the current country's value
                                temp_df['opponent'] = country.upper()
                            
                                # Reorder columns to make 'country' the first column
                                cols = temp_df.columns.tolist()
                                new_order = ['opponent'] + [col for col in cols if col != 'opponent']
                                temp_df = temp_df[new_order]
                                
                            
                                # Concatenate results into result_df
                                if i == 0:
                                    result_df = temp_df
                                    i += 1
                                else:
                                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
                    # Display the final result_df
                        result_df = result_df.drop(columns=['bowler'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        # result_df=round_up_floats(result_df)
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
                     
                        tdf = bpdf[bpdf['bowler'] == player_name]  # Filter data for the specific bowler                        
                                    # Populate an array of unique seasons
                        unique_seasons = sorted(set(tdf['season'].unique()))  # Optional: Sorted list of unique seasons
                        
                                    # Initialize an empty DataFrame to store the final results
                        i = 0
                        for season in unique_seasons:
                                temp_df = tdf[tdf['season'] == season]  # Filter data for the current season
                                temp_df = bowlerstat(temp_df)  # Apply the cumulative function (specific to your logic)
                                temp_df['YEAR'] = season
                                    
                                if i == 0:
                                        result_df = temp_df  # Initialize the result_df with the first season's data
                                        i += 1
                                else:
                                        result_df = pd.concat([result_df, temp_df], ignore_index=True)  # Append subsequent data
                                        
                        result_df = result_df.drop(columns=['bowler'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        result_df=round_up_floats(result_df)
                        # result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            
                        # No need to convert columns to integer (for bowling-specific data)
            
                        # Display the results
                        st.markdown(f"### **Yearwise Bowling Performance**")
                        cols = result_df.columns.tolist()
            
                        # Specify the desired order with 'YEAR' first
                        new_order = ['YEAR'] + [col for col in cols if col != 'YEAR']
            
                        # Reindex the DataFrame with the new column order
                        result_df = result_df[new_order]
            
                        # Display the table with bold headers
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        
            

                        # Filter data for the specific bowler
                        tdf = bpdf[bpdf['bowler'] == player_name]

                                
                        
                        # Process for the first inning
                        temp_df = tdf[(tdf['inning'] == 1)]
                        temp_df = bowlerstat(temp_df)  # Apply the cumulative function specific to bowlers
                        temp_df['inning'] = 1  # Add the inning number
            
                        # Reorder columns to have 'inning' first
                        cols = temp_df.columns.tolist()
                        new_order = ['inning'] + [col for col in cols if col != 'inning']          
                        temp_df = temp_df[new_order] 
            
                        # Initialize result_df with the first inning's data
                        result_df = temp_df
            
                        # Process for the second inning
                        temp_df = tdf[(tdf['inning'] == 2)]
                        temp_df = bowlerstat(temp_df)  # Apply the cumulative function specific to bowlers
                        temp_df['inning'] = 2  # Add the inning number
            
                        # Reorder columns to have 'inning' first
                        cols = temp_df.columns.tolist()
                        new_order = ['inning'] + [col for col in cols if col != 'inning']          
                        temp_df = temp_df[new_order] 
            
                        # Concatenate the results for both innings
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                        temp_df = tdf[(tdf['inning'] == 3)]
                        temp_df = bowlerstat(temp_df)  # Apply the cumulative function specific to bowlers
                        temp_df['inning'] = 3  # Add the inning number
            
                        # Reorder columns to have 'inning' first
                        cols = temp_df.columns.tolist()
                        new_order = ['inning'] + [col for col in cols if col != 'inning']          
                        temp_df = temp_df[new_order] 
            
                        # Concatenate the results for both innings
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                        temp_df = tdf[(tdf['inning'] == 4)]
                        temp_df = bowlerstat(temp_df)  # Apply the cumulative function specific to bowlers
                        temp_df['inning'] = 4  # Add the inning number
            
                        # Reorder columns to have 'inning' first
                        cols = temp_df.columns.tolist()
                        new_order = ['inning'] + [col for col in cols if col != 'inning']          
                        temp_df = temp_df[new_order] 
            
                        # Concatenate the results for both innings
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
            
                        # Drop unnecessary columns
                        result_df = result_df.drop(columns=['bowler'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        result_df=round_up_floats(result_df)
            
                        # Display the results
                        st.markdown(f"### **Inningwise Bowling Performance**")
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))

            
            
                        # Creating a DataFrame to display venues and their corresponding countries            
                        i = 0
                        for country in allowed_countries:
                            temp_df = bpdf[bpdf['bowler'] == player_name] 
                            temp_df = temp_df[(temp_df['country'] == country)]
                            temp_df = bowlerstat(temp_df)
                            temp_df.insert(0, 'country', country.upper())
                
            
                            # If temp_df is empty after applying bcum, skip to the next iteration
                            if len(temp_df) == 0:
                                continue
                            elif i == 0:
                                result_df = temp_df
                                i += 1
                            else:
                                result_df = result_df.reset_index(drop=True)
                                temp_df = temp_df.reset_index(drop=True)
                                result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            
                                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            
                        if 'bowler' in result_df.columns:
                            result_df = result_df.drop(columns=['bowler'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        result_df=round_up_floats(result_df)
            
                        st.markdown(f"### **In Host Country**")
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
    
    with tab2:
            st.header("Current Form")


