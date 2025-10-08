import streamlit as st
import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  


st.set_page_config(page_title='IPL Performance Analysis Portal', layout='wide')
st.title('IPL Performance Analysis Portal')
 # Define pitch zones with boundaries
zones = {
    'SHORT': (8, 10),
    'SHORT_OF_A_GOOD_LENGTH': (6, 8),
    'GOOD_LENGTH': (4, 6),
    'FULL': (2, 4),
    'YORKER': (0, 2),
    'FULL_TOSS': (-2, 0)
}

line_positions = {
    'WIDE_OUTSIDE_OFFSTUMP': 0.25,
    'OUTSIDE_OFFSTUMP': 0.15,
    'ON_THE_STUMPS': 0,
    'DOWN_LEG': -0.15,
    'WIDE_DOWN_LEG': -0.25
}


length_positions = {
    'SHORT': 9,
    'SHORT_OF_A_GOOD_LENGTH': 7,
    'GOOD_LENGTH': 5,
    'FULL': 3,
    'YORKER': 1,
    'FULL_TOSS': -1
} 
@st.cache_data
def categorize_phase(over):
              if over <= 6:
                  return 'Powerplay'
              elif 6 < over < 16:
                  return 'Middle'
              else:
                  return 'Death'
@st.cache_data
def get_current_form(bpdf, player_name):
    # Filter for matches where the player batted or bowled
    bpdf['is_wicket'] = bpdf['out'].astype(int) 
    player_matches = bpdf[(bpdf['batsman'] == player_name) | (bpdf['bowler'] == player_name)]
    # player_matches['is_wicket'] = player_matches['out'].astype(int) 
    # player_matches['start_date'] = pd.to_datetime(player_matches['start_date'], format='%Y/%m/%d')
    player_matches = player_matches.sort_values(by='start_date', ascending=False)
    # bpdf['start_date'] = pd.to_datetime(bpdf['start_date'], format='%m/%d/%Y')
    
    # Get the last 10 unique match IDs
    last_10_matches = player_matches['start_date'].drop_duplicates().sort_values(ascending=False).head(10)

    # Prepare the result DataFrame
    results = []

    for date in last_10_matches:
        # Get batting stats for this match
        bat_match_data = bpdf[(bpdf['start_date'] == date) & (bpdf['batsman'] == player_name)]
        match_id = None
        venue = None
        opp = None
        
        if not bat_match_data.empty:
            runs = bat_match_data['batsman_runs'].sum() 
            balls_faced = bat_match_data['ball'].count()  # Sum balls faced
            SR = (runs / balls_faced) * 100 if balls_faced > 0 else 0.0
            venue = bat_match_data['venue'].iloc[0]
            match_id = bat_match_data['match_id'].iloc[0]
            date = bat_match_data['start_date'].iloc[0]
            opp = bat_match_data['bowling_team'].iloc[0]
        else:
            runs = 0
            balls_faced = 0
            SR = 0.0
        
        # Get bowling stats for this match
        bowl_match_data = bpdf[(bpdf['start_date'] == date) & (bpdf['bowler'] == player_name)]
        
        if not bowl_match_data.empty:
            balls_bowled = bowl_match_data['ball'].count()  # Sum balls bowled
            runs_given = bowl_match_data['total_runs'].sum()  # Sum runs given
            wickets = bowl_match_data['is_wicket'].sum()  # Sum wickets taken
            econ = (runs_given / (balls_bowled / 6)) if balls_bowled > 0 else 0.0  # Calculate Econ
            venue = bowl_match_data['venue'].iloc[0]
            match_id = bowl_match_data['match_id'].iloc[0]
            date = bowl_match_data['start_date'].iloc[0]
            opp = bowl_match_data['batting_team'].iloc[0]
            bowl_avg = (runs_given/wickets)
            bowl_sr = (balls_bowled / wickets)
        else:
            balls_bowled = 0
            runs_given = 0
            wickets = 0
            econ = 0.0
            bowl_avg = float('inf')
            bowl_sr = float('inf')
            
        results.append({
            "Date" : date,
            "Match ID": match_id,
            "Runs": runs,
            "Balls Faced": balls_faced,
            "Batting SR": SR,
            "Balls Bowled": balls_bowled,
            "Runs Given": runs_given,
            "Wickets": wickets,
            "Econ": econ,
            "Venue": venue,
            "Opponent" : opp,
            "Bowling SR": bowl_sr,
            "Bowling Average" : bowl_avg
        })
    
    return pd.DataFrame(results)

@st.cache_data
def round_up_floats(df, decimal_places=2):
    # Select only float columns from the DataFrame
    float_cols = df.select_dtypes(include=['float64', 'float32'])  # Ensure to catch all float types
    
    # Round up the float columns and maintain the same shape
    rounded_floats = np.ceil(float_cols * (10 ** decimal_places)) / (10 ** decimal_places)
    
    # Assign the rounded values back to the original DataFrame
    df[float_cols.columns] = rounded_floats
    
    return df
@st.cache_data
def cumulator(temp_df):
    # First, remove duplicates based on match_id and ball within the same match
    print(f"Before removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")
    temp_df = temp_df.drop_duplicates(subset=['match_id', 'ball_id', 'inning', 'batsman', 'bowler'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")
    # Ensure 'total_runs' exists

    # Calculate runs, balls faced, innings, dismissals, etc.
    runs = temp_df.groupby(['batsman'])['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
    balls = temp_df.groupby(['batsman'])['ball'].count().reset_index()
    inn = temp_df.groupby(['batsman'])['inn_id'].nunique().reset_index().rename(columns={'inn_id': 'innings'})
    matches = temp_df.groupby(['batsman'])['match_id'].nunique().reset_index().rename(columns={'match_id': 'matches'})
    dis = temp_df.groupby(['batsman'])['player_dismissed'].count().reset_index().rename(columns={'player_dismissed': 'dismissals'})

    # Count 100s, 50s, and 30s
    match_runs = temp_df.groupby(['batsman', 'match_id'])['batsman_runs'].sum().reset_index()
    inn_runs = temp_df.groupby(['batsman', 'inn_id'])['batsman_runs'].sum().reset_index()
    hundreds = inn_runs[inn_runs['batsman_runs'] >= 100].groupby('batsman').size().reset_index(name='hundreds')
    fifties = inn_runs[(inn_runs['batsman_runs'] >= 50) & (inn_runs['batsman_runs'] < 100)].groupby('batsman').size().reset_index(name='fifties')
    thirties = inn_runs[(inn_runs['batsman_runs'] >= 30) & (inn_runs['batsman_runs'] < 50)].groupby('batsman').size().reset_index(name='thirties')

    # Calculate the highest score for each batsman
    highest_scores = inn_runs.groupby('batsman')['batsman_runs'].max().reset_index().rename(columns={'batsman_runs': 'highest_score'})

    # Merge all the calculated metrics into a single DataFrame
    summary_df = runs.merge(balls, on='batsman', how='left')
    summary_df = summary_df.merge(inn, on='batsman', how='left')
    summary_df = summary_df.merge(matches, on='batsman', how='left')
    summary_df = summary_df.merge(dis, on='batsman', how='left')
    summary_df = summary_df.merge(hundreds, on='batsman', how='left')
    summary_df = summary_df.merge(fifties, on='batsman', how='left')
    summary_df = summary_df.merge(thirties, on='batsman', how='left')
    summary_df = summary_df.merge(highest_scores, on='batsman', how='left')

    # Calculating additional columns
    def avg(runs, dis, inn):
        return runs / inn if dis == 0 else runs / dis

    summary_df['AVG'] = summary_df.apply(lambda x: avg(x['runs'], x['dismissals'], x['innings']), axis=1)

    debut_year = temp_df.groupby('batsman')['season'].min().reset_index()
    final_year = temp_df.groupby('batsman')['season'].max().reset_index()
    debut_year.rename(columns={'season': 'debut_year'}, inplace=True)
    final_year.rename(columns={'season': 'final_year'}, inplace=True)
    summary_df = summary_df.merge(debut_year, on='batsman').merge(final_year, on='batsman')

    # Merging matches data
    summary_df = summary_df.merge(matches, on='batsman', how='left')
    summary_df = summary_df[summary_df['batsman'] != 0]
    summary_df['matches'] = summary_df['matches_x']
    # Drop both matches_x and matches_y
    summary_df = summary_df.drop(['matches_x', 'matches_y'], axis=1)
    return summary_df
@st.cache_data
def bowlerstat(df):
    # First, remove duplicates based on match_id, ball_id, innings, batsman, and bowler
    print(f"Before removing duplicates based on 'match_id', 'ball_id', 'inning', 'batsman', 'bowler': {df.shape}")
    df = df.drop_duplicates(subset=['match_id', 'ball_id', 'inning', 'batsman', 'bowler'], keep='first')
    print(f"After removing duplicates: {df.shape}")

    # Create an 'is_wicket' column where True indicates a wicket
    df['is_wicket'] = df['out'].astype(int)  # Convert True/False to 1/0

    # Aggregate metrics in a single groupby
    aggregates = {
        'runs': ('batsman_runs', 'sum'),
        'innings': ('inn_id', 'nunique'),
        'balls': ('ball', 'count'),
        'wkts': ('is_wicket', 'sum'),  # Use the new is_wicket column for wickets
        'dots': ('is_dot', 'sum'),
        'ones': ('is_one', 'sum'),
        'twos': ('is_two', 'sum'),
        'threes': ('is_three', 'sum'),
        'fours': ('is_four', 'sum'),
        'sixes': ('is_six', 'sum'),
    }

    # Perform groupby and aggregate all metrics at once
    bowl_rec = df.groupby(['bowler']).agg(**{key: (col, agg_func) for key, (col, agg_func) in aggregates.items()}).reset_index()

    # Calculate dismissals count
    dismissals_count = df.groupby(['bowler', 'match_id'])['is_wicket'].sum()

    # Reset the index to convert the Series to a DataFrame
    dismissals_count_df = dismissals_count.reset_index()

    # Calculate ten-wicket hauls
    ten_wicket_counts = dismissals_count_df[dismissals_count_df['is_wicket'] >= 10].groupby('bowler')['match_id'].nunique()
    ten_wicket_counts = ten_wicket_counts.rename('10W')
    bowl_rec = bowl_rec.merge(ten_wicket_counts, on='bowler', how='left')

    # Fill NaN values in the '10W' column with 0
    bowl_rec['10W'] = bowl_rec['10W'].fillna(0).astype(int)

    # Calculate best bowling in match (bbm)
    bbm_df = dismissals_count_df.groupby('bowler')['is_wicket'].max().reset_index()
    bbm_df = bbm_df.rename(columns={'is_wicket': 'bbm'})
    bowl_rec = bowl_rec.merge(bbm_df, on='bowler', how='left')
    bowl_rec['bbm'] = bowl_rec['bbm'].fillna(0).astype(int)

    # Calculate five-wicket hauls
    dismissals_count_i = df.groupby(['bowler', 'inn_id'])['is_wicket'].sum()
    dismissals_count_dfi = dismissals_count_i.reset_index()
    five_wicket_counts = dismissals_count_dfi[dismissals_count_dfi['is_wicket'] >= 5].groupby('bowler')['inn_id'].nunique()
    five_wicket_counts = five_wicket_counts.rename('5W')
    bowl_rec = bowl_rec.merge(five_wicket_counts, on='bowler', how='left')
    bowl_rec['5W'] = bowl_rec['5W'].fillna(0).astype(int)

    # Calculate best bowling in innings (bbi)
    bbi_df = dismissals_count_dfi.groupby('bowler')['is_wicket'].max().reset_index()
    bbi_df = bbi_df.rename(columns={'is_wicket': 'bbi'})
    bowl_rec = bowl_rec.merge(bbi_df, on='bowler', how='left')
    bowl_rec['bbi'] = bowl_rec['bbi'].fillna(0).astype(int)
    df['over_num'] = df['ball_id'].astype(float).apply(lambda x: int(x))

    # Group by bowler, innings, and over number to identify complete overs with no runs
    maiden_overs = (df.groupby(['bowler', 'inn_id', 'inning', 'over_num'])
                    .agg({
                        'ball_id': 'count',  # number of balls in the over
                        'total_runs': 'sum'  # total runs in the over
                    })
                    .reset_index())

    # Filter for complete overs (6 balls) with zero runs
    maiden_overs_count = (maiden_overs[
        (maiden_overs['ball_id'] == 6) &  # complete over
        (maiden_overs['total_runs'] == 0)  # no runs conceded
    ]
    .groupby('bowler')
    .size()
    .reset_index(name='maiden_overs'))

    # Now merge with bowl_rec
    bowl_rec['Mdns'] = 0  # initialize with zeros
    bowl_rec = bowl_rec.merge(maiden_overs_count,
                            on='bowler',
                            how='left')
    bowl_rec['Mdns'] = bowl_rec['maiden_overs'].fillna(0).astype(int)
    bowl_rec = bowl_rec.drop('maiden_overs', axis=1)

    # Show the results
    print("Sample of bowl_rec with Mdns column:\n", bowl_rec[['bowler', 'Mdns']].head(10))
    print("\nTop 10 bowlers by maiden overs:")
    print(bowl_rec.nlargest(10, 'Mdns')[['bowler', 'Mdns']])

    # Optional: To verify the calculation
    print("\nSample of maiden overs calculation:")
    sample_over = df[df['total_runs'] == 0].head(10)
    print(sample_over[['bowler', 'ball_id', 'over_num', 'total_runs']])
    bowl_rec.head()

    # Calculate derived metrics
    bowl_rec['dot%'] = (bowl_rec['dots'] / bowl_rec['balls']) * 100
    bowl_rec['avg'] = bowl_rec['runs'] / bowl_rec['wkts'].replace(0, np.nan)
    bowl_rec['sr'] = bowl_rec['balls'] / bowl_rec['wkts'].replace(0, np.nan)
    bowl_rec['econ'] = (bowl_rec['runs'] * 6 / bowl_rec['balls'].replace(0, np.nan))
    # bowl_rec=bowl_rec.drop(columns=['Unnamed: 0'])
    bowl_rec['overs'] = bowl_rec['balls'].apply(lambda x: f"{mt.floor(x / 6) + round(0.1 * (x % 6), 1):.1f}".rstrip('0').rstrip('.'))


    return bowl_rec

@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://media.githubusercontent.com/media/Omkarwalunj45/Test_cricket_portal/refs/heads/main/tests_final.csv", 
        low_memory=False
    )
    
    df = df.rename(columns={'innings': 'inning'})
    df['is_wicket'] = df['out'].astype(int)
    
    return df
@st.cache_data
def load_bowling_data():
    try:
        bidf = pd.read_csv("Datasets/lifesaver_bowl_tests.csv", low_memory=False)
        return (bidf
                .drop(columns=['Unnamed: 0', 'overs'], errors='ignore')
                .assign(overs=lambda x: x['balls'].apply(
                    lambda b: f"{mt.floor(b / 6) + round(0.1 * (b % 6), 1):.1f}".rstrip('0').rstrip('.')
                ))
        )
    except Exception as e:
        st.error(f"Error loading bowling data: {e}")
        return pd.DataFrame()

# Load data
pdf = load_data()

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
            current_form_df = get_current_form(bpdf, player_name)
            
            if not current_form_df.empty:
                current_form_df.columns = [col.upper() for col in current_form_df.columns]
                
                # Rearranging columns
                cols = current_form_df.columns.tolist()
                new_order = ['MATCH ID', 'DATE'] + [col for col in cols if col not in ['MATCH ID', 'DATE']]
                current_form_df = current_form_df[new_order]
                current_form_df = current_form_df.loc[:, ~current_form_df.columns.duplicated()]
                
                # Formatting the date
                # current_form_df['DATE'] = pd.to_datetime(current_form_df['DATE'], format='%Y/%m/%d')
                current_form_df = current_form_df.sort_values(by='DATE', ascending=False)
                current_form_df = current_form_df.reset_index(drop=True)
                # current_form_df['DATE'] = current_form_df['DATE'].dt.strftime('%m/%d/%Y')
                # Assuming `current_form_df` is your DataFrame
                cols = current_form_df.columns.tolist()  # Get the current column names as a list

                # Define the new order by putting specific columns at the beginning
                
                # Reorder the DataFrame
                current_form_df = current_form_df[new_order]

                # Displaying the table with clickable MATCH ID
                current_form_df.index = current_form_df.index + 1
                st.markdown(current_form_df.to_html(escape=False), unsafe_allow_html=True)
                # st.table(current_form_df.style.set_table_attributes("style='font-weight: bold;'"))
                
            else:
                st.write("No recent matches found for this player.")
    
elif sidebar_option == "Matchup Analysis":
    
    st.header("Matchup Analysis")
    
    # Filter unique batters and bowlers from the DataFrame
    unique_batters = pdf['batsman'].unique()  # Adjust the column name as per your PDF data structure
    unique_bowlers = pdf['bowler'].unique()    # Adjust the column name as per your PDF data structure
    unique_batters = unique_batters[unique_batters != '0']  # Filter out '0'
    unique_bowlers = unique_bowlers[unique_bowlers != '0']  # Filter out '0'

    # Search box for Batters
    batter_name = st.selectbox("Select a Batter", unique_batters)

    # Search box for Bowlers
    bowler_name = st.selectbox("Select a Bowler", unique_bowlers)

    # Dropdown for grouping options
    grouping_option = st.selectbox("Group By", ["Year", "Match", "Venue", "Inning"])
    matchup_df = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]

    # Step 3: Create a download option for the DataFrame
    if not matchup_df.empty:
        # Convert the DataFrame to CSV format
        csv = matchup_df.to_csv(index=False)  # Generate CSV string
        
        # Step 4: Create the download button
        st.download_button(
            label="Download Matchup Data as CSV",
            data=csv,  # Pass the CSV string directly
            file_name=f"{batter_name}_vs_{bowler_name}_matchup.csv",
            mime="text/csv"  # Specify the MIME type for CSV
        )
        if grouping_option == "Year":
            tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]
            # Populate an array of unique seasons
            unique_seasons = tdf['season'].unique()
            
            # Optional: Convert to a sorted list (if needed)
            unique_seasons = sorted(set(unique_seasons))
    
            # Ensure tdf is a DataFrame
            tdf = pd.DataFrame(tdf)
            tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
            tdf['total_runs'] = tdf['total_runs'].astype(int)
    
            # Initialize an empty result DataFrame
            result_df = pd.DataFrame()
            i=0
            # Run a for loop and pass temp_df to a cumulative function
            for season in unique_seasons:
                temp_df = tdf[tdf['season'] == season]
                temp_df = cumulator(temp_df)
    
                if i==0:
                        result_df = temp_df  # Initialize with the first result_df
                        i=1+i
                else:
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
            # Drop unnecessary columns related to performance metrics
            columns_to_drop = ['batsman', 'bowler', 'debut_year','fifties', 'hundreds', 'thirties', 'highest_score','matches']
            result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
    
            # Convert specific columns to integers and fill NaN values
            columns_to_convert = ['runs','dismissals']
            for col in columns_to_convert:
                result_df[col] = result_df[col].fillna(0).astype(int)
    
            result_df = result_df.rename(columns={'final_year': 'year'})
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
    
            # Display the results
            st.markdown("### **Yearwise Performance**")
            result_df['BATSMAN'] = batter_name.upper()
            result_df['BOWLER'] = bowler_name.upper()
            cols = result_df.columns.tolist()
            
    
            # Specify the desired order with 'year' first
            new_order = ['YEAR','BATSMAN', 'BOWLER'] + [col for col in cols if col not in ['YEAR','BATSMAN', 'BOWLER']]
                      
            # Reindex the DataFrame with the new column order
            result_df = result_df[new_order]
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        elif grouping_option == "Match":
            tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]
    
            # Populate an array of unique match IDs
            unique_matches = sorted(set(tdf['match_id'].unique()))
    
            # Ensure tdf is a DataFrame
            tdf = pd.DataFrame(tdf)
            tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
            tdf['total_runs'] = tdf['total_runs'].astype(int)
    
            # Initialize an empty result DataFrame
            result_df = pd.DataFrame()
            i = 0
    
            # Run a for loop and pass temp_df to a cumulative function
            for match_id in unique_matches:
                temp_df = tdf[tdf['match_id'] == match_id]
                current_match_id = match_id
                temp_df = cumulator(temp_df)
                temp_df.insert(0, 'MATCH_ID', current_match_id)
    
                if i == 0:
                    result_df = temp_df  # Initialize with the first result_df
                    i = 1 + i
                else:
                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
            columns_to_drop = ['debut_year',
                               'fifties', 'hundreds', 'thirties', 'highest_score', 'season','matches']
            result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
    
            # Convert specific columns to integers and fill NaN values
            columns_to_convert = ['runs', 'dismissals']
            for col in columns_to_convert:
                result_df[col] = result_df[col].fillna(0).astype(int)
    
            # Rename columns for better presentation
            result_df = result_df.rename(columns={'match_id': 'MATCH ID'})
            
            
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            # result_df['FINAL YEAR']=result_df['FINAL YEAR'].apply(standardize_season)
            
            result_df = result_df.rename(columns={'FINAL YEAR': 'YEAR'})  
            result_df['BATSMAN'] = batter_name.upper()
            result_df['BOWLER'] = bowler_name.upper()
    
            # Display the results
            st.markdown("### **Matchwise Performance**")
            cols = result_df.columns.tolist()
    
            # Reindex the DataFrame with the new column order
            result_df=result_df.sort_values('YEAR',ascending=True)
            # result_df=result_df[['MATCH ID'] + ['YEAR'] + [col for col in result_df.columns if col not in ['MATCH ID','YEAR']]]
            new_order = ['MATCH ID','YEAR','BATSMAN','BOWLER'] + [col for col in cols if col not in ['MATCH ID','YEAR','BATSMAN','BOWLER']]
            result_df = result_df[new_order]
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
            
            
                     
        elif grouping_option == "Venue":
            # Filter the DataFrame for the selected batsman and bowler
            tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]
        
            # Ensure tdf is a DataFrame and populate unique venue values
            tdf = pd.DataFrame(tdf)
            tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
            tdf['total_runs'] = tdf['total_runs'].astype(int)
        
            # Initialize an empty result DataFrame
            result_df = pd.DataFrame()
            i = 0
        
            # Populate an array of unique venues
            unique_venues = tdf['venue'].unique()
            
            for venue in unique_venues:
                # Filter temp_df for the current venue
                temp_df = tdf[tdf['venue'] == venue]
        
                # Store the current venue in a variable
                current_venue = venue
        
                # Call the cumulator function
                temp_df = cumulator(temp_df)
        
                # Insert the current venue as the first column in temp_df
                temp_df.insert(0, 'VENUE', current_venue)
        
                # Concatenate results
                if i == 0:
                    result_df = temp_df  # Initialize with the first result_df
                    i += 1
                else:
                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
            # Drop unnecessary columns related to performance metrics
            columns_to_drop = ['batsman', 'bowler', 'debut_year','final_year', 'fifties', 'hundreds', 'thirties', 'highest_score', 'matches']
            result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
        
            # Convert specific columns to integers and fill NaN values
            columns_to_convert = ['runs', 'dismissals']
            for col in columns_to_convert:
                result_df[col] = result_df[col].fillna(0).astype(int)
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            
            # result_df = result_df.rename(columns={'FINAL YEAR': 'YEAR'})   
        
            # Display the results
            st.markdown("### **Venuewise Performance**")
            result_df['BATSMAN'] = batter_name.upper()
            result_df['BOWLER'] = bowler_name.upper()
            cols = result_df.columns.tolist()
        
            # Specify the desired order with 'venue' first
            # new_order = ['VENUE'] + [col for col in cols if col != 'VENUE']
            cols = result_df.columns.tolist()
                          
            # Reindex the DataFrame with the new column order
            # result_df = result_df[new_order
            # result_df=result_df.sort_values('YEAR',ascending=True)
            # result_df=result_df[['VENUE'] + ['YEAR'] + [col for col in result_df.columns if col not in ['VENUE','YEAR']]]
            new_order = ['VENUE','BATSMAN','BOWLER'] + [col for col in cols if col not in ['VENUE','BATSMAN','BOWLER']]
            result_df = result_df[new_order]
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
            
        else:
            # Assuming pdf is your main DataFrame
            # Filter for innings 1 and 2 and prepare to accumulate results
            innings = [1,2,3,4]
            result_df = pd.DataFrame()  # Initialize an empty DataFrame for results
            
            for inning in innings:
                # Filter for the specific inning
                tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name) & (pdf['inning'] == inning)]
                
                # Check if there's any data for the current inning
                if not tdf.empty:
                    # Call the cumulator function
                    temp_df = cumulator(tdf)
            
                    # Add the inning as the first column in temp_df
                    temp_df.insert(0, 'INNING', inning)
            
                    # Concatenate to the main result DataFrame
                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
            
            # After processing both innings, drop unnecessary columns if needed
            columns_to_drop = ['batsman', 'bowler', 'debut_year','final_year','fifties', 'hundreds', 'thirties', 'highest_score', 'matches','last_year']
            result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
            result_df['BATSMAN'] = batter_name.upper()
            result_df['BOWLER'] = bowler_name.upper()
            
            # Convert specific columns to integers and fill NaN values
            columns_to_convert = ['runs', 'dismissals']
            for col in columns_to_convert:
                result_df[col] = result_df[col].fillna(0).astype(int)
            
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            
            # Display the results
            st.markdown("### **Innings Performance**")
            cols = result_df.columns.tolist()
            # result_df=result_df[['INNING'] + [col for col in result_df.columns if col not in ['INNING']]]
            new_order = ['INNING','BATSMAN','BOWLER'] + [col for col in cols if col not in ['INNING','BATSMAN','BOWLER']]
            result_df = result_df[new_order]
            st.table(result_df.style.set_table_attributes("style='fsont-weight: bold;'"))
    else:
         st.warning("No data available for the selected matchup.")
    
# elif sidebar_option == "Match by Match Analysis":
#     st.sidebar.title("Match by Match Analysis")
#     match_id = st.sidebar.selectbox("Select Match ID", options=pdf["match_id"].unique())
#     match_data = pdf[pdf["match_id"] == match_id].iloc[0]
#     batting_team = match_data["batting_team"]
#     bowling_team = match_data["bowling_team"]
#     venue = match_data["venue"]
#     start_date = match_data["start_date"]

#     st.write(f"**{batting_team}** vs **{bowling_team}**")
#     st.write(f"**Venue:** {venue}")
#     st.write(f"**Start Date:** {start_date}")
#     temp_df = pdf[pdf["match_id"] == match_id]

#     # Main section - Career Stat Type selection
#     option = st.selectbox("Select Analysis Dimension", ("Batsman Analysis", "Bowler Analysis"))

#     # Batsman Analysis
#     if option == "Batsman Analysis":
#         # Step 1: Select a batsman
#         batsman_selected = st.selectbox("Select Batsman", options=temp_df["batsman"].unique())
        
#         # Filter the data for the selected batsman
#         filtered_df = temp_df[temp_df["batsman"] == batsman_selected]
        
#         # Step 2: Select a bowler with 'All' option included
#         bowler_options = ["All"] + filtered_df["bowler"].unique().tolist()
#         bowler_selected = st.selectbox("Select Bowler", options=bowler_options)
#         # Further filter based on the bowler selection
#         if bowler_selected == "All":
#             final_df = filtered_df  # Only filter by batsman
#         else:
#             final_df = filtered_df[filtered_df["bowler"] == bowler_selected]  # Filter by both batsman and bowler
#         total_runs = filtered_df["batsman_runs"].sum()
#         total_balls = filtered_df["ballfaced"].sum()  # Assuming this column counts each ball faced
#         total_dismissals = filtered_df["is_wicket"].sum()  # Assuming this indicates dismissals for the batsman
#         strike_rate = (total_runs / total_balls) * 100 
#         avg_runs = total_runs / total_dismissals 

#         # Count for each scoring shot type based on provided columns
#         total_zeros = filtered_df["is_dot"].sum()
#         total_ones = filtered_df["is_one"].sum()
#         total_twos = filtered_df["is_two"].sum()
#         total_threes = filtered_df["is_three"].sum()
#         total_fours = filtered_df["is_four"].sum()
#         total_sixes = filtered_df["is_six"].sum()

#         # Total balls to calculate percentages
#         total_balls_for_percentage = total_zeros + total_ones + total_twos + total_threes + total_fours + total_sixes
#         percentages = {
#             "0s": (total_zeros / total_balls_for_percentage * 100) if total_balls_for_percentage > 0 else 0,
#             "1s": (total_ones / total_balls_for_percentage * 100) if total_balls_for_percentage > 0 else 0,
#             "2s": (total_twos / total_balls_for_percentage * 100) if total_balls_for_percentage > 0 else 0,
#             "3s": (total_threes / total_balls_for_percentage * 100) if total_balls_for_percentage > 0 else 0,
#             "4s": (total_fours / total_balls_for_percentage * 100) if total_balls_for_percentage > 0 else 0,
#             "6s": (total_sixes / total_balls_for_percentage * 100) if total_balls_for_percentage > 0 else 0,
#         }

#         # Display stats summary
#         st.write(f"**Analysis for Batsman: {batsman_selected}**")
#         if bowler_selected == "All":
#             st.write("**Against All Bowlers**")
#         else:
#             st.write(f"**Against Bowler: {bowler_selected}**")

#         col1, col2 = st.columns(2)
#         # Step 1: Select a batsman        
#         # Filter the data for the selected batsman            
#         # Calculate statistics
#         total_runs = final_df["batsman_runs"].sum()
#         total_balls = final_df["ballfaced"].sum()
#         total_dismissals = final_df["is_wicket"].sum()
#         strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
#         avg_runs = total_runs / total_dismissals if total_dismissals > 0 else total_runs

#         # Count for each scoring shot type
#         total_zeros = final_df["is_dot"].sum()
#         total_ones = final_df["is_one"].sum()
#         total_twos = final_df["is_two"].sum()
#         total_threes = final_df["is_three"].sum()
#         total_fours = final_df["is_four"].sum()
#         total_sixes = final_df["is_six"].sum()

#         # Calculate percentages
#         total_balls_for_percentage = total_zeros + total_ones + total_twos + total_threes + total_fours + total_sixes
        
#         def calc_percentage(value, total):
#             return f"{(value / total * 100):.1f}%" if total > 0 else "0%"
#         percent_zeros = (total_zeros / total_balls) * 100 if total_balls > 0 else 0
#         percent_ones = (total_ones / total_balls) * 100 if total_balls > 0 else 0
#         percent_twos = (total_twos / total_balls) * 100 if total_balls > 0 else 0
#         percent_threes = (total_threes / total_balls) * 100 if total_balls > 0 else 0
#         percent_fours = (total_fours / total_balls) * 100 if total_balls > 0 else 0
#         percent_sixes = (total_sixes / total_balls) * 100 if total_balls > 0 else 0

#         with st.container():
#             # Create a compact stats box with a grey background and padding
#             st.markdown(
#                 f"""
#                 <style>
#                     .stats-box {{
#                         background-color: #f0f0f0;
#                         padding: 15px;
#                         border-radius: 10px;
#                         font-family: Arial, sans-serif;
#                         color: #333;
#                     }}
#                     .stats-title {{
#                         font-size: 20px;
#                         font-weight: bold;
#                         margin-bottom: 10px;
#                     }}
#                     .stats-details {{
#                         font-size: 16px;
#                         font-weight: bold;
#                     }}
#                     .compact-line {{
#                         font-size: 14px;
#                     }}
#                     .bold {{
#                         font-weight: bold;
#                     }}
#                 </style>
#                 """, unsafe_allow_html=True)
#             st.markdown(f"""
#                 <div class="stats-box">
#                     <div class="stats-title">{batsman_selected} {f'vs {bowler_selected}' if bowler_selected != 'All' else '(All)'}</div>
#                     <div class="stats-details">
#                         Runs: {int(total_runs)}  
#                     </div>
#                     <div class="stats-details">
#                         Balls: {int(total_balls)}  
#                     </div>
#                     <div class="stats-details">
#                         Wickets: {int(total_dismissals)} 🟥
#                     </div>
#                     <div class="stats-details">
#                         S/R: {strike_rate:.1f}  
#                     </div>
#                     <div class="stats-details">
#                         Avg: {avg_runs:.1f}
#                     </div>
#                     <div class="compact-line">
#                         <span class="bold">0s:</span> <span class="white-square"></span> ({percent_zeros:.1f}%) | 
#                         <span class="bold">1s:</span> {int(total_ones)} 🟩 ({percent_ones:.1f}%) | 
#                         <span class="bold">2s:</span> {int(total_twos)} 🟦 ({percent_twos:.1f}%) | 
#                         <span class="bold">3s:</span> {int(total_threes)} 🟪 ({percent_threes:.1f}%) | 
#                         <span class="bold">4s:</span> {int(total_fours)} 🟨 ({percent_fours:.1f}%) | 
#                         <span class="bold">6s:</span> {int(total_sixes)} 🟫 ({percent_sixes:.1f}%)
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#         @st.cache_data
#         def get_sector_angle(zone, batting_style, offset=0):
#             base_angles = {
#                 1: 45,   # Third Man
#                 2: 90,   # Point
#                 3: 135,  # Covers
#                 4: 180,  # Mid-off
#                 5: 225,  # Mid-on
#                 6: 270,  # Mid-wicket
#                 7: 315,  # Square leg
#                 8: 360   # Fine leg
#             }
#             angle = base_angles[zone] + offset
#             if batting_style == 'LHB':
#                 angle = (180 + angle) % 360
#             return np.radians(angle)
#         @st.cache_data
#         def get_line_properties(runs):
#             properties = {
#                 1: {'color': 'darkgreen', 'length': 0.5, 'width': 2.5,'alpha':1},    
#                 2: {'color': 'darkblue', 'length': 0.65, 'width': 2.5},    
#                 3: {'color': 'darkviolet', 'length': 0.8, 'width': 2.5},   
#                 4: {'color': 'goldenrod', 'length': 1.0, 'width': 3},     
#                 6: {'color': 'maroon', 'length': 1.1, 'width': 4}     
#             }
#             return properties.get(runs, {'color': 'white', 'length': 0.4, 'width': 1,'alpha':1})
#         @st.cache_data
#         def draw_cricket_field_with_wagon_wheel(final_df):
#             fig, ax = plt.subplots(figsize=(10, 10))
#             ax.set_aspect('equal')
#             ax.axis('off')
            
#             # Draw base field elements with lighter outer green
#             # boundary = plt.Circle((0, 0), 1, fill=True, color='#228B22', alpha=0.7) 
#             boundary = plt.Circle((0, 0), 1, fill=True, color='#228B22', alpha=1)# Lighter green
#             boundary_line = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=4)
#             boundary_glow = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=4, alpha=1)
#             inner_circle = plt.Circle((0, 0), 0.5, fill=True, color='#90EE90')
#             inner_circle_line = plt.Circle((0, 0), 0.5, fill=False, color='white', linewidth=1)
            
#             # Add title
#             plt.title('WAGON WHEEL', pad=2, color='white', size=8, fontweight='bold')
            
#             # Draw sector lines
#             angles = np.linspace(0, 2*np.pi, 9)[:-1]
#             for angle in angles:
#                 x = np.cos(angle)
#                 y = np.sin(angle)
#                 ax.plot([0, x], [0, y], color='white', alpha=0.2, linewidth=1)
            
#             # Draw pitch rectangle
#             pitch_width = 0.08
#             pitch_length = 0.16
#             pitch_rect = plt.Rectangle((-pitch_width/2, -pitch_length/2), 
#                                     pitch_width, pitch_length, 
#                                     color='tan', alpha=1)
            
#             # Add base elements to plot
#             ax.add_patch(boundary)
#             ax.add_patch(boundary_glow)
#             ax.add_patch(boundary_line)
#             ax.add_patch(inner_circle)
#             ax.add_patch(inner_circle_line)
#             ax.add_patch(pitch_rect)
            
#             # Group shots by zone to handle overlapping
#             for zone in range(1, 9):
#                 zone_shots = final_df[final_df['wagonZone'] == zone]
#                 zone_shots = zone_shots.sort_values('batsman_runs', ascending=False)
                
#                 num_shots = len(zone_shots)
#                 if num_shots > 1:
#                     offsets = np.linspace(-15, 15, num_shots)
#                 else:
#                     offsets = [0]
                
#                 for (_, shot), offset in zip(zone_shots.iterrows(), offsets):
#                     angle = get_sector_angle(shot['wagonZone'], shot['batting_style'], offset)
#                     props = get_line_properties(shot['batsman_runs'])
                    
#                     x = props['length'] * np.cos(angle)
#                     y = props['length'] * np.sin(angle)
                    
#                     ax.plot([0, x], [0, y], 
#                         color=props['color'], 
#                         linewidth=props['width'], 
#                         alpha=0.9,  # Increased line opacity
#                         solid_capstyle='round')
            
#             ax.set_xlim(-1.2, 1.2)
    
#             ax.set_ylim(-1.2, 1.2)
#             plt.tight_layout(pad=0)
            
#             return fig

        
#         left_col, right_col = st.columns([2.5, 4])

#         with left_col:
#             st.markdown("## WAGON WHEEL")
#             fig = draw_cricket_field_with_wagon_wheel(final_df)
#             st.pyplot(fig, use_container_width=True)

#         with right_col:
#            # Define pitch zones with boundaries
#            zones = {
#                'Short': (8, 10),
#                'Back of Length': (6, 8),
#                'Good': (4, 6),
#                'Full': (2, 4),
#                'Yorker': (0, 2),
#                'Full Toss': (-2, 0)
#            }
           
#            # Adjusted line positions for compact spacing
#            line_positions = {
#                'WIDE_OUTSIDE_OFFSTUMP': -0.3,
#                'OUTSIDE_OFFSTUMP': -0.15,
#                'ON_THE_STUMPS': 0,
#                'DOWN_LEG': 0.15,
#                'WIDE_DOWN_LEG': 0.3
#            }
           
#            # Adjusted length positions
#            length_positions = {
#                'SHORT': 9,
#                'SHORT_OF_A_GOOD_LENGTH': 7,
#                'GOOD_LENGTH': 5,
#                'FULL': 3,
#                'YORKER': 1,
#                'FULL_TOSS': -1
#            }
           
#            # Function to apply a small random offset to length and line
#            @st.cache_data
#            def apply_offsets(x_value, y_value, x_offset_range=(-0.04, 0.04), y_offset_range=(-0.95, 0.95), 
#                              x_boundary=(-0.5, 0.5), y_boundary=(-2, 10)):
#                x_offset = np.random.uniform(x_offset_range[0], x_offset_range[1])
#                y_offset = np.random.uniform(y_offset_range[0], y_offset_range[1])
#                x_pos = max(min(x_value + x_offset, x_boundary[1]), x_boundary[0])
#                y_pos = max(min(y_value + y_offset, y_boundary[1]), y_boundary[0])
#                return x_pos, y_pos
           
#            # Set up the 3D plot
#            fig = go.Figure()
           
#            # Define stumps and bails
#            stump_positions = [-0.05, 0, 0.05]
#            stump_height = 0.3
#            stump_thickness = 2
#            bail_height = stump_height + 0.002
           
#            # Add stumps and bails in a loop
#            for x_pos in stump_positions:
#                fig.add_trace(go.Scatter3d(
#                    x=[x_pos, x_pos],
#                    y=[0, 0],
#                    z=[0, stump_height],
#                    mode='lines',
#                    line=dict(color='black', width=stump_thickness),
#                    showlegend=False
#                ))
           
#            fig.add_trace(go.Scatter3d(
#                x=[stump_positions[0], stump_positions[1]],
#                y=[0, 0],
#                z=[bail_height, bail_height],
#                mode='lines',
#                line=dict(color='black', width=2),
#                showlegend=False
#            ))
#            fig.add_trace(go.Scatter3d(
#                x=[stump_positions[1], stump_positions[2]],
#                y=[0, 0],
#                z=[bail_height, bail_height],
#                mode='lines',
#                line=dict(color='black', width=2),
#                showlegend=False
#            ))
           
#            # Add pitch zones in a single loop
#            for zone_name, (y_min, y_max) in zones.items():
#                fig.add_trace(go.Scatter3d(
#                    x=[-0.5, 0.5, 0.5, -0.5, -0.5],
#                    y=[y_min, y_min, y_max, y_max, y_min],
#                    z=[0, 0, 0, 0, 0],
#                    mode='lines+markers',
#                    line=dict(color="gray", width=2),
#                    marker=dict(size=0.1, opacity=0.2),
#                    showlegend=False
#                ))
           
#            # Add length labels
#            for length, y_position in length_positions.items():
#                fig.add_trace(go.Scatter3d(
#                    x=[0.6],
#                    y=[y_position],
#                    z=[0],
#                    mode='text',
#                    text=[length],
#                    textposition="middle right",
#                    textfont=dict(size=10, color="black"),
#                    showlegend=False
#                ))
           
#            # Adjust batting style variable for RHB and LHB
#            batting_style = final_df['batting_style'].iloc[0] if 'batting_style' in final_df else 'RHB'
#            st.write(f"Batting Style: {batting_style}")
           
#            # Set mirroring factor based on RHB or LHB
#            mirror_factor = -1 if batting_style == 'LHB' else 1
           
#            # Collect points to plot all balls at once, avoiding individual traces
#            balls_data = []
           
#            for index, row in final_df.iterrows():
#                if pd.isna(row['line']) or pd.isna(row['length']) or row['batsman_runs'] == 0:
#                    continue
           
#                # Get base X and Y positions from line and length
#                x_base = line_positions.get(row['line'], 0) * mirror_factor
#                y_base = length_positions.get(row['length'], 5)
               
#                # Apply offsets
#                x_pos, y_pos = apply_offsets(x_base, y_base)
#                z_pos = 0
           
#                # Set color and animation based on wicket status
#                if row['is_wkt'] == 1:
#                    color = 'red'
#                    size = 8
#                    opacity = [1, 0.5, 1, 0.8, 1]  # Twinkle effect sequence
#                else:
#                    batsman_runs = row['batsman_runs']
#                    color = {
#                        1: 'green',
#                        2: 'blue',
#                        3: 'violet',
#                        4: 'yellow',
#                        6: 'orange'
#                    }.get(batsman_runs, 'gray')
#                    size = 5
#                    opacity = [1]  # Static for non-wicket balls
           
#                balls_data.append({
#                    'x': [x_pos],
#                    'y': [y_pos],
#                    'z': [z_pos],
#                    'mode': 'markers',
#                    'marker': dict(size=size, color=color, opacity=opacity[0]),
#                    'hoverinfo': "text",
#                    'text': f"Runs: {row['batsman_runs']} - {'Wicket' if row['is_wkt'] else 'Run'}"
#                })
           
#            # Add all balls at once to minimize `add_trace` calls
#            for ball in balls_data:
#                fig.add_trace(go.Scatter3d(**ball))
           
#            # Layout settings
#            fig.update_layout(
#                scene=dict(
#                    xaxis=dict(title='X-axis', range=[-1, 1]),
#                    yaxis=dict(title='Y-axis', range=[-2, 10]),
#                    zaxis=dict(title='Z-axis (Height)', range=[0, 2]),
#                ),
#                width=700,
#                height=800,
#                showlegend=False
#            )
           
#            # Streamlit display
#            st.plotly_chart(fig)


#         line_positions = {
#             'WIDE_OUTSIDE_OFFSTUMP': 0,
#             'OUTSIDE_OFFSTUMP': 1,
#             'ON_THE_STUMPS': 2,
#             'DOWN_LEG': 3,
#             'WIDE_DOWN_LEG': 4
#         }

#         length_positions = {
#             'SHORT': 0,
#             'SHORT_OF_A_GOOD_LENGTH': 1,
#             'GOOD_LENGTH': 2,
#             'FULL': 3,
#             'YORKER': 4
#         }

#         # Initialize 5x5 grids for ball frequency, run accumulation, and wicket counts
#         run_count_grid = np.zeros((5, 5))
#         wicket_count_grid = np.zeros((5, 5))

#         # Filter out rows where line or length is NaN
#         filtered_df = final_df.dropna(subset=['line', 'length'])

#         # Fill the grids based on filtered data
#         for _, row in filtered_df.iterrows():
#             line = row['line']
#             length = row['length']
#             runs = row['batsman_runs']
#             is_wkt = row['is_wkt']

#             # Get the index positions from the mappings
#             line_idx = line_positions.get(line)
#             length_idx = length_positions.get(length)
            
#             if line_idx is not None and length_idx is not None:
#                 # Update run counts and wicket counts in the corresponding cell
#                 run_count_grid[length_idx, line_idx] += runs
#                 if is_wkt == 1:
#                     wicket_count_grid[length_idx, line_idx] += 1

#         # Labels for line and length positions
#         line_labels = ['Wide Outside Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
#         length_labels = ['Short', 'Back of Length', 'Good Length', 'Full', 'Yorker']

#         # Function to create heatmap figure for a 5x5 grid
#         @st.cache_data
#         def create_heatmap(grid, title, annotations):
#             fig = go.Figure(
#                 data=go.Heatmap(
#                     z=grid,
#                     colorscale='Reds',
#                     colorbar=dict(title=title)
#                 )
#             )
#             # Add annotations to show counts
#             for i in range(5):
#                 for j in range(5):
#                     fig.add_annotation(
#                         x=j, y=i,
#                         text=f'{int(annotations[i, j])}',
#                         showarrow=False,
#                         font=dict(color="black", size=12)
#                     )
            
#             # Update layout for labels and orientation
#             fig.update_layout(
#                 xaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=line_labels, title="Line"),
#                 yaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=length_labels, title="Length"),
#                 height=700, width=300  # Adjusted size for display
#             )
#             return fig

#         # Organize layouts in two columns to make them appear side-by-side
#         col1, col2 = st.columns(2)

#         with col1:
#             st.write("### Wicket Count")
#             # Display the wicket count grid
#             st.plotly_chart(create_heatmap(wicket_count_grid, "Wickets", wicket_count_grid), use_container_width=True)

#         with col2:
#             st.write("### Runs Scored")
#             # Display the runs count grid
#             st.plotly_chart(create_heatmap(run_count_grid, "Runs", run_count_grid), use_container_width=True)



# else:
#     st.header("Strength and Weakness Analysis")
#     player_name = st.selectbox("Search for a player", idf['batsman'].unique())
    
#     # Dropdown for Batting or Bowling selection
#     option = st.selectbox("Select Role", ("Batting", "Bowling"))
    
#     if option == "Batting":          
#         # Apply the function to add the 'bowl_kind' column
        
#         result_df = pd.DataFrame()
    
#         final_df = pdf[pdf['batsman'] == player_name]
#         result_df = pd.DataFrame()
#         i = 0
        
#         # Loop over pace and spin bowling types
#         for bowl_kind in ['pace bowler', 'spin bowler']:
#             temp_df = pdf[pdf['batsman'] == player_name]  # Filter data for the selected batsman
            
#             # Filter for the specific 'bowl_kind'
#             temp_df = temp_df[temp_df['bowl_kind'] == bowl_kind]
            
#             # Apply the cumulative function (bcum)
#             temp_df = cumulator(temp_df)
            
#             # If the DataFrame is empty after applying `bcum`, skip this iteration
#             if temp_df.empty:
#                 continue
            
#             # Add the bowl_kind column
#             temp_df['bowl_kind'] = bowl_kind
            
#             # Reorder columns to make 'bowl_kind' the first column
#             cols = temp_df.columns.tolist()
#             new_order = ['bowl_kind'] + [col for col in cols if col != 'bowl_kind']
#             temp_df = temp_df[new_order]
            
#             # Concatenate results into result_df
#             if i == 0:
#                 result_df = temp_df
#                 i += 1
#             else:
#                 result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
#         # Display the final result_df
#         result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year','hundreds','fifties','thirties','highest_score','matches'])
#         result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
#         columns_to_convert = ['RUNS']
        
#         # Fill NaN values with 0
#         # result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
        
#         # Convert the specified columns to integer type
#         # result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
#         result_df = round_up_floats(result_df)
        
#         # Specify the desired order with 'bowl_kind' first
#         cols = result_df.columns.tolist()
#         # new_order = ['BOWL KIND', 'INNINGS'] + [col for col in cols if col not in ['BOWL KIND', 'INNINGS']]
        
#         # Reindex the DataFrame with the new column order
#         # result_df = result_df[new_order]
#         result_df['BOWL KIND'] = result_df['BOWL KIND'].str.capitalize()
#         st.markdown("### Performance Against Bowling Types (Pace vs Spin)")
#         st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        
#         result_df = pd.DataFrame()
#         i = 0
#         allowed_bowling_styles=['Off-break', 'Left-arm medium fast', 'Right-arm medium fast',
#        'Right-arm fast', 'Slow left-arm orthodox',
#        'Left-arm fast',
#        'Leg-break googly', 'Right-arm medium',
#        'Left-arm medium',
#        'Left-arm wrist spin',
#        'Leg-break','Leg-spin',
#        'Left-arm medium fast and slow left-arm orthodox',
#        'Off-break and slow left-arm orthodox',
#        'Right-arm medium fast and off-break']
#         for bowling_style in allowed_bowling_styles:
#             temp_df = pdf[pdf['batsman'] == player_name]  # Filter data for the selected batsman
            
#             # Filter for the specific bowling style
#             temp_df = temp_df[temp_df['bowling_style'] == bowling_style]
            
#             # Apply the cumulative function (bcum)
#             temp_df = cumulator(temp_df)
            
#             # If the DataFrame is empty after applying `bcum`, skip this iteration
#             if temp_df.empty:
#                 continue
            
#             # Add the bowling style column
#             temp_df['bowling_style'] = bowling_style
            
#             # Reorder columns to make 'bowling_style' the first column
#             cols = temp_df.columns.tolist()
#             new_order = ['bowling_style'] + [col for col in cols if col != 'bowling_style']
#             temp_df = temp_df[new_order]
            
#             # Concatenate results into result_df
#             if i == 0:
#                 result_df = temp_df
#                 i += 1
#             else:
#                 result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
#         # Display the final result_df
#         result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year','hundreds','fifties','thirties','highest_score','matches'])
#         result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
#         columns_to_convert = ['RUNS']

#         # Fill NaN values with 0
#         result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)

#         # Convert the specified columns to integer type
#         result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
#         result_df = round_up_floats(result_df)
#         cols = result_df.columns.tolist()

#         # Specify the desired order with 'bowling_style' first
#         new_order = ['BOWLING STYLE', 'INNINGS'] + [col for col in cols if col not in ['BOWLING STYLE','INNINGS',]]

#         # Reindex the DataFrame with the new column order
#         result_df = result_df[new_order]

#         st.markdown("### Performance Against Bowling Styles")
#         st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        

#         @st.cache_data
#         def get_sector_angle(zone, batting_style):
#             # Set base angle for RHB and mirror for LHB
#             base_angles = {
#                 1: 45,   # Third Man
#                 2: 90,   # Point
#                 3: 135,  # Covers
#                 4: 180,  # Mid-off
#                 5: 225,  # Mid-on
#                 6: 270,  # Mid-wicket
#                 7: 315,  # Square leg
#                 8: 0     # Fine leg (360 degrees or 0 degrees)
#             }
#             angle = base_angles[zone]
            
#             # Adjust for left-handed batsman by mirroring the sectors
#             if batting_style == 'LHB':
#                 angle = (180 + angle) % 360
            
#             return np.radians(angle)
#         @st.cache_data
#         def draw_cricket_field_with_run_totals(final_df):
#             fig, ax = plt.subplots(figsize=(10, 10))
#             ax.set_aspect('equal')
#             ax.axis('off')

#             # Draw field elements
#             boundary = plt.Circle((0, 0), 1, fill=True, color='#228B22', alpha=1)
#             boundary_line = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=4)
#             inner_circle = plt.Circle((0, 0), 0.5, fill=True, color='#90EE90')
#             inner_circle_line = plt.Circle((0, 0), 0.5, fill=False, color='white', linewidth=1)
            
#             ax.add_patch(boundary)
#             ax.add_patch(boundary_line)
#             ax.add_patch(inner_circle)
#             ax.add_patch(inner_circle_line)
            
#             # Draw sector lines
#             angles = np.linspace(0, 2*np.pi, 9)[:-1]
#             for angle in angles:
#                 x = np.cos(angle)
#                 y = np.sin(angle)
#                 ax.plot([0, x], [0, y], color='white', alpha=0.2, linewidth=1)
            
#             # Draw pitch rectangle
#             pitch_rect = plt.Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1)
#             ax.add_patch(pitch_rect)
            
#             # Determine total runs per sector
#             batting_style = final_df['batting_style'].iloc[0]  # Assume all rows have the same batting style
#             total_runs = final_df.groupby('wagonZone')['batsman_runs'].sum()

#             # Add title based on batting style
#             title = "Right-handed Batsman" if batting_style == 'RHB' else "Left-handed Batsman"
#             plt.title(title, pad=20, color='white', size=12, fontweight='bold')
            
#             # Label each sector with the total runs scored
#             for zone in range(1, 9):
#                 angle = get_sector_angle(zone, batting_style) + np.radians(22.5)
#                 x = 0.65 * np.cos(angle)  # Reduced radius for labels to be inside the sector
#                 y = 0.65 * np.sin(angle)
                
#                 # Get the total runs for the zone (default to 0 if not present)
#                 runs = total_runs.get(zone, 0)
                
#                 # Display runs in the sector
#                 ax.text(x, y, str(runs), ha='center', va='center', color='white', fontweight='bold', size=20)
            
#             ax.set_xlim(-1.2, 1.2)
#             ax.set_ylim(-1.2, 1.2)
#             plt.tight_layout(pad=0)
            
#             return fig
#         col1, col2 = st.columns(2)

#         with col1:
#             st.write("### Wagon Chart")
#             # Display the wicket count grid
#             fig = draw_cricket_field_with_run_totals(final_df)
#             st.pyplot(fig, use_container_width=True)
            

#         with col2:
#              # Set up line and length mapping
#             import plotly.graph_objects as go          
#             line_positions = {
#                 'WIDE_OUTSIDE_OFFSTUMP': 0,
#                 'OUTSIDE_OFFSTUMP': 1,
#                 'ON_THE_STUMPS':2,
#                 'DOWN_LEG': 3,
#                 'WIDE_DOWN_LEG':4 
#             }


#             length_positions = {
#                 'SHORT': 0 ,
#                 'SHORT_OF_A_GOOD_LENGTH': 1,
#                 'GOOD_LENGTH': 2,
#                 'FULL': 3,
#                 'YORKER': 4,
#                 'FULL_TOSS': 4
#             } 
#             run_count_grid = np.zeros((5, 5))
#             wicket_count_grid = np.zeros((5, 5))
            
#             # Fill the grids based on final_df data
#             for _, row in final_df.iterrows():
#                 line = row['line']
#                 length = row['length']
#                 runs = row['batsman_runs']
#                 is_wkt = row['is_wkt']
                
#                 # Identify the correct cell for run count and wicket count
#                 line_idx = line_positions.get(line, 2)  # Default to 'On Stumps' if line not found
#                 length_idx = length_positions.get(length, 2)  # Default to 'Good Length' if length not found
                
#                 # Update run counts and wicket counts
#                 run_count_grid[length_idx, line_idx] += runs
                
#                 if is_wkt == 1:
#                     wicket_count_grid[length_idx, line_idx] += 1
            
#             # Labels for line and length positions
#             line_labels = ['Wide Outside Off', 'Outside Off', 'On Stumps', 'Outside Leg', 'Wide Outside Leg']
#             length_labels = ['Short', 'Back of Length', 'Good Length', 'Full', 'Yorker']
            
#             # Function to create heatmap figure for a 5x5 grid
#             @st.cache_data
#             def create_heatmap(grid, title, annotations):
#                 fig = go.Figure(
#                     data=go.Heatmap(
#                         z=grid,
#                         colorscale='Reds',
#                         colorbar=dict(title=title)
#                     )
#                 )
#                 # Add black text annotations to show actual counts
#                 for i in range(5):
#                     for j in range(5):
#                         fig.add_annotation(
#                             x=j, y=i,
#                             text=f'{annotations[i, j]}',
#                             showarrow=False,
#                             font=dict(color="black", size=12)
#                         )
                
#                 # Update layout for vertical orientation and labels
#                 fig.update_layout(
#                     xaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=line_labels, title="Line"),
#                     yaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=length_labels, title="Length"),
#                     height=700, width=300  # Adjusted size for compact display
#                 )
#                 return fig
            
            

#             st.write("### Runs Scored")
#             # Display the runs count grid
#             st.plotly_chart(create_heatmap(run_count_grid, "Runs", run_count_grid), use_container_width=True)
         
#         @st.cache_data
#         def apply_length_offset(y_value, offset_range=(-0.95, 0.95), boundary=(-2, 10)):
#             offset = np.random.uniform(offset_range[0], offset_range[1])
#             if boundary[0] <= y_value + offset <= boundary[1]:
#                 return y_value + offset
#             return y_value
        
#         # Function to apply a small random offset to line while keeping length accurate
#         @st.cache_data
#         def apply_line_offset(x_value, offset_range=(-0.05, 0.05), boundary=(-0.5, 0.5)):
#             offset = np.random.uniform(offset_range[0], offset_range[1])
#             if boundary[0] <= x_value + offset <= boundary[1]:
#                 return x_value + offset
#             return x_value
        
#         # Define pitch zones and their respective boundaries
#         zones = {
#             'SHORT': (8, 10),
#             'SHORT_OF_A_GOOD_LENGTH': (6, 8),
#             'GOOD_LENGTH': (4, 6),
#             'FULL': (2, 4),
#             'YORKER': (0, 2),
#             'FULL_TOSS': (-2, 0)
#         }
        
#         line_positions = {
#             'WIDE_OUTSIDE_OFFSTUMP': 0.25,
#             'OUTSIDE_OFFSTUMP': 0.15,
#             'ON_THE_STUMPS': 0,
#             'DOWN_LEG': -0.15,
#             'WIDE_DOWN_LEG': -0.25
#         }
        
#         length_positions = {
#             'SHORT': 9,
#             'SHORT_OF_A_GOOD_LENGTH': 7,
#             'GOOD_LENGTH': 5,
#             'FULL': 3,
#             'YORKER': 1,
#             'FULL_TOSS': -1
#         }
        
#         # Function to create a 3D pitch map based on handedness
#         @st.cache_data
#         def create_pitch_map(data, handedness):
#             fig = go.Figure()
        
#             # Define stumps and bails positions
#             stump_positions = [-0.05, 0, 0.05]
#             stump_height = 0.3
#             stump_thickness = 2
#             bail_height = stump_height + 0.002
        
#             # Add stumps and bails to the figure
#             for x_pos in stump_positions:
#                 fig.add_trace(go.Scatter3d(
#                     x=[x_pos, x_pos],
#                     y=[0, 0],
#                     z=[0, stump_height],
#                     mode='lines',
#                     line=dict(color='black', width=stump_thickness),
#                     showlegend=False
#                 ))
        
#             fig.add_trace(go.Scatter3d(
#                 x=[stump_positions[0], stump_positions[1]],
#                 y=[0, 0],
#                 z=[bail_height, bail_height],
#                 mode='lines',
#                 line=dict(color='black', width=2),
#                 showlegend=False
#             ))
#             fig.add_trace(go.Scatter3d(
#                 x=[stump_positions[1], stump_positions[2]],
#                 y=[0, 0],
#                 z=[bail_height, bail_height],
#                 mode='lines',
#                 line=dict(color='black', width=2),
#                 showlegend=False
#             ))
        
#             # Add pitch zones
#             for zone_name, (y_min, y_max) in zones.items():
#                 fig.add_trace(go.Scatter3d(
#                     x=[-0.5, 0.5, 0.5, -0.5, -0.5],
#                     y=[y_min, y_min, y_max, y_max, y_min],
#                     z=[0, 0, 0, 0, 0],
#                     mode='lines+markers',
#                     line=dict(color="gray", width=2),
#                     marker=dict(size=0.1, opacity=0.2),
#                     showlegend=False
#                 ))
        
#             # Add length labels on the side of the pitch
#             for length, y_position in length_positions.items():
#                 fig.add_trace(go.Scatter3d(
#                     x=[0.6],  # Position length labels to the side of the pitch
#                     y=[y_position],
#                     z=[0],
#                     mode='text',
#                     text=[length],
#                     textposition="middle right",
#                     textfont=dict(size=10, color="black"),
#                     showlegend=False
#                 ))
        
#             # Set mirroring factor based on handedness
#             mirror_factor = -1 if handedness == 'LHB' else 1 if handedness == 'RHB' else 0
        
#             # Filter data to only include wicket balls
#             wicket_data = data[data['bowler_wkt'] == 1]
        
#             # Plot wicket balls
#             for index, row in wicket_data.iterrows():
#                 if pd.isna(row['line']) or pd.isna(row['length']):
#                     continue  # Skip missing data
#                 x_base = line_positions.get(row['line'], 0) * mirror_factor
#                 y_base = length_positions.get(row['length'], 5)
        
#                 # Apply offset to length while keeping line accurate
#                 x_pos = apply_line_offset(x_base, boundary=(-0.5, 0.5))
#                 y_pos = apply_length_offset(y_base, boundary=(-2, 10))
#                 z_pos = 0
        
#                 # Plot the wicket ball
#                 fig.add_trace(go.Scatter3d(
#                     x=[x_pos],
#                     y=[y_pos],
#                     z=[z_pos],
#                     mode='markers',
#                     marker=dict(size=5, color='red', opacity=1),
#                     hoverinfo="text",
#                     text=f"Line: {row['line']}<br>Length: {row['length']}<br>Runs: {row['batsman_runs']} - Wicket"
#                 ))
        
#             # Final layout settings
#             fig.update_layout(
#                 scene=dict(
#                     xaxis=dict(title='X-axis', range=[-1, 1]),
#                     yaxis=dict(title='Y-axis', range=[-2, 10]),
#                     zaxis=dict(title='Z-axis (Height)', range=[0, 2]),
#                 ),
#                 width=1200,
#                 height=1000,
#                 showlegend=False
#             )
        
#             return fig


#         bat_hand = final_df['batting_style'].iloc[0]
#         # Display each plot in the respective column
#         with col1:
#             st.write("### Against Pace")
#             if pace_df.empty:
#                 st.write("No data for Pace")
#             else:
#                 st.plotly_chart(create_pitch_map(pace_df, bat_hand ))
        
#         with col2:
#             st.write("### Against Spin")
#             if spin_df.empty:
#                 st.write("No data for Right-Handed Batsmen")
#             else:
#                 st.plotly_chart(create_pitch_map(spin_df,bat_hand))
                
                
#     else:
#         final_df=pdf[pdf['bowler']==player_name]  
#         allowed_batting_styles = ['LHB', 'RHB']  # Define the two batting styles
#         result_df = pd.DataFrame()
#         temp_df = pdf[pdf['bowler'] == player_name]
#         if temp_df.empty :
#             st.markdown('Bowling stats do not exist')
#         else:
#             # Loop over left-hand and right-hand batting styles
#             for bat_style in allowed_batting_styles:
#                 temp_df = pdf[pdf['bowler'] == player_name]  # Filter data for the selected bowler
                
#                 # Filter for the specific batting style
#                 temp_df = temp_df[temp_df['batting_style'] == bat_style]
                
#                 # Apply the cumulative function (bcum) for bowling
#                 temp_df = bowlerstat(temp_df)
                
#                 # If the DataFrame is empty after applying bcum, skip this iteration
#                 if temp_df.empty:
#                     continue
                
#                 # Add the batting style as a column for later distinction
#                 temp_df['batting_style'] = bat_style
                
#                 # Concatenate results into result_df
#                 result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
#             # Drop unwanted columns from the result DataFrame
#             result_df = result_df.drop(columns=['bowler'])
        
#             # Standardize column names
#             result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            
#             # Convert the relevant columns to integers and fill NaN values
#             columns_to_convert = ['WKTS']
#             result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0).astype(int)
#             result_df = round_up_floats(result_df)
#             cols = result_df.columns.tolist()
              
#               # Specify the desired order with 'phase' first
#             new_order = ['BATTING STYLE'] + [col for col in cols if col not in 'BATTING STYLE']
#             result_df = result_df[new_order]
        
#             # Display the final table
#             st.markdown("### Cumulative Bowling Performance Against Batting Styles")
#             st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
#         if not final_df.empty:     
#             st.markdown("## PITCH MAP VS LEFT-HANDED AND RIGHT-HANDED BATSMEN")
            
#             zones = {
#                 'SHORT': (8, 10),
#                 'SHORT_OF_A_GOOD_LENGTH': (6, 8),
#                 'GOOD_LENGTH': (4, 6),
#                 'FULL': (2, 4),
#                 'YORKER': (0, 2),
#                 'FULL_TOSS': (-2, 0)
#             }

#             line_positions = {
#                 'WIDE_OUTSIDE_OFFSTUMP': -0.25,
#                 'OUTSIDE_OFFSTUMP': -0.15,
#                 'ON_THE_STUMPS': 0,
#                 'DOWN_LEG': 0.15,
#                 'WIDE_DOWN_LEG': 0.25
#             }


#             length_positions = {
#                 'SHORT': 9,
#                 'SHORT_OF_A_GOOD_LENGTH': 7,
#                 'GOOD_LENGTH': 5,
#                 'FULL': 3,
#                 'YORKER': 1,
#                 'FULL_TOSS': -1
#             } 
            
#             # Function to apply a small random offset to length while keeping line accurate
#             @st.cache_data
#             def apply_length_offset(y_value, offset_range=(-0.95, 0.95), boundary=(-2, 10)):
#                 offset = np.random.uniform(offset_range[0], offset_range[1])
#                 if boundary[0] <= y_value + offset <= boundary[1]:
#                     return y_value + offset
#                 return y_value
#             @st.cache_data
#             def apply_line_offset(x_value, offset_range=(-0.05, 0.05), boundary=(-0.5, 0.5)):
#                 offset = np.random.uniform(offset_range[0], offset_range[1])
#                 if boundary[0] <= x_value + offset <= boundary[1]:
#                     return x_value + offset
#                 return x_value
            
#             # Set up two columns for LHB and RHB views
#             col1, col2 = st.columns(2)
#             lhb_data = final_df[final_df['batting_style'] == 'LHB']
#             rhb_data = final_df[final_df['batting_style'] == 'RHB']
            
#             # Function to create a 3D pitch map based on handedness
#             @st.cache_data
#             def create_pitch_map(data, handedness):
#                 fig = go.Figure()
            
#                 # Define stumps and bails
#                 stump_positions = [-0.05, 0, 0.05]
#                 stump_height = 0.3
#                 stump_thickness = 2
#                 bail_height = stump_height + 0.002
            
#                 # Add stumps
#                 for x_pos in stump_positions:
#                     fig.add_trace(go.Scatter3d(
#                         x=[x_pos, x_pos],
#                         y=[0, 0],
#                         z=[0, stump_height],
#                         mode='lines',
#                         line=dict(color='black', width=stump_thickness),
#                         showlegend=False
#                     ))
            
#                 # Add bails
#                 fig.add_trace(go.Scatter3d(
#                     x=[stump_positions[0], stump_positions[1]],
#                     y=[0, 0],
#                     z=[bail_height, bail_height],
#                     mode='lines',
#                     line=dict(color='black', width=2),
#                     showlegend=False
#                 ))
#                 fig.add_trace(go.Scatter3d(
#                     x=[stump_positions[1], stump_positions[2]],
#                     y=[0, 0],
#                     z=[bail_height, bail_height],
#                     mode='lines',
#                     line=dict(color='black', width=2),
#                     showlegend=False
#                 ))
            
#                 # Add pitch zones
#                 for zone_name, (y_min, y_max) in zones.items():
#                     fig.add_trace(go.Scatter3d(
#                         x=[-0.5, 0.5, 0.5, -0.5, -0.5],
#                         y=[y_min, y_min, y_max, y_max, y_min],
#                         z=[0, 0, 0, 0, 0],
#                         mode='lines+markers',
#                         line=dict(color="gray", width=2),
#                         marker=dict(size=0.1, opacity=0.2),
#                         showlegend=False
#                     ))
            
#                 # Add length labels on the side of the pitch
#                 for length, y_position in length_positions.items():
#                     fig.add_trace(go.Scatter3d(
#                         x=[0.6],  # Adjust X position to be to the side of the pitch
#                         y=[y_position],
#                         z=[0],
#                         mode='text',
#                         text=[length],
#                         textposition="middle right",
#                         textfont=dict(size=10, color="black"),
#                         showlegend=False
#                     ))
            
#                 # Set mirroring factor based on handedness
#                 if handedness == 'LHB':
#                     mirror_factor = -1
#                 elif handedness == 'RHB':
#                     mirror_factor = 1
#                 else:
#                     mirror_factor = 0  # Default case if handedness is neither "Left-hand bat" nor "Right-hand bat"
            
#                 # Separate the data into wicket and non-wicket balls
#                 wicket_data = data[data['bowler_wkt'] == 1]        
#                 # Plot wicket balls first
#                 for index, row in wicket_data.iterrows():
#                     if pd.isna(row['line']) or pd.isna(row['length']):
#                        continue  # Skip this row and move to the next one
#                     # Determine base X and Y positions from line and length
#                     x_base = line_positions.get(row['line'], 0) * mirror_factor
#                     y_base = length_positions.get(row['length'], 5)
            
#                     # Apply offset to length (y) while keeping line (x) accurate
#                     x_pos = apply_line_offset(x_base, boundary=(-0.5, 0.5))
#                     y_pos = apply_length_offset(y_base, boundary=(-2, 10))
#                     z_pos = 0
            
#                     # Set color and size for wickets
#                     color = 'red'
#                     size = 5
#                     opacity = 1  # Set opacity to a single value
            
#                     # Plot the wicket ball
#                     fig.add_trace(go.Scatter3d(
#                         x=[x_pos],
#                         y=[y_pos],
#                         z=[z_pos],
#                         mode='markers',
#                         marker=dict(size=size, color=color, opacity=opacity),
#                         hoverinfo="text",
#                         text=f"Line: {row['line']}<br>Length: {row['length']}<br>Runs: {row['batsman_runs']} - Wicket"

#                     ))
#                 # Twinkle effect for wickets (already added in the wicket balls loop)
            
#                 fig.update_layout(
#                     scene=dict(
#                         xaxis=dict(title='X-axis', range=[-1, 1]),
#                         yaxis=dict(title='Y-axis', range=[-2, 10]),
#                         zaxis=dict(title='Z-axis (Height)', range=[0, 2]),
#                     ),
#                     width=1200,
#                     height=1000,
#                     showlegend=False
#                 )
#                 return fig

            
#             # Display each plot in the respective column
#             with col1:
#                 st.write("### Against Left-Handed Batsmen")
#                 if lhb_data.empty:
#                     st.write("No data for Left-Handed Batsmen")
#                 else:
#                     st.plotly_chart(create_pitch_map(lhb_data, 'LHB'), key="LHB")
            
#             with col2:
#                 st.write("### Against Right-Handed Batsmen")
#                 if rhb_data.empty:
#                     st.write("No data for Right-Handed Batsmen")
#                 else:
#                     st.plotly_chart(create_pitch_map(rhb_data, 'RHB'), key="RHB")

#             import numpy as np
#             import plotly.graph_objects as go
#             import streamlit as st
            
#             # Assuming final_df already exists with the following columns: 'line', 'length', 'batsman_runs', 'is_wkt'
            
#             # Set up line and length mapping
#             line_positions = {
#                 'WIDE_OUTSIDE_OFFSTUMP': 0,
#                 'OUTSIDE_OFFSTUMP': 1,
#                 'ON_THE_STUMPS':2,
#                 'DOWN_LEG': 3,
#                 'WIDE_DOWN_LEG':4 
#             }


#             length_positions = {
#                 'SHORT': 0 ,
#                 'SHORT_OF_A_GOOD_LENGTH': 1,
#                 'GOOD_LENGTH': 2,
#                 'FULL': 3,
#                 'YORKER': 4,
#                 'FULL_TOSS': 4
#             }
            
#             # Initialize 5x5 grids for wicket count and run accumulation
#             wicket_count_grid = np.zeros((5, 5))
#             run_count_grid_bowler = np.zeros((5, 5))
            
#             # Fill the grids based on final_df data
#             for _, row in final_df.iterrows():
#                 if pd.isna(row['line']) or pd.isna(row['length']):
#                     continue  # Skip this row and move to the next one
#                 line = row['line']
#                 length = row['length']
#                 runs = row['batsman_runs']
#                 is_wkt = row['bowler_wkt']
            
#                 # Identify the correct cell for wicket count and run count
#                 line_idx = line_positions.get(line, 2)  # Default to 'On Stumps' if line not found
#                 length_idx = length_positions.get(length, 2)  # Default to 'Good Length' if length not found
            
#                 if is_wkt == 1:
#                     # Update wicket count in the grid for the given line and length
#                     wicket_count_grid[length_idx, line_idx] += 1
                
#                 # Update run count in the grid for the given line and length
#                 run_count_grid_bowler[length_idx, line_idx] += runs
            
#             # Labels for line and length positions
#             line_labels = ['Wide Outside Off', 'Outside Off', 'On Stumps', 'Outside Leg', 'Wide Outside Leg']
#             length_labels = ['Short', 'Back of Length', 'Good Length', 'Full', 'Yorker']
            
#             # Function to create heatmap figure for a 5x5 grid
#             @st.cache_data
#             def create_heatmap(grid, title, annotations):
#                 fig = go.Figure(
#                     data=go.Heatmap(
#                         z=grid,
#                         colorscale='Reds',
#                         colorbar=dict(title=f'{title}')
#                     )
#                 )
#                 # Add black text annotations to show values
#                 for i in range(5):
#                     for j in range(5):
#                         fig.add_annotation(
#                             x=j, y=i,
#                             text=f'{annotations[i, j]}',
#                             showarrow=False,
#                             font=dict(color="black", size=12)
#                         )
                
#                 # Update layout for vertical orientation and labels
#                 fig.update_layout(
#                     xaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=line_labels, title="Line"),
#                     yaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=length_labels, title="Length"),
#                     height=700, width=300  # Adjusted size for compact display
#                 )
#                 return fig
            
#             # Organize layouts in two columns to make them appear side-by-side
#             col1, col2 = st.columns(2)
            
#             # First Column - Wicket Heatmap
#             with col1:
#                 st.write("### Wicket Distribution")
#                 wicket_fig = create_heatmap(wicket_count_grid, "Wickets", wicket_count_grid)
#                 st.plotly_chart(wicket_fig, use_container_width=True)
            
#             # Second Column - Run Distribution for Bowler
#             with col2:
#                 st.write("### Runs Given")
#                 run_fig_bowler = create_heatmap(run_count_grid_bowler, "Runs", run_count_grid_bowler)
#                 st.plotly_chart(run_fig_bowler, use_container_width=True)
#         else:
#             st.write("## No Bowling Data Available")
