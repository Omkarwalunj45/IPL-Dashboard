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

# ---- helpers ----
def bp6(balls, sixes):
    return balls / sixes if (sixes is not None and sixes > 0) else balls

def bp4(balls, fours):
    return balls / fours if (fours is not None and fours > 0) else balls

def bpb(balls, boundary_count):
    return balls / boundary_count if (boundary_count is not None and boundary_count > 0) else np.nan

def bpd(balls, dismissals):
    return balls / dismissals if (dismissals is not None and dismissals > 0) else np.nan

def avg(runs, dismissals, innings):
    # batting average: runs/dismissals ; fallback runs/innings if no dismissals
    try:
        if pd.notna(dismissals) and dismissals > 0:
            return runs / dismissals
        elif pd.notna(innings) and innings > 0:
            return runs / innings
        else:
            return np.nan
    except Exception:
        return np.nan

def categorize_phase(over):
    # Over should be numeric; return phase string
    try:
        o = float(over)
    except Exception:
        return "Unknown"
    if o <= 6:
        return 'Powerplay'
    elif 6 < o <= 11:
        return 'Middle 1'
    elif 11 < o <= 16:
        return 'Middle 2'
    else:
        return 'Death'

# ---- main function ----
def Custom(df):
    """
    Adapted to the given dataset columns:
    Expected columns (at minimum): 
      p_match, inns, bat, team_bat, ball_id, out, dismissal, over, noball, wide, batruns,
      cur_bat_runs, cur_bat_bf, cur_bowl_ovr, cur_bowl_wkts, cur_bowl_runs, inns_runs, inns_wkts, inns_balls, ...
    Key design:
      - Uses cur_bat_runs / cur_bat_bf per (batsman, match) taking the LAST record in that match
        to represent that batsman's runs/balls for that match (because cur_* are cumulative).
      - Derives per-ball flags (is_one/is_two/is_four/is_six/is_dot) from 'batruns'.
      - Builds phase = categorize_phase(over)
      - Creates legal_ball from noball & wide.
    Returns:
      bat_rec DataFrame with aggregated batting metrics + phase-wise pivot metrics.
    """

    d = df.copy()

    # ---- column normalization ----
    # rename to internal names used here
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'inns' in d.columns:
        d = d.rename(columns={'inns': 'inning'})
    if 'bat' in d.columns:
        d = d.rename(columns={'bat': 'batsman'})
    if 'team_bat' in d.columns:
        d = d.rename(columns={'team_bat': 'batting_team'})
    # per-ball runs off bat: prefer 'batruns' (explicit)
    if 'batruns' in d.columns:
        d = d.rename(columns={'batruns': 'runs_off_bat'})
    elif 'score' in d.columns:
        # fallback if no explicit batruns column
        d = d.rename(columns={'score': 'runs_off_bat'})
    else:
        # create runs_off_bat if not present (assume 0)
        d['runs_off_bat'] = 0

    # ---- legal ball ----
    # convert noball/wide to numeric and create legal_ball (1 if neither noball nor wide)
    if 'noball' in d.columns:
        d['noball'] = pd.to_numeric(d['noball'], errors='coerce').fillna(0).astype(int)
    else:
        d['noball'] = 0
    if 'wide' in d.columns:
        d['wide'] = pd.to_numeric(d['wide'], errors='coerce').fillna(0).astype(int)
    else:
        d['wide'] = 0
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # ---- wicket flag ----
    # 'out' is present as 0/1 in your schema; ensure numeric
    if 'out' in d.columns:
        d['is_wkt'] = pd.to_numeric(d['out'], errors='coerce').fillna(0).astype(int)
    else:
        # fallback to dismissal column presence
        d['is_wkt'] = d['dismissal'].notna().astype(int) if 'dismissal' in d.columns else 0

    # ---- per-delivery flags based on runs_off_bat ----
    d['runs_off_bat'] = pd.to_numeric(d.get('runs_off_bat', 0), errors='coerce').fillna(0).astype(int)
    d['is_dot'] = ((d['runs_off_bat'] == 0) & (d['legal_ball'] == 1)).astype(int)
    d['is_one'] = (d['runs_off_bat'] == 1).astype(int)
    d['is_two'] = (d['runs_off_bat'] == 2).astype(int)
    d['is_three'] = (d['runs_off_bat'] == 3).astype(int)
    d['is_four'] = (d['runs_off_bat'] == 4).astype(int)
    d['is_six'] = (d['runs_off_bat'] == 6).astype(int)

    # ---- ensure ordering so 'last' per match is correct ----
    # If ball_id encodes sequential order (it usually does), sort by match and ball_id.
    if 'ball_id' in d.columns:
        try:
            # try numeric sort if possible
            d['__ball_sort__'] = pd.to_numeric(d['ball_id'], errors='coerce')
        except Exception:
            d['__ball_sort__'] = d['ball_id']
    else:
        # fallback to index order
        d['__ball_sort__'] = np.arange(len(d))

    d.sort_values(['match_id', 'batsman', '__ball_sort__'], inplace=True, kind='stable')

    # ---- compute per-(batsman, match) last cumulative values ----
    # these represent that batsman's final runs & balls in that match
    # use cur_bat_runs and cur_bat_bf (they are cumulative)
    # defensive: ensure the columns exist
    if 'cur_bat_runs' in d.columns:
        d['cur_bat_runs'] = pd.to_numeric(d['cur_bat_runs'], errors='coerce').fillna(0).astype(int)
    else:
        d['cur_bat_runs'] = 0

    if 'cur_bat_bf' in d.columns:
        d['cur_bat_bf'] = pd.to_numeric(d['cur_bat_bf'], errors='coerce').fillna(0).astype(int)
    else:
        d['cur_bat_bf'] = 0

    # last per-match snapshot for batter
    last_bat_snapshot = (
        d.groupby(['batsman', 'match_id'], sort=False, as_index=False)
         .agg({'cur_bat_runs': 'last', 'cur_bat_bf': 'last'})
         .rename(columns={'cur_bat_runs': 'match_runs', 'cur_bat_bf': 'match_balls'})
    )

    # Similarly, for bowlers we can use cur_bowl_ovr/cur_bowl_wkts/cur_bowl_runs (last per match per bowler)
    if 'cur_bowl_ovr' in d.columns:
        d['cur_bowl_ovr'] = pd.to_numeric(d['cur_bowl_ovr'], errors='coerce').fillna(0)
    else:
        d['cur_bowl_ovr'] = 0

    if 'cur_bowl_wkts' in d.columns:
        d['cur_bowl_wkts'] = pd.to_numeric(d['cur_bowl_wkts'], errors='coerce').fillna(0).astype(int)
    else:
        d['cur_bowl_wkts'] = 0

    if 'cur_bowl_runs' in d.columns:
        d['cur_bowl_runs'] = pd.to_numeric(d['cur_bowl_runs'], errors='coerce').fillna(0).astype(int)
    else:
        d['cur_bowl_runs'] = 0

    last_bowl_snapshot = (
        d.groupby(['bowl', 'match_id'], sort=False, as_index=False)
         .agg({'cur_bowl_ovr': 'last', 'cur_bowl_wkts': 'last', 'cur_bowl_runs': 'last'})
    )

    # ---- per-match derived metrics for batters ----
    # total runs across dataset = sum of match_runs
    runs_per_match = last_bat_snapshot[['batsman', 'match_runs', 'match_balls', 'match_id']].copy()

    # number of innings (distinct matches with an appearance)
    innings_count = runs_per_match.groupby('batsman')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings'})

    # total runs & balls across matches (summing final per-match values)
    total_runs = runs_per_match.groupby('batsman')['match_runs'].sum().reset_index().rename(columns={'match_runs': 'runs'})
    total_balls = runs_per_match.groupby('batsman')['match_balls'].sum().reset_index().rename(columns={'match_balls': 'balls'})

    # dismissals: use sum of 'is_wkt' across all deliveries (since out flag appears on the ball where wicket happened)
    dismissals = d.groupby('batsman')['is_wkt'].sum().reset_index().rename(columns={'is_wkt': 'dismissals'})

    # boundary & running counts from per-delivery 'runs_off_bat' flags
    fours = d.groupby('batsman')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = d.groupby('batsman')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
    dots = d.groupby('batsman')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    ones = d.groupby('batsman')['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
    twos = d.groupby('batsman')['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
    threes = d.groupby('batsman')['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})

    # match-level thresholds (30/50/100) from runs_per_match last values
    thirties = runs_per_match[runs_per_match['match_runs'] >= 30].groupby('batsman').size().reset_index(name='30s')
    fifties = runs_per_match[runs_per_match['match_runs'] >= 50].groupby('batsman').size().reset_index(name='50s')
    hundreds = runs_per_match[runs_per_match['match_runs'] >= 100].groupby('batsman').size().reset_index(name='100s')

    highest_score = runs_per_match.groupby('batsman')['match_runs'].max().reset_index().rename(columns={'match_runs': 'HS'})
    median_runs = runs_per_match.groupby('batsman')['match_runs'].median().reset_index().rename(columns={'match_runs': 'median'})

    # boundary_runs & running_runs (total runs from boundaries and from running)
    boundary_runs = (d.groupby('batsman').apply(lambda x: int((x['is_four'] * 4).sum() + (x['is_six'] * 6).sum()))
                        .reset_index(name='boundary_runs'))
    running_runs = (d.groupby('batsman').apply(lambda x: int((x['is_one'] * 1).sum() + (x['is_two'] * 2).sum() + (x['is_three'] * 3).sum()))
                        .reset_index(name='running_runs'))

    # ---- merge master batting record ----
    bat_rec = innings_count.merge(total_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(total_balls, on='batsman', how='left')
    bat_rec = bat_rec.merge(dismissals, on='batsman', how='left')
    bat_rec = bat_rec.merge(sixes, on='batsman', how='left')
    bat_rec = bat_rec.merge(fours, on='batsman', how='left')
    bat_rec = bat_rec.merge(dots, on='batsman', how='left')
    bat_rec = bat_rec.merge(ones, on='batsman', how='left')
    bat_rec = bat_rec.merge(twos, on='batsman', how='left')
    bat_rec = bat_rec.merge(threes, on='batsman', how='left')
    bat_rec = bat_rec.merge(boundary_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(running_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(thirties, on='batsman', how='left')
    bat_rec = bat_rec.merge(fifties, on='batsman', how='left')
    bat_rec = bat_rec.merge(hundreds, on='batsman', how='left')
    bat_rec = bat_rec.merge(highest_score, on='batsman', how='left')
    bat_rec = bat_rec.merge(median_runs, on='batsman', how='left')

    # fill NaNs & cast types
    fill_zero_cols = ['30s', '50s', '100s', 'runs', 'balls', 'dismissals', 'sixes', 'fours',
                      'dots', 'ones', 'twos', 'threes', 'boundary_runs', 'running_runs', 'HS', 'median']
    for col in fill_zero_cols:
        if col in bat_rec.columns:
            bat_rec[col] = bat_rec[col].fillna(0)

    int_cols = ['30s', '50s', '100s', 'runs', 'balls', 'dismissals', 'sixes', 'fours',
                'dots', 'ones', 'twos', 'threes', 'boundary_runs', 'running_runs']
    for col in int_cols:
        if col in bat_rec.columns:
            bat_rec[col] = bat_rec[col].astype(int)

    # ---- basic ratios & metrics ----
    bat_rec['RPI'] = bat_rec.apply(lambda x: (x['runs'] / x['innings']) if x['innings'] > 0 else np.nan, axis=1)
    bat_rec['SR'] = bat_rec.apply(lambda x: (x['runs'] / x['balls'] * 100) if x['balls'] > 0 else np.nan, axis=1)
    bat_rec['BPD'] = bat_rec.apply(lambda x: bpd(x['balls'], x['dismissals']), axis=1)
    bat_rec['BPB'] = bat_rec.apply(lambda x: bpb(x['balls'], (x['fours'] + x['sixes'])), axis=1)
    bat_rec['BP6'] = bat_rec.apply(lambda x: bp6(x['balls'], x['sixes']), axis=1)
    bat_rec['BP4'] = bat_rec.apply(lambda x: bp4(x['balls'], x['fours']), axis=1)

    # nbdry_sr: average runs per running delivery (ones/twos/threes)
    def compute_nbdry_sr(row):
        run_count = (row['dots'] * 0 + row['ones'] * 1 + row['twos'] * 2 + row['threes'] * 3)
        denom = (row['dots'] + row['ones'] + row['twos'] + row['threes'])
        return (run_count / denom * 100) if denom > 0 else 0
    bat_rec['nbdry_sr'] = bat_rec.apply(compute_nbdry_sr, axis=1)

    bat_rec['AVG'] = bat_rec.apply(lambda x: avg(x['runs'], x['dismissals'], x['innings']), axis=1)
    bat_rec['dot_percentage'] = bat_rec.apply(lambda x: (x['dots'] / x['balls'] * 100) if x['balls'] > 0 else 0, axis=1)
    bat_rec['Bdry%'] = bat_rec.apply(lambda x: (x['boundary_runs'] / x['runs'] * 100) if x['runs'] > 0 else 0, axis=1)
    bat_rec['Running%'] = bat_rec.apply(lambda x: (x['running_runs'] / x['runs'] * 100) if x['runs'] > 0 else 0, axis=1)

    # ---- latest team for each batsman (based on last match occurrence) ----
    latest_team = (d.sort_values(['match_id', '__ball_sort__'])
                     .drop_duplicates(subset=['batsman'], keep='last')
                     [['batsman', 'batting_team']])
    bat_rec = bat_rec.merge(latest_team, on='batsman', how='left')

    # ---- phase-wise aggregation using per-delivery runs_off_bat & legal_ball ----
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'

    phase_stats = d.groupby(['batsman', 'phase']).agg({
        'runs_off_bat': 'sum',
        'legal_ball': 'sum',
        'is_dot': 'sum',
        'is_four': 'sum',
        'is_six': 'sum',
        'match_id': 'nunique',
        'is_wkt': 'sum'
    }).reset_index()

    phase_stats.rename(columns={
        'runs_off_bat': 'Runs',
        'legal_ball': 'Balls',
        'is_dot': 'Dots',
        'is_four': 'Fours',
        'is_six': 'Sixes',
        'match_id': 'Innings',
        'is_wkt': 'Dismissals'
    }, inplace=True)

    # Derived phase metrics
    phase_stats['BPB'] = phase_stats.apply(lambda x: bpb(x['Balls'], (x['Fours'] + x['Sixes'])), axis=1)
    phase_stats['BPD'] = phase_stats.apply(lambda x: bpd(x['Balls'], x['Dismissals']), axis=1)
    phase_stats['SR'] = phase_stats.apply(lambda x: (x['Runs'] / x['Balls'] * 100) if x['Balls'] > 0 else 0, axis=1)
    phase_stats['AVG'] = phase_stats.apply(lambda x: avg(x['Runs'], x['Dismissals'], x['Innings']), axis=1)
    phase_stats['DOT%'] = phase_stats.apply(lambda x: (x['Dots'] / x['Balls'] * 100) if x['Balls'] > 0 else 0, axis=1)

    # Pivot to wide format: "Phase_Metric" (e.g., Powerplay_SR)
    phase_pivot = phase_stats.pivot(index='batsman', columns='phase',
                                    values=['SR', 'AVG', 'DOT%', 'BPB', 'BPD', 'Innings', 'Runs', 'Balls'])
    if isinstance(phase_pivot.columns, pd.MultiIndex):
        phase_pivot.columns = [f"{col[1]}_{col[0]}" for col in phase_pivot.columns]
    phase_pivot.reset_index(inplace=True)

    # Merge phase metrics into bat_rec
    bat_rec = bat_rec.merge(phase_pivot, on='batsman', how='left')

    # ---- average batting position if lineupOrder exists ----
    if 'lineupOrder' in d.columns:
        avg_pos = d.groupby('batsman')['lineupOrder'].mean().reset_index().rename(columns={'lineupOrder': 'avg_batting_position'})
        bat_rec = bat_rec.merge(avg_pos, on='batsman', how='left')
    else:
        bat_rec['avg_batting_position'] = np.nan

    # final cleanup: drop helper column if present
    if '__ball_sort__' in bat_rec.columns:
        bat_rec.drop(columns=['__ball_sort__'], errors='ignore', inplace=True)

    bat_rec.reset_index(drop=True, inplace=True)
    return bat_rec
import pandas as pd
import numpy as np

def bowlerstat(df):
    """
    Aggregates bowler-level statistics from ball-by-ball DataFrame with columns:
    (p_match, inns, bat, team_bat, bowl, team_bowl, ball_id, outcome, score, out, dismissal,
     over, noball, wide, byes, legbyes, cur_bat_runs, cur_bat_bf, cur_bowl_ovr, cur_bowl_wkts,
     cur_bowl_runs, inns_runs, inns_wkts, inns_balls, ..., batruns, ballfaced, bowlruns, season, ...)
    
    Returns a DataFrame 'bowl_rec' with summary metrics per bowler.
    """

    d = df.copy()

    # --- normalize column names to internal names ---
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'inns' in d.columns:
        d = d.rename(columns={'inns': 'inning'})
    if 'bowl' in d.columns:
        d = d.rename(columns={'bowl': 'bowler'})
    if 'ball_id' in d.columns:
        d = d.rename(columns={'ball_id': 'ball'})
    # per-ball runs off bat (explicit)
    if 'batruns' in d.columns:
        d = d.rename(columns={'batruns': 'batsman_runs'})
    elif 'score' in d.columns:
        d = d.rename(columns={'score': 'batsman_runs'})
    else:
        d['batsman_runs'] = 0
    # per-ball runs charged to bowler (including extras if provided)
    if 'bowlruns' in d.columns:
        d = d.rename(columns={'bowlruns': 'total_runs'})
    else:
        # if not present, approximate total_runs = batsman_runs + byes + legbyes + wides + noballs
        d['byes'] = pd.to_numeric(d.get('byes', 0), errors='coerce').fillna(0).astype(int)
        d['legbyes'] = pd.to_numeric(d.get('legbyes', 0), errors='coerce').fillna(0).astype(int)
        d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
        d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
        d['total_runs'] = pd.to_numeric(d['batsman_runs'], errors='coerce').fillna(0).astype(int) + d['byes'] + d['legbyes'] + d['noball'] + d['wide']

    # --- legal_ball (1 if not wide and not noball) ---
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide']   = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # --- derive bowler wicket flag: bowler_wkt ---
    # Use numeric 'out' if present (0/1). Else use presence of dismissal text.
    if 'out' in d.columns:
        d['bowler_wkt'] = pd.to_numeric(d['out'], errors='coerce').fillna(0).astype(int)
    else:
        d['bowler_wkt'] = d.get('dismissal', pd.Series([None]*len(d))).notna().astype(int)

    # --- per-delivery flags from batsman_runs ---
    d['batsman_runs'] = pd.to_numeric(d.get('batsman_runs', 0), errors='coerce').fillna(0).astype(int)
    d['is_dot']  = ((d['batsman_runs'] == 0) & (d['legal_ball'] == 1)).astype(int)
    d['is_four'] = (d['batsman_runs'] == 4).astype(int)
    d['is_six']  = (d['batsman_runs'] == 6).astype(int)

    # ensure ball ordering
    if 'ball' in d.columns:
        # try numeric conversion; if fails, leave as-is
        try:
            d['__ball_sort__'] = pd.to_numeric(d['ball'], errors='coerce')
        except Exception:
            d['__ball_sort__'] = np.arange(len(d))
    else:
        d['__ball_sort__'] = np.arange(len(d))

    # sort by match and ball so aggregations like maiden detection & mega over detection work reliably
    sort_keys = []
    if 'match_id' in d.columns:
        sort_keys.append('match_id')
    sort_keys.append('__ball_sort__')
    d.sort_values(sort_keys, inplace=True, kind='stable')

    # ---- Basic aggregates (bowler level) ----
    runs = d.groupby('bowler')['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
    innings = d.groupby('bowler')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings'})
    # use legal_ball sum as number of balls bowled
    balls = d.groupby('bowler')['legal_ball'].sum().reset_index().rename(columns={'legal_ball': 'balls'})
    wkts = d.groupby('bowler')['bowler_wkt'].sum().reset_index().rename(columns={'bowler_wkt': 'wkts'})
    dots = d.groupby('bowler')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    fours = d.groupby('bowler')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = d.groupby('bowler')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})

    # ---- 3W hauls & best-bbi (best wickets in a match) ----
    dismissals_count = d.groupby(['bowler', 'match_id'])['bowler_wkt'].sum().reset_index(name='wkts_in_match')
    three_wicket_hauls = dismissals_count[dismissals_count['wkts_in_match'] >= 3].groupby('bowler').size().reset_index(name='three_wicket_hauls')
    bbi = dismissals_count.groupby('bowler')['wkts_in_match'].max().reset_index().rename(columns={'wkts_in_match': 'bbi'})

    # ---- Maiden overs: require over and total_runs (per over total_runs == 0 and 6 legal balls) ----
    # Ensure 'over' numeric
    if 'over' in d.columns:
        try:
            d['over_num'] = pd.to_numeric(d['over'], errors='coerce').fillna(0).astype(int)
        except Exception:
            d['over_num'] = d['over'].astype(str).str.split('.').str[0].astype(int)
    else:
        # try extracting integer part from ball if ball is like 12.3
        try:
            d['over_num'] = d['__ball_sort__'].fillna(0).astype(float).astype(int)
        except Exception:
            d['over_num'] = 0

    over_agg = d.groupby(['bowler', 'match_id', 'over_num']).agg(
        balls_in_over=('legal_ball', 'sum'),
        runs_in_over=('total_runs', 'sum')
    ).reset_index()
    maiden_overs_count = over_agg[(over_agg['balls_in_over'] == 6) & (over_agg['runs_in_over'] == 0)].groupby('bowler').size().reset_index(name='maiden_overs')

    # ---- Phase-wise metrics (Powerplay / Middle1 / Middle2 / Death) ---
    def categorize_phase_local(over):
        try:
            o = float(over)
        except Exception:
            return 'Unknown'
        if o <= 6:
            return 'Powerplay'
        elif 6 < o <= 11:
            return 'Middle1'
        elif 11 < o <= 16:
            return 'Middle2'
        else:
            return 'Death'

    d['phase'] = d['over'].apply(categorize_phase_local) if 'over' in d.columns else 'Unknown'

    phase_group = d.groupby(['bowler', 'phase']).agg(
        phase_balls=('legal_ball', 'sum'),
        phase_runs=('batsman_runs', 'sum'),
        phase_wkts=('bowler_wkt', 'sum'),
        phase_dots=('is_dot', 'sum'),
        phase_innings=('match_id', 'nunique')
    ).reset_index()

    # pivot each metric to columns (safe handling if some phases are missing)
    def pivot_metric(df_pg, metric):
        pivoted = df_pg.pivot(index='bowler', columns='phase', values=metric)
        # ensure expected phase columns exist
        for ph in ['Powerplay', 'Middle1', 'Middle2', 'Death']:
            if ph not in pivoted.columns:
                pivoted[ph] = 0
        # rename columns to include metric prefix
        pivoted = pivoted.rename(columns={ph: f"{metric}_{ph}" for ph in pivoted.columns})
        pivoted = pivoted.reset_index()
        return pivoted

    pb = pivot_metric(phase_group, 'phase_balls')
    pr = pivot_metric(phase_group, 'phase_runs')
    pw = pivot_metric(phase_group, 'phase_wkts')
    pdot = pivot_metric(phase_group, 'phase_dots')
    pi = pivot_metric(phase_group, 'phase_innings')

    # Merge phase pivots (start with pb)
    phase_df = pb.merge(pr, on='bowler', how='outer').merge(pw, on='bowler', how='outer') \
                 .merge(pdot, on='bowler', how='outer').merge(pi, on='bowler', how='outer')
    phase_df = phase_df.fillna(0)

    # derive phase-level rate metrics
    for ph in ['Powerplay', 'Middle1', 'Middle2', 'Death']:
        balls_col = f"phase_balls_{ph}"
        runs_col = f"phase_runs_{ph}"
        wkts_col = f"phase_wkts_{ph}"
        dots_col = f"phase_dots_{ph}"

        # avoid divide-by-zero by using replace
        phase_df[f'{ph}_sr'] = phase_df.apply(lambda r: (r[balls_col] / r[wkts_col]) if r[wkts_col] > 0 else np.nan, axis=1)  # balls per wicket (SR)
        phase_df[f'{ph}_econ'] = phase_df.apply(lambda r: (r[runs_col] * 6.0 / r[balls_col]) if r[balls_col] > 0 else np.nan, axis=1)
        phase_df[f'{ph}_avg'] = phase_df.apply(lambda r: (r[runs_col] / r[wkts_col]) if r[wkts_col] > 0 else np.nan, axis=1)
        phase_df[f'{ph}_dot%'] = phase_df.apply(lambda r: (r[dots_col] / r[balls_col] * 100) if r[balls_col] > 0 else 0.0, axis=1)

        # also rename base columns to consistent naming
        phase_df.rename(columns={
            balls_col: f'{ph}_balls', runs_col: f'{ph}_runs', wkts_col: f'{ph}_wkts', dots_col: f'{ph}_dots'
        }, inplace=True)

    # ---- Mega Over (consecutive over continuation by same bowler) detection ----
    # We consider a "mega over" when the same bowler bowls at the start of consecutive overs (i.e., over.x -> over+1.1 by same bowler)
    # We'll look for sequences where decimal part indicates first legal delivery of an over (e.g., ball 12.1) with same bowler as previous row.
    df_sorted = d.sort_values(['match_id', '__ball_sort__']).reset_index(drop=True).copy()
    # Prepare string version of ball to extract fractional part if present
    df_sorted['ball_str'] = df_sorted['ball'].astype(str)
    df_sorted['frac'] = df_sorted['ball_str'].str.split('.').str[1].fillna('0')
    df_sorted['frac_int'] = pd.to_numeric(df_sorted['frac'], errors='coerce').fillna(0).astype(int)
    df_sorted['prev_bowler'] = df_sorted['bowler'].shift(1)
    df_sorted['prev_match'] = df_sorted['match_id'].shift(1)
    df_sorted['prev_bowler_same'] = (df_sorted['prev_bowler'] == df_sorted['bowler']) & (df_sorted['prev_match'] == df_sorted['match_id'])
    # we treat frac_int == 1 as start of over (ball x.1), detect when that occurs while prev_bowler_same is True
    df_sorted['Mega_Over'] = (df_sorted['frac_int'] == 1) & (df_sorted['prev_bowler_same'])
    mega_over_count = df_sorted[df_sorted['Mega_Over']].groupby('bowler').size().reset_index(name='Mega_Over_Count')

    # ---- merge all components into bowl_rec ----
    # Start merging base components; use outer merges to avoid losing bowlers
    bowl_rec = innings.merge(balls, on='bowler', how='outer') \
                      .merge(runs, on='bowler', how='outer') \
                      .merge(wkts, on='bowler', how='outer') \
                      .merge(sixes, on='bowler', how='outer') \
                      .merge(fours, on='bowler', how='outer') \
                      .merge(dots, on='bowler', how='outer') \
                      .merge(three_wicket_hauls, on='bowler', how='left') \
                      .merge(maiden_overs_count, on='bowler', how='left') \
                      .merge(bbi, on='bowler', how='left') \
                      .merge(phase_df, on='bowler', how='left') \
                      .merge(mega_over_count, on='bowler', how='left')

    # fill NaNs
    for c in ['three_wicket_hauls', 'maiden_overs', 'Mega_Over_Count', 'bbi']:
        if c in bowl_rec.columns:
            bowl_rec[c] = bowl_rec[c].fillna(0).astype(int)

    # debut/final seasons if 'season' exists
    if 'season' in d.columns:
        debut_final = d.groupby('bowler')['season'].agg(debut_year='min', final_year='max').reset_index()
        bowl_rec = bowl_rec.merge(debut_final, on='bowler', how='left')
    else:
        bowl_rec['debut_year'] = np.nan
        bowl_rec['final_year'] = np.nan

    # Fill numeric columns defaults and cast types
    numeric_defaults = ['balls', 'runs', 'wkts', 'sixes', 'fours', 'dots']
    for col in numeric_defaults:
        if col in bowl_rec.columns:
            bowl_rec[col] = pd.to_numeric(bowl_rec[col], errors='coerce').fillna(0)

    # rate metrics
    bowl_rec['dot%'] = bowl_rec.apply(lambda r: (r['dots'] / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['avg'] = bowl_rec.apply(lambda r: (r['runs'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['sr'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['econ'] = bowl_rec.apply(lambda r: (r['runs'] * 6.0 / r['balls']) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['WPI'] = bowl_rec.apply(lambda r: (r['wkts'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)
    bowl_rec['DPI'] = bowl_rec.apply(lambda r: (r['dots'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)
    bowl_rec['RPI'] = bowl_rec.apply(lambda r: (r['runs'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)

    # boundary percentage (how many legal balls conceded a boundary)
    bowl_rec['bdry%'] = bowl_rec.apply(lambda r: ((r.get('fours',0) + r.get('sixes',0)) / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)

    # BPB, BPD, BP6
    bowl_rec['BPB'] = bowl_rec.apply(lambda r: (r['balls'] / (r.get('fours',0) + r.get('sixes',0))) if (r.get('fours',0) + r.get('sixes',0)) > 0 else np.nan, axis=1)
    bowl_rec['BPD'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['BP6'] = bowl_rec.apply(lambda r: (r['balls'] / r['sixes']) if r['sixes'] > 0 else np.nan, axis=1)

    # Over-wise runs (ten-run overs, 7+ overs, 6- overs)
    overwise_runs = d.groupby(['bowler', 'match_id', 'over_num'])['total_runs'].sum().reset_index()
    ten_run_overs = overwise_runs[overwise_runs['total_runs'] >= 10].groupby('bowler').size().reset_index(name='10_run_overs')
    seven_plus_overs = overwise_runs[overwise_runs['total_runs'] >= 7].groupby('bowler').size().reset_index(name='7_plus_run_overs')
    six_minus_overs = overwise_runs[overwise_runs['total_runs'] <= 6].groupby('bowler').size().reset_index(name='6_minus_run_overs')

    bowl_rec = bowl_rec.merge(ten_run_overs, on='bowler', how='left')
    bowl_rec = bowl_rec.merge(seven_plus_overs, on='bowler', how='left')
    bowl_rec = bowl_rec.merge(six_minus_overs, on='bowler', how='left')

    for col in ['10_run_overs', '7_plus_run_overs', '6_minus_run_overs']:
        if col in bowl_rec.columns:
            bowl_rec[col] = bowl_rec[col].fillna(0).astype(int)

    # Total overs bowled string (6-ball standard)
    bowl_rec['overs'] = bowl_rec['balls'].apply(lambda x: f"{int(x // 6)}.{int(x % 6)}" if pd.notna(x) else "0.0")

    # Ensure no accidental zero-string bowler rows
    bowl_rec = bowl_rec[bowl_rec['bowler'].notna()]

    # Reset index and return
    bowl_rec.reset_index(drop=True, inplace=True)
    return bowl_rec
# -------------------
# Player Profile -> Batting
# -------------------
if option == "Batting":
    # Defensive column handling: idf should already have 'batsman'
    if 'batsman' not in idf.columns:
        if 'bat' in idf.columns:
            idf = idf.rename(columns={'bat': 'batsman'})
        else:
            st.error("idf missing 'batsman' column. Ensure Custom(df) produced per-batsman summary.")
            st.stop()

    player_stats = idf[idf['batsman'] == player_name].copy()
    # Drop final_year if present (safe)
    player_stats = player_stats.drop(columns=['final_year'], errors='ignore')

    # Standardize display column names
    player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]

    # Round floats (use your helper)
    try:
        player_stats = round_up_floats(player_stats)
    except Exception:
        float_cols = player_stats.select_dtypes(include=['float64', 'float32']).columns
        player_stats[float_cols] = player_stats[float_cols].round(2)

    st.markdown("### Batting Statistics")
    columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
    # Fill & cast sensible integer columns if present
    for c in columns_to_convert:
        if c in player_stats.columns:
            player_stats[c] = player_stats[c].fillna(0).astype(int)

    st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'"))

    # ------------------------
    # Opponent-wise performance (team_bowl / team_bowl)
    # ------------------------
    # Defensive detection of raw-data column names
    bat_col = 'batsman' if 'batsman' in pdf.columns else ('bat' if 'bat' in pdf.columns else None)
    opp_col = 'team_bowl' if 'team_bowl' in pdf.columns else ('team_bowl' if 'team_bowl' in pdf.columns else ('team_bowl' if 'team_bowl' in pdf.columns else 'team_bowl'))
    # prefer 'team_bowl' per your schema; fall back to 'team_bowl' string (keeps consistent)
    if bat_col is None:
        st.error("Raw dataframe missing batter column ('bat' or 'batsman').")
        st.stop()

    # get list of opponents (unique bowling teams where this batter faced)
    opponents = sorted(pdf[pdf[bat_col] == player_name][ 'team_bowl' ].dropna().unique().tolist())

    opponent_df_list = []
    for opp in opponents:
        # filter raw PDF for this player vs current opponent
        temp_df = pdf[(pdf[bat_col] == player_name) & (pdf['team_bowl'] == opp)].copy()
        # Normalize names expected by cumulator: match_id, inning, batsman
        rename_map = {}
        if 'p_match' in temp_df.columns and 'match_id' not in temp_df.columns:
            rename_map['p_match'] = 'match_id'
        if 'inns' in temp_df.columns and 'inning' not in temp_df.columns:
            rename_map['inns'] = 'inning'
        if 'bat' in temp_df.columns and 'batsman' not in temp_df.columns:
            rename_map['bat'] = 'batsman'
        if 'batruns' in temp_df.columns and 'batsman_runs' not in temp_df.columns:
            rename_map['batruns'] = 'batsman_runs'
        temp_df = temp_df.rename(columns=rename_map)

        # If temp_df empty after filtering, skip
        if temp_df.empty:
            continue

        # Apply cumulator (expects per-ball df with cumulative or per-ball columns)
        try:
            temp_summary = cumulator(temp_df)
        except Exception as e:
            st.error(f"cumulator failed for opponent {opp}: {e}")
            continue

        if temp_summary.empty:
            continue

        temp_summary['OPPONENT'] = opp
        # place opponent as first column
        cols = temp_summary.columns.tolist()
        cols = ['OPPONENT'] + [c for c in cols if c != 'OPPONENT']
        temp_summary = temp_summary[cols]
        opponent_df_list.append(temp_summary)

    # combine and clean
    if opponent_df_list:
        result_df = pd.concat(opponent_df_list, ignore_index=True).drop(columns=['batsman','debut_year','final_year'], errors='ignore')
        # uppercase col names for display
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
        conv_cols = ['HUNDREDS', 'FIFTIES', 'THIRTIES', 'RUNS', 'HIGHEST SCORE']
        for c in conv_cols:
            if c in result_df.columns:
                result_df[c] = result_df[c].fillna(0).astype(int)
        # Round floats
        try:
            result_df = round_up_floats(result_df)
        except Exception:
            float_cols = result_df.select_dtypes(include=['float']).columns
            result_df[float_cols] = result_df[float_cols].round(2)

        # Reorder so opponent & matches appear first if present
        cols = result_df.columns.tolist()
        new_order = []
        if 'OPPONENT' in cols:
            new_order.append('OPPONENT')
        if 'MATCHES' in cols:
            new_order.append('MATCHES')
        new_order += [c for c in cols if c not in new_order]
        result_df = result_df[new_order]

        st.markdown("### Opponent-wise Performance")
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
    else:
        st.markdown("### Opponent-wise Performance")
        st.write("No opponent-wise data available for this player.")

    # ------------------------
    # Season-wise performance (use 'year' column in raw df)
    # ------------------------
    # find seasons where player has data
    if 'year' in pdf.columns:
        seasons = sorted(pdf[pdf[bat_col] == player_name]['year'].dropna().unique().tolist())
    else:
        seasons = []

    season_list = []
    for season in seasons:
        temp_df = pdf[(pdf[bat_col] == player_name) & (pdf['year'] == season)].copy()
        # normalize names for cumulator
        rename_map = {}
        if 'p_match' in temp_df.columns and 'match_id' not in temp_df.columns:
            rename_map['p_match'] = 'match_id'
        if 'inns' in temp_df.columns and 'inning' not in temp_df.columns:
            rename_map['inns'] = 'inning'
        if 'bat' in temp_df.columns and 'batsman' not in temp_df.columns:
            rename_map['bat'] = 'batsman'
        if 'batruns' in temp_df.columns and 'batsman_runs' not in temp_df.columns:
            rename_map['batruns'] = 'batsman_runs'
        temp_df = temp_df.rename(columns=rename_map)

        if temp_df.empty:
            continue

        try:
            temp_summary = cumulator(temp_df)
        except Exception as e:
            st.warning(f"cumulator failed for season {season}: {e}")
            continue

        temp_summary['YEAR'] = season
        season_list.append(temp_summary)

    if season_list:
        result_seasons = pd.concat(season_list, ignore_index=True).drop(columns=['batsman','debut_year','final_year'], errors='ignore')
        # rename columns for display
        result_seasons.columns = [col.upper().replace('_', ' ') for col in result_seasons.columns]
        # ensure integer columns
        conv_cols = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
        for c in conv_cols:
            if c in result_seasons.columns:
                result_seasons[c] = result_seasons[c].fillna(0).astype(int)
        # Round floats
        try:
            result_seasons = round_up_floats(result_seasons)
        except Exception:
            float_cols = result_seasons.select_dtypes(include=['float']).columns
            result_seasons[float_cols] = result_seasons[float_cols].round(2)

        # Reorder columns to show YEAR first if present
        cols = result_seasons.columns.tolist()
        new_order = ['YEAR']
        if 'MATCHES' in cols:
            new_order.append('MATCHES')
        new_order += [c for c in cols if c not in new_order]
        result_seasons = result_seasons[new_order]

        st.markdown("### Year-wise Performance")
        st.table(result_seasons.style.set_table_attributes("style='font-weight: bold;'"))
    else:
        st.markdown("### Year-wise Performance")
        st.write("No season-wise data available for this player.")

    # ------------------------
    # Inning-wise performance (IPL mostly has innings 1 & 2)
    # ------------------------
    # Determine inning column name in raw df
    inning_col = 'inning' if 'inning' in pdf.columns else ('inns' if 'inns' in pdf.columns else None)
    if inning_col is None:
        st.write("No inning column (inns/inning) found in raw data; skipping inning-wise breakdown.")
    else:
        tdf = pdf[pdf[bat_col] == player_name].copy()
        inning_results = []
        for inn in [1, 2]:  # IPL typically 1 or 2
            temp = tdf[tdf[inning_col] == inn].copy()
            # normalize and run cumulator
            rename_map = {}
            if 'p_match' in temp.columns and 'match_id' not in temp.columns:
                rename_map['p_match'] = 'match_id'
            if inning_col == 'inns' and 'inning' not in temp.columns:
                rename_map['inns'] = 'inning'
            if 'bat' in temp.columns and 'batsman' not in temp.columns:
                rename_map['bat'] = 'batsman'
            if 'batruns' in temp.columns and 'batsman_runs' not in temp.columns:
                rename_map['batruns'] = 'batsman_runs'
            temp = temp.rename(columns=rename_map)

            if temp.empty:
                continue

            try:
                s = cumulator(temp)
            except Exception as e:
                st.warning(f"cumulator failed for inning {inn}: {e}")
                continue

            s['INNING'] = inn
            inning_results.append(s)

        if inning_results:
            final_innings = pd.concat(inning_results, ignore_index=True).drop(columns=['batsman','debut_year','final_year'], errors='ignore')
            final_innings.columns = [col.upper().replace('_', ' ') for col in final_innings.columns]
            # ensure numeric columns
            conv_cols = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
            for c in conv_cols:
                if c in final_innings.columns:
                    final_innings[c] = final_innings[c].fillna(0).astype(int)
            final_innings = round_up_floats(final_innings)
            final_innings = final_innings.drop(columns=['MATCHES'], errors='ignore')
            st.markdown("### Inning-wise Performance")
            st.table(final_innings.reset_index(drop=True).style.set_table_attributes("style='font-weight: bold;'"))
        else:
            st.markdown("### Inning-wise Performance")
            st.write("No inning-wise data available for this player.")






