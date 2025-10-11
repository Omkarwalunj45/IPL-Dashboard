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
    path = "Dataset/ipl_bbb_21_25.xlsx"
    df = pd.read_excel(path)
    return df

df = load_data()    
def rename_rcb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames 'Royal Challengers Bangalore' to 'Royal Challengers Bengaluru' in team_bat, team_bowl, and winner columns.
    Returns the modified DataFrame.
    """
    d = df.copy()
    
    # List of columns to check and rename
    columns_to_rename = ['team_bat', 'team_bowl', 'winner']
    
    # Rename 'Royal Challengers Bangalore' to 'Royal Challengers Bengaluru' in specified columns
    for col in columns_to_rename:
        if col in d.columns:
            d[col] = d[col].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    
    return d
df = rename_rcb(df)
df['is_wicket'] = df['out'].astype(int)


# -----------------------
# Utility helpers
# -----------------------
def safe_get_col(df: pd.DataFrame, choices, default=None):
    for c in choices:
        if c in df.columns:
            return c
    return default

def round_up_floats(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].round(decimals)
    return df
def bpd(balls, dismissals):
    return balls / dismissals if dismissals > 0 else np.nan

def bpb(balls, boundaries):
    return balls / boundaries if boundaries > 0 else np.nan

def bp6(balls, sixes):
    return balls / sixes if sixes > 0 else np.nan

def bp4(balls, fours):
    return balls / fours if fours > 0 else np.nan

def avg(runs, dismissals, innings):
    return runs / dismissals if dismissals > 0 else np.nan

def categorize_phase(over):
    if over < 6:
        return 'Powerplay'
    elif over < 15:
        return 'Middle'
    else:
        return 'Death'

def bpd(balls, dismissals):
    return balls / dismissals if dismissals > 0 else np.nan

def bpb(balls, boundaries):
    return balls / boundaries if boundaries > 0 else np.nan

def bp6(balls, sixes):
    return balls / sixes if sixes > 0 else np.nan

def bp4(balls, fours):
    return balls / fours if fours > 0 else np.nan

def avg(runs, dismissals, innings):
    return runs / dismissals if dismissals > 0 else np.nan

def categorize_phase(over):
    if over < 6:
        return 'Powerplay'
    elif over < 15:
        return 'Middle'
    else:
        return 'Death'

def cumulator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batting summary builder implementing:
      - legal_ball: both wide & noball == 0
      - 50s counted only if innings score >=50 and <100
      - Dismissal logic: non-striker is last different batsman in same inns and p_match with out=False
      - No dismissal attribution if non-striker invalid or already dismissed
    Returns bat_rec (one row per batsman).
    """
    if df is None:
        return pd.DataFrame()
    d = df.copy()

    # ---- normalize column names ----
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'inns' in d.columns and 'inning' not in d.columns:
        d = d.rename(columns={'inns': 'inning'})
    if 'bat' in d.columns and 'batsman' not in d.columns:
        d = d.rename(columns={'bat': 'batsman'})
    if 'team_bat' in d.columns and 'batting_team' not in d.columns:
        d = d.rename(columns={'team_bat': 'batting_team'})
    if 'batruns' in d.columns and 'runs_off_bat' not in d.columns:
        d = d.rename(columns={'batruns': 'runs_off_bat'})
    elif 'score' in d.columns and 'runs_off_bat' not in d.columns:
        d = d.rename(columns={'score': 'runs_off_bat'})
    elif 'runs_off_bat' not in d.columns:
        d['runs_off_bat'] = 0

    # ensure stable RangeIndex
    d.index = pd.RangeIndex(len(d))

    # ---- safe ball ordering ----
    if 'ball_id' in d.columns:
        tmp = pd.to_numeric(d['ball_id'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    else:
        d['__ball_sort__'] = pd.Series(np.arange(len(d)), index=d.index)

    # ---- dismissal normalization & flags ----
    special_runout_types = set(['run out', 'obstructing the field', 'retired out', 'retired not out (hurt)'])
    d['dismissal_clean'] = d.get('dismissal', '').astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan': '', 'none': ''})

    # p_bat and p_out are integers, handle missing as NaN
    d['p_bat_num'] = d.get('p_bat', pd.Series(np.nan, index=d.index)).astype(float)
    d['p_out_num'] = d.get('p_out', pd.Series(np.nan, index=d.index)).astype(float)

    # out is boolean (True/False), convert to 0/1
    d['out_flag'] = d.get('out', False).astype(int)

    # ensure match_id exists
    if 'match_id' not in d.columns:
        d['match_id'] = 0

    # sort by match, inning, and ball order
    d.sort_values(['match_id', 'inning', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # initialize dismissal outputs
    d['dismissed_player'] = None
    d['bowler_wkt'] = 0

    # resolve dismissals per specified logic
    for match in d['match_id'].unique():
        for inning in d[d['match_id'] == match]['inning'].unique():
            idxs = d[(d['match_id'] == match) & (d['inning'] == inning)].index.tolist()
            idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
            for pos, i in enumerate(idxs):
                if d.at[i, 'out_flag'] != 1:  # Check if out=True
                    continue

                disc = (d.at[i, 'dismissal_clean'] or '').strip()
                striker = d.at[i, 'batsman'] if 'batsman' in d.columns else None

                # Rule 1: out=True and dismissal not in [blank, nan, special] -> striker out, bowler credit
                if disc and disc not in special_runout_types:
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 1
                    continue

                # Rule 2: out=True and dismissal in special
                if disc in special_runout_types:
                    pbat = d.at[i, 'p_bat_num']
                    pout = d.at[i, 'p_out_num']

                    if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                        d.at[i, 'dismissed_player'] = striker
                        d.at[i, 'bowler_wkt'] = 0
                        continue

                    # Nonstriker dismissed: find last different batter in same match and inning
                    nonstriker = None
                    last_idx_of_nonstriker = None
                    for j in reversed(idxs[:pos]):
                        prev_bat = d.at[j, 'batsman'] if 'batsman' in d.columns else None
                        if prev_bat is not None and prev_bat != striker:
                            nonstriker = prev_bat
                            last_idx_of_nonstriker = j
                            break

                    if nonstriker is None:
                        # No valid nonstriker: do not attribute dismissal
                        d.at[i, 'dismissed_player'] = None
                        d.at[i, 'bowler_wkt'] = 0
                        continue

                    prev_out_flag = d.at[last_idx_of_nonstriker, 'out_flag'] if last_idx_of_nonstriker is not None else 0
                    if prev_out_flag == 0:
                        d.at[i, 'dismissed_player'] = nonstriker
                        d.at[i, 'bowler_wkt'] = 0
                    else:
                        # Nonstriker already dismissed: do not attribute dismissal
                        d.at[i, 'dismissed_player'] = None
                        d.at[i, 'bowler_wkt'] = 0

    # ---- compute per-delivery summaries ----
    d['cur_bat_runs'] = pd.to_numeric(d.get('cur_bat_runs', 0), errors='coerce').fillna(0).astype(int)
    d['cur_bat_bf'] = pd.to_numeric(d.get('cur_bat_bf', 0), errors='coerce').fillna(0).astype(int)

    # legal ball: both wide & noball must be 0
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # per-delivery run flags
    d['runs_off_bat'] = pd.to_numeric(d.get('runs_off_bat', 0), errors='coerce').fillna(0).astype(int)
    d['is_dot'] = ((d['runs_off_bat'] == 0) & (d['legal_ball'] == 1)).astype(int)
    d['is_one'] = (d['runs_off_bat'] == 1).astype(int)
    d['is_two'] = (d['runs_off_bat'] == 2).astype(int)
    d['is_three'] = (d['runs_off_bat'] == 3).astype(int)
    d['is_four'] = (d['runs_off_bat'] == 4).astype(int)
    d['is_six'] = (d['runs_off_bat'] == 6).astype(int)

    # last snapshot per batsman per match
    last_bat_snapshot = (
        d.groupby(['batsman', 'match_id'], sort=False, as_index=False)
         .agg({'cur_bat_runs': 'last', 'cur_bat_bf': 'last'})
         .rename(columns={'cur_bat_runs': 'match_runs', 'cur_bat_bf': 'match_balls'})
    )

    runs_per_match = last_bat_snapshot[['batsman', 'match_runs', 'match_balls', 'match_id']].copy()
    innings_count = runs_per_match.groupby('batsman')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings'})
    total_runs = runs_per_match.groupby('batsman')['match_runs'].sum().reset_index().rename(columns={'match_runs': 'runs'})
    total_balls = runs_per_match.groupby('batsman')['match_balls'].sum().reset_index().rename(columns={'match_balls': 'balls'})

    # dismissals: count per resolved dismissed_player, ensuring unique dismissals
    dismissals_df = d[d['dismissed_player'].notna()].groupby(['dismissed_player', 'match_id', 'inning']).size().reset_index(name='dismissals')
    dismissals_df = dismissals_df.groupby('dismissed_player')['dismissals'].sum().reset_index().rename(columns={'dismissed_player': 'batsman'})

    # boundary & running counts
    fours = d.groupby('batsman')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = d.groupby('batsman')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
    dots = d.groupby('batsman')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    ones = d.groupby('batsman')['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
    twos = d.groupby('batsman')['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
    threes = d.groupby('batsman')['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})

    # match-level thresholds: 30s (30-49), 50s (50-99), 100s (>=100)
    thirties = runs_per_match[(runs_per_match['match_runs'] >= 30) & (runs_per_match['match_runs'] < 50)].groupby('batsman').size().reset_index(name='30s')
    fifties = runs_per_match[(runs_per_match['match_runs'] >= 50) & (runs_per_match['match_runs'] < 100)].groupby('batsman').size().reset_index(name='50s')
    hundreds = runs_per_match[runs_per_match['match_runs'] >= 100].groupby('batsman').size().reset_index(name='100s')

    highest_score = runs_per_match.groupby('batsman')['match_runs'].max().reset_index().rename(columns={'match_runs': 'HS'})
    median_runs = runs_per_match.groupby('batsman')['match_runs'].median().reset_index().rename(columns={'match_runs': 'median'})

    boundary_runs = (d.groupby('batsman').apply(lambda x: int((x['is_four'] * 4).sum() + (x['is_six'] * 6).sum()))
                     .reset_index(name='boundary_runs'))
    running_runs = (d.groupby('batsman').apply(lambda x: int((x['is_one'] * 1).sum() + (x['is_two'] * 2).sum() + (x['is_three'] * 3).sum()))
                    .reset_index(name='running_runs'))

    # Merge master batting record
    bat_rec = innings_count.merge(total_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(total_balls, on='batsman', how='left')
    bat_rec = bat_rec.merge(dismissals_df, on='batsman', how='left')
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

    # basic ratios & metrics
    bat_rec['RPI'] = bat_rec.apply(lambda x: (x['runs'] / x['innings']) if x['innings'] > 0 else np.nan, axis=1)
    bat_rec['SR'] = bat_rec.apply(lambda x: (x['runs'] / x['balls'] * 100) if x['balls'] > 0 else np.nan, axis=1)
    bat_rec['BPD'] = bat_rec.apply(lambda x: bpd(x['balls'], x['dismissals']), axis=1)
    bat_rec['BPB'] = bat_rec.apply(lambda x: bpb(x['balls'], (x.get('fours', 0) + x.get('sixes', 0))), axis=1)
    bat_rec['BP6'] = bat_rec.apply(lambda x: bp6(x['balls'], x.get('sixes', 0)), axis=1)
    bat_rec['BP4'] = bat_rec.apply(lambda x: bp4(x['balls'], x.get('fours', 0)), axis=1)

    def compute_nbdry_sr(row):
        run_count = (row.get('dots', 0) * 0 + row.get('ones', 0) * 1 + row.get('twos', 0) * 2 + row.get('threes', 0) * 3)
        denom = (row.get('dots', 0) + row.get('ones', 0) + row.get('twos', 0) + row.get('threes', 0))
        return (run_count / denom * 100) if denom > 0 else 0
    bat_rec['nbdry_sr'] = bat_rec.apply(compute_nbdry_sr, axis=1)

    bat_rec['AVG'] = bat_rec.apply(lambda x: avg(x['runs'], x.get('dismissals', 0), x['innings']), axis=1)
    bat_rec['dot_percentage'] = bat_rec.apply(lambda x: (x['dots'] / x['balls'] * 100) if x['balls'] > 0 else 0, axis=1)
    bat_rec['Bdry%'] = bat_rec.apply(lambda x: (x['boundary_runs'] / x['runs'] * 100) if x['runs'] > 0 else 0, axis=1)
    bat_rec['Running%'] = bat_rec.apply(lambda x: (x['running_runs'] / x['runs'] * 100) if x['runs'] > 0 else 0, axis=1)

    # latest team
    if 'batting_team' in d.columns:
        latest_team = (d.sort_values(['match_id', 'inning', '__ball_sort__'])
                       .drop_duplicates(subset=['batsman'], keep='last')
                       [['batsman', 'batting_team']])
        bat_rec = bat_rec.merge(latest_team, on='batsman', how='left')
    else:
        bat_rec['batting_team'] = np.nan

    # phase-wise aggregation
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
    }).reset_index()

    phase_stats.rename(columns={
        'runs_off_bat': 'Runs',
        'legal_ball': 'Balls',
        'is_dot': 'Dots',
        'is_four': 'Fours',
        'is_six': 'Sixes',
        'match_id': 'Innings'
    }, inplace=True)

    # Add phase dismissals
    phase_dismissals = d[d['dismissed_player'].notna()].groupby(['dismissed_player', 'phase', 'match_id', 'inning']).size().reset_index(name='Dismissals')
    phase_dismissals = phase_dismissals.groupby(['dismissed_player', 'phase'])['Dismissals'].sum().reset_index()
    phase_dismissals.rename(columns={'dismissed_player': 'batsman'}, inplace=True)
    phase_stats = phase_stats.merge(phase_dismissals, on=['batsman', 'phase'], how='left')
    phase_stats['Dismissals'] = phase_stats['Dismissals'].fillna(0).astype(int)

    phase_stats['BPB'] = phase_stats.apply(lambda x: bpb(x['Balls'], (x['Fours'] + x['Sixes'])), axis=1)
    phase_stats['BPD'] = phase_stats.apply(lambda x: bpd(x['Balls'], x['Dismissals']), axis=1)
    phase_stats['SR'] = phase_stats.apply(lambda x: (x['Runs'] / x['Balls'] * 100) if x['Balls'] > 0 else 0, axis=1)
    phase_stats['AVG'] = phase_stats.apply(lambda x: avg(x['Runs'], x['Dismissals'], x['Innings']), axis=1)
    phase_stats['DOT%'] = phase_stats.apply(lambda x: (x['Dots'] / x['Balls'] * 100) if x['Balls'] > 0 else 0, axis=1)

    if not phase_stats.empty:
        phase_pivot = phase_stats.pivot(index='batsman', columns='phase',
                                        values=['SR', 'AVG', 'DOT%', 'BPB', 'BPD', 'Innings', 'Runs', 'Balls'])
        if isinstance(phase_pivot.columns, pd.MultiIndex):
            phase_pivot.columns = [f"{col[1]}_{col[0]}" for col in phase_pivot.columns]
        phase_pivot.reset_index(inplace=True)
    else:
        phase_pivot = pd.DataFrame({'batsman': []})

    bat_rec = bat_rec.merge(phase_pivot, on='batsman', how='left')

    bat_rec.reset_index(drop=True, inplace=True)
    return bat_rec
# bowlerstat - bowling summary with exact dismissal rules
# -------------------------
def bowlerstat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build bowling summary and apply the same dismissal resolution rules as Custom.
      - legal_ball requires both wide and noball be zero
      - bowler wicket credit only when dismissal is non-special (i.e., caught, bowled, lbw, stumped etc.)
    Returns bowl_rec DataFrame.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    # normalize names
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'bowl' not in d.columns:
        d['bowl'] = None
    # prefer ball_id or ball order
    if 'ball_id' in d.columns:
        tmp = pd.to_numeric(d['ball_id'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    elif 'ball' in d.columns:
        tmp = pd.to_numeric(d['ball'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    else:
        d['__ball_sort__'] = pd.Series(np.arange(len(d)), index=d.index)

    # legal_ball
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # batsman_runs (batruns or score)
    if 'batruns' in d.columns:
        d['batsman_runs'] = pd.to_numeric(d['batruns'], errors='coerce').fillna(0).astype(int)
    elif 'score' in d.columns:
        d['batsman_runs'] = pd.to_numeric(d['score'], errors='coerce').fillna(0).astype(int)
    else:
        d['batsman_runs'] = 0

    # dismissal normalization (same rules)
    d['dismissal_clean'] = d.get('dismissal', pd.Series([None]*len(d))).astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan':'', 'none':''})
    special_runout_types = set([
        'run out', 'runout',
        'obstructing the field', 'obstructing thefield', 'obstructing',
        'retired out', 'retired not out (hurt)', 'retired not out', 'retired'
    ])

    d['p_bat_num'] = pd.to_numeric(d.get('p_bat', np.nan), errors='coerce')
    d['p_out_num'] = pd.to_numeric(d.get('p_out', np.nan), errors='coerce')
    d['out_flag'] = pd.to_numeric(d.get('out', 0), errors='coerce').fillna(0).astype(int)

    if 'match_id' not in d.columns:
        d['match_id'] = 0
    d.sort_values(['match_id', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # initialize resolved fields
    d['dismissed_player'] = None
    d['bowler_wkt'] = 0

    # resolve dismissals per match using same exact logic
    for m in d['match_id'].unique():
        idxs = d.index[d['match_id'] == m].tolist()
        idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
        for pos, i in enumerate(idxs):
            out_flag = int(d.at[i, 'out_flag']) if not pd.isna(d.at[i, 'out_flag']) else 0
            disc = (d.at[i, 'dismissal_clean'] or '').strip()
            striker = d.at[i, 'bat'] if 'bat' in d.columns else None
            # If out_flag True:
            if out_flag == 1:
                if disc and (disc not in special_runout_types):
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 1
                    continue
                pbat = d.at[i, 'p_bat_num']
                pout = d.at[i, 'p_out_num']
                if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
                    continue
                # find nonstriker
                nonstriker = None
                last_idx_of_nonstriker = None
                for j in reversed(idxs[:pos]):
                    prev_bat = d.at[j, 'bat'] if 'bat' in d.columns else None
                    if prev_bat is not None and prev_bat != striker:
                        nonstriker = prev_bat
                        last_idx_of_nonstriker = j
                        break
                if nonstriker is None:
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
                    continue
                prev_out_flag = int(d.at[last_idx_of_nonstriker, 'out_flag']) if last_idx_of_nonstriker is not None else 0
                if prev_out_flag == 0:
                    d.at[i, 'dismissed_player'] = nonstriker
                    d.at[i, 'bowler_wkt'] = 0
                else:
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
            else:
                # If out==0 => mark striker dismissed per instruction
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0

    # Now aggregate bowler-level stats using bowler_wkt
    runs = d.groupby('bowl')['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs', 'bowl':'bowler'})
    innings = d.groupby('bowl')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings', 'bowl':'bowler'})
    balls = d.groupby('bowl')['legal_ball'].sum().reset_index().rename(columns={'legal_ball': 'balls', 'bowl':'bowler'})
    wkts = d.groupby('bowl')['bowler_wkt'].sum().reset_index().rename(columns={'bowler_wkt': 'wkts', 'bowl':'bowler'})
    dots = d.groupby('bowl')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots', 'bowl':'bowler'})
    fours = d.groupby('bowl')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours', 'bowl':'bowler'})
    sixes = d.groupby('bowl')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes', 'bowl':'bowler'})

    dismissals_count = d.groupby(['bowl', 'match_id'])['bowler_wkt'].sum().reset_index(name='wkts_in_match')
    three_wicket_hauls = dismissals_count[dismissals_count['wkts_in_match'] >= 3].groupby('bowl').size().reset_index(name='three_wicket_hauls').rename(columns={'bowl':'bowler'})
    bbi = dismissals_count.groupby('bowl')['wkts_in_match'].max().reset_index().rename(columns={'wkts_in_match': 'bbi', 'bowl':'bowler'})

    # over_num extraction
    if 'over' in d.columns:
        try:
            d['over_num'] = pd.to_numeric(d['over'], errors='coerce').fillna(0).astype(int)
        except Exception:
            d['over_num'] = d['over'].astype(str).str.split('.').str[0].astype(int)
    else:
        d['over_num'] = 0

    over_agg = d.groupby(['bowl', 'match_id', 'over_num']).agg(
        balls_in_over=('legal_ball', 'sum'),
        runs_in_over=('bowlruns', 'sum' if 'bowlruns' in d.columns else 'sum')
    ).reset_index()

    # safe runs_in_over (if above syntax caused issues fallback)
    if 'runs_in_over' not in over_agg.columns:
        over_agg = d.groupby(['bowl', 'match_id', 'over_num']).agg(
            balls_in_over=('legal_ball', 'sum'),
            runs_in_over=('batsman_runs', 'sum')
        ).reset_index()

    maiden_overs_count = over_agg[(over_agg['balls_in_over'] == 6) & (over_agg['runs_in_over'] == 0)].groupby('bowl').size().reset_index(name='maiden_overs').rename(columns={'bowl':'bowler'})

    # phase grouping
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'
    phase_group = d.groupby(['bowl', 'phase']).agg(
        phase_balls=('legal_ball','sum'),
        phase_runs=('batsman_runs','sum'),
        phase_wkts=('bowler_wkt','sum'),
        phase_dots=('is_dot','sum'),
        phase_innings=('match_id','nunique')
    ).reset_index().rename(columns={'bowl':'bowler'})

    def pivot_metric(df_pg, metric):
        if df_pg.empty:
            return pd.DataFrame({'bowler':[]})
        pivoted = df_pg.pivot(index='bowler', columns='phase', values=metric).fillna(0)
        for ph in ['Powerplay','Middle1','Middle2','Death']:
            if ph not in pivoted.columns:
                pivoted[ph] = 0
        pivoted = pivoted.rename(columns={ph: f"{metric}_{ph}" for ph in pivoted.columns})
        pivoted = pivoted.reset_index()
        return pivoted

    pb = pivot_metric(phase_group, 'phase_balls')
    pr = pivot_metric(phase_group, 'phase_runs')
    pw = pivot_metric(phase_group, 'phase_wkts')
    pdot = pivot_metric(phase_group, 'phase_dots')
    pi = pivot_metric(phase_group, 'phase_innings')

    # merge everything
    # rename frames to common 'bowler' index
    frames = [innings.rename(columns={'bowl':'bowler'}) if 'bowl' in innings.columns else innings,
              balls.rename(columns={'bowl':'bowler'}) if 'bowl' in balls.columns else balls,
              runs.rename(columns={'bowl':'bowler'}) if 'bowl' in runs.columns else runs,
              wkts.rename(columns={'bowl':'bowler'}) if 'bowl' in wkts.columns else wkts,
              sixes.rename(columns={'bowl':'bowler'}) if 'bowl' in sixes.columns else sixes,
              fours.rename(columns={'bowl':'bowler'}) if 'bowl' in fours.columns else fours,
              dots.rename(columns={'bowl':'bowler'}) if 'bowl' in dots.columns else dots,
              three_wicket_hauls, maiden_overs_count, bbi, pb, pr, pw, pdot, pi]

    bowl_rec = None
    for fr in frames:
        if fr is None or fr.empty:
            continue
        if bowl_rec is None:
            bowl_rec = fr.copy()
        else:
            bowl_rec = bowl_rec.merge(fr, on='bowler', how='outer')

    if bowl_rec is None:
        bowl_rec = pd.DataFrame(columns=['bowler'])

    # fill NaNs and finalize numerical fields
    for col in ['innings','balls','runs','wkts','sixes','fours','dots','three_wicket_hauls','maiden_overs','bbi','Mega_Over_Count']:
        if col in bowl_rec.columns:
            bowl_rec[col] = pd.to_numeric(bowl_rec[col], errors='coerce').fillna(0)

    bowl_rec['dot%'] = bowl_rec.apply(lambda r: (r['dots'] / r['balls'] * 100) if r.get('balls',0) > 0 else np.nan, axis=1)
    bowl_rec['avg'] = bowl_rec.apply(lambda r: (r['runs'] / r['wkts']) if r.get('wkts',0) > 0 else np.nan, axis=1)
    bowl_rec['sr'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r.get('wkts',0) > 0 else np.nan, axis=1)
    bowl_rec['econ'] = bowl_rec.apply(lambda r: (r['runs'] * 6.0 / r['balls']) if r.get('balls',0) > 0 else np.nan, axis=1)
    bowl_rec['WPI'] = bowl_rec.apply(lambda r: (r['wkts'] / r['innings']) if r.get('innings',0) > 0 else np.nan, axis=1)
    bowl_rec['RPI'] = bowl_rec.apply(lambda r: (r['runs'] / r['innings']) if r.get('innings',0) > 0 else np.nan, axis=1)

    bowl_rec.reset_index(drop=True, inplace=True)
    return bowl_rec


def bowlerstat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bowler aggregation implementing:
      - legal_ball requires both wide & noball == 0
      - dismissal resolution aligned with Custom (dismissed_player attribution)
      - bowler wicket credit only for non-runout-like dismissals
    Returns bowl_rec DataFrame.
    """
    if df is None:
        return pd.DataFrame()
    d = df.copy()

    # normalize minimal names
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'inns' in d.columns and 'inning' not in d.columns:
        d = d.rename(columns={'inns': 'inning'})
    if 'bowl' in d.columns and 'bowler' not in d.columns:
        d = d.rename(columns={'bowl': 'bowler'})
    if 'ball_id' in d.columns and 'ball' not in d.columns:
        d = d.rename(columns={'ball_id': 'ball'})
    if 'batruns' in d.columns and 'batsman_runs' not in d.columns:
        d = d.rename(columns={'batruns': 'batsman_runs'})
    elif 'score' in d.columns and 'batsman_runs' not in d.columns:
        d = d.rename(columns={'score': 'batsman_runs'})
    elif 'batsman_runs' not in d.columns:
        d['batsman_runs'] = 0

    # total_runs fallback
    if 'bowlruns' in d.columns:
        d = d.rename(columns={'bowlruns': 'total_runs'})
    else:
        d['byes'] = pd.to_numeric(d.get('byes', 0), errors='coerce').fillna(0).astype(int)
        d['legbyes'] = pd.to_numeric(d.get('legbyes', 0), errors='coerce').fillna(0).astype(int)
        d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
        d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
        d['total_runs'] = pd.to_numeric(d['batsman_runs'], errors='coerce').fillna(0).astype(int) + d['byes'] + d['legbyes'] + d['noball'] + d['wide']

    # legal_ball: both wide & noball must be 0
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide']   = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # dismissal normalization same as Custom
    special_runout_types = set([
        'run out', 'runout',
        'obstructing the field', 'obstructing thefield', 'obstructing',
        'retired out', 'retired not out (hurt)', 'retired not out', 'retired'
    ])
    d['dismissal_clean'] = d.get('dismissal', pd.Series([None]*len(d), index=d.index)).astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan': '', 'none': ''})
    d['p_bat_num'] = pd.to_numeric(d.get('p_bat', np.nan), errors='coerce')
    d['p_out_num'] = pd.to_numeric(d.get('p_out', np.nan), errors='coerce')
    d['out_flag'] = pd.to_numeric(d.get('out', 0), errors='coerce').fillna(0).astype(int)

    # ball ordering
    if 'ball' in d.columns:
        tmp = pd.to_numeric(d['ball'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    else:
        d['__ball_sort__'] = pd.Series(np.arange(len(d)), index=d.index)

    if 'match_id' not in d.columns:
        d['match_id'] = 0
    d.sort_values(['match_id', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # initialize dismissal fields
    d['dismissed_player'] = None
    d['bowler_wkt'] = 0

    # resolve dismissal per match using same exact logic as Custom
    for match in d['match_id'].unique():
        idxs = d.index[d['match_id'] == match].tolist()
        idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
        for pos, i in enumerate(idxs):
            if int(d.at[i, 'out_flag']) != 1:
                continue

            disc = (d.at[i, 'dismissal_clean'] or '').strip()
            striker = d.at[i, 'batsman'] if 'batsman' in d.columns else None

            # normal dismissal -> striker out and bowler gets credit
            if disc and (disc not in special_runout_types):
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 1
                continue

            # special or blank -> check p_bat/p_out
            pbat = d.at[i, 'p_bat_num']
            pout = d.at[i, 'p_out_num']
            if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0
                continue

            # find nonstriker (last different batter in same match)
            nonstriker = None
            last_idx_of_nonstriker = None
            for j in reversed(idxs[:pos]):
                prev_bat = d.at[j, 'batsman'] if 'batsman' in d.columns else None
                if prev_bat is not None and prev_bat != striker:
                    nonstriker = prev_bat
                    last_idx_of_nonstriker = j
                    break

            if nonstriker is None:
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0
                continue

            prev_out_flag = int(d.at[last_idx_of_nonstriker, 'out_flag']) if last_idx_of_nonstriker is not None else 0
            if prev_out_flag == 0:
                d.at[i, 'dismissed_player'] = nonstriker
                d.at[i, 'bowler_wkt'] = 0
            else:
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0

    # Now aggregate bowler-level stats using these resolved flags
    runs = d.groupby('bowler')['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
    innings = d.groupby('bowler')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings'})
    balls = d.groupby('bowler')['legal_ball'].sum().reset_index().rename(columns={'legal_ball': 'balls'})
    wkts = d.groupby('bowler')['bowler_wkt'].sum().reset_index().rename(columns={'bowler_wkt': 'wkts'})
    dots = d.groupby('bowler')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    fours = d.groupby('bowler')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = d.groupby('bowler')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})

    # three-wicket hauls / bbi from bowler_wkt (credited wickets)
    dismissals_count = d.groupby(['bowler', 'match_id'])['bowler_wkt'].sum().reset_index(name='wkts_in_match')
    three_wicket_hauls = dismissals_count[dismissals_count['wkts_in_match'] >= 3].groupby('bowler').size().reset_index(name='three_wicket_hauls')
    bbi = dismissals_count.groupby('bowler')['wkts_in_match'].max().reset_index().rename(columns={'wkts_in_match': 'bbi'})

    # over/maiden logic
    if 'over' in d.columns:
        try:
            d['over_num'] = pd.to_numeric(d['over'], errors='coerce').fillna(0).astype(int)
        except Exception:
            d['over_num'] = d['over'].astype(str).str.split('.').str[0].astype(int)
    else:
        d['over_num'] = 0

    over_agg = d.groupby(['bowler', 'match_id', 'over_num']).agg(
        balls_in_over=('legal_ball', 'sum'),
        runs_in_over=('total_runs', 'sum')
    ).reset_index()
    maiden_overs_count = over_agg[(over_agg['balls_in_over'] == 6) & (over_agg['runs_in_over'] == 0)].groupby('bowler').size().reset_index(name='maiden_overs')

    # phase metrics
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'

    phase_group = d.groupby(['bowler', 'phase']).agg(
        phase_balls=('legal_ball', 'sum'),
        phase_runs=('batsman_runs', 'sum'),
        phase_wkts=('bowler_wkt', 'sum'),
        phase_dots=('is_dot', 'sum'),
        phase_innings=('match_id', 'nunique')
    ).reset_index()

    def pivot_metric(df_pg, metric):
        if df_pg.empty:
            return pd.DataFrame({'bowler': []})
        pivoted = df_pg.pivot(index='bowler', columns='phase', values=metric).fillna(0)
        for ph in ['Powerplay', 'Middle1', 'Middle2', 'Death']:
            if ph not in pivoted.columns:
                pivoted[ph] = 0
        pivoted = pivoted.rename(columns={ph: f"{metric}_{ph}" for ph in pivoted.columns})
        pivoted = pivoted.reset_index()
        return pivoted

    pb = pivot_metric(phase_group, 'phase_balls')
    pr = pivot_metric(phase_group, 'phase_runs')
    pw = pivot_metric(phase_group, 'phase_wkts')
    pdot = pivot_metric(phase_group, 'phase_dots')
    pi = pivot_metric(phase_group, 'phase_innings')

    phase_df = pb.merge(pr, on='bowler', how='outer').merge(pw, on='bowler', how='outer') \
                 .merge(pdot, on='bowler', how='outer').merge(pi, on='bowler', how='outer').fillna(0)

    # mega over detection (unchanged)
    df_sorted = d.sort_values(['match_id', '__ball_sort__']).reset_index(drop=True).copy()
    df_sorted['ball_str'] = df_sorted.get('ball', df_sorted['__ball_sort__']).astype(str)
    df_sorted['frac'] = df_sorted['ball_str'].str.split('.').str[1].fillna('0')
    df_sorted['frac_int'] = pd.to_numeric(df_sorted['frac'], errors='coerce').fillna(0).astype(int)
    df_sorted['prev_bowler'] = df_sorted['bowler'].shift(1)
    df_sorted['prev_match'] = df_sorted['match_id'].shift(1)
    df_sorted['prev_bowler_same'] = (df_sorted['prev_bowler'] == df_sorted['bowler']) & (df_sorted['prev_match'] == df_sorted['match_id'])
    df_sorted['Mega_Over'] = (df_sorted['frac_int'] == 1) & (df_sorted['prev_bowler_same'])
    mega_over_count = df_sorted[df_sorted['Mega_Over']].groupby('bowler').size().reset_index(name='Mega_Over_Count')

    # combine into bowl_rec
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

    for c in ['three_wicket_hauls', 'maiden_overs', 'Mega_Over_Count', 'bbi']:
        if c in bowl_rec.columns:
            bowl_rec[c] = bowl_rec[c].fillna(0).astype(int)

    if 'season' in d.columns:
        debut_final = d.groupby('bowler')['season'].agg(debut_year='min', final_year='max').reset_index()
        bowl_rec = bowl_rec.merge(debut_final, on='bowler', how='left')
    else:
        bowl_rec['debut_year'] = np.nan
        bowl_rec['final_year'] = np.nan

    numeric_defaults = ['balls', 'runs', 'wkts', 'sixes', 'fours', 'dots']
    for col in numeric_defaults:
        if col in bowl_rec.columns:
            bowl_rec[col] = pd.to_numeric(bowl_rec[col], errors='coerce').fillna(0)

    bowl_rec['dot%'] = bowl_rec.apply(lambda r: (r['dots'] / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['avg'] = bowl_rec.apply(lambda r: (r['runs'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['sr'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['econ'] = bowl_rec.apply(lambda r: (r['runs'] * 6.0 / r['balls']) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['WPI'] = bowl_rec.apply(lambda r: (r['wkts'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)
    bowl_rec['DPI'] = bowl_rec.apply(lambda r: (r['dots'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)
    bowl_rec['RPI'] = bowl_rec.apply(lambda r: (r['runs'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)

    bowl_rec['bdry%'] = bowl_rec.apply(lambda r: ((r.get('fours',0) + r.get('sixes',0)) / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['BPB'] = bowl_rec.apply(lambda r: (r['balls'] / (r.get('fours',0) + r.get('sixes',0))) if (r.get('fours',0) + r.get('sixes',0)) > 0 else np.nan, axis=1)
    bowl_rec['BPD'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['BP6'] = bowl_rec.apply(lambda r: (r['balls'] / r['sixes']) if r['sixes'] > 0 else np.nan, axis=1)

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

    bowl_rec['overs'] = bowl_rec['balls'].apply(lambda x: f"{int(x // 6)}.{int(x % 6)}" if pd.notna(x) else "0.0")
    bowl_rec = bowl_rec[bowl_rec['bowler'].notna()]
    bowl_rec.reset_index(drop=True, inplace=True)
    return bowl_rec

# -----------------------
# Streamlit integration
# -----------------------
@st.cache_data
def build_idf(df_local):
    return cumulator(df_local)


sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis", "Strength vs Weakness", "Match by Match Analysis")
)

if df is not None:
    idf = build_idf(df)
else:
    idf = pd.DataFrame()

if sidebar_option == "Player Profile":
    st.header("Player Profile")

    if idf is None or idf.empty:
        st.error("⚠️ Please run idf = Custom(df) before showing Player Profile (ensure raw 'df' is loaded).")
        st.stop()
    if df is None:
        st.error("⚠️ This view requires the original raw 'df' (ball-by-ball / match-level dataframe). Please ensure 'df' is loaded.")
        st.stop()

    def as_dataframe(x):
        if isinstance(x, pd.Series):
            return x.to_frame().T.reset_index(drop=True)
        elif isinstance(x, pd.DataFrame):
            return x.copy()
        else:
            try:
                return pd.DataFrame(x)
            except Exception:
                return pd.DataFrame()

    idf = as_dataframe(idf)
    df  = as_dataframe(df)

    if 'batsman' not in idf.columns:
        if 'bat' in idf.columns:
            idf = idf.rename(columns={'bat': 'batsman'})
        else:
            st.error("Dataset must contain a 'batsman' or 'bat' column in idf.")
            st.stop()

    players = sorted(idf['batsman'].dropna().unique().tolist())
    if not players:
        st.error("No players found in idf dataset.")
        st.stop()

    player_name = st.selectbox("Search for a player", players, index=0)

    tabs = st.tabs(["Career Statistics"])
    with tabs[0]:
        st.header("Career Statistics")
        option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))

if option == "Batting":
    player_stats = as_dataframe(idf[idf['batsman'] == player_name])
    if player_stats is None or player_stats.empty:
        st.warning(f"No data available for {player_name}.")
        st.stop()

    # cleanup & formatting
    player_stats = player_stats.drop(columns=['final_year'], errors='ignore')
    player_stats.columns = [str(col).upper().replace('_', ' ') for col in player_stats.columns]
    player_stats = round_up_floats(player_stats)

    int_cols = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
    for c in int_cols:
        if c in player_stats.columns:
            player_stats[c] = pd.to_numeric(player_stats[c], errors='coerce').fillna(0).astype(int)

    # Header / metric cards (no emojis)
    st.markdown("### Batting Statistics")

    # helper to find column by candidates
    def find_col(df, candidates):
        for cand in candidates:
            if cand in df.columns:
                return cand
        return None

    # preferred top metrics (ordered)
    top_metric_mapping = {
        "Runs": ["RUNS", "RUNS "],
        "Innings": ["INNINGS", "MATCHES"],
        "Average": ["AVG", "AVERAGE"],
        "Strike Rate": ["SR", "STRIKE RATE"],
        "Highest Score": ["HIGHEST SCORE", "HS"],
        "50s": ["FIFTIES", "50S", "FIFTY"],
        "100s": ["HUNDREDS", "100S"],
    }

    # collect values for display
    found_top_cols = {}
    for label, candidates in top_metric_mapping.items():
        col = find_col(player_stats, candidates)
        val = None
        if col is not None:
            try:
                val = player_stats.iloc[0][col]
            except Exception:
                val = player_stats[col].values[0] if len(player_stats[col].values) > 0 else None
        found_top_cols[label] = val

    # Display top metrics as columns (show only those that exist)
    visible_metrics = [(k, v) for k, v in found_top_cols.items() if v is not None and (not (isinstance(v, float) and np.isnan(v)))]
    if visible_metrics:
        cols = st.columns(len(visible_metrics))
        for (label, val), col in zip(visible_metrics, cols):
            if isinstance(val, (int, np.integer)):
                disp = f"{int(val)}"
            elif isinstance(val, (float, np.floating)) and not np.isnan(val):
                disp = f"{val:.2f}"
            else:
                disp = str(val)
            col.metric(label, disp)
    else:
        st.write("Top metrics not available for this player.")

    # --------------------
    # Show the rest of the single-row summary as vertical key:value table
    # --------------------
    # Remove displayed top columns from the transposed view EXCEPT keep 'RUNS' (show Runs in Detailed)
    top_cols_used = [find_col(player_stats, cand) for cand in top_metric_mapping.values()]
    top_cols_used = [c for c in top_cols_used if c is not None]
    # ensure RUNS is not removed from the detailed view
    top_cols_used_excluding_runs = [c for c in top_cols_used if c is not None and c.upper() != 'RUNS']

    try:
        rest_series = player_stats.iloc[0].drop(labels=top_cols_used_excluding_runs, errors='ignore')
    except Exception:
        rest_series = pd.Series(dtype=object)

    if not rest_series.empty:
        rest_df = rest_series.reset_index()
        rest_df.columns = ["Metric", "Value"]

        def fmt_val(x):
            if pd.isna(x):
                return ""
            if isinstance(x, (int, np.integer)):
                return int(x)
            if isinstance(x, (float, np.floating)):
                return round(x, 2)
            return x

        rest_df["Value"] = rest_df["Value"].apply(fmt_val)

        # Light "skin" header color for Detailed stats (peach / light skin tone)
        detailed_header_color = "#fff0e6"  # light skin/peach
        detailed_table_styles = [
            {"selector": "thead th", "props": [("background-color", detailed_header_color), ("color", "#000"), ("font-weight", "600")]},
            {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fff9f4")]},
        ]

        st.markdown("#### Detailed stats")
        st.dataframe(rest_df.style.set_table_styles(detailed_table_styles), use_container_width=True)
    else:
        st.write("No additional per-player summary metrics available.")

    # --------------------
    # Opponent / Year / Inning breakdowns: use scrollable, lightly-colored tables
    # --------------------
    bat_col = 'batsman' if 'batsman' in df.columns else ('bat' if 'bat' in df.columns else None)
    if bat_col:
        opp_col = safe_get_col(df, ['team_bowl', 'team_bow', 'team_bowling'], default=None)
        if opp_col:
            opponents = sorted(df[df[bat_col] == player_name][opp_col].dropna().unique().tolist())
            all_opp = []
            for opp in opponents:
                temp = df[(df[bat_col] == player_name) & (df[opp_col] == opp)].copy()
                if temp.empty:
                    continue
                temp_summary = cumulator(temp)
                if not temp_summary.empty:
                    temp_summary['OPPONENT'] = opp
                    all_opp.append(temp_summary)
            if all_opp:
                result_df = pd.concat(all_opp, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')

                # Upper-case column names, replace underscores with spaces, and normalize Middle1/Middle2 to "Middle 1"/"Middle 2"
                new_cols = []
                for col in result_df.columns:
                    cname = str(col).upper().replace('_', ' ')
                    cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                    new_cols.append(cname)
                result_df.columns = new_cols

                # Ensure Opponent is first column
                if 'OPPONENT' in result_df.columns:
                    cols = ['OPPONENT'] + [c for c in result_df.columns if c != 'OPPONENT']
                    result_df = result_df[cols]

                # cast a few numeric cols safely
                for c in ['HUNDREDS', 'FIFTIES', '30S', 'RUNS', 'HIGHEST SCORE']:
                    if c in result_df.columns:
                        result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                result_df = round_up_floats(result_df)

                # Light blue header color for Opponentwise table
                opp_header_color = "#e6f2ff"
                opp_table_styles = [
                    {"selector": "thead th", "props": [("background-color", opp_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7fbff")]},
                ]
                st.markdown("### Opponentwise Performance")
                st.dataframe(result_df.style.set_table_styles(opp_table_styles), use_container_width=True)

        # Yearwise
        if 'year' in df.columns:
            seasons = sorted(df[df[bat_col] == player_name]['year'].dropna().unique().tolist())
            all_seasons = []
            for season in seasons:
                temp = df[(df[bat_col] == player_name) & (df['year'] == season)].copy()
                if temp.empty:
                    continue
                temp_summary = cumulator(temp)
                if not temp_summary.empty:
                    temp_summary['YEAR'] = season
                    all_seasons.append(temp_summary)
            if all_seasons:
                result_df = pd.concat(all_seasons, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')

                # Upper-case column names, replace underscores with spaces, normalize Middle1/Middle2
                new_cols = []
                for col in result_df.columns:
                    cname = str(col).upper().replace('_', ' ')
                    cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                    new_cols.append(cname)
                result_df.columns = new_cols

                # Ensure YEAR is first column
                if 'YEAR' in result_df.columns:
                    cols = ['YEAR'] + [c for c in result_df.columns if c != 'YEAR']
                    result_df = result_df[cols]

                for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                    if c in result_df.columns:
                        result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                result_df = round_up_floats(result_df)

                # Light purple header for Yearwise table
                year_header_color = "#efe6ff"  # light purple
                year_table_styles = [
                    {"selector": "thead th", "props": [("background-color", year_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fbf7ff")]},
                ]
                st.markdown("### Yearwise Performance")
                st.dataframe(result_df.style.set_table_styles(year_table_styles), use_container_width=True)

        # Inningwise
        inning_col = 'inns' if 'inns' in df.columns else ('inning' if 'inning' in df.columns else None)
        if inning_col:
            innings_list = []
            for inn in sorted(df[inning_col].dropna().unique()):
                temp = df[(df[bat_col] == player_name) & (df[inning_col] == inn)].copy()
                if temp.empty:
                    continue
                temp_summary = cumulator(temp)
                if not temp_summary.empty:
                    temp_summary['INNING'] = inn
                    innings_list.append(temp_summary)
            if innings_list:
                result_df = pd.concat(innings_list, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')

                # Upper-case column names, replace underscores with spaces, normalize Middle1/Middle2
                new_cols = []
                for col in result_df.columns:
                    cname = str(col).upper().replace('_', ' ')
                    cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                    new_cols.append(cname)
                result_df.columns = new_cols

                # Ensure INNING is first col (and show 1/2 if that is the value)
                if 'INNING' in result_df.columns:
                    cols = ['INNING'] + [c for c in result_df.columns if c != 'INNING']
                    result_df = result_df[cols]

                for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                    if c in result_df.columns:
                        result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                result_df = round_up_floats(result_df)
                result_df = result_df.drop(columns=['MATCHES'], errors='ignore')

                # Light green header for Inningwise table
                inning_header_color = "#e9f9ea"
                inning_table_styles = [
                    {"selector": "thead th", "props": [("background-color", inning_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f3fff3")]},
                ]
                st.markdown("### Inningwise Performance")
                st.dataframe(result_df.reset_index(drop=True).style.set_table_styles(inning_table_styles), use_container_width=True)




elif option == "Bowling":
    # Defensive: ensure bidf (summary idf for bowlers) and bpdf (ball-by-ball for bowlers) exist
    bpdf=df
    bidf=bowlerstat(df)
    # try:
    #     bidf
    #     bpdf
    # except NameError:
    #     st.error("Bowling view requires 'bidf' (bowling summary) and 'bpdf' (bowling ball-by-ball). Please load them.")
    #     st.stop()

    # Normalize to DataFrame and ensure bowler column exists
    temp_df = as_dataframe(bidf)
    if 'bowler' not in temp_df.columns:
        st.error("bidf must contain a 'bowler' column.")
        st.stop()

    # Player-specific summary row(s)
    player_stats = temp_df[temp_df['bowler'] == player_name].copy()
    player_stats = as_dataframe(player_stats)
    if player_stats.empty:
        st.markdown("No bowling stats available for the selected player.")
    else:
        # prepare display copy: uppercase names & round floats
        disp_stats = player_stats.copy()
        disp_stats.columns = [str(col).upper().replace('_', ' ') for col in disp_stats.columns]
        disp_stats = round_up_floats(disp_stats)

        # Top metric mapping for bowlers
        def find_col(df, candidates):
            for cand in candidates:
                if cand in df.columns:
                    return cand
            return None

        top_metric_mapping = {
            "Overs": ["OVERS", "OVERS BOWLED", "OVERS "],
            "Runs": ["RUNS", "RUNS CONCEDED"],
            "Wickets": ["WKTS", "WICKETS", "WICKETS "],
            "Innings": ["INNINGS", "MATCHES"],
            "Econ": ["ECON", "ECONOMY"],
            "Average": ["AVG", "AVERAGE"],
            "Strike Rate": ["SR", "STRIKE RATE"],
            "Maidens": ["MAIDEN OVERS", "MAIDENS"],
            "3w Hauls": ["THREE_WICKET_HAULS", "3WICKETHAULS", "3W"],
            "BBI": ["BBI", "BEST BBI", "BEST"]
        }

        # Collect top metrics
        found_top_cols = {}
        for label, candidates in top_metric_mapping.items():
            col = find_col(disp_stats, candidates)
            val = None
            if col is not None:
                try:
                    val = disp_stats.iloc[0][col]
                except Exception:
                    val = disp_stats[col].values[0] if len(disp_stats[col].values) > 0 else None
            found_top_cols[label] = val

        st.markdown("### Bowling Statistics")

        visible_metrics = [(k, v) for k, v in found_top_cols.items() if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if visible_metrics:
            cols = st.columns(len(visible_metrics))
            for (label, val), col in zip(visible_metrics, cols):
                if isinstance(val, (int, np.integer)):
                    disp = f"{int(val)}"
                elif isinstance(val, (float, np.floating)) and not np.isnan(val):
                    disp = f"{val:.2f}"
                else:
                    disp = str(val)
                col.metric(label, disp)
        else:
            st.write("Top bowling metrics not available.")

        # Detailed stats vertical table (include RUNS)
        # Remove top metrics except keep 'RUNS' in the detailed table
        top_cols_used = [find_col(disp_stats, cand) for cand in top_metric_mapping.values()]
        top_cols_used = [c for c in top_cols_used if c is not None]
        top_cols_used_excluding_runs = [c for c in top_cols_used if c is not None and c.upper() != 'RUNS']

        try:
            rest_series = disp_stats.iloc[0].drop(labels=top_cols_used_excluding_runs, errors='ignore')
        except Exception:
            rest_series = pd.Series(dtype=object)

        if not rest_series.empty:
            rest_df = rest_series.reset_index()
            rest_df.columns = ["Metric", "Value"]

            def fmt_val(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, (int, np.integer)):
                    return int(x)
                if isinstance(x, (float, np.floating)):
                    # show two decimals for floats
                    return round(x, 2)
                return x

            rest_df["Value"] = rest_df["Value"].apply(fmt_val)

            # Light skin (peach) header for detailed stats
            detailed_header_color = "#fff0e6"
            detailed_table_styles = [
                {"selector": "thead th", "props": [("background-color", detailed_header_color), ("color", "#000"), ("font-weight", "600")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fff9f4")]},
            ]

            st.markdown("#### Detailed stats")
            st.dataframe(rest_df.style.set_table_styles(detailed_table_styles), use_container_width=True)
        else:
            st.write("No detailed bowling metrics available.")

    # --------------------
    # Opponentwise Performance
    # --------------------
    # Determine appropriate bat/team columns
    bat_col = 'batsman' if 'batsman' in bpdf.columns else ('bat' if 'bat' in bpdf.columns else None)
    opp_col = safe_get_col(bpdf, ['team_bat', 'team_bowl', 'team_bowling', 'batting_team'], default=None)

    if bat_col is None or opp_col is None:
        st.info("Opponentwise breakdown not available (missing team/batsman columns).")
    else:
        # Find opponents list for this bowler
        opponents = sorted(bpdf[bpdf['bowler'] == player_name][opp_col].dropna().unique().tolist())
        all_opp = []
        for opp in opponents:
            temp = bpdf[(bpdf['bowler'] == player_name) & (bpdf[opp_col] == opp)].copy()
            if temp.empty:
                continue
            temp_summary = bowlerstat(temp)
            if not temp_summary.empty:
                temp_summary['OPPONENT'] = opp.upper()
                # ensure opponent is first column
                cols = temp_summary.columns.tolist()
                if 'OPPONENT' in temp_summary.columns:
                    temp_summary = temp_summary[['OPPONENT'] + [c for c in cols if c != 'OPPONENT']]
                all_opp.append(temp_summary)
        if all_opp:
            result_df = pd.concat(all_opp, ignore_index=True).drop(columns=['bowler'], errors='ignore')
            result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
            # safe numeric casts for typical columns
            for c in ['RUNS', 'WKTS', 'BALLS', 'OVERS', 'ECON', 'AVG']:
                if c in result_df.columns:
                    result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0)
            result_df = round_up_floats(result_df)

            # Light blue header for opponent table
            opp_header_color = "#e6f2ff"
            opp_table_styles = [
                {"selector": "thead th", "props": [("background-color", opp_header_color), ("color", "#000"), ("font-weight", "600")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7fbff")]},
            ]
            st.markdown("### Opponentwise Performance")
            st.dataframe(result_df.style.set_table_styles(opp_table_styles), use_container_width=True)
        else:
            st.info("No opponentwise bowling summary available for this player.")

    # --------------------
    # Yearwise Performance
    # --------------------
    if 'season' in bpdf.columns:
        tdf = bpdf[bpdf['bowler'] == player_name].copy()
        unique_seasons = sorted(tdf['season'].dropna().unique().tolist())
        all_seasons = []
        for season in unique_seasons:
            temp = tdf[tdf['season'] == season].copy()
            temp_summary = bowlerstat(temp)
            if not temp_summary.empty:
                temp_summary['YEAR'] = season
                # place YEAR first
                cols = temp_summary.columns.tolist()
                if 'YEAR' in temp_summary.columns:
                    temp_summary = temp_summary[['YEAR'] + [c for c in cols if c != 'YEAR']]
                all_seasons.append(temp_summary)
        if all_seasons:
            result_df = pd.concat(all_seasons, ignore_index=True).drop(columns=['bowler'], errors='ignore')
            result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
            result_df = round_up_floats(result_df)

            # Light purple header for year table
            year_header_color = "#efe6ff"
            year_table_styles = [
                {"selector": "thead th", "props": [("background-color", year_header_color), ("color", "#000"), ("font-weight", "600")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fbf7ff")]},
            ]

            st.markdown("### Yearwise Bowling Performance")
            st.dataframe(result_df.style.set_table_styles(year_table_styles), use_container_width=True)
        else:
            st.info("No yearwise bowling summary available for this player.")

    # --------------------
    # Inningwise Performance
    # --------------------
    inning_col = 'inns' if 'inns' in bpdf.columns else ('inning' if 'inning' in bpdf.columns else None)
    if inning_col is None:
        st.info("Inningwise breakdown not available (missing 'inns' / 'inning' column).")
    else:
        tdf = bpdf[bpdf['bowler'] == player_name].copy()
        innings_list = []
        for inn in sorted(tdf[inning_col].dropna().unique()):
            temp = tdf[tdf[inning_col] == inn].copy()
            if temp.empty:
                continue
            temp_summary = bowlerstat(temp)
            if not temp_summary.empty:
                temp_summary['INNING'] = inn
                cols = temp_summary.columns.tolist()
                if 'INNING' in temp_summary.columns:
                    temp_summary = temp_summary[['INNING'] + [c for c in cols if c != 'INNING']]
                innings_list.append(temp_summary)
        if innings_list:
            result_df = pd.concat(innings_list, ignore_index=True).drop(columns=['bowler'], errors='ignore')
            result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
            result_df = round_up_floats(result_df)

            # Light green header for inning table
            inning_header_color = "#e9f9ea"
            inning_table_styles = [
                {"selector": "thead th", "props": [("background-color", inning_header_color), ("color", "#000"), ("font-weight", "600")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f3fff3")]},
            ]

            st.markdown("### Inningwise Bowling Performance")
            st.dataframe(result_df.style.set_table_styles(inning_table_styles), use_container_width=True)
        else:
            st.info("No inningwise bowling summary available for this player.")



        
