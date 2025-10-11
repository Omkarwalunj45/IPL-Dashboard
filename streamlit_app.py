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
    try:
        balls = float(balls); dismissals = float(dismissals)
    except Exception:
        return np.nan
    return (balls / dismissals) if dismissals > 0 else np.nan

def bpb(balls, boundaries_count):
    try:
        balls = float(balls); boundaries_count = float(boundaries_count)
    except Exception:
        return np.nan
    return (balls / boundaries_count) if boundaries_count > 0 else np.nan

def bp6(balls, sixes):
    try:
        balls = float(balls); sixes = float(sixes)
    except Exception:
        return np.nan
    return (balls / sixes) if sixes > 0 else np.nan

def bp4(balls, fours):
    try:
        balls = float(balls); fours = float(fours)
    except Exception:
        return np.nan
    return (balls / fours) if fours > 0 else np.nan

def avg(runs, dismissals, innings):
    try:
        runs = float(runs); dismissals = float(dismissals); innings = float(innings)
    except Exception:
        return np.nan
    if dismissals > 0:
        return runs / dismissals
    if innings > 0:
        return runs / innings
    return np.nan

def categorize_phase(over_val):
    try:
        o = float(over_val)
    except Exception:
        return "Unknown"
    if o <= 6:
        return "Powerplay"
    if 6 < o <= 11:
        return "Middle 1"
    if 11 < o <= 16:
        return "Middle 2"
    return "Death"

import pandas as pd
import numpy as np

# -------------------------
# small helper functions (replace with your versions if present)
# -------------------------
def avg(runs, dismissals, innings):
    try:
        if dismissals > 0:
            return runs / dismissals
        if innings > 0:
            return runs / innings
        return np.nan
    except Exception:
        return np.nan

def bpd(balls, dismissals):
    try:
        return balls / dismissals if dismissals > 0 else np.nan
    except Exception:
        return np.nan

def bpb(balls, boundaries):
    try:
        return balls / boundaries if boundaries > 0 else np.nan
    except Exception:
        return np.nan

def bp6(balls, sixes):
    try:
        return balls / sixes if sixes > 0 else np.nan
    except Exception:
        return np.nan

def bp4(balls, fours):
    try:
        return balls / fours if fours > 0 else np.nan
    except Exception:
        return np.nan

def categorize_phase(over):
    # simple phase split: Powerplay (<=6), Middle1 (7-11), Middle2 (12-16), Death (17+)
    try:
        o = float(over)
    except Exception:
        return "Unknown"
    if o <= 6:
        return "Powerplay"
    if 6 < o <= 11:
        return "Middle1"
    if 11 < o <= 16:
        return "Middle2"
    return "Death"

# -------------------------
# Custom - batting summary with exact dismissal rules
# -------------------------
def Custom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build batting summary (one row per batsman) and apply the user-specified dismissal rules:
      - legal_ball counts only if both wide==0 and noball==0
      - 50s counted only when match_runs >=50 and <100
      - dismissal resolution exactly as the user specified (special run-out-like handling)
    Returns bat_rec DataFrame (one row per batsman).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    # Normalize expected columns (use user's column names)
    # p_match -> match_id
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    # ensure batsman column is 'bat' (as you said)
    if 'bat' not in d.columns:
        d['bat'] = None

    # create safe ball order (ball_id preferred, then ball, else index)
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

    # legal ball: both wide & noball must be 0
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # per-delivery run flags from batruns / score
    if 'batruns' in d.columns:
        d['runs_off_bat'] = pd.to_numeric(d['batruns'], errors='coerce').fillna(0).astype(int)
    elif 'score' in d.columns:
        d['runs_off_bat'] = pd.to_numeric(d['score'], errors='coerce').fillna(0).astype(int)
    else:
        d['runs_off_bat'] = 0
    d['is_dot']  = ((d['runs_off_bat'] == 0) & (d['legal_ball'] == 1)).astype(int)
    d['is_one']  = (d['runs_off_bat'] == 1).astype(int)
    d['is_two']  = (d['runs_off_bat'] == 2).astype(int)
    d['is_three']= (d['runs_off_bat'] == 3).astype(int)
    d['is_four'] = (d['runs_off_bat'] == 4).astype(int)
    d['is_six']  = (d['runs_off_bat'] == 6).astype(int)

    # normalize dismissal text and special set (case-insensitive)
    d['dismissal_clean'] = d.get('dismissal', pd.Series([None]*len(d))).astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan': '', 'none': ''})
    special_runout_types = set([
        'run out', 'runout',
        'obstructing the field', 'obstructing thefield', 'obstructing',
        'retired out', 'retired not out (hurt)', 'retired not out', 'retired'
    ])

    # numeric p_bat / p_out and numeric out flag
    d['p_bat_num'] = pd.to_numeric(d.get('p_bat', np.nan), errors='coerce')
    d['p_out_num'] = pd.to_numeric(d.get('p_out', np.nan), errors='coerce')
    d['out_flag'] = pd.to_numeric(d.get('out', 0), errors='coerce').fillna(0).astype(int)

    # sort by match & ball
    if 'match_id' not in d.columns:
        d['match_id'] = 0
    d.sort_values(['match_id', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # initialize resolved fields
    d['dismissed_player'] = None     # will contain the actual dismissed batter name
    d['bowler_wkt'] = 0              # 1 if bowler gets wicket credit

    # iterate per match to apply the exact dismissal rules
    for m in d['match_id'].unique():
        idxs = d.index[d['match_id'] == m].tolist()
        idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
        for pos, i in enumerate(idxs):
            out_flag = int(d.at[i, 'out_flag']) if not pd.isna(d.at[i, 'out_flag']) else 0
            disc = (d.at[i, 'dismissal_clean'] or '').strip()
            striker = d.at[i, 'bat'] if 'bat' in d.columns else None

            # If out_flag True:
            if out_flag == 1:
                # If dismissal text exists AND is NOT in special set -> striker is out; bowler gets credit
                if disc and (disc not in special_runout_types):
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 1
                    continue

                # Else disc is in special set OR blank -> check p_bat and p_out
                pbat = d.at[i, 'p_bat_num']
                pout = d.at[i, 'p_out_num']

                # If both numeric and equal -> striker out (no bowler credit)
                if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
                    continue

                # Else non-striker is out: find last different batter earlier in this match
                nonstriker = None
                last_idx_of_nonstriker = None
                for j in reversed(idxs[:pos]):
                    prev_bat = d.at[j, 'bat'] if 'bat' in d.columns else None
                    if prev_bat is not None and prev_bat != striker:
                        nonstriker = prev_bat
                        last_idx_of_nonstriker = j
                        break

                if nonstriker is None:
                    # fallback to striker
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
                    continue

                # Inspect last ball that nonstriker played
                prev_out_flag = int(d.at[last_idx_of_nonstriker, 'out_flag']) if last_idx_of_nonstriker is not None else 0
                if prev_out_flag == 0:
                    # other batter's last ball out_flag == 0 -> nonstriker is dismissed now (no bowler credit)
                    d.at[i, 'dismissed_player'] = nonstriker
                    d.at[i, 'bowler_wkt'] = 0
                else:
                    # fallback to striker
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0

            else:
                # If out_flag is False -> per instruction mark the striker as dismissed
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0

    # -------------------------
    # Now compute batting aggregates using resolved 'dismissed_player'
    # -------------------------

    # ensure cur_bat_runs/cur_bat_bf exist for match-level snapshot
    d['cur_bat_runs'] = pd.to_numeric(d.get('cur_bat_runs', 0), errors='coerce').fillna(0).astype(int)
    d['cur_bat_bf'] = pd.to_numeric(d.get('cur_bat_bf', 0), errors='coerce').fillna(0).astype(int)

    last_bat_snapshot = (
        d.groupby(['bat', 'match_id'], sort=False, as_index=False)
         .agg({'cur_bat_runs': 'last', 'cur_bat_bf': 'last'})
         .rename(columns={'cur_bat_runs': 'match_runs', 'cur_bat_bf': 'match_balls'})
    )

    runs_per_match = last_bat_snapshot[['bat', 'match_runs', 'match_balls', 'match_id']].copy()
    innings_count = runs_per_match.groupby('bat')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings', 'bat': 'batsman'})
    total_runs = runs_per_match.groupby('bat')['match_runs'].sum().reset_index().rename(columns={'match_runs': 'runs', 'bat': 'batsman'})
    total_balls = runs_per_match.groupby('bat')['match_balls'].sum().reset_index().rename(columns={'match_balls': 'balls', 'bat': 'batsman'})

    # Dismissals counted using resolved 'dismissed_player'
    dismissals_df = d[d['dismissed_player'].notna()].groupby('dismissed_player').size().reset_index(name='dismissals')
    dismissals_df = dismissals_df.rename(columns={'dismissed_player': 'batsman'})

    # boundary & running counts
    fours = d.groupby('bat')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours', 'bat': 'batsman'})
    sixes = d.groupby('bat')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes', 'bat': 'batsman'})
    dots = d.groupby('bat')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots', 'bat': 'batsman'})
    ones = d.groupby('bat')['is_one'].sum().reset_index().rename(columns={'is_one': 'ones', 'bat': 'batsman'})
    twos = d.groupby('bat')['is_two'].sum().reset_index().rename(columns={'is_two': 'twos', 'bat': 'batsman'})
    threes = d.groupby('bat')['is_three'].sum().reset_index().rename(columns={'is_three': 'threes', 'bat': 'batsman'})

    # thresholds: 30s (30-49), 50s (50-99), 100s (>=100)
    thirties = runs_per_match[(runs_per_match['match_runs'] >= 30) & (runs_per_match['match_runs'] < 50)].groupby('bat').size().reset_index(name='30s').rename(columns={'bat':'batsman'})
    fifties = runs_per_match[(runs_per_match['match_runs'] >= 50) & (runs_per_match['match_runs'] < 100)].groupby('bat').size().reset_index(name='50s').rename(columns={'bat':'batsman'})
    hundreds = runs_per_match[runs_per_match['match_runs'] >= 100].groupby('bat').size().reset_index(name='100s').rename(columns={'bat':'batsman'})

    highest_score = runs_per_match.groupby('bat')['match_runs'].max().reset_index().rename(columns={'match_runs': 'HS', 'bat':'batsman'})
    median_runs = runs_per_match.groupby('bat')['match_runs'].median().reset_index().rename(columns={'match_runs': 'median', 'bat':'batsman'})

    boundary_runs = (d.groupby('bat').apply(lambda x: int((x['is_four'] * 4).sum() + (x['is_six'] * 6).sum()))
                     .reset_index(name='boundary_runs').rename(columns={'level_1':'batsman', 'bat':'batsman'}))
    running_runs = (d.groupby('bat').apply(lambda x: int((x['is_one'] * 1).sum() + (x['is_two'] * 2).sum() + (x['is_three'] * 3).sum()))
                    .reset_index(name='running_runs').rename(columns={'level_1':'batsman', 'bat':'batsman'}))

    # Merge batting record
    bat_rec = innings_count.merge(total_runs, left_on='batsman', right_on='batsman', how='left')
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

    # fill NaNs & cast
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

    # metrics
    bat_rec['RPI'] = bat_rec.apply(lambda x: (x['runs'] / x['innings']) if x['innings'] > 0 else np.nan, axis=1)
    bat_rec['SR'] = bat_rec.apply(lambda x: (x['runs'] / x['balls'] * 100) if x['balls'] > 0 else np.nan, axis=1)
    bat_rec['BPD'] = bat_rec.apply(lambda x: bpd(x['balls'], x['dismissals']), axis=1)
    bat_rec['BPB'] = bat_rec.apply(lambda x: bpb(x['balls'], (x.get('fours',0) + x.get('sixes',0))), axis=1)
    bat_rec['BP6'] = bat_rec.apply(lambda x: bp6(x['balls'], x.get('sixes',0)), axis=1)
    bat_rec['BP4'] = bat_rec.apply(lambda x: bp4(x['balls'], x.get('fours',0)), axis=1)
    bat_rec['AVG'] = bat_rec.apply(lambda x: avg(x['runs'], x.get('dismissals', 0), x['innings']), axis=1)

    # phase-wise aggregation (optional)
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'

    phase_stats = d.groupby(['bat', 'phase']).agg({
        'runs_off_bat': 'sum',
        'legal_ball': 'sum',
        'is_dot': 'sum',
        'is_four': 'sum',
        'is_six': 'sum',
        'match_id': 'nunique'
    }).reset_index().rename(columns={'bat':'batsman', 'match_id':'Innings'})

    # pivot phase stats
    if not phase_stats.empty:
        phase_pivot = phase_stats.pivot(index='batsman', columns='phase', values=['runs_off_bat','legal_ball','is_dot','is_four','is_six','Innings'])
        if isinstance(phase_pivot.columns, pd.MultiIndex):
            phase_pivot.columns = [f"{col[1]}_{col[0]}" for col in phase_pivot.columns]
        phase_pivot.reset_index(inplace=True)
        bat_rec = bat_rec.merge(phase_pivot, on='batsman', how='left')

    bat_rec.reset_index(drop=True, inplace=True)
    return bat_rec

# -------------------------
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
    return Custom(df_local)


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

            # cleanup & formatting (same as before)
            player_stats = player_stats.drop(columns=['final_year'], errors='ignore')
            player_stats.columns = [str(col).upper().replace('_', ' ') for col in player_stats.columns]
            player_stats = round_up_floats(player_stats)

            int_cols = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
            for c in int_cols:
                if c in player_stats.columns:
                    player_stats[c] = pd.to_numeric(player_stats[c], errors='coerce').fillna(0).astype(int)

            # --------------------
            # Nicely formatted header / metric cards
            # --------------------
            st.markdown("### 🏏 Batting Statistics")
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
                    # get first row value safely
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
                    # format numeric nicely
                    if isinstance(val, (int, np.integer)):
                        disp = f"{int(val)}"
                    elif isinstance(val, (float, np.floating)) and not np.isnan(val):
                        # show two decimals for floats
                        disp = f"{val:.2f}"
                    else:
                        disp = str(val)
                    col.metric(label, disp)
            else:
                st.write("Top metrics not available for this player.")

            # --------------------
            # Show the rest of the single-row summary as vertical key:value table
            # --------------------
            # Remove displayed top columns from the transposed view
            top_cols_used = [find_col(player_stats, cand) for cand in top_metric_mapping.values()]
            top_cols_used = [c for c in top_cols_used if c is not None]

            # Build "other" key-value pairs
            try:
                rest_series = player_stats.iloc[0].drop(labels=top_cols_used, errors='ignore')
            except Exception:
                rest_series = pd.Series(dtype=object)

            if not rest_series.empty:
                rest_df = rest_series.reset_index()
                rest_df.columns = ["Metric", "Value"]
                # Pretty formatting for some numeric columns
                def fmt_val(x):
                    if pd.isna(x):
                        return ""
                    if isinstance(x, (int, np.integer)):
                        return int(x)
                    if isinstance(x, (float, np.floating)):
                        return round(x, 2)
                    return x
                rest_df["Value"] = rest_df["Value"].apply(fmt_val)
                st.markdown("#### Detailed stats")
                st.dataframe(rest_df, use_container_width=True)
            else:
                st.write("No additional per-player summary metrics available.")

            # --------------------
            # Opponent / Year / Inning breakdowns: use scrollable dataframes
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
                        result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
                        # cast a few numeric cols safely
                        for c in ['HUNDREDS', 'FIFTIES', '30S', 'RUNS', 'HIGHEST SCORE']:
                            if c in result_df.columns:
                                result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                        result_df = round_up_floats(result_df)
                        st.markdown("### 🆚 Opponentwise Performance")
                        # show interactive, scrollable table
                        st.dataframe(result_df, use_container_width=True)

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
                        result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
                        for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                            if c in result_df.columns:
                                result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                        result_df = round_up_floats(result_df)
                        st.markdown("### 📅 Yearwise Performance")
                        st.dataframe(result_df, use_container_width=True)

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
                        result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
                        for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                            if c in result_df.columns:
                                result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                        result_df = round_up_floats(result_df)
                        st.markdown("### 🏟️ Inningwise Performance")
                        st.dataframe(result_df.reset_index(drop=True), use_container_width=True)

        elif option == "Bowling":
            st.info("🎯 Bowling module will be integrated after validation of bowl_rec().")



        
