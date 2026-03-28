"""
train_model.py
==============
This script does 3 things:
1. Pulls historical NBA game logs from nba_api for the 2023-24, 2024-25, and 2025-26 seasons
2. Engineers features for each game (rolling averages, back-to-back, opponent defense, etc.)
3. Trains one XGBoost model per stat category using TimeSeriesSplit for honest evaluation
   and saves them all to a single pickle file (nba_models.pkl)

Run this script once locally to generate nba_models.pkl.
GitHub Actions will re-run it every morning to keep the models fresh.
"""

import pandas as pd
import numpy as np
import pickle
import time
import os
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguedashplayerstats
from nba_api.stats.static import players, teams
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# CONSTANTS
# ============================================================

# Seasons to pull — 3 seasons of recent data keeps the model
# relevant to how the game is played today
SEASONS = ['2023-24', '2024-25', '2025-26']

# FEATURES is the single source of truth for what goes into the model.
# This exact list, in this exact order, must be used in both train_model.py
# and app.py. If these are out of sync, the model will make predictions
# using the wrong values with no error message — silent and dangerous.
FEATURES = [
    'pts_l5',           # avg points last 5 games
    'pts_l10',          # avg points last 10 games
    'reb_l5',           # avg rebounds last 5 games
    'reb_l10',          # avg rebounds last 10 games
    'ast_l5',           # avg assists last 5 games
    'ast_l10',          # avg assists last 10 games
    'fg3m_l5',          # avg 3PT made last 5 games
    'fg3m_l10',         # avg 3PT made last 10 games
    'stl_l5',           # avg steals last 5 games
    'blk_l5',           # avg blocks last 5 games
    'tov_l5',           # avg turnovers last 5 games
    'min_l5',           # avg minutes last 5 games
    'is_home',          # 1 = home game, 0 = away game
    'is_b2b',           # 1 = back to back, 0 = not
    'opp_def_rtg',      # opponent defensive rating (lower = tougher defense)
    'opp_pace',         # opponent pace (possessions per 48 min)
    'season_pts_avg',   # season avg points up to this game
    'season_reb_avg',   # season avg rebounds up to this game
    'season_ast_avg',   # season avg assists up to this game
    'season_min_avg',   # season avg minutes up to this game
]

# One XGBoost model gets trained per stat category.
# Key = model name used in app.py, Value = column name in the dataframe.
TARGETS = {
    'pts':  'PTS',
    'reb':  'REB',
    'ast':  'AST',
    'fg3m': 'FG3M',
    'stl':  'STL',
    'blk':  'BLK',
    'tov':  'TOV',
    'pra':  'PRA',
    'pr':   'PR',
    'pa':   'PA',
}

MIN_GAMES    = 20
API_SLEEP    = 1.0
PROGRESS_FILE = 'pull_progress.pkl'


# ============================================================
# STEP 1: GET TOP PLAYERS BY MINUTES
# ============================================================

def get_top_players(season, n=150):
    """
    Returns the top N players by minutes played for a given season
    as a list of (player_id, player_name) tuples.
    """
    print(f"Fetching top {n} players for {season}...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        df = df.sort_values('MIN', ascending=False).head(n)
        return list(zip(df['PLAYER_ID'].tolist(), df['PLAYER_NAME'].tolist()))
    except Exception as e:
        print(f"  Error fetching players for {season}: {e}")
        return []


# ============================================================
# STEP 2: PULL GAME LOGS FOR EACH PLAYER
# ============================================================

def pull_game_logs(player_list, season):
    """
    Pulls game-by-game stats for every player in player_list.
    Saves progress incrementally so interrupted runs can resume.
    Retries each player up to 3 times before skipping.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'rb') as f:
            progress = pickle.load(f)
        print(f"  Resuming — {len(progress)} players already cached.")
    else:
        progress = {}

    all_logs = []

    for player_id, player_name in player_list:
        cache_key = f"{season}_{player_id}"

        if cache_key in progress:
            all_logs.append(progress[cache_key])
            continue

        for attempt in range(3):
            try:
                log = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                )
                df = log.get_data_frames()[0]

                if df.empty:
                    print(f"  No data for {player_name} ({season}), skipping.")
                    break

                df['PLAYER_ID']   = player_id
                df['PLAYER_NAME'] = player_name
                df['SEASON']      = season

                progress[cache_key] = df
                with open(PROGRESS_FILE, 'wb') as f:
                    pickle.dump(progress, f)

                all_logs.append(df)
                print(f"  {player_name}: {len(df)} games")
                time.sleep(API_SLEEP)
                break

            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {player_name}: {e}")
                time.sleep(5)
                if attempt == 2:
                    print(f"  Skipping {player_name}.")

    if not all_logs:
        return pd.DataFrame()

    return pd.concat(all_logs, ignore_index=True)


# ============================================================
# STEP 3: PULL TEAM DEFENSIVE STATS
# ============================================================

def get_team_defense_stats(season):
    """
    Pulls defensive rating and pace for every team.
    These become the opponent context features in training.
    """
    print(f"  Fetching team defense stats for {season}...")
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced'
        )
        df = stats.get_data_frames()[0]
        defense = df[['TEAM_ID', 'TEAM_NAME', 'DEF_RATING', 'PACE']].copy()
        defense.columns = ['OPP_TEAM_ID', 'OPP_TEAM_NAME', 'opp_def_rtg', 'opp_pace']
        time.sleep(API_SLEEP)
        return defense
    except Exception as e:
        print(f"  Error fetching team defense for {season}: {e}")
        return pd.DataFrame()


# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================

def engineer_features(df, defense_df):
    """
    Builds all model features from raw game logs.

    CRITICAL RULE: always shift(1) before rolling to prevent data leakage.
    shift(1) ensures that game N's rolling average only uses games 1..N-1.
    """
    all_players = []

    nba_teams    = teams.get_teams()
    teams_df     = pd.DataFrame(nba_teams)
    abbrev_to_id = dict(zip(teams_df['abbreviation'], teams_df['id']))

    for player_id in df['PLAYER_ID'].unique():
        p = df[df['PLAYER_ID'] == player_id].copy()
        p['GAME_DATE'] = pd.to_datetime(p['GAME_DATE'])
        p = p.sort_values('GAME_DATE').reset_index(drop=True)

        if len(p) < MIN_GAMES:
            continue

        # --- Rolling averages with shift(1) leakage fix ---
        stat_cols = {
            'PTS':  ('pts_l5',  'pts_l10'),
            'REB':  ('reb_l5',  'reb_l10'),
            'AST':  ('ast_l5',  'ast_l10'),
            'FG3M': ('fg3m_l5', 'fg3m_l10'),
            'STL':  ('stl_l5',  None),
            'BLK':  ('blk_l5',  None),
            'TOV':  ('tov_l5',  None),
            'MIN':  ('min_l5',  None),
        }

        for col, (feat_l5, feat_l10) in stat_cols.items():
            if col not in p.columns:
                continue
            shifted = p[col].shift(1)
            p[feat_l5] = shifted.rolling(5,  min_periods=1).mean()
            if feat_l10:
                p[feat_l10] = shifted.rolling(10, min_periods=1).mean()

        # --- Season averages (expanding window, shifted) ---
        for col, feat in [('PTS', 'season_pts_avg'), ('REB', 'season_reb_avg'),
                           ('AST', 'season_ast_avg'), ('MIN', 'season_min_avg')]:
            if col in p.columns:
                p[feat] = p[col].shift(1).expanding().mean()

        # --- Home / Away ---
        p['is_home'] = p['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)

        # --- Back to Back ---
        p['prev_date'] = p['GAME_DATE'].shift(1)
        p['days_rest'] = (p['GAME_DATE'] - p['prev_date']).dt.days
        p['is_b2b']    = p['days_rest'].apply(lambda x: 1 if x == 1 else 0)

        # --- Opponent abbreviation for joining defense stats ---
        p['OPP_ABBREV']  = p['MATCHUP'].apply(lambda x: str(x).split()[-1])
        p['OPP_TEAM_ID'] = p['OPP_ABBREV'].map(abbrev_to_id)

        # --- Combo stat targets ---
        p['PRA'] = p['PTS'] + p['REB'] + p['AST']
        p['PR']  = p['PTS'] + p['REB']
        p['PA']  = p['PTS'] + p['AST']

        all_players.append(p)

    if not all_players:
        return pd.DataFrame()

    result = pd.concat(all_players, ignore_index=True)

    # --- Join opponent defensive stats ---
    if not defense_df.empty:
        result = result.merge(
            defense_df[['OPP_TEAM_ID', 'opp_def_rtg', 'opp_pace']],
            on='OPP_TEAM_ID',
            how='left'
        )
    else:
        result['opp_def_rtg'] = 112.0
        result['opp_pace']    = 98.0

    null_def = result['opp_def_rtg'].isna().sum()
    if null_def > 0:
        print(f"  Warning: {null_def} rows missing opponent defense — filling with league avg.")
        result['opp_def_rtg'] = result['opp_def_rtg'].fillna(112.0)
        result['opp_pace']    = result['opp_pace'].fillna(98.0)

    return result


# ============================================================
# STEP 5: TRAIN ONE MODEL PER STAT CATEGORY
# ============================================================

def train_models(df):
    """
    Trains one XGBoost regression model per stat category.

    Uses TimeSeriesSplit for honest cross-validation.
    Trains final model on all data with chronological early stopping holdout.

    Returns: {stat_name: {'model': XGBRegressor, 'cv_mae': float, 'cv_std': float}}
    """
    models = {}

    if 'GAME_DATE' in df.columns:
        df = df.sort_values('GAME_DATE').reset_index(drop=True)

    for stat_name, target_col in TARGETS.items():
        print(f"\n  Training: {stat_name}")

        if target_col not in df.columns:
            print(f"    Skipping — column {target_col} not found.")
            continue

        model_df = df[FEATURES + [target_col]].dropna()

        if len(model_df) < 200:
            print(f"    Skipping — only {len(model_df)} rows.")
            continue

        X = model_df[FEATURES]
        y = model_df[target_col]

        # --- TimeSeriesSplit cross validation ---
        # 5 folds, always training on past and validating on future
        tscv      = TimeSeriesSplit(n_splits=5)
        fold_maes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.1,
                random_state=42,
                verbosity=0
            )
            fold_model.fit(X_tr, y_tr)

            preds = fold_model.predict(X_v)
            mae   = np.mean(np.abs(preds - y_v))
            fold_maes.append(mae)

        cv_mae = np.mean(fold_maes)
        cv_std = np.std(fold_maes)
        print(f"    CV MAE: {cv_mae:.2f} +/- {cv_std:.2f}")

        # --- Final model on all data with early stopping ---
        split_idx           = int(len(X) * 0.85)
        X_train, X_val      = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val      = y.iloc[:split_idx], y.iloc[split_idx:]

        final_model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            verbosity=0
        )

        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        best_trees = getattr(final_model, 'best_iteration', final_model.n_estimators)
        print(f"    Final model: {best_trees} trees.")

        models[stat_name] = {
            'model':  final_model,
            'cv_mae': round(cv_mae, 2),
            'cv_std': round(cv_std, 2),
        }

    return models


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    all_featured = []

    for season in SEASONS:
        print(f"\n{'='*55}")
        print(f"  SEASON: {season}")
        print(f"{'='*55}")

        player_list = get_top_players(season)
        if not player_list:
            continue

        print(f"\nPulling game logs for {len(player_list)} players...")
        game_logs = pull_game_logs(player_list, season)
        if game_logs.empty:
            continue

        defense_df = get_team_defense_stats(season)

        print(f"\nEngineering features...")
        featured = engineer_features(game_logs, defense_df)

        if not featured.empty:
            all_featured.append(featured)
            print(f"  Added {len(featured)} rows from {season}.")

    if not all_featured:
        print("No data collected. Exiting.")
        exit()

    full_df = pd.concat(all_featured, ignore_index=True)
    print(f"\nTotal rows: {len(full_df)} | Unique players: {full_df['PLAYER_ID'].nunique()}")

    print("\nTraining models...")
    models = train_models(full_df)

    save_data = {
        'models':   models,
        'features': FEATURES,
        'targets':  TARGETS,
    }

    with open('nba_models.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nSaved {len(models)} models to nba_models.pkl")

    print("\n--- Accuracy Summary ---")
    for stat, data in models.items():
        print(f"  {stat:6s}: MAE {data['cv_mae']:.2f} +/- {data['cv_std']:.2f}")

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("\nProgress cache cleaned up.")