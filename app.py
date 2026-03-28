"""
app.py
======
The full Streamlit app for the NBA Props Predictor.

Flow:
1. Load trained models from nba_models.pkl
2. Pull today's games from nba_api
3. User selects: Game → Team → Player → Stat → Line → Over/Under
4. App builds feature vector and runs XGBoost prediction
5. Converts prediction to probability and displays recommendation

Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

ET = ZoneInfo("America/New_York")

from nba_api.stats.endpoints import (
    scoreboardv2,
    playergamelog,
    leaguedashteamstats,
    commonteamroster,
)
from nba_api.stats.static import teams as nba_teams_static

# ============================================================
# PAGE CONFIG — must be the first Streamlit call
# ============================================================

st.set_page_config(
    page_title="NBA Props",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — dark sports app theme
# ============================================================

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

    /* Base dark theme */
    .stApp {
        background-color: #0a0a0f;
        color: #e8e8e8;
        font-family: 'Barlow', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #1e1e2e;
        margin-bottom: 2rem;
    }

    .app-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ffffff 0%, #c084fc 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
    }

    .app-subtitle {
        font-family: 'Barlow', sans-serif;
        font-size: 0.85rem;
        font-weight: 400;
        color: #6b7280;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* Section labels */
    .section-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }

    /* Game cards */
    .game-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .game-card:hover {
        border-color: #7c3aed;
        background: #16161f;
    }

    .game-card.selected {
        border-color: #7c3aed;
        background: #1a1028;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.15);
    }

    .game-card-teams {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.05em;
    }

    .game-card-time {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Player context card */
    .context-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .context-player-name {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }

    .context-matchup {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }

    .context-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-right: 0.4rem;
    }

    .badge-home { background: #052e16; color: #4ade80; border: 1px solid #166534; }
    .badge-away { background: #1c1917; color: #a8a29e; border: 1px solid #44403c; }
    .badge-b2b  { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
    .badge-rest { background: #0c1a3a; color: #60a5fa; border: 1px solid #1e3a5f; }
    .badge-low  { background: #451a03; color: #fb923c; border: 1px solid #7c2d12; }

    /* Stat pills */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .stat-pill {
        background: #1a1a2e;
        border: 1px solid #2d2d4a;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        text-align: center;
        min-width: 80px;
    }

    .stat-pill-value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }

    .stat-pill-label {
        font-size: 0.65rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
    }

    /* Defense meter */
    .defense-bar-container {
        margin-top: 1rem;
    }

    .defense-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }

    /* Result card */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }

    .result-card-over {
        background: linear-gradient(135deg, #052e16 0%, #0d3321 100%);
        border: 1px solid #166534;
    }

    .result-card-under {
        background: linear-gradient(135deg, #450a0a 0%, #3d1515 100%);
        border: 1px solid #7f1d1d;
    }

    .result-card-neutral {
        background: linear-gradient(135deg, #111118 0%, #1a1a2e 100%);
        border: 1px solid #2d2d4a;
    }

    .result-probability {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }

    .result-probability-over { color: #4ade80; }
    .result-probability-under { color: #f87171; }
    .result-probability-neutral { color: #9ca3af; }

    .result-recommendation {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .result-predicted {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-top: 0.75rem;
    }

    .result-accuracy {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }

    /* Selectbox and input styling */
    .stSelectbox > div > div {
        background-color: #111118 !important;
        border-color: #2d2d4a !important;
        color: #ffffff !important;
    }

    .stNumberInput > div > div > input {
        background-color: #111118 !important;
        border-color: #2d2d4a !important;
        color: #ffffff !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid #1e1e2e;
        margin: 1.5rem 0;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.7rem;
        color: #4b5563;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1e1e2e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS — must match train_model.py exactly
# ============================================================

FEATURES = [
    'pts_l5', 'pts_l10', 'reb_l5', 'reb_l10', 'ast_l5', 'ast_l10',
    'fg3m_l5', 'fg3m_l10', 'stl_l5', 'blk_l5', 'tov_l5', 'min_l5',
    'is_home', 'is_b2b', 'opp_def_rtg', 'opp_pace',
    'season_pts_avg', 'season_reb_avg', 'season_ast_avg', 'season_min_avg',
]

STAT_LABELS = {
    'pts':  'Points',
    'reb':  'Rebounds',
    'ast':  'Assists',
    'fg3m': '3PT Made',
    'stl':  'Steals',
    'blk':  'Blocks',
    'tov':  'Turnovers',
    'pra':  'Pts+Reb+Ast',
    'pr':   'Pts+Reb',
    'pa':   'Pts+Ast',
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():
    """
    Loads the trained models from nba_models.pkl.
    cache_resource means this only runs once per app session —
    the models stay in memory rather than reloading on every interaction.
    """
    try:
        with open('nba_models.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("nba_models.pkl not found. Run train_model.py first.")
        st.stop()

# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================

def _minutes_to_float(val):
    """nba_api sometimes returns MIN as float minutes or 'MM:SS'."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if ':' in s:
        m, sec = s.split(':', 1)
        return float(m) + float(sec) / 60.0
    return float(s)


@st.cache_data(ttl=3600)
def get_todays_games():
    """
    Pulls today's NBA schedule.
    ttl=3600 means this refreshes every hour automatically.
    Returns a list of game dicts with team names, IDs, and game ID.
    """
    try:
        now_et = datetime.now(ET)
        today = now_et.strftime('%m/%d/%Y')
        board = scoreboardv2.ScoreboardV2(game_date=today)
        games_df = board.get_data_frames()[0]

        if games_df.empty:
            return []

        # Get team info for name lookup
        all_teams = nba_teams_static.get_teams()
        id_to_team = {t['id']: t for t in all_teams}

        slate_date = pd.Timestamp(now_et.date())
        games = []
        for _, row in games_df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            home = id_to_team.get(home_id, {})
            away = id_to_team.get(away_id, {})

            games.append({
                'game_id':        row['GAME_ID'],
                'home_team_id':   home_id,
                'away_team_id':   away_id,
                'home_team_name': home.get('full_name', 'Unknown'),
                'away_team_name': away.get('full_name', 'Unknown'),
                'home_abbrev':    home.get('abbreviation', '???'),
                'away_abbrev':    away.get('abbreviation', '???'),
                'game_date_et':   slate_date,
            })

        return games

    except Exception as e:
        st.warning(f"Could not load today's games: {e}")
        return []


@st.cache_data(ttl=3600)
def get_team_roster(team_id):
    """
    Gets the current roster for a team.
    Returns list of (player_id, player_name) tuples.
    """
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        df = roster.get_data_frames()[0]
        time.sleep(0.6)
        return list(zip(df['PLAYER_ID'].tolist(), df['PLAYER'].tolist()))
    except Exception as e:
        return []


@st.cache_data(ttl=3600)
def get_player_recent_games(player_id, season='2025-26'):
    """
    Pulls the player's game log for the current season.
    We use this to build the feature vector at prediction time.
    """
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season
        )
        df = log.get_data_frames()[0]
        time.sleep(0.6)

        if df.empty:
            return pd.DataFrame()

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        if 'MIN' in df.columns:
            df['MIN'] = df['MIN'].map(_minutes_to_float)
        return df

    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_team_defense_current():
    """
    Pulls current season defensive ratings and pace for all teams.
    Used to get tonight's opponent's defensive context.
    """
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season='2025-26',
            measure_type_detailed_defense='Advanced'
        )
        df = stats.get_data_frames()[0]
        time.sleep(0.6)

        defense = df[['TEAM_ID', 'DEF_RATING', 'PACE']].copy()
        defense.columns = ['team_id', 'def_rtg', 'pace']
        return defense

    except Exception as e:
        return pd.DataFrame()

# ============================================================
# FEATURE BUILDER
# ============================================================

def build_feature_vector(player_logs, is_home, is_b2b, opp_def_rtg, opp_pace):
    """
    Builds the exact feature vector the model expects.
    Uses the player's most recent games to calculate rolling averages.

    This must produce features in the same order as FEATURES and
    use the same definitions as train_model.py — otherwise the model
    makes predictions on the wrong values.

    Returns a 1-row numpy array ready for model.predict().
    """
    if player_logs.empty or len(player_logs) < 1:
        return None

    # Use all available games — most recent last
    df = player_logs.copy()

    def safe_mean(series, n):
        """Average of last n values, or all values if fewer than n exist."""
        vals = series.dropna().tail(n)
        return float(vals.mean()) if len(vals) > 0 else 0.0

    # Rolling averages — last 5 and last 10 games
    pts_l5   = safe_mean(df['PTS'],  5)
    pts_l10  = safe_mean(df['PTS'],  10)
    reb_l5   = safe_mean(df['REB'],  5)
    reb_l10  = safe_mean(df['REB'],  10)
    ast_l5   = safe_mean(df['AST'],  5)
    ast_l10  = safe_mean(df['AST'],  10)
    fg3m_l5  = safe_mean(df['FG3M'], 5)
    fg3m_l10 = safe_mean(df['FG3M'], 10)
    stl_l5   = safe_mean(df['STL'],  5)
    blk_l5   = safe_mean(df['BLK'],  5)
    tov_l5   = safe_mean(df['TOV'],  5)
    min_l5   = safe_mean(df['MIN'],  5)

    # Season averages — mean of all completed games in the log.
    # Training used shift(1).expanding().mean(): on row i that's mean(games 0..i-1).
    # Predicting the *next* game after the last log row (index N) matches row N+1,
    # where season avg = mean(games 0..N) = full log mean.
    season_pts_avg = float(df['PTS'].mean())
    season_reb_avg = float(df['REB'].mean())
    season_ast_avg = float(df['AST'].mean())
    season_min_avg = float(df['MIN'].mean())

    # Build the feature vector in the exact order of FEATURES
    vector = [
        pts_l5, pts_l10, reb_l5, reb_l10, ast_l5, ast_l10,
        fg3m_l5, fg3m_l10, stl_l5, blk_l5, tov_l5, min_l5,
        float(is_home), float(is_b2b),
        float(opp_def_rtg), float(opp_pace),
        season_pts_avg, season_reb_avg, season_ast_avg, season_min_avg,
    ]

    return np.array(vector).reshape(1, -1)


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_prop(model_data, stat_key, feature_vector):
    """
    Runs the feature vector through the XGBoost model and returns
    the predicted stat value.
    """
    model_info = model_data['models'].get(stat_key)
    if model_info is None:
        return None, None

    model  = model_info['model']
    cv_mae = model_info['cv_mae']

    prediction = float(model.predict(feature_vector)[0])
    prediction = max(0, prediction)  # stats can't be negative

    return prediction, cv_mae


def prediction_to_probability(predicted, line, mae):
    """
    Converts a regression prediction to a probability of going over the line.

    We assume the model's errors follow a normal distribution centered
    at the predicted value with standard deviation = MAE.
    Then we ask: what fraction of that distribution falls above the line?

    Example:
      Predicted: 27.3 pts, Line: 24.5, MAE: 5.73
      → probability that actual value > 24.5 given N(27.3, 5.73)
      → scipy.stats.norm.sf(24.5, loc=27.3, scale=5.73)
      → about 0.69 = 69% probability of going over
    """
    if mae <= 0:
        mae = 1.0

    # sf = survival function = 1 - CDF = P(X > line)
    prob_over = float(stats.norm.sf(line, loc=predicted, scale=mae))
    prob_over = max(0.01, min(0.99, prob_over))  # clamp between 1% and 99%
    return prob_over


def get_recommendation(prob_over, direction):
    """
    Maps a probability to a betting recommendation label.

    If user picked Over:
      prob_over is directly the probability of their bet hitting.
    If user picked Under:
      probability of their bet hitting = 1 - prob_over.
    """
    prob_bet = prob_over if direction == 'Over' else (1 - prob_over)

    if prob_bet >= 0.65:
        return f"STRONG {direction.upper()}", "strong"
    elif prob_bet >= 0.55:
        return f"LEAN {direction.upper()}", "lean"
    else:
        return "NO EDGE", "neutral"


# ============================================================
# APP HEADER
# ============================================================

st.markdown("""
<div class="app-header">
    <p class="app-title">🏀 NBA Props</p>
    <p class="app-subtitle">AI-Powered Prop Predictions · Updated Daily</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

model_data   = load_models()
todays_games = get_todays_games()
defense_df   = get_team_defense_current()

if not todays_games:
    st.markdown("""
    <div style="text-align:center; padding: 4rem; color: #6b7280;">
        <div style="font-size:3rem">🏀</div>
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.5rem; 
                    font-weight:700; color:#9ca3af; margin-top:1rem;">
            No Games Today
        </div>
        <div style="font-size:0.85rem; margin-top:0.5rem;">
            Check back on a game day
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# STEP 1: SELECT GAME
# ============================================================

st.markdown('<p class="section-label">Step 1 — Select Game</p>', unsafe_allow_html=True)

game_options = {
    f"{g['away_abbrev']} @ {g['home_abbrev']}": g
    for g in todays_games
}

selected_game_label = st.selectbox(
    "Game",
    options=list(game_options.keys()),
    label_visibility="collapsed"
)

selected_game = game_options[selected_game_label]

# ============================================================
# STEP 2: SELECT TEAM
# ============================================================

st.markdown('<p class="section-label">Step 2 — Select Team</p>', unsafe_allow_html=True)

team_options = {
    selected_game['away_team_name']: {
        'team_id':   selected_game['away_team_id'],
        'is_home':   0,
        'opp_id':    selected_game['home_team_id'],
    },
    selected_game['home_team_name']: {
        'team_id':   selected_game['home_team_id'],
        'is_home':   1,
        'opp_id':    selected_game['away_team_id'],
    },
}

col1, col2 = st.columns(2)
with col1:
    away_name = selected_game['away_team_name']
    if st.button(f"✈️ {away_name}", key="away_btn"):
        st.session_state['selected_team'] = away_name
with col2:
    home_name = selected_game['home_team_name']
    if st.button(f"🏠 {home_name}", key="home_btn"):
        st.session_state['selected_team'] = home_name

# Default to away team if nothing selected yet
if 'selected_team' not in st.session_state:
    st.session_state['selected_team'] = away_name

# Check if the selected team is still valid for the current game
if st.session_state['selected_team'] not in team_options:
    st.session_state['selected_team'] = away_name

selected_team_name = st.session_state['selected_team']
selected_team_info = team_options[selected_team_name]

# ============================================================
# STEP 3: SELECT PLAYER
# ============================================================

st.markdown('<p class="section-label">Step 3 — Select Player</p>', unsafe_allow_html=True)

roster = get_team_roster(selected_team_info['team_id'])

if not roster:
    st.warning("Could not load roster. Try again in a moment.")
    st.stop()

player_names  = [name for _, name in roster]
player_id_map = {name: pid for pid, name in roster}

selected_player_name = st.selectbox(
    "Player",
    options=player_names,
    label_visibility="collapsed"
)

selected_player_id = player_id_map[selected_player_name]

# ============================================================
# LOAD PLAYER DATA
# ============================================================

with st.spinner(f"Loading {selected_player_name}'s stats..."):
    player_logs = get_player_recent_games(selected_player_id)

if player_logs.empty:
    st.warning(f"No 2025-26 data found for {selected_player_name}.")
    st.stop()

# Low confidence warning for players with few games
low_confidence = len(player_logs) < 30

# Get opponent defensive stats
opp_team_id  = selected_team_info['opp_id']
opp_def_rtg  = 112.0  # league average fallback
opp_pace     = 98.0

if not defense_df.empty:
    opp_row = defense_df[defense_df['team_id'] == opp_team_id]
    if not opp_row.empty:
        opp_def_rtg = float(opp_row['def_rtg'].values[0])
        opp_pace    = float(opp_row['pace'].values[0])

# Back to back — matches training: days between consecutive games == 1.
# Use slate date (ET) from scoreboard so it matches the game we're predicting,
# not the laptop's local calendar.
is_home = selected_team_info['is_home']
is_b2b  = 0

if len(player_logs) >= 1 and 'game_date_et' in selected_game:
    last_game = pd.Timestamp(player_logs['GAME_DATE'].iloc[-1]).normalize()
    slate     = pd.Timestamp(selected_game['game_date_et']).normalize()
    days_rest = (slate - last_game).days
    is_b2b = 1 if days_rest == 1 else 0

# ============================================================
# PLAYER CONTEXT CARD
# ============================================================

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Calculate display stats
pts_l5  = float(player_logs['PTS'].tail(5).mean())
reb_l5  = float(player_logs['REB'].tail(5).mean())
ast_l5  = float(player_logs['AST'].tail(5).mean())
pts_avg = float(player_logs['PTS'].mean())
reb_avg = float(player_logs['REB'].mean())
ast_avg = float(player_logs['AST'].mean())
min_avg = float(player_logs['MIN'].mean())

opp_name = selected_game['home_team_name'] if is_home == 0 else selected_game['away_team_name']

# Defensive rating label
if opp_def_rtg <= 108:
    def_label = "🔒 Elite Defense"
    def_color = "#f87171"
elif opp_def_rtg <= 112:
    def_label = "💪 Good Defense"
    def_color = "#fb923c"
elif opp_def_rtg <= 116:
    def_label = "😐 Average Defense"
    def_color = "#facc15"
else:
    def_label = "🧁 Weak Defense"
    def_color = "#4ade80"

home_away_badge = '<span class="context-badge badge-home">HOME</span>' if is_home else '<span class="context-badge badge-away">AWAY</span>'
b2b_badge       = '<span class="context-badge badge-b2b">BACK TO BACK</span>' if is_b2b else '<span class="context-badge badge-rest">RESTED</span>'
conf_badge      = '<span class="context-badge badge-low">⚠️ LIMITED DATA</span>' if low_confidence else ''

st.markdown(f"""
<div class="context-card">
    <p class="context-player-name">{selected_player_name}</p>
    <p class="context-matchup">vs {opp_name} &nbsp;·&nbsp; {selected_team_name}</p>
    <div style="margin-top:0.75rem;">
        {home_away_badge}{b2b_badge}{conf_badge}
    </div>
    <div style="margin-top:1rem; display:flex; gap:2rem; flex-wrap:wrap;">
        <div>
            <div style="font-size:0.7rem; color:#6b7280; text-transform:uppercase; 
                        letter-spacing:0.1em;">Opp Defense</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; 
                        font-weight:700; color:{def_color};">{def_label} ({opp_def_rtg:.1f})</div>
        </div>
        <div>
            <div style="font-size:0.7rem; color:#6b7280; text-transform:uppercase; 
                        letter-spacing:0.1em;">Opp Pace</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; 
                        font-weight:700; color:#e8e8e8;">{opp_pace:.1f}</div>
        </div>
        <div>
            <div style="font-size:0.7rem; color:#6b7280; text-transform:uppercase; 
                        letter-spacing:0.1em;">Avg MIN</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; 
                        font-weight:700; color:#e8e8e8;">{min_avg:.1f}</div>
        </div>
    </div>
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-pill-value">{pts_l5:.1f}</div>
            <div class="stat-pill-label">PTS L5</div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-value">{reb_l5:.1f}</div>
            <div class="stat-pill-label">REB L5</div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-value">{ast_l5:.1f}</div>
            <div class="stat-pill-label">AST L5</div>
        </div>
        <div class="stat-pill" style="background:#0f1f0f; border-color:#1a3a1a;">
            <div class="stat-pill-value" style="color:#9ca3af;">{pts_avg:.1f}</div>
            <div class="stat-pill-label">PTS AVG</div>
        </div>
        <div class="stat-pill" style="background:#0f1f0f; border-color:#1a3a1a;">
            <div class="stat-pill-value" style="color:#9ca3af;">{reb_avg:.1f}</div>
            <div class="stat-pill-label">REB AVG</div>
        </div>
        <div class="stat-pill" style="background:#0f1f0f; border-color:#1a3a1a;">
            <div class="stat-pill-value" style="color:#9ca3af;">{ast_avg:.1f}</div>
            <div class="stat-pill-label">AST AVG</div>
        </div>
    </div>
    <div style="font-size:0.7rem; color:#4b5563; margin-top:1rem;">
        Based on {len(player_logs)} games this season
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STEP 4: PROP INPUTS
# ============================================================

st.markdown('<p class="section-label">Step 4 — Enter Prop</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1.5, 1.5])

with col1:
    stat_label    = st.selectbox("Stat", list(STAT_LABELS.values()), label_visibility="collapsed")
    stat_key      = [k for k, v in STAT_LABELS.items() if v == stat_label][0]

with col2:
    prop_line = st.number_input("Line", min_value=0.0, max_value=100.0,
                                 value=20.5, step=0.5, label_visibility="collapsed")

with col3:
    direction = st.radio("Direction", ["Over", "Under"],
                          horizontal=True, label_visibility="collapsed")

# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
predict_clicked = st.button("🔮  PREDICT", key="predict_btn")
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION RESULT
# ============================================================

if predict_clicked:
    # Build feature vector
    feature_vector = build_feature_vector(
        player_logs=player_logs,
        is_home=is_home,
        is_b2b=is_b2b,
        opp_def_rtg=opp_def_rtg,
        opp_pace=opp_pace,
    )

    if feature_vector is None:
        st.error("Not enough data to make a prediction.")
    else:
        # Run prediction
        predicted, cv_mae = predict_prop(model_data, stat_key, feature_vector)

        if predicted is None:
            st.error(f"No model found for {stat_label}.")
        else:
            # Convert to probability
            prob_over = prediction_to_probability(predicted, prop_line, cv_mae)
            prob_bet  = prob_over if direction == 'Over' else (1 - prob_over)
            recommendation, rec_type = get_recommendation(prob_over, direction)

            # Pick colors based on recommendation
            if rec_type == "strong":
                card_class = "result-card-over" if direction == "Over" else "result-card-under"
                prob_class = "result-probability-over" if direction == "Over" else "result-probability-under"
            elif rec_type == "lean":
                card_class = "result-card-over" if direction == "Over" else "result-card-under"
                prob_class = "result-probability-over" if direction == "Over" else "result-probability-under"
            else:
                card_class = "result-card-neutral"
                prob_class = "result-probability-neutral"

            rec_color = "#4ade80" if (rec_type != "neutral" and direction == "Over") else \
                        "#f87171" if (rec_type != "neutral" and direction == "Under") else "#9ca3af"

            st.markdown(f"""
            <div class="result-card {card_class}">
                <p class="result-probability {prob_class}">{prob_bet:.0%}</p>
                <p class="result-recommendation" style="color:{rec_color};">
                    {recommendation}
                </p>
                <p class="result-predicted">
                    Model projection: <strong>{predicted:.1f} {stat_label}</strong> 
                    &nbsp;·&nbsp; Line: {prop_line} {direction}
                </p>
                <p class="result-accuracy">
                    Model avg error: ±{cv_mae:.1f} · Based on {len(player_logs)} games this season
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown(f"""
            <div style="display:flex; gap:1rem; margin-top:1rem;">
                <div class="stat-pill" style="flex:1; background:#052e16; border-color:#166534;">
                    <div class="stat-pill-value" style="color:#4ade80;">{prob_over:.0%}</div>
                    <div class="stat-pill-label">Over {prop_line}</div>
                </div>
                <div class="stat-pill" style="flex:1; background:#450a0a; border-color:#7f1d1d;">
                    <div class="stat-pill-value" style="color:#f87171;">{1-prob_over:.0%}</div>
                    <div class="stat-pill-label">Under {prop_line}</div>
                </div>
                <div class="stat-pill" style="flex:1;">
                    <div class="stat-pill-value">{predicted:.1f}</div>
                    <div class="stat-pill-label">Projection</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="disclaimer">
    For entertainment purposes only. Always check injury reports before betting. 
    Past model performance does not guarantee future results. Bet responsibly.
</div>
""", unsafe_allow_html=True)