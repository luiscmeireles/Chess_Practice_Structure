import pandas as pd
from pandas import Period
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
import chess
from stockfish import Stockfish
import re
from datetime import datetime
from collections import Counter
from functools import partial
import ast
from scipy.stats import linregress
from collections import defaultdict
import chess

def convert_monthly_games_to_dict(monthly_games_str):
    """
    Converts a stringified monthly_played_games (e.g., "{Period('2014-10', 'M'): 66, ...}")
    into a dictionary with keys in 'MM-YYYY' format and game counts as values.

    Parameters:
        monthly_games_str (str): String representation of a dictionary with Period objects.

    Returns:
        dict: Dictionary with keys as 'MM-YYYY' strings and values as game counts.
              Returns empty dict if parsing fails or input is invalid.
    """
    if not isinstance(monthly_games_str, str) or not monthly_games_str:
        return {}
    
    # Remove outer braces and check if empty
    s = monthly_games_str.strip('{}')
    if not s:
        return {}
    
    # Use regex to find Period('YYYY-MM', 'M') and counts
    pattern = r"Period\('(\d{4}-\d{2})', 'M'\):\s*(\d+)"
    matches = re.findall(pattern, s)
    
    # Create dictionary with MM-YYYY keys
    result = {}
    for period, count in matches:
        year, month = period.split('-')
        mm_yyyy = f"{month}-{year}"
        result[mm_yyyy] = int(count)
    
    return result

def filter_last_month_players(df):
    # Step 1: Find the most recent month across all players
    def get_all_months(games_dict):
        if not isinstance(games_dict, dict) or not games_dict:
            return set()
        try:
            return {datetime.strptime(month, '%m-%Y') for month in games_dict.keys()}
        except ValueError:
            return set()

    # Collect all unique months in the dataset
    all_months = set()
    for games_dict in df['monthly_played_games']:
        all_months.update(get_all_months(games_dict))
    
    if not all_months:
        return df.iloc[0:0]  # Return empty DataFrame if no valid months
    
    # Find the latest month in the dataset
    latest_month = max(all_months)
    latest_month_str = latest_month.strftime('%m-%Y')

    # Step 2: Identify players with games in the latest month
    def has_games_in_last_month(games_dict):
        if not isinstance(games_dict, dict) or not games_dict:
            return False
        try:
            return latest_month_str in games_dict and games_dict[latest_month_str] > 0
        except ValueError:
            return False

    # Get player IDs with games in the last month
    valid_players = df[df['monthly_played_games'].apply(has_games_in_last_month)]['played_id'].unique()

    # Step 3: Return all data for those players
    return df[df['played_id'].isin(valid_players)]

def has_enough_recent_months(monthly_dict):
    final_months = pd.period_range(start='2014-01', end='2015-12', freq='M')
    if isinstance(monthly_dict, str):
        monthly_dict = eval(monthly_dict)
    months_played = pd.to_datetime(list(monthly_dict.keys()), errors='coerce', format='%m-%Y').to_period('M')
    recent_overlap = sum(month in months_played for month in final_months)
    return recent_overlap >= 18  # You can tune this threshold (e.g., 18, 20, etc.)

def parse_to_dict(x):
    """Convert string to dict, return {} for invalid inputs."""
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return x if isinstance(x, dict) else {}

def standardize_month_format(month_dict):
    """Convert month keys from YYYY-MM to MM-YYYY in a dictionary."""
    if not isinstance(month_dict, dict):
        return {}
    new_dict = {}
    for month, value in month_dict.items():
        try:
            # Convert YYYY-MM to MM-YYYY (e.g., '2013-11' to '11-2013')
            year, mon = month.split('-')
            new_month = f"{mon}-{year}"
            new_dict[new_month] = value
        except (ValueError, AttributeError):
            continue
    return new_dict

def assign_player_id_and_color(row, players_list):
    if row['White'] in players_list:
        return pd.Series({'player_id': row['White'], 'player_color': 'white'})
    elif row['Black'] in players_list:
        return pd.Series({'player_id': row['Black'], 'player_color': 'black'})
    return pd.Series({'player_id': None, 'player_color': None})

def to_month_year(date_str):
    """
    Convert a date string into 'MM-YY' format.

    Parameters:
    - date_str: A string representing a date.

    Returns:
    - A string in 'MM-YY' format if conversion is successful, otherwise None.
    """
    try:
        date = pd.to_datetime(date_str)
        return date.strftime('%m-%y')
    except:
        return None


def add_first_last_match_month(df):
    """
    Adds 'first_match_month' and 'last_match_month' columns to the DataFrame
    based on the 'monthly_played_games' column.

    Args:
        df (pd.DataFrame): DataFrame with a 'monthly_played_games' column
                           (dictionary of pandas Period or 'YYYY-MM' string: count).

    Returns:
        pd.DataFrame: DataFrame with added 'first_match_month' and
                      'last_match_month' columns.
    """
    first_match_months = {}
    last_match_months = {}

    for player_id, row in df.iterrows():
        monthly_games = row['monthly_played_games']
        if isinstance(monthly_games, dict) and monthly_games:
            active_months = []
            for period, count in monthly_games.items():
                if count > 0:
                    if isinstance(period, pd.Period):
                        active_months.append(period)
                    elif isinstance(period, str):
                        try:
                            active_months.append(pd.Period(period, freq='M'))
                        except ValueError:
                            print(f"Warning: Could not parse month string '{period}' for player {player_id}")

            if active_months:
                first_month = min(active_months)
                last_month = max(active_months)
                first_match_months[player_id] = str(first_month)
                last_match_months[player_id] = str(last_month)
            else:
                first_match_months[player_id] = None
                last_match_months[player_id] = None
        else:
            first_match_months[player_id] = None
            last_match_months[player_id] = None

    df['first_match_month'] = pd.Series(first_match_months)
    df['last_match_month'] = pd.Series(last_match_months)
    return df

def filter_last_month_players(df):
    # Step 1: Find the most recent month across all players
    def get_all_months(games_dict):
        if not isinstance(games_dict, dict) or not games_dict:
            return set()
        try:
            return {datetime.strptime(month, '%m-%Y') for month in games_dict.keys()}
        except ValueError:
            return set()

    # Collect all unique months in the dataset
    all_months = set()
    for games_dict in df['monthly_played_games']:
        all_months.update(get_all_months(games_dict))
    
    if not all_months:
        return df.iloc[0:0]  # Return empty DataFrame if no valid months
    
    # Find the latest month in the dataset
    latest_month = max(all_months)
    latest_month_str = latest_month.strftime('%m-%Y')

    # Step 2: Identify players with games in the latest month
    def has_games_in_last_month(games_dict):
        if not isinstance(games_dict, dict) or not games_dict:
            return False
        try:
            return latest_month_str in games_dict and games_dict[latest_month_str] > 0
        except ValueError:
            return False

    # Get player IDs with games in the last month
    valid_players = df[df['monthly_played_games'].apply(has_games_in_last_month)]['played_id'].unique()

    # Step 3: Return all data for those players
    return df[df['played_id'].isin(valid_players)]

def convert_to_datetime(date_string):
    try:
        return datetime.strptime(date_string + '-01', '%Y-%m-%d')
    except ValueError as e:
        print(f"Error converting '{date_string}': {e}")
        return pd.NaT  # Or some other appropriate missing value
    
def calculate_active_months(row):
    try:
        first_month = pd.to_datetime(row['first_match_month']).to_period('M')
        last_month = pd.to_datetime(row['last_match_month']).to_period('M')
        return (last_month - first_month).n + 1
    except ValueError:
        return None # Or some other default value, like 0, np.nan, or pd.NaT
    
from collections import Counter
import numpy as np
import pandas as pd
from functools import partial

from collections import Counter
import numpy as np

from collections import Counter
import numpy as np

def compute_monthly_positional_ngram_entropy(player_id, player_games, n_gram_size=2):
    """
    Compute monthly n-gram entropy separately for games where the player is white and black.
    Returns two dictionaries: one for white games, one for black games.
    """

    if n_gram_size < 2:
        raise ValueError("n_gram_size must be at least 2")

    games = player_games[player_games['player_id'] == player_id]
    if games.empty:
        return {'no_data': 0}, {'no_data': 0}

    monthly_entropy_white = {}
    monthly_entropy_black = {}

    for color, entropy_dict in [('white', monthly_entropy_white), ('black', monthly_entropy_black)]:
        color_games = games[games['player_color'].str.lower() == color]
        if color_games.empty:
            continue

        for month in color_games['month_year'].unique():
            month_games = color_games[color_games['month_year'] == month]

            ngram_counts = Counter()
            total_possible_ngrams = 0

            for game in month_games['Moves']:
                if not isinstance(game, list) or len(game) < n_gram_size:
                    continue

                start_idx = 0 if color == 'white' else 1
                player_moves = game[start_idx::2]
                if len(player_moves) < n_gram_size:
                    continue

                num_ngrams = len(player_moves) - n_gram_size + 1
                total_possible_ngrams += num_ngrams

                for i in range(num_ngrams):
                    ngram = tuple(player_moves[i:i + n_gram_size])
                    if all(ngram):  # Skip empty strings or None
                        ngram_counts[ngram] += 1

            if not ngram_counts:
                entropy_dict[month] = 0
                continue

            total_ngrams = sum(ngram_counts.values())
            entropy = -sum((count / total_ngrams) * np.log2(count / total_ngrams)
                           for count in ngram_counts.values() if count > 0)
            entropy_dict[month] = entropy if not np.isnan(entropy) else 0

    if not monthly_entropy_white:
        monthly_entropy_white = {'no_data': 0}
    if not monthly_entropy_black:
        monthly_entropy_black = {'no_data': 0}

    return monthly_entropy_white, monthly_entropy_black

def preprocess_games(player_games):
    """
    Pre-process player_games by converting Moves to lists and optimizing dtypes.
    
    Parameters:
    - player_games: DataFrame with player_id, month_year, Moves, player_color.
    
    Returns:
    - Processed DataFrame.
    """
    player_games = player_games.astype({'player_color': 'category'})
    player_games['Moves'] = player_games['Moves'].apply(
        lambda x: x.strip().split() if isinstance(x, str) and x.strip() else []
    )
    return player_games

# Usage: Add multiple n-gram entropy columns
def add_ngram_entropy_columns(df, player_games, n_gram_sizes=[2, 4, 6, 8]):
    """
    Add columns to df for different n-gram entropy values.
    
    Parameters:
    - df: DataFrame with player_id.
    - player_games: Pre-processed DataFrame.
    - n_gram_sizes: List of n-gram sizes to compute (e.g., [2, 4, 6, 8]).
    
    Returns:
    - df with new columns (e.g., monthly_entropy_2, monthly_entropy_4).
    """
    player_games = preprocess_games(player_games)
    for n in n_gram_sizes:
        column_name = f'monthly_entropy_{n}'
        entropy_func = partial(compute_monthly_positional_ngram_entropy, 
                              player_games=player_games, n_gram_size=n)
        df[column_name] = df['player_id'].apply(entropy_func)
    return df

def extract_entropy_trend_features(entropy_dict, monthly_played_games):
    if not isinstance(entropy_dict, dict) or len(entropy_dict) < 3:
        return {k: np.nan for k in [
            "mean_slope", "early_slope", "late_slope",
            "slope_std", "max_slope", "min_slope",
            "sign_flips", "consistency"
        ]}

    try:
        # Convert to DataFrame for aligned processing
        entropy_series = pd.Series(entropy_dict)
        games_series = pd.Series(monthly_played_games)

        # Parse dates and align
        entropy_series.index = pd.to_datetime(entropy_series.index.astype(str).str.strip(), format='%m-%Y', errors='coerce')
        games_series.index = pd.to_datetime(games_series.index.astype(str).str.strip(), format='%m-%Y', errors='coerce')


        df = pd.DataFrame({
            "entropy": entropy_series,
            "games": games_series
        }).dropna().sort_index()
    except:
        return {k: np.nan for k in [
            "mean_slope", "early_slope", "late_slope",
            "slope_std", "max_slope", "min_slope",
            "sign_flips", "consistency"
        ]}

    if len(df) < 3:
        return {k: np.nan for k in [
            "mean_slope", "early_slope", "late_slope",
            "slope_std", "max_slope", "min_slope",
            "sign_flips", "consistency"
        ]}

    x = np.arange(len(df))
    y = df["entropy"].values
    weights = df["games"].values

    local_slopes = np.diff(y)
    local_weights = (weights[:-1] + weights[1:]) / 2  # average weight between two months

    # Weighted slope statistics
    mean_slope = np.average(local_slopes, weights=local_weights)
    slope_std = np.sqrt(np.average((local_slopes - mean_slope) ** 2, weights=local_weights))
    max_slope = local_slopes[np.argmax(local_weights)]
    min_slope = local_slopes[np.argmin(local_weights)]

    slope_early = slope_late = np.nan
    if len(df) >= 3:
        slope_early, *_ = linregress(x[:3], y[:3])
        slope_late, *_ = linregress(x[-3:], y[-3:])

    signs = np.sign(local_slopes)
    sign_flips = np.sum(signs[:-1] != signs[1:])
    consistency = 1 - sign_flips / len(local_slopes) if len(local_slopes) > 0 else np.nan

    return {
        "mean_slope": mean_slope,
        "early_slope": slope_early,
        "late_slope": slope_late,
        "slope_std": slope_std,
        "max_slope": max_slope,
        "min_slope": min_slope,
        "sign_flips": sign_flips,
        "consistency": consistency,
    }


def extract_entropy_features(df, entropy_columns, extractor_func, game_count_col='monthly_played_games'):
    """
    Applies entropy trend feature extraction to each specified entropy column.

    Parameters:
    - df: pandas DataFrame with entropy columns and a game count column.
    - entropy_columns: list of column names containing entropy dictionaries.
    - extractor_func: function to extract features, must accept (entropy_dict, monthly_played_games).
    - game_count_col: name of the column containing monthly game count dictionaries.

    Returns:
    - Updated DataFrame with extracted features appended.
    """
    for col in entropy_columns:
        prefix = col.replace("monthly_entropy_", "entropy_") + "_"

        # ✅ Apply using both entropy and game count dictionaries
        features = df.apply(
            lambda row: extractor_func(row[col], row[game_count_col]),
            axis=1
        )

        # Convert extracted features into DataFrame
        features_df = pd.DataFrame(features.tolist(), index=df.index).add_prefix(prefix)
        df = pd.concat([df, features_df], axis=1)

    return df



def extract_engagement_features(monthly_game_series):
    values = monthly_game_series.values

    if np.count_nonzero(values) < 3:
        return {
            "overall_slope_month_games": np.nan,
            "engagement_consistency": np.nan
        }

    x = np.arange(len(values))
    weights = values

    # Weighted slope (more active months contribute more)
    mean_x = np.average(x, weights=weights)
    mean_y = np.average(values, weights=weights)
    cov_xy = np.average((x - mean_x) * (values - mean_y), weights=weights)
    var_x = np.average((x - mean_x) ** 2, weights=weights)
    weighted_slope = cov_xy / var_x if var_x > 0 else np.nan

    # Engagement consistency: how much volume is in months >= median
    median = np.median(values)
    above_or_equal = values >= median
    consistency = np.sum(values[above_or_equal]) / np.sum(values) if np.sum(values) > 0 else np.nan

    return {
        "overall_slope_month_games": weighted_slope,
        "engagement_consistency": consistency
    }

def compute_session_spacing_features(player_games):
    """
    Compute session spacing features per player.

    A session is a distinct day when the player played at least one game.
    We measure time between sessions, weighted by number of games in the prior session.

    Returns per-player metrics:
        - weighted_mean_days_between_sessions
        - std_days_between_sessions
        - max_days_between_sessions
        - percent_sessions_within_2_days
        - num_sessions_last_14_days
    """
    # Ensure UTCDate is datetime
    player_games['UTCDate'] = pd.to_datetime(player_games['UTCDate'], errors='coerce')
    player_games = player_games.dropna(subset=['UTCDate'])

    # Create session date column
    player_games['session_date'] = player_games['UTCDate'].dt.date

    # Count games per session (player_id + session_date)
    session_counts = player_games.groupby(['player_id', 'session_date']).size().reset_index(name='games_in_session')
    session_counts = session_counts.sort_values(by=['player_id', 'session_date'])

    # Compute gap to previous session
    session_counts['prev_session_date'] = session_counts.groupby('player_id')['session_date'].shift(1)
    session_counts['days_between'] = (
        pd.to_datetime(session_counts['session_date']) -
        pd.to_datetime(session_counts['prev_session_date'])
    ).dt.days

    # Get game count of prior session (for weighting)
    session_counts['prev_session_games'] = session_counts.groupby('player_id')['games_in_session'].shift(1)

    # Drop rows without prior session
    session_gaps = session_counts.dropna(subset=['days_between', 'prev_session_games'])

    def per_player_stats(group):
        days_between = group['days_between']
        weights = group['prev_session_games']
        total_weight = weights.sum()

        # Weighted mean and std
        if total_weight > 0:
            weighted_mean = np.average(days_between, weights=weights)
            weighted_std = np.sqrt(np.average((days_between - weighted_mean) ** 2, weights=weights))
        else:
            weighted_mean = np.nan
            weighted_std = np.nan

        # Max spacing (drop-off risk)
        max_gap = days_between.max()

        # % sessions with <= 2-day gap
        fast_return_pct = np.sum(days_between <= 2) / len(days_between) if len(days_between) > 0 else np.nan

        # Count sessions in last 14 days (momentum)
        most_recent_date = group['session_date'].max()
        cutoff_date = most_recent_date - pd.Timedelta(days=14)
        recent_sessions = group[group['session_date'] >= cutoff_date]
        recent_session_count = len(recent_sessions)

        return pd.Series({
            'weighted_mean_days_between_sessions': weighted_mean,
            'std_days_between_sessions': weighted_std,
            'max_days_between_sessions': max_gap,
            'percent_sessions_within_2_days': fast_return_pct,
            'num_sessions_last_14_days': recent_session_count
        })

    # Apply feature extraction per player
    spacing_stats = session_counts.groupby('player_id').apply(per_player_stats).reset_index()

    return spacing_stats

def compute_challenge_features(games_df, players_list):
    """
    Compute challenge-level features per player in players_list:
    - Mean Elo difference
    - Std dev of Elo difference
    - Proportion of games vs stronger opponents

    Parameters:
    - games_df: DataFrame with 'White', 'Black', 'WhiteElo', 'BlackElo'
    - players_list: list of target players

    Returns:
    - DataFrame indexed by player_id with challenge features
    """
    df = games_df.copy()

    # Assign player role
    def extract_player_info(row):
        if row['White'] in players_list:
            return pd.Series({
                'player_id': row['White'],
                'player_elo': row['WhiteElo'],
                'opponent_elo': row['BlackElo']
            })
        elif row['Black'] in players_list:
            return pd.Series({
                'player_id': row['Black'],
                'player_elo': row['BlackElo'],
                'opponent_elo': row['WhiteElo']
            })
        else:
            return pd.Series({'player_id': None, 'player_elo': None, 'opponent_elo': None})

    df[['player_id', 'player_elo', 'opponent_elo']] = df.apply(extract_player_info, axis=1)
    df = df.dropna(subset=['player_id'])  # Keep only games involving players in players_list

    df['player_elo'] = pd.to_numeric(df['player_elo'], errors='coerce')
    df['opponent_elo'] = pd.to_numeric(df['opponent_elo'], errors='coerce') 

    df['elo_diff'] = df['player_elo'] - df['opponent_elo']

    # Aggregate features per player
    challenge_stats = df.groupby('player_id').agg(
        mean_elo_diff=('elo_diff', 'mean'),
        std_challenge=('elo_diff', 'std'),
        proportion_vs_stronger=('elo_diff', lambda x: np.mean(x < 0))
    )

    return challenge_stats

import chess

stockfish = Stockfish(path="stockfish/stockfish-macos-m1-apple-silicon", parameters={"Threads": 2, "Minimum Thinking Time": 30})
stockfish.set_depth(10)

import numpy as np

def centipawn_to_win_prob(cp):
    # Logistic model fitted to human games (approximation)
    return 1 / (1 + np.exp(-0.004 * cp))
from tqdm import tqdm
import numpy as np
from stockfish import Stockfish
import chess

def rounds_until_win_prob_70(moves, stockfish):
    """
    Returns the number of moves until win probability exceeds 0.70.
    """
    try:
        stockfish.set_fen_position(chess.STARTING_FEN)
        for i, move in enumerate(moves):
            stockfish.make_moves_from_current_position([move])
            eval_ = stockfish.get_evaluation()
            if eval_['type'] != 'cp':
                continue  # Skip mate evaluations
            cp = eval_['value']
            prob = 1 / (1 + np.exp(-0.004 * cp))  # Logit transform
            if prob > 0.7:
                return i + 1
        return len(moves)
    except Exception:
        return np.nan

def compute_avg_normalized_efficiency_sampled(player_games, sample_size=3, epsilon=50, stockfish_path="stockfish"):
    """
    Computes normalized efficiency: (rounds_to_70 / (elo_diff + epsilon))
    using a shared Stockfish engine.
    """
    stockfish = Stockfish(path="chess_25/stockfish/stockfish-macos-m1-apple-silicon", depth=10)
    sampled_data = []
    grouped = player_games.groupby(['player_id', 'month_year'])

    for (player, month), group in tqdm(grouped, desc="Sampling per player-month"):
        sample = group.sample(n=min(sample_size, len(group)), random_state=42)
        normalized_scores = []

        for _, row in sample.iterrows():
            try:
                # Determine player role and Elos
                if row['White'] == player:
                    player_elo = float(row['WhiteElo'])
                    opponent_elo = float(row['BlackElo'])
                elif row['Black'] == player:
                    player_elo = float(row['BlackElo'])
                    opponent_elo = float(row['WhiteElo'])
                else:
                    continue

                elo_diff = abs(opponent_elo - player_elo)
                rounds = rounds_until_win_prob_70(row['Moves'], stockfish)

                if not np.isnan(rounds):
                    normalized = rounds / (elo_diff + epsilon)
                    normalized_scores.append(normalized)

            except (KeyError, ValueError, TypeError):
                continue

        avg_normalized = np.mean(normalized_scores) if normalized_scores else np.nan

        sampled_data.append({
            'player_id': player,
            'month_year': month,
            'avg_normalized_efficiency': avg_normalized
        })

    return pd.DataFrame(sampled_data)


def extract_efficiency_features(df, column, feature_func):
    """
    Apply trend feature extraction to a dictionary column (e.g., monthly_efficiency_dict).

    Parameters:
    - df: DataFrame containing one row per player
    - column: string name of the column containing dicts {month: value}
    - feature_func: function to extract trend-based features from each dict

    Returns:
    - DataFrame with extracted trend features merged back to input df
    """
    tqdm.pandas(desc=f"Processing: {column}")
    features = df[column].progress_apply(feature_func)
    features_df = pd.DataFrame(features.tolist())
    features_df.columns = [f'efficiency_{col}' for col in features_df.columns]
    features_df['player_id'] = df['player_id']
    return pd.merge(df, features_df, on='player_id', how='left')

def compute_monthly_ngram_entropy_df(player_games, n_gram_size=2):
    """
    Compute monthly n-gram entropy per player (white games only).

    Parameters:
    - player_games: DataFrame with ['player_id', 'player_color', 'UTCDate', 'Moves']
    - n_gram_size: size of n-gram (>= 2)

    Returns:
    - DataFrame with columns: player_id, month, entropy, n_gram_size
    """
    if n_gram_size < 2:
        raise ValueError("n_gram_size must be at least 2")

    # Parse month from UTCDate
    player_games = player_games.copy()
    player_games['month'] = pd.to_datetime(player_games['UTCDate'], errors='coerce').dt.to_period("M")
    
    # Filter to white games only
    white_games = player_games[player_games['player_color'].str.lower() == 'white']

    results = []

    for (player_id, month), group in white_games.groupby(['player_id', 'month']):
        ngram_counts = Counter()
        total_possible_ngrams = 0

        for moves in group['Moves']:
            if not isinstance(moves, list) or len(moves) < n_gram_size:
                continue

            player_moves = moves[0::2]  # White moves are even-indexed
            if len(player_moves) < n_gram_size:
                continue

            num_ngrams = len(player_moves) - n_gram_size + 1
            total_possible_ngrams += num_ngrams

            for i in range(num_ngrams):
                ngram = tuple(player_moves[i:i + n_gram_size])
                if all(ngram):  # Avoid empty/None
                    ngram_counts[ngram] += 1

        if not ngram_counts:
            entropy = np.nan
        else:
            total = sum(ngram_counts.values())
            probs = np.array(list(ngram_counts.values())) / total
            entropy = -np.sum(probs * np.log2(probs))

        results.append({
            'player_id': player_id,
            'month': month,
            'entropy': entropy,
            'n_gram_size': n_gram_size
        })

    return pd.DataFrame(results)

def compute_trend_features(series, weights=None, prefix="feature", include_consistency=False):
    """
    Compute trend features for a monthly time series.

    Parameters:
    - series: pd.Series with datetime or period index and values
    - weights: pd.Series with same index (optional)
    - prefix: str prefix for feature names
    - include_consistency: bool, whether to compute % of volume above median

    Returns:
    - dict of computed features
    """
    s = series.dropna().sort_index()
    if weights is None:
        weights = pd.Series(1, index=s.index)
    else:
        weights = weights.reindex(s.index).fillna(0)

    if len(s) < 3:
        output = {
            f"{prefix}__steady_magnitude": np.nan,
            f"{prefix}__slope": np.nan,
            f"{prefix}__early_slope": np.nan,
            f"{prefix}__late_slope": np.nan,
            f"{prefix}__mean": np.nan
        }
        if include_consistency:
            output[f"{prefix}__consistency"] = np.nan
        return output

    x = np.arange(len(s))
    y = s.values
    w = weights.values

    # Weighted slope
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)
    cov_xy = np.average((x - mean_x) * (y - mean_y), weights=w)
    var_x = np.average((x - mean_x) ** 2, weights=w)
    slope = cov_xy / var_x if var_x > 0 else np.nan

    # Net change, volatility, early/late slopes
    net_change = y[-1] - y[0]
    local_slopes = np.diff(y)
    slope_std = np.std(local_slopes)
    early_slope = linregress(x[:3], y[:3])[0] if len(x) >= 3 else np.nan
    late_slope = linregress(x[-3:], y[-3:])[0] if len(x) >= 3 else np.nan

    # Directional dominance
    pos_sum = np.sum(local_slopes[local_slopes > 0])
    neg_sum = np.sum(np.abs(local_slopes[local_slopes < 0]))
    dominance_ratio = pos_sum / (pos_sum + neg_sum) if (pos_sum + neg_sum) > 0 else np.nan

    # SteadyMagnitude: strong, smooth, directional change
    steady_magnitude = (net_change / (1 + slope_std)) * dominance_ratio if not np.isnan(dominance_ratio) else np.nan

    output = {
        f"{prefix}__steady_magnitude": steady_magnitude,
        f"{prefix}__slope": slope,
        f"{prefix}__early_slope": early_slope,
        f"{prefix}__late_slope": late_slope,
        f"{prefix}__mean": np.average(y, weights=w)
    }

    # Optional: consistency metric (% of volume in high months)
    if include_consistency:
        median = np.median(y)
        consistency = np.sum(y[y >= median]) / np.sum(y) if np.sum(y) > 0 else np.nan
        output[f"{prefix}__consistency"] = consistency

    return output

def compute_monthly_spacing(filtered_games_df, players_list):
    """
    Compute session spacing features per player per month.
    Returns a dict: {player_id: {month: spacing_metrics_dict}}
    """

    spacing_data = defaultdict(dict)

    for player_id in tqdm(players_list):
        player_games = filtered_games_df[
            (filtered_games_df["White"] == player_id) | (filtered_games_df["Black"] == player_id)
        ].copy()

        if player_games.empty:
            continue

        player_games["UTCDate"] = pd.to_datetime(player_games["UTCDate"], errors="coerce")
        player_games = player_games.dropna(subset=["UTCDate"])
        player_games["month"] = player_games["UTCDate"].dt.to_period("M")

        for month, month_games in player_games.groupby("month"):
            session_counts = (
                month_games.groupby(month_games["UTCDate"].dt.date)
                .size()
                .sort_index()
            )

            if len(session_counts) < 2:
                continue

            session_dates = pd.to_datetime(session_counts.index)
            days_between = session_dates.to_series().diff().dt.days.dropna()
            weights = session_counts.shift(1).dropna().reindex(days_between.index)

            if weights.sum() == 0 or len(days_between) != len(weights):
                continue

            # Compute metrics
            weighted_mean = np.average(days_between, weights=weights)
            weighted_std = np.sqrt(np.average((days_between - weighted_mean) ** 2, weights=weights))
            max_gap = days_between.max()
            fast_return_pct = np.mean(days_between <= 2)

            last_date = session_dates.max()
            cutoff = last_date - pd.Timedelta(days=14)
            recent_sessions = (session_dates >= cutoff).sum()

            spacing_data[player_id][str(month)] = {
                "weighted_mean_days_between_sessions": weighted_mean,
                "std_days_between_sessions": weighted_std,
                "max_days_between_sessions": max_gap,
                "percent_sessions_within_2_days": fast_return_pct,
                "num_sessions_last_14_days": recent_sessions,
            }

    return spacing_data

def flatten_spacing_all_metrics(spacing_data):
    rows = []
    for player_id, month_dict in spacing_data.items():
        for month, metrics in month_dict.items():
            row = {
                "player_id": player_id,
                "month": pd.Period(month, freq="M")
            }
            for key, value in metrics.items():
                row[f"spacing_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)

def elo_win_probability(diff):
    return 1 / (1 + 10 ** (-diff / 400))


def categorize_by_win_prob(p):
    if 0.35 <= p <= 0.65:
        return "optimal"
    elif p > 0.65:
        return "underchallenged"
    else:
        return "overchallenged"

def rounds_until_win_prob_70(moves, stockfish):
    """
    Returns the number of moves until win probability exceeds 0.70.
    """
    try:
        stockfish.set_fen_position(chess.STARTING_FEN)
        for i, move in enumerate(moves):
            stockfish.make_moves_from_current_position([move])
            eval_ = stockfish.get_evaluation()
            if eval_['type'] != 'cp':
                continue  # Skip mate evaluations
            cp = eval_['value']
            prob = 1 / (1 + np.exp(-0.004 * cp))  # Logit transform
            if prob > 0.7:
                return i + 1
        return len(moves)
    except Exception:
        return np.nan

def compute_avg_normalized_efficiency_sampled(player_games, sample_size=3, epsilon=50, stockfish_path="stockfish"):
    """
    Computes normalized efficiency: (rounds_to_70 / (elo_diff + epsilon))
    using a shared Stockfish engine.
    """
    stockfish = Stockfish(path="stockfish/stockfish-macos-m1-apple-silicon", depth=10)
    sampled_data = []
    grouped = player_games.groupby(['player_id', 'month_year'])

    for (player, month), group in tqdm(grouped, desc="Sampling per player-month"):
        sample = group.sample(n=min(sample_size, len(group)), random_state=42)
        normalized_scores = []

        for _, row in sample.iterrows():
            try:
                # Determine player role and Elos
                if row['White'] == player:
                    player_elo = float(row['WhiteElo'])
                    opponent_elo = float(row['BlackElo'])
                elif row['Black'] == player:
                    player_elo = float(row['BlackElo'])
                    opponent_elo = float(row['WhiteElo'])
                else:
                    continue

                elo_diff = abs(opponent_elo - player_elo)
                rounds = rounds_until_win_prob_70(row['Moves'], stockfish)

                if not np.isnan(rounds):
                    normalized = rounds / (elo_diff + epsilon)
                    normalized_scores.append(normalized)

            except (KeyError, ValueError, TypeError):
                continue

        avg_normalized = np.mean(normalized_scores) if normalized_scores else np.nan

        sampled_data.append({
            'player_id': player,
            'month_year': month,
            'avg_normalized_efficiency': avg_normalized
        })

    return pd.DataFrame(sampled_data)
