import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import norm

def concatenate_feather_files_to_df(folder_path):
    """
    Imports and concatenates all Feather files (.feather) from a given folder
    into a single Pandas DataFrame, displaying a progress bar using tqdm.

    Args:
        folder_path (str): The path to the folder containing the Feather files.

    Returns:
        pandas.DataFrame: A DataFrame containing the concatenated data from all Feather files,
                          or an empty DataFrame if no Feather files are found.
    """
    feather_files = [f for f in os.listdir(folder_path) if f.endswith(".feather")]
    num_files = len(feather_files)
    all_dfs = []

    with tqdm(total=num_files, desc="Processing Feather Files") as pbar:
        for filename in feather_files:
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_feather(file_path)
                all_dfs.append(df)
                pbar.update(1)
                pbar.set_postfix({"File": filename, "Status": "Imported"})
            except Exception as e:
                pbar.update(1)
                pbar.set_postfix({"File": filename, "Status": f"Error: {e}"})

    if all_dfs:
        concatenated_df = pd.concat(all_dfs, ignore_index=True)
        return concatenated_df
    else:
        print(f"No Feather files found in the specified folder: {folder_path}")
        return pd.DataFrame()




def filter_players_by_games(df, threshold=30):
    """
    Filter chess games DataFrame to include only players with more than a specified number of games.
    
    Parameters:
    - df: DataFrame with "White" and "Black" columns representing players
    - threshold: Minimum number of games a player must have (default: 30)
    
    Returns:
    - player_names: List of player names with more than 'threshold' games
    - filtered_df: DataFrame filtered to include only games with those players
    """
    # Ensure required columns exist
    required_columns = {"White", "Black"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Count games as White
    white_counts = df["White"].value_counts()

    # Count games as Black
    black_counts = df["Black"].value_counts()

    # Combine counts
    total_counts = white_counts.add(black_counts, fill_value=0).astype(int)

    # Filter players with more than the threshold
    players_above_threshold = total_counts[total_counts > threshold]

    # Get the list of player names
    player_names = players_above_threshold.index.tolist()

    # If no player meets the threshold, return empty filtered_df
    if not player_names:
        return [], df.iloc[0:0]  # Return empty DataFrame with same structure

    # Filter the original DataFrame
    filtered_df = df[df["White"].isin(player_names) | df["Black"].isin(player_names)]

    return filtered_df

def filter_players_by_game_type(df, game_type = 'Rated Classical game'):
    filtered_df = df[df['Event'] == game_type]
    return filtered_df

def add_total_game_counts(df):
    """
    Adds two new columns to the DataFrame:
    - 'WhiteTotalGames': Total number of games played by the White player in that row.
    - 'BlackTotalGames': Total number of games played by the Black player in that row.

    Args:
        df (pd.DataFrame): DataFrame containing chess game data with 'White' and 'Black' columns.

    Returns:
        pd.DataFrame: The DataFrame with the two new columns added.
    """
    white_counts = df['White'].value_counts()
    black_counts = df['Black'].value_counts()
    all_player_counts = white_counts.add(black_counts, fill_value=0).astype(int)

    df['WhiteTotalGames'] = df['White'].map(all_player_counts)
    df['BlackTotalGames'] = df['Black'].map(all_player_counts)

    return df

def monthly_player_totals(df):
    """
    Adds wide-format columns to the DataFrame showing the total number of games
    played by the White and Black players in each month of the year (version 2).

    Args:
        df (pd.DataFrame): DataFrame with 'White', 'Black', and 'UTCDate' columns.

    Returns:
        pd.DataFrame: DataFrame with new columns:
                      'WhiteTotalGames_YYYY_MM' and 'BlackTotalGames_YYYY_MM' for each month.
    """
    if 'UTCDate' not in df.columns:
        raise ValueError("DataFrame must contain a 'UTCDate' column.")

    df['UTCDate'] = pd.to_datetime(df['UTCDate'], errors='coerce')
    df['Year'] = df['UTCDate'].dt.year
    df['Month'] = df['UTCDate'].dt.month.astype(str).str.zfill(2) # Ensure month is two digits

    # Calculate total white games per month
    white_monthly_counts = df.groupby(['Year', 'Month', 'White']).size().reset_index(name='WhiteTotal')
    white_pivot = white_monthly_counts.pivot_table(index='White', columns=['Year', 'Month'], values='WhiteTotal', fill_value=0)
    white_pivot.columns = ['WhiteTotalGames_' + str(year) + '_' + month for year, month in white_pivot.columns]
    white_pivot = white_pivot.reset_index(names='White')

    # Calculate total black games per month
    black_monthly_counts = df.groupby(['Year', 'Month', 'Black']).size().reset_index(name='BlackTotal')
    black_pivot = black_monthly_counts.pivot_table(index='Black', columns=['Year', 'Month'], values='BlackTotal', fill_value=0)
    black_pivot.columns = ['BlackTotalGames_' + str(year) + '_' + month for year, month in black_pivot.columns]
    black_pivot = black_pivot.reset_index(names='Black')

    # Merge the wide format tables back into the original DataFrame
    df = pd.merge(df, white_pivot, on='White', how='left')
    df = pd.merge(df, black_pivot, on='Black', how='left')

    return df

def compute_quarterly_player_totals(df):
    """
    Computes quarterly totals of games played by White and Black players.

    Args:
        df (pd.DataFrame): DataFrame with 'White', 'Black', and 'UTCDate' columns.

    Returns:
        pd.DataFrame: DataFrame with new columns:
                      'WhiteTotalGames_YYYY_Q', 'BlackTotalGames_YYYY_Q' for each quarter.
    """
    if 'UTCDate' not in df.columns:
        raise ValueError("DataFrame must contain a 'UTCDate' column.")

    df['UTCDate'] = pd.to_datetime(df['UTCDate'], errors='coerce')
    df['Year'] = df['UTCDate'].dt.year
    df['Quarter'] = df['UTCDate'].dt.to_period('Q').astype(str).str.replace('Q', '') # Get quarter number

    # Calculate total white games per quarter
    white_quarterly_counts = df.groupby(['Year', 'Quarter', 'White']).size().reset_index(name='WhiteTotal')
    white_pivot_total = white_quarterly_counts.pivot_table(index='White', columns=['Year', 'Quarter'], values='WhiteTotal', fill_value=0)
    white_pivot_total.columns = ['WhiteTotalGames_' + str(year) + '_Q' + quarter for year, quarter in white_pivot_total.columns]
    white_pivot_total = white_pivot_total.reset_index(names='White')

    # Calculate total black games per quarter
    black_quarterly_counts = df.groupby(['Year', 'Quarter', 'Black']).size().reset_index(name='BlackTotal')
    black_pivot_total = black_quarterly_counts.pivot_table(index='Black', columns=['Year', 'Quarter'], values='BlackTotal', fill_value=0)
    black_pivot_total.columns = ['BlackTotalGames_' + str(year) + '_Q' + quarter for year, quarter in black_pivot_total.columns]
    black_pivot_total = black_pivot_total.reset_index(names='Black')

    # Merge the total quarterly counts back into the original DataFrame
    df = pd.merge(df, white_pivot_total, on='White', how='left')
    df = pd.merge(df, black_pivot_total, on='Black', how='left')

    df.drop(columns=['Year', 'Quarter'], inplace=True, errors='ignore')

    return df

def select_players_by_running_quarterly_threshold(df, threshold=90, date_column='UTCDate'):
    """
    Selects players whose running total of quarterly games played (as White or Black)
    reaches or exceeds a given threshold at any point in the dataset.

    Args:
        df (pd.DataFrame): DataFrame with 'White', 'Black', and a date column.
        threshold (int): The minimum running total of quarterly games to qualify.
        date_column (str): Name of the column containing the game date.

    Returns:
        list: A list of unique player names who meet the threshold.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame.")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df['Quarter'] = df[date_column].dt.to_period('Q')

    player_quarters = {}

    for index, row in df.iterrows():
        quarter = row['Quarter']
        white_player = row['White']
        black_player = row['Black']

        if white_player not in player_quarters:
            player_quarters[white_player] = {}
        if black_player not in player_quarters:
            player_quarters[black_player] = {}

        player_quarters[white_player][quarter] = player_quarters[white_player].get(quarter, 0) + 1
        if white_player != black_player:  # Avoid double counting if White and Black are the same (rare but possible in some datasets)
            player_quarters[black_player][quarter] = player_quarters[black_player].get(quarter, 0) + 1

    selected_players = set()
    for player, quarterly_counts in player_quarters.items():
        running_total = 0
        sorted_quarters = sorted(quarterly_counts.keys())
        for quarter in sorted_quarters:
            running_total += quarterly_counts[quarter]
            if running_total >= threshold:
                selected_players.add(player)
                break  # Once threshold is reached, no need to check further quarters for this player

    return sorted(list(selected_players))

def select_matches_with_frequent_players(df, frequent_player_list):
    """
    Selects rows (matches) from the DataFrame where either the 'White'
    or the 'Black' player is present in the provided list of frequent players.

    Args:
        df (pd.DataFrame): The main DataFrame containing chess game data
                           with 'White' and 'Black' columns.
        frequent_player_list (list): A list of player names considered frequent.

    Returns:
        pd.DataFrame: A new DataFrame containing only the matches where
                      at least one of the players ('White' or 'Black')
                      is in the 'frequent_player_list'.
    """
    if 'White' not in df.columns or 'Black' not in df.columns:
        raise ValueError("DataFrame must contain 'White' and 'Black' columns.")

    # Create a boolean mask indicating if the White player is frequent
    white_mask = df['White'].isin(frequent_player_list)

    # Create a boolean mask indicating if the Black player is frequent
    black_mask = df['Black'].isin(frequent_player_list)

    # Combine the masks: a match is selected if either White OR Black is frequent
    combined_mask = white_mask | black_mask

    # Use the combined mask to select the corresponding rows
    frequent_player_matches_df = df[combined_mask].copy()

    return frequent_player_matches_df

import pandas as pd
import numpy as np
import random

def select_random_unique_players(df, num_players=2000, white_col='White', black_col='Black', random_seed=None):
    """
    Selects a random number (up to num_players) of unique players from a chess dataset.

    Args:
        df (pd.DataFrame): The input DataFrame with columns for White and Black players.
        num_players (int): The maximum number of unique players to select.
        white_col (str): The name of the column containing White player IDs.
        black_col (str): The name of the column containing Black player IDs.
        random_seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        list: A list containing the names (IDs) of the randomly selected unique players.
              The actual number of players might be less than num_players if there
              aren't that many unique players in the dataset.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    unique_white_players = df[white_col].unique()
    unique_black_players = df[black_col].unique()
    all_unique_players = np.union1d(unique_white_players, unique_black_players)

    num_available_players = len(all_unique_players)
    num_to_select = min(num_players, num_available_players)

    if num_to_select > 0:
        selected_players = np.random.choice(all_unique_players, size=num_to_select, replace=False)
        return list(selected_players)
    else:
        return []
    
def filter_dataframe_by_player_list(df, player_list, white_col='White', black_col='Black'):
    """
    Filters a DataFrame to keep only the rows (games) where either the White
    player or the Black player is present in the provided player list.

    Args:
        df (pd.DataFrame): The input DataFrame with columns for White and Black players.
        player_list (list): A list of player names (IDs) to filter by.
        white_col (str): The name of the column containing White player IDs.
        black_col (str): The name of the column containing Black player IDs.

    Returns:
        pd.DataFrame: A new DataFrame containing only the games played by the
                      players in the provided list.
    """
    filtered_df = df[df[white_col].isin(player_list) | df[black_col].isin(player_list)].copy()
    return filtered_df

def get_all_active_player_mean_elo(df, white_col='White', black_col='Black', white_elo_col='WhiteElo', black_elo_col='BlackElo'):
    white_elos = df[[white_col, white_elo_col]].rename(columns={white_col: 'Player', white_elo_col: 'Elo'}).dropna(subset=['Elo'])
    black_elos = df[[black_col, black_elo_col]].rename(columns={black_col: 'Player', black_elo_col: 'Elo'}).dropna(subset=['Elo'])
    player_elos = pd.concat([white_elos, black_elos])

    player_median_elo = player_elos.groupby('Player')['Elo'].mean()

    return player_median_elo

def get_random_sample_from_player_list(player_list, num_samples=2000, random_seed=None):
    """
    Selects a random sample of a specified number of players from a given list.

    Args:
        player_list (list): A list of player identifiers (e.g., IDs, names).
        num_samples (int): The desired number of players in the random sample.
        random_seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        list: A list containing a random sample of players from the input list.
              The length of the returned list will be min(num_samples, len(player_list)).
    """
    if random_seed is not None:
        random.seed(random_seed)

    if len(player_list) <= num_samples:
        return random.sample(player_list, len(player_list))
    else:
        return random.sample(player_list, num_samples)
    
def filter_games_by_player_list(df, player_list, white_col='White', black_col='Black'):
    """
    Filters the game DataFrame to include only the games played by the
    players in the provided list (as either White or Black).

    Args:
        df (pd.DataFrame): The original game DataFrame.
        player_list (list): A list of player identifiers.
        white_col (str): The name of the column containing the White player.
        black_col (str): The name of the column containing the Black player.

    Returns:
        pd.DataFrame: A new DataFrame containing only the games played by the
                      players in the input list.
    """
    filtered_df = df[df[white_col].isin(player_list) | df[black_col].isin(player_list)].copy()
    return filtered_df

def compute_player_stats_extended(df, player_list, date_col='UTCDate', white_col='White', black_col='Black',
                                 white_elo_col='WhiteElo', black_elo_col='BlackElo'):
    """
    Computes the initial Elo rating, final Elo rating, Elo improvement,
    average Elo rating, number of monthly played games, and
    quarterly Elo growth rate per player.

    Args:
        df (pd.DataFrame): DataFrame with game data.
        player_list (list): List of player IDs to compute statistics for.
        date_col (str): Name of the date column.
        white_col (str): Name of the White player column.
        black_col (str): Name of the Black player column.
        white_elo_col (str): Name of the White Elo column.
        black_elo_col (str): Name of the Black Elo column.

    Returns:
        pd.DataFrame: DataFrame grouped by player_id with computed statistics.
    """
    player_stats = {}
    for player_id in tqdm(player_list, desc="Computing Player Stats"):
        white_games = df[df[white_col] == player_id].copy()
        black_games = df[df[black_col] == player_id].copy()
        player_games = pd.concat([white_games, black_games], ignore_index=True)

        if player_games.empty:
            continue

        player_games[date_col] = pd.to_datetime(player_games[date_col])
        player_games['YearMonth'] = player_games[date_col].dt.to_period('M')
        player_games['Quarter'] = player_games[date_col].dt.to_period('Q')
        player_games.sort_values(by=date_col, inplace=True)

        # Initial and Final Elo
        initial_elo = np.nan
        final_elo = np.nan

        first_game = player_games.iloc[0]
        last_game = player_games.iloc[-1]

        if first_game[white_col] == player_id:
            initial_elo = first_game[white_elo_col]
        elif first_game[black_col] == player_id:
            initial_elo = first_game[black_elo_col]

        if last_game[white_col] == player_id:
            final_elo = last_game[white_elo_col]
        elif last_game[black_col] == player_id:
            final_elo = last_game[black_elo_col]

        elo_improvement = final_elo - initial_elo

        # Average Elo Rating
        white_elo_player = player_games[player_games[white_col] == player_id][white_elo_col].dropna()
        black_elo_player = player_games[player_games[black_col] == player_id][black_elo_col].dropna()
        all_elos = pd.concat([white_elo_player, black_elo_player])
        avg_elo = all_elos.mean() if not all_elos.empty else np.nan

        # Number of Monthly Played Games
        monthly_games = player_games['YearMonth'].value_counts().to_dict()

        # Quarterly Elo Growth Rate
        quarterly_elo_data = {}
        for quarter, group in player_games.groupby('Quarter'):
            first_elo_q = np.nan
            last_elo_q = np.nan
            if not group.empty:
                first_row_q = group.iloc[0]
                last_row_q = group.iloc[-1]
                if first_row_q[white_col] == player_id:
                    first_elo_q = first_row_q[white_elo_col]
                elif first_row_q[black_col] == player_id:
                    first_elo_q = first_row_q[black_elo_col]

                if last_row_q[white_col] == player_id:
                    last_elo_q = last_row_q[white_elo_col]
                elif last_row_q[black_col] == player_id:
                    last_elo_q = last_row_q[black_elo_col]
            quarterly_elo_data[quarter] = {'first_elo': first_elo_q, 'last_elo': last_elo_q}

        quarterly_elo_df = pd.DataFrame.from_dict(quarterly_elo_data, orient='index')
        quarterly_elo_df['elo_growth_rate'] = (quarterly_elo_df['last_elo'] - quarterly_elo_df['first_elo']) / quarterly_elo_df['first_elo'] if (quarterly_elo_df['first_elo'] != 0).all() else np.nan
        avg_quarterly_elo_growth = quarterly_elo_df['elo_growth_rate'].mean()

        player_stats[player_id] = {
            'initial_elo_rating': initial_elo,
            'final_elo_rating': final_elo,
            'elo_improvement': elo_improvement,
            'average_elo_rating': avg_elo,
            'monthly_played_games': monthly_games,
            'average_quarterly_elo_growth_rate': avg_quarterly_elo_growth
        }

    return pd.DataFrame.from_dict(player_stats, orient='index')

def get_average_monthly_games(monthly_games_dict):
    if isinstance(monthly_games_dict, dict) and monthly_games_dict:
        return sum(monthly_games_dict.values()) / len(monthly_games_dict)
    return 0

def calculate_elo_diff(row, player_id, white_col='White', black_col='Black', white_elo_col='WhiteElo', black_elo_col='BlackElo'):
    if row[white_col] == player_id:
        return row[white_elo_col] - row[black_elo_col]
    elif row[black_col] == player_id:
        return row[black_elo_col] - row[white_elo_col]
    return None

def compute_player_stats_with_quarterly_growth(df, player_list, date_col='UTCDate', white_col='White', black_col='Black',
                                                white_elo_col='WhiteElo', black_elo_col='BlackElo'):
    """
    Computes average Elo, monthly games, and quarterly Elo growth rates per player (as a series).
    """
    player_stats = {}
    for player_id in tqdm(player_list, desc="Computing Player Stats"):
        white_games = df[df[white_col] == player_id].copy()
        black_games = df[df[black_col] == player_id].copy()
        player_games = pd.concat([white_games, black_games], ignore_index=True)

        if player_games.empty:
            continue

        player_games[date_col] = pd.to_datetime(player_games[date_col])
        player_games['YearMonth'] = player_games[date_col].dt.to_period('M')
        player_games['Quarter'] = player_games[date_col].dt.to_period('Q')
        player_games.sort_values(by=date_col, inplace=True)

        # Average Elo Rating
        white_elo_player = player_games[player_games[white_col] == player_id][white_elo_col].dropna()
        black_elo_player = player_games[player_games[black_col] == player_id][black_elo_col].dropna()
        all_elos = pd.concat([white_elo_player, black_elo_player])
        avg_elo = all_elos.mean() if not all_elos.empty else np.nan

        # Number of Monthly Played Games
        monthly_games = player_games['YearMonth'].value_counts().to_dict()
        avg_monthly_games = sum(monthly_games.values()) / len(monthly_games) if monthly_games else 0

        # Quarterly Elo Growth Rates
        quarterly_elo_data = {}
        for quarter, group in player_games.groupby('Quarter'):
            first_elo = np.nan
            last_elo = np.nan
            if not group.empty:
                first_row = group.iloc[0]
                last_row = group.iloc[-1]
                if first_row[white_col] == player_id:
                    first_elo = first_row[white_elo_col]
                elif first_row[black_col] == player_id:
                    first_elo = first_row[black_elo_col]

                if last_row[white_col] == player_id:
                    last_elo = last_row[white_elo_col]
                elif last_row[black_col] == player_id:
                    last_elo = last_row[black_elo_col]
            quarterly_elo_data[quarter] = {'first_elo': first_elo, 'last_elo': last_elo}

        quarterly_elo_df = pd.DataFrame.from_dict(quarterly_elo_data, orient='index')
        quarterly_elo_df['elo_growth_rate'] = (quarterly_elo_df['last_elo'] - quarterly_elo_df['first_elo']) / quarterly_elo_df['first_elo'] if (quarterly_elo_df['first_elo'] != 0).all() else np.nan

        player_stats[player_id] = {
            'average_elo_rating': avg_elo,
            'average_monthly_games': avg_monthly_games,
            'quarterly_elo_growth_rates': quarterly_elo_df['elo_growth_rate'].to_dict() # Store as a dictionary
        }

    return pd.DataFrame.from_dict(player_stats, orient='index')


def compute_player_stats_with_quarterly_growth(df, player_list, date_col='UTCDate', white_col='White', black_col='Black',
                                                white_elo_col='WhiteElo', black_elo_col='BlackElo'):
    """
    Computes average Elo, monthly games, and quarterly Elo growth rates per player (with year-quarter).
    """
    player_stats = {}
    for player_id in tqdm(player_list, desc="Computing Player Stats"):
        white_games = df[df[white_col] == player_id].copy()
        black_games = df[df[black_col] == player_id].copy()
        player_games = pd.concat([white_games, black_games], ignore_index=True)

        if player_games.empty:
            continue

        player_games[date_col] = pd.to_datetime(player_games[date_col])
        player_games['YearMonth'] = player_games[date_col].dt.to_period('M')
        player_games['QuarterPeriod'] = player_games[date_col].dt.to_period('Q')
        player_games['Year'] = player_games['QuarterPeriod'].dt.year
        player_games['QuarterNum'] = player_games['QuarterPeriod'].dt.quarter
        player_games.sort_values(by=date_col, inplace=True)

        # Average Elo Rating
        white_elo_player = player_games[player_games[white_col] == player_id][white_elo_col].dropna()
        black_elo_player = player_games[player_games[black_col] == player_id][black_elo_col].dropna()
        all_elos = pd.concat([white_elo_player, black_elo_player])
        avg_elo = all_elos.mean() if not all_elos.empty else np.nan

        # Number of Monthly Played Games
        monthly_games = player_games['YearMonth'].value_counts().to_dict()
        avg_monthly_games = sum(monthly_games.values()) / len(monthly_games) if monthly_games else 0

        # Quarterly Elo Growth Rates
        quarterly_elo_data = {}
        for (year, q), group in player_games.groupby(['Year', 'QuarterNum']):
            first_elo = np.nan
            last_elo = np.nan
            if not group.empty:
                first_row = group.iloc[0]
                last_row = group.iloc[-1]
                if first_row[white_col] == player_id:
                    first_elo = first_row[white_elo_col]
                elif first_row[black_col] == player_id:
                    first_elo = first_row[black_elo_col]

                if last_row[white_col] == player_id:
                    last_elo = last_row[white_elo_col]
                elif last_row[black_col] == player_id:
                    last_elo = last_row[black_elo_col]
            quarter_label = f'{year}-Q{int(q)}' # Include the year
            quarterly_elo_data[quarter_label] = {'first_elo': first_elo, 'last_elo': last_elo}

        quarterly_elo_df = pd.DataFrame.from_dict(quarterly_elo_data, orient='index')
        if not quarterly_elo_df.empty:
            quarterly_elo_df['elo_growth_rate'] = (quarterly_elo_df['last_elo'] - quarterly_elo_df['first_elo']) / quarterly_elo_df['first_elo'] if (quarterly_elo_df['first_elo'] != 0).all() else np.nan
            player_stats[player_id] = {
                'average_elo_rating': avg_elo,
                'average_monthly_games': avg_monthly_games,
                'quarterly_elo_growth_rates': quarterly_elo_df['elo_growth_rate'].to_dict()
            }
        else:
            player_stats[player_id] = {
                'average_elo_rating': avg_elo,
                'average_monthly_games': avg_monthly_games,
                'quarterly_elo_growth_rates': {}
            }

    return pd.DataFrame.from_dict(player_stats, orient='index')

