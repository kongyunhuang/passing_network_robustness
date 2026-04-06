"""
data_loader.py — StatsBomb JSON event data parsing and pass extraction
======================================================================
From StatsBomb open-data format JSON event files:
1. Load single-match events
2. Extract successful open-play passes (excluding set-piece passes themselves)
3. Build substitution mappings and map substitutes back to starter identities
4. Load authoritative home/away information from metadata JSON

Input:  StatsBomb event JSON files (*_events.json) and match info JSON
Output: List of pass records (List[Dict]), each containing player_id,
        recipient_id, coordinates, etc.

Usage:
    from pipeline.data_loader import safe_load_match_events, extract_passes
    events = safe_load_match_events("path/to/events.json")
    passes = extract_passes(events, match_id=12345, team_id=1)

Last modified: 2026-03-27
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ============== Constants ==============

# StatsBomb event type IDs
EVENT_TYPE_PASS = 30
EVENT_TYPE_SUBSTITUTION = 19

# League IDs
COMPETITION_ID_PREMIER_LEAGUE = 2
COMPETITION_ID_LA_LIGA = 11

# Season IDs
SEASON_ID_2022_23 = 235
SEASON_ID_2023_24 = 281
SEASON_ID_2024_25 = 317


# ============== Core functions ==============

def load_match_events(filepath: str) -> List[Dict]:
    """
    Load event data for a single match.

    Args:
        filepath: Event JSON file path (e.g., "3837230_events.json")

    Returns:
        events: List of events
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        events = json.load(f)

    return events


def safe_load_match_events(filepath: str) -> Optional[List[Dict]]:
    """
    Safely load single-match events, catching JSON errors.

    Args:
        filepath: File path

    Returns:
        events: List of events, or None if loading fails
    """
    try:
        return load_match_events(filepath)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"[DataError] Cannot load file {Path(filepath).name}: {e}")
        return None
    except Exception as e:
        print(f"[UnknownError] Unknown error loading file {Path(filepath).name}: {e}")
        return None


def build_substitution_map(
    events: List[Dict],
    team_id: Optional[int] = None
) -> Dict[int, int]:
    """
    Extract substitution mapping from event data.

    Maps substitute player_id to the replaced player's player_id,
    ensuring substitution inheritance within a single match only.
    This means a single match will not introduce a "12th on-pitch identity",
    but when aggregating multiple matches to season level, different starting
    lineups will merge, so the season-aggregated network's unique player_id
    count can exceed 11.

    Handles chained substitutions: if A is replaced by B, and B is later
    replaced by C, then B->A and C->A.

    Args:
        events: StatsBomb event list
        team_id: Extract substitutions for this team only (None = all teams)

    Returns:
        sub_map: {substitute_player_id: original_starter_player_id}
    """
    sub_map = {}

    for event in events:
        if event.get('type', {}).get('id') != EVENT_TYPE_SUBSTITUTION:
            continue

        if team_id is not None and event.get('team', {}).get('id') != team_id:
            continue

        # player = substituted out player, replacement = substituted in player
        player_out_id = event.get('player', {}).get('id')
        player_in_id = event.get('substitution', {}).get('replacement', {}).get('id')

        if player_out_id is None or player_in_id is None:
            continue

        # Chained substitution handling: if the substituted-out player is
        # themselves a substitute (already in sub_map), the new substitute
        # should map to the original starter
        original_id = sub_map.get(player_out_id, player_out_id)
        sub_map[player_in_id] = original_id

    return sub_map


def apply_substitution_map(
    passes: List[Dict],
    sub_map: Dict[int, int]
) -> List[Dict]:
    """
    Apply substitution mapping to pass data: replace player_id and recipient_id.

    Args:
        passes: List of pass records
        sub_map: Substitution mapping {substitute_id: original_starter_id}

    Returns:
        passes: Modified pass list (in-place modification)
    """
    if not sub_map:
        return passes

    for p in passes:
        if p['player_id'] in sub_map:
            p['player_id'] = sub_map[p['player_id']]
        if p['recipient_id'] in sub_map:
            p['recipient_id'] = sub_map[p['recipient_id']]

    return passes


# Set-piece pass types (pass.type.name): exclude these passes themselves,
# but retain subsequent open-play passes.
# Note: play_pattern is a possession-level attribute (shared across the entire
# possession chain), not a single-pass attribute. Therefore, play_pattern
# cannot be used for filtering, as that would also exclude normal passes
# following set pieces.
SET_PIECE_PASS_TYPES = {'Throw-in', 'Free Kick', 'Corner', 'Kick Off', 'Goal Kick'}


def extract_passes(
    events: List[Dict],
    match_id: Optional[int] = None,
    team_id: Optional[int] = None,
    successful_only: bool = True,
    exclude_set_pieces: bool = True
) -> List[Dict]:
    """
    Extract passes from event data.

    Args:
        events: StatsBomb event list
        match_id: Match ID (from filename or metadata, as events typically
                  do not contain this field)
        team_id: Extract passes for this team only (None = all)
        successful_only: Keep only successful passes (default True)
        exclude_set_pieces: Exclude set-piece passes themselves (default True).
            Excludes passes where pass.type is Throw-in/Free Kick/Corner/
            Kick Off/Goal Kick. Retains all open-play passes within the
            subsequent possession chain.

    Returns:
        passes: Extracted pass list
    """
    passes = []

    for event in events:
        # 1. Filter Pass type events
        if event.get('type', {}).get('id') != EVENT_TYPE_PASS:
            continue

        # 2. Filter by team (if specified)
        if team_id is not None and event.get('team', {}).get('id') != team_id:
            continue

        pass_data = event.get('pass', {})

        # 3. Filter successful passes (no outcome field = successful)
        if successful_only and 'outcome' in pass_data:
            continue

        # 4. Exclude set-piece passes themselves (by pass.type, not play_pattern)
        if exclude_set_pieces:
            pass_type = pass_data.get('type', {}).get('name', '')
            if pass_type in SET_PIECE_PASS_TYPES:
                continue

        # 5. Extract coordinates
        location = event.get('location')
        end_location = pass_data.get('end_location')

        # Skip passes with missing coordinates (location may be None or contain None elements)
        if not location or not end_location:
            continue
        if None in location or None in end_location:
            continue

        # Skip passes without a recipient
        recipient = pass_data.get('recipient', {})
        if not recipient.get('id'):
            continue

        # Skip passes with missing passer ID (prevent None nodes entering the network)
        player_id = event.get('player', {}).get('id')
        if player_id is None:
            continue

        # 6. Build pass record
        passes.append({
            'match_id': match_id if match_id is not None else event.get('match_id'),
            'team_id': event.get('team', {}).get('id'),
            'team_name': event.get('team', {}).get('name'),
            'player_id': player_id,
            'player_name': event.get('player', {}).get('name'),
            'recipient_id': recipient.get('id'),
            'recipient_name': recipient.get('name'),
            'x_start': location[0],
            'y_start': location[1],
            'x_end': end_location[0],
            'y_end': end_location[1],
        })

    return passes


def get_match_files(data_dir: str, competition_id: int, season_id: int) -> List[Path]:
    """
    Get all match event files for a given league and season.

    Args:
        data_dir: Data root directory
        competition_id: League ID (2=EPL, 11=La Liga)
        season_id: Season ID (235=22/23, 281=23/24, 317=24/25)

    Returns:
        List of file paths
    """
    folder_name = f"{competition_id}_{season_id}_*"
    data_path = Path(data_dir)

    # Match directory
    matching_dirs = list(data_path.glob(folder_name))
    if not matching_dirs:
        raise FileNotFoundError(f"No matching directory found: {folder_name}")

    match_dir = matching_dirs[0]

    # Get all event files
    event_files = list(match_dir.glob("*_events.json"))

    return event_files


def load_match_info(data_dir: str, competition_id: int, season_id: int) -> Dict[int, Dict]:
    """
    Load season match metadata from 00_info_all_matches_*.json.

    This is the authoritative source for home/away information (including
    home_team/away_team/scores, etc.).

    Args:
        data_dir: Data root directory
        competition_id: League ID (2=EPL, 11=La Liga)
        season_id: Season ID

    Returns:
        {match_id: {"home_team": {...}, "away_team": {...}, "home_score": int, ...}}
    """
    folder_name = f"{competition_id}_{season_id}_*"
    data_path = Path(data_dir)
    matching_dirs = list(data_path.glob(folder_name))
    if not matching_dirs:
        raise FileNotFoundError(f"No matching directory found: {folder_name}")

    match_dir = matching_dirs[0]
    info_pattern = f"00_info_all_matches_{competition_id}_{season_id}.json"
    info_files = list(match_dir.glob(info_pattern))
    if not info_files:
        # Fallback: try any 00_info_all_matches file
        info_files = list(match_dir.glob("00_info_all_matches_*.json"))
    if not info_files:
        raise FileNotFoundError(f"Match info file not found: {match_dir / info_pattern}")

    with open(info_files[0], 'r', encoding='utf-8') as f:
        matches = json.load(f)

    lookup = {}
    for m in matches:
        mid = m.get('match_id')
        if mid is None:
            continue
        lookup[mid] = {
            'home_team': {
                'id': m.get('home_team', {}).get('home_team_id'),
                'name': m.get('home_team', {}).get('home_team_name'),
            },
            'away_team': {
                'id': m.get('away_team', {}).get('away_team_id'),
                'name': m.get('away_team', {}).get('away_team_name'),
            },
            'home_score': m.get('home_score'),
            'away_score': m.get('away_score'),
            'match_date': m.get('match_date'),
            'match_week': m.get('match_week'),
        }
    return lookup


def get_teams_from_events(events: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Extract both teams' information from event data (fallback method).

    Note: The main pipeline should use load_match_info() to obtain
    authoritative home/away information. This function serves only as
    a fallback for standalone scripts and testing.

    Returns:
        (home_team, away_team): Two dicts containing id and name
    """
    # Find Starting XI events to determine teams
    teams = []
    for event in events:
        if event.get('type', {}).get('name') == 'Starting XI':
            team_info = event.get('team', {})
            if team_info.get('id') not in [t.get('id') for t in teams]:
                teams.append(team_info)
        if len(teams) == 2:
            break

    # StatsBomb Starting XI order matches match info home/away (verified across 2280 matches)
    # However, this method is only a fallback; the main pipeline should use load_match_info()
    if len(teams) == 2:
        return teams[0], teams[1]
    else:
        return {}, {}
