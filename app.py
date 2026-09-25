import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Moteur Prono Foot Auto",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Moteur de Prédiction Football")
st.caption("Analyse basée sur les statistiques récentes des équipes")

BASE_URL = "https://v3.football.api-sports.io"


# ============================================================
# API
# ============================================================

def get_api_key():
    try:
        return st.secrets["API_FOOTBALL_KEY"]
    except Exception:
        return None


def api_get(endpoint, params=None):
    api_key = get_api_key()

    if not api_key:
        return None, "Clé API absente."

    headers = {
        "x-apisports-key": api_key
    }

    try:
        response = requests.get(
            BASE_URL + endpoint,
            headers=headers,
            params=params,
            timeout=20
        )

        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        data = response.json()

        if "errors" in data and data["errors"]:
            return None, str(data["errors"])

        return data, None

    except requests.RequestException as e:
        return None, str(e)


# ============================================================
# RECHERCHE D'UNE ÉQUIPE
# ============================================================

@st.cache_data(ttl=3600)
def search_team(team_name):

    data, error = api_get(
        "/teams",
        {"search": team_name}
    )

    if error:
        return None, error

    response = data.get("response", [])

    if not response:
        return None, f"Aucune équipe trouvée pour : {team_name}"

    # Recherche la correspondance la plus proche
    team = response[0]["team"]

    return {
        "id": team["id"],
        "name": team["name"],
        "country": team.get("country", "")
    }, None


# ============================================================
# DERNIERS MATCHS
# ============================================================

@st.cache_data(ttl=900)
def get_last_matches(team_id, last=10):

    data, error = api_get(
        "/fixtures",
        {
            "team": team_id,
            "last": last
        }
    )

    if error:
        return [], error

    matches = []

    for item in data.get("response", []):

        fixture = item["fixture"]
        teams = item["teams"]
        goals = item["goals"]

        # On ne garde que les matchs terminés
        status = fixture["status"]["short"]

        if status not in ["FT", "AET", "PEN"]:
            continue

        home_id = teams["home"]["id"]
        away_id = teams["away"]["id"]

        home_goals = goals["home"]
        away_goals = goals["away"]

        if home_goals is None or away_goals is None:
            continue

        matches.append({
            "fixture_id": fixture["id"],
            "date": fixture["date"],
            "home_id": home_id,
            "away_id": away_id,
            "home_name": teams["home"]["name"],
            "away_name": teams["away"]["name"],
            "home_goals": home_goals,
            "away_goals": away_goals
        })

    return matches, None


# ============================================================
# STATISTIQUES D'UN MATCH
# ============================================================

@st.cache_data(ttl=3600)
def get_match_statistics(fixture_id):

    data, error = api_get(
        "/fixtures/statistics",
        {"fixture": fixture_id}
    )

    if error:
        return {}

    result = data.get("response", [])

    stats = {}

    for team_block in result:

        team_id = team_block["team"]["id"]

        stats[team_id] = {}

        for item in team_block.get("statistics", []):

            stat_name = item["type"]
            value = item["value"]

            stats[team_id][stat_name] = value

    return stats


# ============================================================
# CONVERSION DES STATISTIQUES
# ============================================================

def number(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):

        value = value.replace("%", "").strip()

        try:
            return float(value)
        except:
            return None

    return None


# ============================================================
# ANALYSE D'UNE ÉQUIPE
# ============================================================

def analyze_team(team, matches):

    team_id = team["id"]

    data = {
        "played": 0,

        "goals_for": [],
        "goals_against": [],

        "home_goals_for": [],
        "home_goals_against": [],

        "away_goals_for": [],
        "away_goals_against": [],

        "shots": [],
        "shots_on_target": [],
        "corners": [],

        "wins": 0,
        "draws": 0,
        "losses": 0,

        "btts_yes": 0,

        "over_1_5": 0,
        "over_2_5": 0,
        "over_3_5": 0
    }

    for match in matches:

        is_home = match["home_id"] == team_id

        if is_home:
            gf = match["home_goals"]
            ga = match["away_goals"]
        else:
            gf = match["away_goals"]
            ga = match["home_goals"]

        data["played"] += 1

        data["goals_for"].append(gf)
        data["goals_against"].append(ga)

        if is_home:

            data["home_goals_for"].append(gf)
            data["home_goals_against"].append(ga)

        else:

            data["away_goals_for"].append(gf)
            data["away_goals_against"].append(ga)

        # Résultat
        if gf > ga:
            data["wins"] += 1

        elif gf == ga:
            data["draws"] += 1

        else:
            data["losses"] += 1

        # BTTS
        if gf > 0 and ga > 0:
            data["btts_yes"] += 1

        total = gf + ga

        if total >= 2:
            data["over_1_5"] += 1

        if total >= 3:
            data["over_2_5"] += 1

        if total >= 4:
            data["over_3_5"] += 1

        # Statistiques du match
        stats = get_match_statistics(match["fixture_id"])

        team_stats = stats.get(team_id, {})

        shots = number(team_stats.get("Total Shots"))
        shots_on = number(team_stats.get("Shots on Goal"))
        corners = number(team_stats.get("Corner Kicks"))

        if shots is not None:
            data["shots"].append(shots)

        if shots_on is not None:
            data["shots_on_target"].append(shots_on)

        if corners is not None:
            data["corners"].append(corners)

    return data


# ============================================================
# MOYENNE SÉCURISÉE
# ============================================================

def avg(values, default=0):

    if not values:
        return default

    return float(np.mean(values))


# ============================================================
# CALCUL DES PARAMÈTRES DE POISSON
# ============================================================

def calculate_expected_goals(home_data, away_data):

    # --------------------------------------------------------
    # DOMICILE
    # --------------------------------------------------------

    home_attack = avg(
        home_data["home_goals_for"],
        avg(home_data["goals_for"], 1.2)
    )

    away_defence = avg(
        away_data["away_goals_against"],
        avg(away_data["goals_against"], 1.2)
    )

    # --------------------------------------------------------
    # EXTÉRIEUR
    # --------------------------------------------------------

    away_attack = avg(
        away_data["away_goals_for"],
        avg(away_data["goals_for"], 1.0)
    )

    home_defence = avg(
        home_data["home_goals_against"],
        avg(home_data["goals_against"], 1.2)
    )

    # --------------------------------------------------------
    # Mélange attaque / défense
    # --------------------------------------------------------

    expected_home = (
        home_attack * 0.55 +
        away_defence * 0.45
    )

    expected_away = (
        away_attack * 0.55 +
        home_defence * 0.45
    )

    # Bornes de sécurité
    expected_home = max(0.15, min(expected_home, 4.5))
    expected_away = max(0.15, min(expected_away, 4.5))

    return expected_home, expected_away


# ============================================================
# MATRICE DES SCORES
# ============================================================

def build_score_matrix(lambda_home, lambda_away, max_goals=8):

    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for home_goals in range(max_goals + 1):

        for away_goals in range(max_goals + 1):

            matrix[home_goals, away_goals] = (
                poisson.pmf(home_goals, lambda_home)
                *
                poisson.pmf(away_goals, lambda_away)
            )

    # Renormalisation
    total = matrix.sum()

    if total > 0:
        matrix = matrix / total

    return matrix


# ============================================================
# 1X2
# ============================================================

def calculate_1x2(matrix):

    home = 0
    draw = 0
    away = 0

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            if i > j:
                home += matrix[i, j]

            elif i == j:
                draw += matrix[i, j]

            else:
                away += matrix[i, j]

    return home, draw, away


# ============================================================
# TOP SCORES
# ============================================================

def top_scores(matrix, number_scores=5):

    scores = []

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            scores.append(
                (
                    i,
                    j,
                    matrix[i, j]
                )
            )

    scores.sort(key=lambda x: x[2], reverse=True)

    return scores[:number_scores]


# ============================================================
# OVER / UNDER
# ============================================================

def total_goals_probabilities(matrix):

    result = {
        "over_1_5": 0,
        "under_1_5": 0,

        "over_2_5": 0,
        "under_2_5": 0,

        "over_3_5": 0,
        "under_3_5": 0,

        "btts_yes": 0,
        "btts_no": 0
    }

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            p = matrix[i, j]

            total = i + j

            if total >= 2:
                result["over_1_5"] += p
            else:
                result["under_1_5"] += p

            if total >= 3:
                result["over_2_5"] += p
            else:
                result["under_2_5"] += p

            if total >= 4:
                result["over_3_5"] += p
            else:
                result["under_3_5"] += p

            if i > 0 and j > 0:
                result["btts_yes"] += p
            else:
                result["btts_no"] += p

    return result


# ============================================================
# INTERFACE
# ============================================================

st.subheader("⚽ Match à analyser")

col1, col2 = st.columns(2)

with col1:
    home_team_input = st.text_input(
        "Équipe domicile",
        value="Arsenal"
    )

with col2:
    away_team_input = st.text_input(
        "Équipe extérieure",
        value="Chelsea"
    )


if st.button("🔎 Lancer l'analyse", use_container_width=True):

    if not home_team_input or not away_team_input:

        st.error("Veuillez saisir les deux équipes.")

        st.stop()

    if not get_api_key():

        st.error(
            "Clé API absente. Ajoutez API_FOOTBALL_KEY dans les secrets Streamlit."
        )

        st.stop()

    with st.spinner("Recherche des équipes et analyse des derniers matchs..."):

        # ----------------------------------------------------
        # Recherche équipes
        # ----------------------------------------------------

        home_team, error_home = search_team(home_team_input)
        away_team, error_away = search_team(away_team_input)

        if error_home:
            st.error(error_home)
            st.stop()

        if error_away:
            st.error(error_away)
            st.stop()

        # ----------------------------------------------------
        # Derniers matchs
        # ----------------------------------------------------

        home_matches, error1 = get_last_matches(
            home_team["id"],
            last=10
        )

        away_matches, error2 = get_last_matches(
            away_team["id"],
            last=10
        )

        if error1:
            st.warning(error1)

        if error2:
            st.warning(error2)

        if not home_matches or not away_matches:

            st.error(
                "Impossible de récupérer suffisamment de matchs récents."
            )

            st.stop()

        # ----------------------------------------------------
        # Analyse
        # ----------------------------------------------------

        home_data = analyze_team(
            home_team,
            home_matches
        )

        away_data = analyze_team(
            away_team,
            away_matches
        )

        # ----------------------------------------------------
        # Expected goals du modèle
        # ----------------------------------------------------

        lambda_home, lambda_away = calculate_expected_goals(
            home_data,
            away_data
        )

        # ----------------------------------------------------
        # Matrice
        # ----------------------------------------------------

        matrix = build_score_matrix(
            lambda_home,
            lambda_away
        )

        # ----------------------------------------------------
        # 1X2
        # ----------------------------------------------------

        p_home, p_draw, p_away = calculate_1x2(matrix)

        # ----------------------------------------------------
        # Marchés buts
        # ----------------------------------------------------

        markets = total_goals_probabilities(matrix)

        # ----------------------------------------------------
        # Top scores
        # ----------------------------------------------------

        scores = top_scores(matrix, 5)

    # ========================================================
    # AFFICHAGE
    # ========================================================

    st.success("Analyse terminée.")

    st.subheader(
        f"📊 {home_team['name']} — {away_team['name']}"
    )

    # --------------------------------------------------------
    # Probabilités 1X2
    # --------------------------------------------------------

    st.markdown("### 🎯 Probabilités 1X2")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Victoire domicile",
            f"{p_home * 100:.1f}%"
        )

    with c2:
        st.metric(
            "Match nul",
            f"{p_draw * 100:.1f}%"
        )

    with c3:
        st.metric(
            "Victoire extérieur",
            f"{p_away * 100:.1f}%"
        )

    # --------------------------------------------------------
    # Expected goals
    # --------------------------------------------------------

    st.markdown("### ⚽ Buts attendus du modèle")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            home_team["name"],
            f"{lambda_home:.2f}"
        )

    with c2:
        st.metric(
            "Total",
            f"{lambda_home + lambda_away:.2f}"
        )

    with c3:
        st.metric(
            away_team["name"],
            f"{lambda_away:.2f}"
        )

    # --------------------------------------------------------
    # SCORE EXACT
    # --------------------------------------------------------

    st.markdown("### 🔢 Top 5 des scores exacts")

    for position, (hg, ag, probability) in enumerate(scores, 1):

        st.write(
            f"**{position}. {hg} – {ag}** "
            f"→ **{probability * 100:.2f}%**"
        )

    best_score = scores[0]

    st.success(
        f"🎯 Score le plus probable : "
        f"**{best_score[0]} – {best_score[1]}** "
        f"({best_score[2] * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # MARCHÉS BUTS
    # --------------------------------------------------------

    st.markdown("### 📈 Marchés de buts")

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            f"Over 1,5 : **{markets['over_1_5'] * 100:.1f}%**"
        )

        st.write(
            f"Under 1,5 : **{markets['under_1_5'] * 100:.1f}%**"
        )

        st.write(
            f"Over 2,5 : **{markets['over_2_5'] * 100:.1f}%**"
        )

        st.write(
            f"Under 2,5 : **{markets['under_2_5'] * 100:.1f}%**"
        )

    with c2:

        st.write(
            f"Over 3,5 : **{markets['over_3_5'] * 100:.1f}%**"
        )

        st.write(
            f"Under 3,5 : **{markets['under_3_5'] * 100:.1f}%**"
        )

        st.write(
            f"BTTS Oui : **{markets['btts_yes'] * 100:.1f}%**"
        )

        st.write(
            f"BTTS Non : **{markets['btts_no'] * 100:.1f}%**"
        )

    # --------------------------------------------------------
    # STATISTIQUES RÉELLES
    # --------------------------------------------------------

    st.markdown("### 📋 Statistiques des derniers matchs")

    c1, c2 = st.columns(2)

    with c1:

        st.write(f"**{home_team['name']}**")

        st.write(
            f"Matchs analysés : {home_data['played']}"
        )

        st.write(
            f"Buts marqués/match : "
            f"{avg(home_data['goals_for']):.2f}"
        )

        st.write(
            f"Buts encaissés/match : "
            f"{avg(home_data['goals_against']):.2f}"
        )

        st.write(
            f"BTTS : "
            f"{home_data['btts_yes'] / max(1, home_data['played']) * 100:.1f}%"
        )

        st.write(
            f"Over 2,5 : "
            f"{home_data['over_2_5'] / max(1, home_data['played']) * 100:.1f}%"
        )

        if home_data["shots"]:
            st.write(
                f"Tirs/match : "
                f"{avg(home_data['shots']):.1f}"
            )

        if home_data["corners"]:
            st.write(
                f"Corners/match : "
                f"{avg(home_data['corners']):.1f}"
            )

    with c2:

        st.write(f"**{away_team['name']}**")

        st.write(
            f"Matchs analysés : {away_data['played']}"
        )

        st.write(
            f"Buts marqués/match : "
            f"{avg(away_data['goals_for']):.2f}"
        )

        st.write(
            f"Buts encaissés/match : "
            f"{avg(away_data['goals_against']):.2f}"
        )

        st.write(
            f"BTTS : "
            f"{away_data['btts_yes'] / max(1, away_data['played']) * 100:.1f}%"
        )

        st.write(
            f"Over 2,5 : "
            f"{away_data['over_2_5'] / max(1, away_data['played']) * 100:.1f}%"
        )

        if away_data["shots"]:
            st.write(
                f"Tirs/match : "
                f"{avg(away_data['shots']):.1f}"
            )

        if away_data["corners"]:
            st.write(
                f"Corners/match : "
                f"{avg(away_data['corners']):.1f}"
            )

    # --------------------------------------------------------
    # DERNIERS MATCHS
    # --------------------------------------------------------

    with st.expander("📜 Voir les derniers matchs analysés"):

        st.write(f"**{home_team['name']}**")

        for match in home_matches:
            st.write(
                f"{match['home_name']} "
                f"{match['home_goals']} - "
                f"{match['away_goals']} "
                f"{match['away_name']}"
            )

        st.write("---")

        st.write(f"**{away_team['name']}**")

        for match in away_matches:
            st.write(
                f"{match['home_name']} "
                f"{match['home_goals']} - "
                f"{match['away_goals']} "
                f"{match['away_name']}"
        )                                                                                                                                                     with c1:
                                                                                                                                                                                                                                                                                                                                                                                                                    st.metric("Total Tirs Estimé", f"{total_shots:.1f}")
                                                                                                                                                                                                                                                                                                                                                                                                                                st.write(f"Over 22.5 Tirs : **{'Oui' if total_shots > 22.5 else 'Non'}**")
                                                                                                                                                                                                                                                                                                                                                                                                                                        with c2:
                                                                                                                                                                                                                                                                                                                                                                                                                                                    st.metric("Total Corners Estimé", f"{total_corners:.1f}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                st.write(f"Over 9.5 Corners : **{'Oui' if total_corners > 9.5 else 'Non'}**")
                                                                                                                                                                                                                                                                                                                                                                                                                                                    
