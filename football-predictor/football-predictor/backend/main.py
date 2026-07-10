"""
API de prédiction de résultats de matchs de football.

Charge une bonne fois pour toutes (au démarrage du serveur) le modèle XGBoost
entraîné ainsi que l'historique des matchs, puis expose :

  GET  /api/teams    -> liste des équipes connues (pour peupler le frontend)
  POST /api/predict  -> probabilités Victoire domicile / Nul / Victoire extérieur

Lancer en local :
    uvicorn main:app --reload --port 8000
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "model_assets"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Ordre exact des 9 features attendues par le scaler et le modèle.
# NB : le fichier `model_extras.pkl` contient une liste `features` tronquée
# (7 éléments ; il manquait 'h2h_win_rate' et 'momentum_diff' suite à une
# réexécution partielle du notebook d'entraînement). Cette liste-ci a été
# vérifiée directement contre les dimensions du scaler (`scaler.mean_`,
# 9 valeurs) et du modèle (`num_feature=9`) — c'est la version fiable.
FEATURES = [
    "home_rank", "away_rank", "home_points", "away_points",
    "rank_diff", "point_diff", "neutral", "h2h_win_rate", "momentum_diff",
]

app = FastAPI(
    title="Football Match Predictor API",
    description="Prédiction du résultat d'un match à partir d'un modèle XGBoost",
    version="1.0.0",
)

# En développement on autorise tout le monde. En production, restreins
# `allow_origins` au(x) domaine(s) réel(s) de ton frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Chargement des ressources (une seule fois, au démarrage du process)
# ----------------------------------------------------------------------
try:
    # Modèle au format natif XGBoost (robuste entre versions, contrairement
    # à un XGBClassifier pickled directement avec joblib).
    _model = xgb.XGBClassifier()
    _model.load_model(str(ASSETS_DIR / "xgb_model.json"))

    # Scaler sklearn (léger, pickling classique OK pour ça).
    _extras = joblib.load(ASSETS_DIR / "model_extras.pkl")
    _scaler = _extras["scaler"]
    _features = FEATURES  # cf. note ci-dessus, on n'utilise pas _extras["features"]

    _history = joblib.load(ASSETS_DIR / "football_historical_data.pkl")
    _history = _history.sort_values("date").reset_index(drop=True)
except FileNotFoundError as exc:
    raise RuntimeError(
        "Fichiers modèle introuvables. Place 'xgb_model.json', "
        "'model_extras.pkl' et 'football_historical_data.pkl' "
        f"dans {ASSETS_DIR}"
    ) from exc

if _scaler.mean_.shape[0] != len(_features):
    raise RuntimeError(
        f"Incohérence : le scaler attend {_scaler.mean_.shape[0]} features "
        f"mais FEATURES en définit {len(_features)}. Vérifie la liste FEATURES "
        "dans main.py."
    )

_ALL_TEAMS = sorted(set(_history["home_team"]).union(_history["away_team"]))
_RESULT_MAP = {0: "away", 1: "draw", 2: "home"}


# ----------------------------------------------------------------------
# Schémas API
# ----------------------------------------------------------------------
class PredictRequest(BaseModel):
    team_home: str = Field(..., description="Équipe recevante", examples=["France"])
    team_away: str = Field(..., description="Équipe visiteuse", examples=["Belgium"])
    is_neutral: bool = Field(False, description="Match joué sur terrain neutre")


class PredictResponse(BaseModel):
    team_home: str
    team_away: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    predicted_result: str          # "home" | "draw" | "away"
    predicted_team: str            # nom de l'équipe gagnante, ou "Match Nul"


class TeamsResponse(BaseModel):
    teams: list[str]


# ----------------------------------------------------------------------
# Logique métier
# ----------------------------------------------------------------------
def _last_team_stats(team: str) -> dict:
    """Dernier rang / points / momentum connus pour une équipe donnée."""
    matches = _history[(_history["home_team"] == team) | (_history["away_team"] == team)]
    if matches.empty:
        return None
    row = matches.iloc[-1]
    if row["home_team"] == team:
        return {"rank": row["home_rank"], "points": row["home_points"], "momentum": row["home_momentum"]}
    return {"rank": row["away_rank"], "points": row["away_points"], "momentum": row["away_momentum"]}


def _h2h_win_rate(team_home: str, team_away: str) -> float:
    """
    Taux de victoire historique de `team_home` face à `team_away`,
    tous matchs confondus (peu importe qui recevait à l'époque).
    Renvoie 0.5 (neutre) s'il n'y a jamais eu de confrontation directe.
    """
    direct = (_history["home_team"] == team_home) & (_history["away_team"] == team_away)
    reverse = (_history["home_team"] == team_away) & (_history["away_team"] == team_home)
    meetings = _history[direct | reverse]

    if meetings.empty:
        return 0.5

    wins = 0
    for _, row in meetings.iterrows():
        if row["home_team"] == team_home and row["target"] == 2:
            wins += 1
        elif row["away_team"] == team_home and row["target"] == 0:
            wins += 1
    return wins / len(meetings)


def _build_feature_vector(team_home: str, team_away: str, is_neutral: bool) -> pd.DataFrame:
    stats_h = _last_team_stats(team_home)
    stats_a = _last_team_stats(team_away)

    h2h = _h2h_win_rate(team_home, team_away)
    momentum_diff = stats_h["momentum"] - stats_a["momentum"]

    row = [
        stats_h["rank"], stats_a["rank"],
        stats_h["points"], stats_a["points"],
        stats_h["rank"] - stats_a["rank"],
        stats_h["points"] - stats_a["points"],
        1 if is_neutral else 0,
        h2h,
        momentum_diff,
    ]
    return pd.DataFrame([row], columns=_features)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/api/teams", response_model=TeamsResponse)
def get_teams():
    """Liste des équipes présentes dans l'historique (pour les menus déroulants)."""
    return TeamsResponse(teams=_ALL_TEAMS)


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    team_home = payload.team_home.strip()
    team_away = payload.team_away.strip()

    if team_home.lower() == team_away.lower():
        raise HTTPException(400, "Les deux équipes doivent être différentes.")
    if team_home not in _ALL_TEAMS:
        raise HTTPException(404, f"Équipe inconnue dans l'historique : « {team_home} »")
    if team_away not in _ALL_TEAMS:
        raise HTTPException(404, f"Équipe inconnue dans l'historique : « {team_away} »")

    input_df = _build_feature_vector(team_home, team_away, payload.is_neutral)
    input_scaled = _scaler.transform(input_df)
    prob = _model.predict_proba(input_scaled)[0]  # [P(away), P(draw), P(home)]

    winner_idx = int(np.argmax(prob))
    predicted_team = {0: team_away, 1: "Match Nul", 2: team_home}[winner_idx]

    return PredictResponse(
        team_home=team_home,
        team_away=team_away,
        prob_home_win=round(float(prob[2]), 4),
        prob_draw=round(float(prob[1]), 4),
        prob_away_win=round(float(prob[0]), 4),
        predicted_result=_RESULT_MAP[winner_idx],
        predicted_team=predicted_team,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "teams_loaded": len(_ALL_TEAMS)}


# ----------------------------------------------------------------------
# Sert le frontend statique (index.html, style.css, app.js) à la racine.
# Doit être monté APRÈS les routes /api/... pour ne pas les masquer.
# ----------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
