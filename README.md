# Floodlight — Prédicteur de matchs de football

API FastAPI + frontend statique pour prédire l'issue d'un match international
(victoire domicile / nul / victoire extérieur) à partir d'un modèle XGBoost
entraîné en amont.

## Structure du projet

```
football-predictor/
├── backend/
│   ├── main.py                # API FastAPI
│   ├── requirements.txt
│   └── model_assets/
│       ├── xgb_model.json           # modèle XGBoost, format natif (robuste entre versions)
│       ├── model_extras.pkl         # scaler sklearn (le champ "features" qu'il contient est ignoré, voir main.py)
│       └── football_historical_data.pkl
├── frontend/
│   └── index.html             # UI (HTML/CSS/JS, aucun build nécessaire)
├── Dockerfile
└── README.md
```

> **Pourquoi deux formats différents ?** Le modèle XGBoost est chargé via
> `model.load_model("xgb_model.json")` (format natif XGBoost) plutôt que via
> `joblib.load(...)`, car sérialiser un `XGBClassifier` directement avec
> pickle/joblib s'est révélé fragile d'une version d'XGBoost à l'autre
> (erreur `XGBoostError: input stream corrupted` rencontrée en cours de
> route). Le `scaler` sklearn, lui, reste un objet pickle classique dans
> `model_extras.pkl` — ce n'est pas lui qui posait problème.

## Lancer en local

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Ouvre ensuite **http://localhost:8000** — FastAPI sert à la fois l'API
(`/api/teams`, `/api/predict`) et le frontend statique (`index.html`) sur le
même port, donc aucune configuration CORS particulière n'est nécessaire.

Documentation interactive de l'API (Swagger) : **http://localhost:8000/docs**

## Endpoints

| Méthode | Route          | Description                                   |
|---------|----------------|------------------------------------------------|
| GET     | `/api/teams`   | Liste des 193 équipes connues de l'historique  |
| POST    | `/api/predict` | Prédiction pour un match `team_home` vs `team_away` |
| GET     | `/api/health`  | Vérification que le modèle est bien chargé     |

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"team_home": "France", "team_away": "Belgium", "is_neutral": true}'
```

```json
{
  "team_home": "France",
  "team_away": "Belgium",
  "prob_home_win": 0.55,
  "prob_draw": 0.23,
  "prob_away_win": 0.22,
  "predicted_result": "home",
  "predicted_team": "France"
}
```

## Ce qui a changé par rapport à ta fonction notebook

- **Chargement unique au démarrage** : `joblib.load(...)` n'est appelé qu'une
  fois quand le serveur démarre, pas à chaque requête (ta fonction rechargeait
  les deux fichiers — modèle inclus — à chaque appel, ce qui serait bien trop
  lent en prod).
- **`h2h_win_rate` et `momentum_diff` calculés réellement** à partir de
  l'historique (taux de victoires passées entre les deux équipes, et écart de
  forme récente), plutôt que fixés à `0.5` et `0` par défaut. Si les deux
  équipes ne se sont jamais rencontrées, `h2h_win_rate` retombe sur `0.5`
  (neutre). Si tu préfères garder exactement le comportement du notebook,
  remplace l'appel à `_h2h_win_rate(...)` et à `momentum_diff` dans
  `_build_feature_vector` par des constantes `0.5` et `0`.
- **Validation des entrées** : noms d'équipes inconnus (404), équipes
  identiques (400), avec des messages d'erreur clairs renvoyés au frontend.
- **Ordre des features codé en dur** (`FEATURES` en haut de `main.py`) plutôt
  que lu depuis `model_extras.pkl["features"]` : ce dernier ne contenait que
  7 des 9 noms attendus par le scaler (probablement une réexécution partielle
  du notebook d'entraînement avant la sauvegarde). Si tu retouches le modèle,
  pense à vérifier que `scaler.mean_.shape[0] == len(FEATURES)` — l'API refuse
  de démarrer si ce n'est pas le cas, avec un message explicite.

## Déploiement avec Docker

```bash
docker build -t football-predictor .
docker run -p 8000:8000 football-predictor
```

## Déploiement sur une plateforme cloud (ex: Render, Railway, Fly.io, VPS)

1. Pousse ce dossier dans un dépôt Git.
2. Configure la commande de démarrage : `uvicorn main:app --host 0.0.0.0 --port $PORT`
   (adapte `$PORT` à la variable d'environnement fournie par la plateforme).
3. Vérifie que `backend/model_assets/*.pkl` est bien inclus dans le déploiement
   (les fichiers volumineux sont parfois exclus par `.gitignore` par défaut —
   utilise Git LFS si un fichier dépasse les limites de ton hébergeur).
4. En production, remplace `allow_origins=["*"]` dans `main.py` par le(s)
   domaine(s) réel(s) de ton frontend si tu le sépares du backend.

## Limites connues du modèle actuel

- Les statistiques (rang, points, momentum) utilisées sont celles du **dernier
  match connu** de chaque équipe dans l'historique — si `football_historical_data.pkl`
  n'est pas mis à jour régulièrement, les prédictions se baseront sur des
  données de plus en plus anciennes.
- Aucune queue d'attente / rate limiting n'est en place : à prévoir avant une
  mise en production à fort trafic.
