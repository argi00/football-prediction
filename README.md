# Floodlight — Prédicteur de matchs de football

API FastAPI + frontend React (Vite) pour prédire l'issue d'un match international
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
│   ├── package.json
│   ├── vite.config.js         # proxy /api -> localhost:8000 en dev
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── index.css
├── vercel.json                 # config de déploiement Vercel (build statique + fonction Python)
├── Dockerfile
└── README.md
```

> **Pourquoi deux formats différents pour le modèle ?** Le modèle XGBoost est
> chargé via `model.load_model("xgb_model.json")` (format natif XGBoost)
> plutôt que via `joblib.load(...)`, car sérialiser un `XGBClassifier`
> directement avec pickle/joblib s'est révélé fragile d'une version
> d'XGBoost à l'autre (erreur `XGBoostError: input stream corrupted`
> rencontrée en cours de route). Le `scaler` sklearn, lui, reste un objet
> pickle classique dans `model_extras.pkl` — ce n'est pas lui qui posait
> problème.

## Lancer en local (deux terminaux)

**Terminal 1 — backend :**
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend :**
```bash
cd frontend
npm install
npm run dev
```

Ouvre **http://localhost:5173** (Vite). Le fichier `vite.config.js` redirige
automatiquement tous les appels `/api/*` vers `http://localhost:8000`, donc le
code du frontend appelle simplement `fetch('/api/teams')` sans se soucier du
port, aussi bien en dev qu'une fois déployé.

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

- **Chargement unique au démarrage** : les ressources sont chargées une fois
  quand le serveur démarre, pas à chaque requête (ta fonction rechargeait tout
  — modèle inclus — à chaque appel, ce qui serait bien trop lent en prod).
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

## Déploiement sur Vercel (via GitHub)

Le fichier `vercel.json` à la racine utilise la config classique et largement
supportée (`builds`/`routes`, compatible avec toutes les versions de la CLI
Vercel) :

- **`frontend/package.json`** est construit avec `@vercel/static-build`
  (exécute `npm run build`, sert le contenu de `frontend/dist/`).
- **`backend/main.py`** est exécuté comme fonction Python (`@vercel/python`).
- Les requêtes `/api/*` vont au backend, tout le reste va aux fichiers
  statiques buildés du frontend.

**Étapes :**
1. Pousse tout le dépôt (avec `vercel.json` à la racine) sur GitHub.
2. Sur [vercel.com](https://vercel.com), importe le dépôt.
3. **Laisse le champ "Root Directory" vide** dans les paramètres du projet —
   le routage se fait entièrement via `vercel.json`.
4. Déploie. Teste `https://ton-projet.vercel.app/api/health` (doit répondre
   `{"status":"ok",...}`), puis `https://ton-projet.vercel.app/` pour
   l'interface.

**Pour tester en local exactement comme en production**, utilise `vercel dev`
à la racine du projet :
```bash
npm i -g vercel   # si pas déjà installé
vercel dev
```

**Si tu obtiens encore un 404 :**
- Vérifie que `vercel.json` est bien à la racine du dépôt (pas dans `backend/`
  ni `frontend/`).
- Regarde les logs de build : la ligne `@vercel/static-build` doit montrer que
  `npm run build` s'est bien exécuté dans `frontend/` et a produit un dossier
  `dist/`.
- Regarde aussi que `@vercel/python` a bien détecté `backend/requirements.txt`
  et installé les dépendances sans erreur.
- Assure-toi que le champ "Root Directory" du projet Vercel est vide.

**Limite à surveiller** : les fonctions Python de Vercel ont une limite de
bundle de 500 Mo non compressés — nos fichiers (`xgb_model.json` ~5,5 Mo,
`football_historical_data.pkl` ~3,3 Mo) sont largement dans les clous.

## Déploiement avec Docker

```bash
docker build -t football-predictor .
docker run -p 8000:8000 football-predictor
```

Le `Dockerfile` fait un build multi-étapes : il compile d'abord le frontend
React (`npm run build`), puis sert les fichiers statiques générés directement
depuis FastAPI.

## Déploiement sur une plateforme cloud (ex: Render, Railway, Fly.io, VPS)

1. Pousse ce dossier dans un dépôt Git.
2. Configure la commande de démarrage : `uvicorn main:app --host 0.0.0.0 --port $PORT`
   (adapte `$PORT` à la variable d'environnement fournie par la plateforme).
3. Vérifie que `backend/model_assets/*` est bien inclus dans le déploiement
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

## ⚠️ Limite de mon environnement de test

Je n'ai pas d'accès réseau dans le sandbox où j'ai développé ce projet, donc
je n'ai pas pu exécuter `npm install` pour le frontend (le registre npm a
renvoyé une erreur 403). Le code React a été écrit avec soin selon les
conventions Vite standard, mais **teste `npm install && npm run build` en
local avant de déployer**, et dis-moi si une erreur apparaît.
