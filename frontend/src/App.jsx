import { useEffect, useState } from 'react';

// Chemin relatif : fonctionne en dev (proxy Vite -> localhost:8000) comme en
// prod une fois déployé, tant que /api/* pointe vers le même backend.
const API_BASE = '';

function PitchBackground() {
  return (
    <div className="pitch-bg" aria-hidden="true">
      <svg viewBox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">
        <line x1="0" y1="400" x2="800" y2="400" stroke="#F2F5EC" strokeWidth="2" />
        <circle cx="400" cy="400" r="140" fill="none" stroke="#F2F5EC" strokeWidth="2" />
        <circle cx="400" cy="400" r="4" fill="#F2F5EC" />
      </svg>
    </div>
  );
}

export default function App() {
  const [teams, setTeams] = useState([]);
  const [teamHome, setTeamHome] = useState('');
  const [teamAway, setTeamAway] = useState('');
  const [isNeutral, setIsNeutral] = useState(false);
  const [loadingTeams, setLoadingTeams] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [barsVisible, setBarsVisible] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadTeams() {
      try {
        const res = await fetch(`${API_BASE}/api/teams`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (cancelled) return;

        setTeams(data.teams);
        if (data.teams.includes('France')) setTeamHome('France');
        if (data.teams.includes('Belgium')) setTeamAway('Belgium');
      } catch {
        if (!cancelled) {
          setError("Le serveur ne répond pas. Vérifie que l'API FastAPI est bien lancée.");
        }
      } finally {
        if (!cancelled) setLoadingTeams(false);
      }
    }

    loadTeams();
    return () => { cancelled = true; };
  }, []);

  async function handlePredict() {
    setError('');

    if (!teamHome || !teamAway) {
      setError('Choisis les deux équipes avant de lancer la prédiction.');
      return;
    }
    if (teamHome === teamAway) {
      setError('Les deux équipes doivent être différentes.');
      return;
    }

    setPredicting(true);
    setBarsVisible(false);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          team_home: teamHome,
          team_away: teamAway,
          is_neutral: isNeutral,
        }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Erreur lors de la prédiction.');
      }

      setResult(data);
      requestAnimationFrame(() => setBarsVisible(true));
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  const pHome = result ? Math.round(result.prob_home_win * 100) : 0;
  const pDraw = result ? Math.round(result.prob_draw * 100) : 0;
  const pAway = result ? Math.round(result.prob_away_win * 100) : 0;
  const winnerLabel = result
    ? (result.predicted_result === 'draw' ? 'Match Nul' : result.predicted_team)
    : '';

  return (
    <div className="page">
      <PitchBackground />
      <div className="wrap">
        <div className="eyebrow">Floodlight Predictor</div>
        <h1>Qui gagne ce soir&nbsp;?</h1>
        <p className="subtitle">
          Choisis deux sélections nationales, le modèle calcule les probabilités
          de victoire, nul et défaite à partir des dernières statistiques connues.
        </p>

        <div className="card">
          <div className="matchup">
            <div className="team-field home">
              <label htmlFor="teamHome">Domicile</label>
              <select
                id="teamHome"
                value={teamHome}
                onChange={(e) => setTeamHome(e.target.value)}
              >
                <option value="">{loadingTeams ? 'Chargement…' : 'Sélectionner…'}</option>
                {teams.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>

            <div className="vs">VS</div>

            <div className="team-field away">
              <label htmlFor="teamAway">Extérieur</label>
              <select
                id="teamAway"
                value={teamAway}
                onChange={(e) => setTeamAway(e.target.value)}
              >
                <option value="">{loadingTeams ? 'Chargement…' : 'Sélectionner…'}</option>
                {teams.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>

          <div className="options-row">
            <label className="toggle" htmlFor="neutralToggle">
              <input
                type="checkbox"
                id="neutralToggle"
                checked={isNeutral}
                onChange={(e) => setIsNeutral(e.target.checked)}
              />
              Match sur terrain neutre
            </label>
          </div>

          <button className="predict-btn" onClick={handlePredict} disabled={predicting}>
            {predicting ? 'Calcul…' : 'Prédire le match'}
          </button>

          {error && <p className="error-msg">{error}</p>}

          {result && (
            <div className="result">
              <p className="verdict">
                Résultat le plus probable&nbsp;: <span className="winner">{winnerLabel}</span>
              </p>

              <div className="score-row">
                <div className="score-cell home">
                  <div className="name">{result.team_home}</div>
                  <div className="pct">{pHome}%</div>
                </div>
                <div className="score-cell draw">
                  <div className="name">Nul</div>
                  <div className="pct">{pDraw}%</div>
                </div>
                <div className="score-cell away">
                  <div className="name">{result.team_away}</div>
                  <div className="pct">{pAway}%</div>
                </div>
              </div>

              <div className="bar">
                <span className="seg-home" style={{ width: barsVisible ? `${pHome}%` : 0 }} />
                <span className="seg-draw" style={{ width: barsVisible ? `${pDraw}%` : 0 }} />
                <span className="seg-away" style={{ width: barsVisible ? `${pAway}%` : 0 }} />
              </div>
            </div>
          )}
        </div>

        <footer>Modèle XGBoost entraîné sur données historiques FIFA · usage informatif uniquement</footer>
      </div>
    </div>
  );
}
