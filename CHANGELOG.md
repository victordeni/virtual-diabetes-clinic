# Changelog
## v0.1 – Baseline
- Pipeline: StandardScaler + LinearRegression
- API: /health, /predict
- Docker: modèle inclus
- CI: lint, tests, entraînement, artifacts
## v0.2 – Improvement
- Model: Ridge(alpha=1.0) au lieu de LinearRegression.
- Résultat: RMSE(test) v0.2 = <remplacer> vs v0.1 = <remplacer> (Δ = <remplacer>%).
- Rationale: la régularisation L2 stabilise les coefficients et réduit l’overfit → meilleure généralisation.
