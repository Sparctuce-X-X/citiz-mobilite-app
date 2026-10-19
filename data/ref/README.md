# Fichiers de référence

Petits fichiers versionnés dans Git (quelques Ko), par exemple `couts.csv` (coûts par type d'équipement, tâche #18, voir `docs/COUTS.md`). Tout le reste de `data/` est ignoré : les données lourdes restent dans le Lakehouse Fabric.

- `projections_omphale_dep.csv` : projections de population INSEE Omphale 2022 par département (scénario central), reconstruit par `python optim/build_projections.py <projections_scenario_central.xlsx>` ; lu par `optim/projections.py`.
