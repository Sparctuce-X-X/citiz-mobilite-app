"""Copie locale des tables utiles à l'optimiseur (data/snapshot/<table>.parquet).

    python optim/export_snapshot.py                 # Strasbourg : schéma dbo -> data/snapshot/
    python optim/export_snapshot.py --ville nancy   # autre ville : son schéma -> data/snapshot/nancy/

Lit les tables Delta du Lakehouse sur OneLake. Jeton : variable d'environnement ONELAKE_TOKEN si elle
existe (jeton « storage » valable ~1 h), sinon la connexion en cache de fabric/onelake_sync.py.
Le snapshot permet à l'application et aux tests de tourner sans Fabric (data/ est ignoré par Git).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "snapshot"

# table -> colonnes gardées (None = toutes)
TABLES = {
    "gold_candidates": ["candidate_id", "type", "lat", "lon", "x", "y", "carreau_id",
                        "covered_carreaux", "new_pop", "pop_couverte"],
    "gold_carreau_access": ["carreau_id", "lat", "lon", "pop", "i_est_200", "n_tram", "n_bus",
                            "n_citiz", "n_velhop", "options_ponderees", "classe"],
    "gold_predictions": ["candidate_id", "p_usage", "model_version"],
    "silver_carreaux": ["carreau_id", "pop", "menages", "menages_pauvres", "niveau_vie_somme",
                        "commune_code", "iris_code", "i_est_200"],
}
# absentes sans erreur : gold_costs (repli sur data/ref/couts.csv), gold_carreau_qpv (autre tâche),
# silver_citiz_stations (test « Ouverture prochaine » de optim/benchmark.py)
OPTIONAL = {"gold_costs": None, "gold_carreau_qpv": None,
            "silver_citiz_stations": ["station_id", "name", "lat", "lon", "x", "y", "ouverture_prochaine"]}


def token() -> str:
    if os.environ.get("ONELAKE_TOKEN"):
        return os.environ["ONELAKE_TOKEN"]
    sys.path.insert(0, str(ROOT / "fabric"))
    import onelake_sync as o
    return o.credential().get_token("https://storage.azure.com/.default").token


def snapshot_dir(ville: str) -> Path:
    """Strasbourg : data/snapshot/ (inchangé) ; autre ville : data/snapshot/<ville>/."""
    return SNAPSHOT if ville == "strasbourg" else SNAPSHOT / ville


def main() -> None:
    from deltalake import DeltaTable

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ville", default="strasbourg", help="config/villes/<ville>.json (défaut : strasbourg)")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "fabric"))
    import onelake_sync as o

    cfg = o.city_config(args.ville)
    base = o.tables_uri(cfg["schema"])
    opts = {"bearer_token": token(), "use_fabric_endpoint": "true"}
    dest = snapshot_dir(args.ville)
    dest.mkdir(parents=True, exist_ok=True)
    if args.ville != "strasbourg":  # l'application y lit le nom de la ville (titres)
        (dest / "ville.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{cfg['nom']} : schéma {cfg['schema']} -> {dest}")
    for name, cols in {**TABLES, **OPTIONAL}.items():
        try:
            df = DeltaTable(base + name, storage_options=opts).to_pandas(columns=cols)
        except Exception as e:  # noqa: BLE001
            if name in OPTIONAL:
                print(f"{name} : absente ({type(e).__name__}), ignorée")
                continue
            raise
        df.to_parquet(dest / f"{name}.parquet", index=False)
        print(f"{name} : {len(df):,} lignes -> {dest / (name + '.parquet')}")


if __name__ == "__main__":
    main()
