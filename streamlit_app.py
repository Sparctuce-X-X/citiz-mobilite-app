"""Point d'entrée Streamlit Community Cloud : télécharge les données au premier démarrage, puis lance app/app.py.

Les données (snapshots Strasbourg, Nancy, intercommunalités et vue France) sont une archive du dépôt de données
Hugging Face Sparctuce358/citiz-mobilite-data ; le code de l'application est inchangé.
"""
import os
import runpy
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
os.environ.setdefault("DATA_FRANCE", str(DATA / "france"))

if not (DATA / "snapshot" / "gold_carreau_access.parquet").exists():
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download("Sparctuce358/citiz-mobilite-data", "data.tar.gz", repo_type="dataset")
    with tarfile.open(archive) as tar:
        tar.extractall(DATA, filter="data")

runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
