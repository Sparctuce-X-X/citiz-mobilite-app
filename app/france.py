"""Territoires et vue « France métropolitaine » (issue 68, docs/FRANCE.md), sans Streamlit, testable avec pytest.

- Territoires : la démo Strasbourg (data/snapshot/), chaque ville exportée (data/snapshot/<ville>/) et chaque snapshot
  d'EPCI (data/snapshot/epci/<epci_code>/, même format + ville.json), plus l'entrée « France métropolitaine ».
- Vue France : indicateurs agrégés par EPCI (DATA_FRANCE/epci_access.parquet), contours (epci_geo.parquet) et, s'il
  existe, le plan national (plan_france.parquet). Vie privée : seulement des agrégats EPCI, jamais un carreau.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind

FRANCE = "france"  # clé du territoire « France métropolitaine » (?territoire=france)
FRANCE_LABEL = "France métropolitaine"
STRASBOURG_EPCI = "246700488"

# Fichiers nationaux lus par l'application (DATA_FRANCE, docs/FRANCE.md)
EPCI_ACCESS = "epci_access.parquet"
EPCI_GEO = "epci_geo.parquet"
PLAN_FRANCE = "plan_france.parquet"
COLONNES_ACCESS = ("epci_code", "epci_nom", "pop", "part_sans_option", "part_citiz", "citiz_present", "potentiel_500k")
COLONNES_GEOMETRIE = ("geometry", "wkt", "geom", "geometry_wkt")  # WKT (contrat) ou WKB (GeoParquet)

# Indicateurs de la carte de France : (libellé, format, chaud quand la valeur est haute). Même lecture que la carte
# des carreaux : froid = peu d'options de mobilité, chaud = beaucoup ; pour le potentiel, chaud = fort potentiel.
INDICATEURS = {
    "part_sans_option": ("Part des habitants sans option à 5 min", "pct", False),
    "part_citiz": ("Part des habitants avec une station Citiz à 5 min", "pct", True),
    "potentiel_500k": ("Potentiel d'un plan à 500 k€ (habitants nouvellement desservis)", "int", True),
}


def dossier_france() -> Path:
    """DATA_FRANCE (variable d'environnement prioritaire), sinon le dossier par défaut du contrat."""
    return Path(os.environ.get("DATA_FRANCE", "/Users/huang/EpitechHackaton/data/france"))


def dossier_epci(snapshot_dir: Path | str | None = None) -> Path:
    """Snapshots par EPCI : SNAPSHOT_EPCI, sinon <SNAPSHOT_DIR>/epci."""
    if os.environ.get("SNAPSHOT_EPCI"):
        return Path(os.environ["SNAPSHOT_EPCI"])
    return Path(snapshot_dir if snapshot_dir is not None else ind.SNAPSHOT_DIR) / "epci"


def fmt(x, kind: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return ind.fmt_pct(x) if kind == "pct" else ind.fmt_int(x)


# --------------------------------------------------------------------------- territoires


def snapshots_epci(base: Path | str) -> pd.DataFrame:
    """Snapshots d'EPCI disponibles : epci_code, nom, pop, dossier. Index unique `index.parquet` (epci_code, nom,
    pop) s'il existe, sinon lecture des ville.json. Seuls les dossiers qui contiennent gold_carreau_access comptent."""
    base = Path(base)
    cols = ["epci_code", "nom", "pop", "dossier"]
    if not base.is_dir():
        return pd.DataFrame(columns=cols)
    index = base / "index.parquet"
    if index.exists():
        df = pd.read_parquet(index)
        df = df.rename(columns={"epci_nom": "nom"})
        df["epci_code"] = df["epci_code"].astype(str)
        if "pop" not in df:
            df["pop"] = np.nan
        df["dossier"] = [str(base / c) for c in df["epci_code"]]
    else:
        rows = []
        for f in base.glob("*/ville.json"):
            try:
                v = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            code = str(v.get("epci") or f.parent.name)
            rows.append((code, v.get("nom") or v.get("ville") or code, v.get("pop", np.nan), str(f.parent)))
        df = pd.DataFrame(rows, columns=cols)
    df = df[[(Path(d) / "gold_carreau_access.parquet").exists() for d in df["dossier"]]]
    return df[cols].sort_values("nom", key=lambda s: s.str.lower()).reset_index(drop=True)


def territoires(snapshot_dir: Path | str | None = None, epci_dir: Path | str | None = None) -> dict[str, dict]:
    """Territoires sélectionnables, dans l'ordre d'affichage : {clé: {label, dossier, epci}}.
    « France métropolitaine » d'abord, puis les villes exportées (Strasbourg en tête), puis les EPCI par nom.
    Clé = code SIREN de l'EPCI (sinon nom de la ville) : c'est la valeur de ?territoire= dans l'URL. Un EPCI déjà
    couvert par une ville exportée (Strasbourg, Nancy) garde le snapshot de la ville (la démo ne change pas)."""
    snapshot_dir = Path(snapshot_dir if snapshot_dir is not None else ind.SNAPSHOT_DIR)
    out: dict[str, dict] = {FRANCE: {"label": FRANCE_LABEL, "dossier": None, "epci": None}}
    for dossier in ind.villes_disponibles(snapshot_dir):
        v = ind.load_city(dossier)
        cle = str(v.get("epci") or v["ville"])
        if cle in out:
            continue
        code = f" ({v['epci']})" if v.get("epci") else ""
        out[cle] = {"label": f"{v['titre']} · {v['nom']}{code}", "dossier": dossier, "epci": v.get("epci")}
    epci = snapshots_epci(epci_dir if epci_dir is not None else dossier_epci(snapshot_dir))
    for r in epci.itertuples():
        if r.epci_code not in out:
            out[r.epci_code] = {"label": f"{r.nom} ({r.epci_code})", "dossier": r.dossier, "epci": r.epci_code}
    return out


def territoire_par_defaut(terr: dict[str, dict]) -> str:
    """Strasbourg (démo) si présent, sinon le premier territoire avec des données, sinon la France."""
    if STRASBOURG_EPCI in terr:
        return STRASBOURG_EPCI
    return next((k for k in terr if k != FRANCE), FRANCE)


def resoudre(terr: dict[str, dict], valeur: str | None) -> str | None:
    """Clé d'un territoire à partir de ?territoire= : clé exacte, sinon nom de ville/EPCI (insensible à la casse)."""
    if not valeur:
        return None
    valeur = str(valeur).strip()
    if valeur in terr:
        return valeur
    bas = valeur.lower()
    return next((k for k, t in terr.items() if t["label"].lower().split(" (")[0] == bas
                 or t["label"].lower().split(" · ")[0] == bas), None)


# --------------------------------------------------------------------------- données nationales


def charger_france(directory: Path | str | None = None) -> dict[str, pd.DataFrame | None]:
    """epci_access, epci_geo, plan_france ; un fichier absent vaut None."""
    d = Path(directory) if directory is not None else dossier_france()
    out = {}
    for nom, fichier in (("epci_access", EPCI_ACCESS), ("epci_geo", EPCI_GEO), ("plan_france", PLAN_FRANCE)):
        p = d / fichier
        out[nom] = pd.read_parquet(p) if p.exists() else None
    for nom in ("epci_access", "epci_geo", "plan_france"):
        if out[nom] is not None and "epci_code" in out[nom]:
            out[nom]["epci_code"] = out[nom]["epci_code"].astype(str)
    return out


def colonnes_manquantes(access: pd.DataFrame | None) -> list[str]:
    if access is None:
        return list(COLONNES_ACCESS)
    return [c for c in COLONNES_ACCESS if c not in access]


def kpis_france(access: pd.DataFrame) -> dict:
    """Chiffres clés nationaux, pondérés par la population des EPCI."""
    pop = access["pop"].fillna(0)
    total = float(pop.sum())
    if "pop_sans_option" in access:
        sans = float(access["pop_sans_option"].fillna(0).sum())
    else:
        sans = float((access["part_sans_option"].fillna(0) * pop).sum())
    citiz = access["citiz_present"].fillna(False).astype(bool)
    return {
        "population": total,
        "part_sans_option": sans / total if total else None,
        "part_citiz": float((access["part_citiz"].fillna(0) * pop).sum() / total) if total else None,
        "n_epci": int(len(access)),
        "n_epci_citiz": int(citiz.sum()),
        "potentiel_500k": float(access["potentiel_500k"].fillna(0).sum()),
    }


def classes_indicateur(values: pd.Series, chaud_si_haut: bool = True) -> tuple[pd.Series, dict[int, tuple]]:
    """Classes 1 (froid) à 5 (chaud) d'un indicateur par EPCI, même règle que la carte des carreaux : si au moins
    un EPCI sur cinq est au plancher (aucune valeur), ces EPCI forment la classe 1 et les autres les quartiles
    2 à 5 ; sinon, quintiles. Égalités : même classe. Renvoie (classe par ligne, {classe: (min, max)})."""
    v = pd.to_numeric(values, errors="coerce")
    s = v if chaud_si_haut else -v
    classe = pd.Series(np.nan, index=v.index)
    ok = s.notna()
    plancher = s[ok].min() if ok.any() else 0
    bas = ok & (s == plancher)
    if bas.sum() >= 0.2 * ok.sum() and (ok & ~bas).sum() >= 4:
        classe[bas] = 1
        reste = ok & ~bas
        seuils = s[reste].quantile([0.25, 0.5, 0.75]).to_numpy()
        classe[reste] = 2 + (s[reste].to_numpy()[:, None] > seuils).sum(axis=1)
    elif ok.sum() >= 5:
        seuils = s[ok].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        classe[ok] = 1 + (s[ok].to_numpy()[:, None] > seuils).sum(axis=1)
    else:
        classe[ok] = 3
    bornes = {int(c): (float(v[classe == c].min()), float(v[classe == c].max())) for c in sorted(classe.dropna().unique())}
    return classe, bornes


def legende_indicateur(cle: str, bornes: dict[int, tuple]) -> list[tuple[int, str]]:
    """(classe, libellé « 12,0 % à 18,5 % ») du froid au chaud."""
    kind = INDICATEURS[cle][1]
    out = []
    for c in sorted(bornes):
        a, b = sorted(bornes[c])
        out.append((c, fmt(a, kind) if fmt(a, kind) == fmt(b, kind) else f"{fmt(a, kind)} à {fmt(b, kind)}"))
    return out


def infobulle(r) -> str:
    """Info-bulle d'un EPCI : seulement des agrégats (population totale et parts), aucun carreau."""
    citiz = "oui" if bool(getattr(r, "citiz_present", False)) else "pas encore"
    lignes = [f"<b>{r.epci_nom}</b>", f"{ind.fmt_int(r.pop)} habitants",
              f"Sans option à 5 min : {fmt(r.part_sans_option, 'pct')}",
              f"Station Citiz à 5 min : {fmt(r.part_citiz, 'pct')} (Citiz présent : {citiz})",
              f"Plan à 500 k€ : {fmt(r.potentiel_500k, 'int')} habitants nouvellement desservis"]
    cph = getattr(r, "cout_par_habitant_500k", None)
    if cph is not None and pd.notna(cph):
        lignes[-1] += f" ({ind.fmt_int(cph)} € par habitant)"
    return "<br/>".join(lignes)


def geometries(geo: pd.DataFrame):
    """Colonne géométrie d'epci_geo -> tableau shapely (WKT ou WKB)."""
    import shapely

    col = next((c for c in COLONNES_GEOMETRIE if c in geo), None)
    if col is None:
        raise KeyError(f"epci_geo : colonne géométrie attendue parmi {COLONNES_GEOMETRIE}")
    vals = geo[col]
    first = vals.dropna().iloc[0] if vals.notna().any() else None
    if isinstance(first, (bytes, bytearray, memoryview)):
        return shapely.from_wkb(vals.to_numpy())
    return shapely.from_wkt(vals.to_numpy())


def geojson_epci(access: pd.DataFrame, geo: pd.DataFrame, tolerance: float = 0.01, precision: float = 1e-3) -> dict:
    """Carte des EPCI pour folium : {geojson, legendes, bounds}. GeoJSON léger : contours simplifiés à ≈ 1 km et
    coordonnées à 3 décimales (invisible à l'échelle de la France), info-bulle et couleur de chaque indicateur
    précalculées (propriétés `tip` et `c_<indicateur>`). `legendes` : classes de chaque indicateur ; `bounds` :
    emprise [[lat, lon], [lat, lon]]."""
    import shapely

    df = geo[["epci_code"]].assign(_g=geometries(geo)).merge(access, on="epci_code", how="inner")
    g = shapely.simplify(df["_g"].to_numpy(), tolerance, preserve_topology=True)
    g = shapely.set_precision(g, precision)
    geojsons = shapely.to_geojson(g)
    couleurs, legendes = {}, {}
    for cle, (_, _, chaud) in INDICATEURS.items():
        classe, bornes = classes_indicateur(df[cle], chaud)
        couleurs[cle] = [ind.CLASS_COLORS[int(c)] if pd.notna(c) else ind.PROTECTED_COLOR for c in classe]
        legendes[cle] = legende_indicateur(cle, bornes)
    features = []
    for i, r in enumerate(df.itertuples()):
        if geojsons[i] is None:
            continue
        props = {"epci_code": r.epci_code, "tip": infobulle(r)}
        props.update({f"c_{cle}": couleurs[cle][i] for cle in INDICATEURS})
        features.append({"type": "Feature", "geometry": json.loads(geojsons[i]), "properties": props})
    lon_min, lat_min, lon_max, lat_max = shapely.total_bounds(g)
    return {"geojson": {"type": "FeatureCollection", "features": features}, "legendes": legendes,
            "bounds": [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]}


def classement(access: pd.DataFrame, citiz: str = "tous", n: int = 20) -> pd.DataFrame:
    """« Où investir en premier ? » : EPCI par potentiel d'un plan à 500 k€ (habitants nouvellement desservis),
    filtrés sur la présence de Citiz aujourd'hui (« tous », « avec », « sans »)."""
    df = access.copy()
    present = df["citiz_present"].fillna(False).astype(bool)
    if citiz == "avec":
        df = df[present]
    elif citiz == "sans":
        df = df[~present]
    df = df.sort_values(["potentiel_500k", "pop"], ascending=[False, False]).head(n)
    return df.reset_index(drop=True)


def classement_fr(top: pd.DataFrame, avec_detail: set) -> pd.DataFrame:
    """Version affichée du classement (formats français)."""
    cph = top["cout_par_habitant_500k"] if "cout_par_habitant_500k" in top else pd.Series(np.nan, index=top.index)
    return pd.DataFrame({
        "Rang": range(1, len(top) + 1),
        "Intercommunalité (EPCI)": top["epci_nom"],
        "Habitants": top["pop"].map(ind.fmt_int),
        "Habitants nouvellement desservis (500 k€)": top["potentiel_500k"].map(lambda v: fmt(v, "int")),
        "Coût par habitant": cph.map(lambda v: "—" if pd.isna(v) else f"{ind.fmt_int(v)} €"),
        "Sans option à 5 min": top["part_sans_option"].map(lambda v: fmt(v, "pct")),
        "Citiz aujourd'hui": top["citiz_present"].fillna(False).astype(bool).map({True: "Oui", False: "Pas encore"}),
        "Détail disponible": top["epci_code"].isin(avec_detail).map({True: "Oui", False: "—"}),
    })


# --------------------------------------------------------------------------- plan national


def budgets_plan(plan: pd.DataFrame) -> list[int]:
    """Budgets proposés pour le plan national : paliers jusqu'au budget du plan calculé."""
    maxi = float(plan["budget_eur"].max()) if "budget_eur" in plan and plan["budget_eur"].notna().any() \
        else float(plan["cumul_cost_eur"].max())
    paliers = [1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6, 1e9]
    out = [int(b) for b in paliers if b < maxi]
    return out + [int(maxi)]


def plan_sous_budget(plan: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Emplacements du plan national financés par `budget` : le plan est classé (glouton), on garde les premiers
    rangs dont le coût cumulé tient dans le budget. Si le fichier contient plusieurs plans (budget_eur), on part
    du plus grand."""
    p = plan
    if "budget_eur" in p and p["budget_eur"].nunique() > 1:
        p = p[p["budget_eur"] == p["budget_eur"].max()]
    p = p.sort_values("rank")
    return p[p["cumul_cost_eur"] <= budget + 1e-6]


def plan_par_epci(plan: pd.DataFrame, access: pd.DataFrame | None) -> pd.DataFrame:
    """Répartition par EPCI d'un plan : emplacements par type, coût, habitants nouvellement desservis (somme par
    équipement), position moyenne (pour la carte de points)."""
    cols = ["epci_code", "epci_nom", "n_emplacements", "citiz", "bus", "velhop", "cout_eur", "new_pop", "lat", "lon"]
    if plan.empty:
        return pd.DataFrame(columns=cols)
    g = plan.groupby("epci_code")
    out = pd.DataFrame({"n_emplacements": g.size(), "cout_eur": g["cost_eur"].sum(), "new_pop": g["new_pop"].sum(),
                        "lat": g["lat"].mean(), "lon": g["lon"].mean()})
    par_type = plan.pivot_table(index="epci_code", columns="type", values="rank", aggfunc="count", fill_value=0)
    for t in ("citiz", "bus", "velhop"):
        out[t] = par_type[t] if t in par_type else 0
    out = out.reset_index()
    noms = access.set_index("epci_code")["epci_nom"] if access is not None and "epci_nom" in access else pd.Series(dtype=str)
    out["epci_nom"] = out["epci_code"].map(noms).fillna(out["epci_code"])
    return out[cols].sort_values("new_pop", ascending=False).reset_index(drop=True)


def plan_par_epci_fr(rep: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Intercommunalité (EPCI)": rep["epci_nom"],
        "Emplacements": rep["n_emplacements"].astype(int),
        "Citiz": rep["citiz"].astype(int), "Bus": rep["bus"].astype(int), "Vélo": rep["velhop"].astype(int),
        "Habitants nouvellement desservis": rep["new_pop"].map(ind.fmt_int),
        "Coût": rep["cout_eur"].map(ind.fmt_eur_plein),
    })
