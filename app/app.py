"""Application de démonstration — Hackathon Mobility (Citiz x Microsoft x Epitech).

    streamlit run app/app.py

Lit le snapshot local data/snapshot/*.parquet (exporté par app/export_snapshot.py) : fonctionne hors ligne.
Seule la rubrique affichée est calculée (st.segmented_control + st.fragment), les plans sont mis en cache.
Territoires : Strasbourg (défaut), villes exportées, snapshots d'EPCI (data/snapshot/epci/<code>/) et vue
« France métropolitaine » (DATA_FRANCE, docs/FRANCE.md) ; le territoire choisi est dans l'URL (?territoire=<code>).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import folium
import pandas as pd
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (str(ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import france as fr  # noqa: E402
import indicators as ind  # noqa: E402

REPO = "https://github.com/Sparctuce-X-X/Hackaton-EPITECH-citizxMicrosoft/blob/main"
BUDGETS = [50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000,
           1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
EQUITE = [round(i / 10, 1) for i in range(11)]
MIN_P = [round(i * 0.05, 2) for i in range(19)]
TYPES_PLAN = ("citiz", "bus", "velhop")
CONTOUR_DIFF = "#F4A100"  # emplacement présent dans un seul des deux plans (comparaison)
TEXTE = "#2B3445"


@st.cache_data(show_spinner=False, ttl=300)
def liste_territoires(snapshot_dir: str, epci_dir: str) -> dict[str, dict]:
    """France métropolitaine, villes exportées (data/snapshot/, data/snapshot/<ville>/) et snapshots d'EPCI
    (data/snapshot/epci/<code>/ ou index.parquet) : lu une fois, rafraîchi toutes les 5 minutes."""
    return fr.territoires(snapshot_dir, epci_dir)


# Territoire affiché : ?territoire=<code EPCI | france> dans l'URL (lien partageable), sinon Strasbourg (démo).
TERRITOIRES = liste_territoires(str(ind.SNAPSHOT_DIR), str(fr.dossier_epci(ind.SNAPSHOT_DIR)))
DEFAUT = fr.territoire_par_defaut(TERRITOIRES)
territoire_url = st.query_params.get("territoire")
territoire_inconnu = None
if "territoire" not in st.session_state:
    cle_url = fr.resoudre(TERRITOIRES, territoire_url)
    territoire_inconnu = territoire_url if territoire_url and cle_url is None else None
    st.session_state["territoire"] = cle_url or DEFAUT
elif st.session_state["territoire"] not in TERRITOIRES:
    st.session_state["territoire"] = DEFAUT
TERRITOIRE = st.session_state["territoire"]
EST_FRANCE = TERRITOIRE == fr.FRANCE
data_dir = None if EST_FRANCE else TERRITOIRES[TERRITOIRE]["dossier"]
VILLE = ({"ville": fr.FRANCE, "nom": fr.FRANCE_LABEL, "titre": fr.FRANCE_LABEL, "velo_nom": ind.VELO_GENERIQUE,
          "a_velo": True, "est_epci": False} if EST_FRANCE else ind.load_city(data_dir))
ind.nommer_velo(VILLE["velo_nom"])
STATIC = APP_DIR / "static"
NOM = "À deux pas"
st.set_page_config(page_title=f"{NOM} · {VILLE['titre']}", page_icon=str(STATIC / "favicon.png"), layout="wide")
LOGO = base64.b64encode((STATIC / "logo_128.png").read_bytes()).decode()

# Mise en page « carte plein écran » : la carte (conteneur `fond`) occupe toute la fenêtre ; l'en-tête (`entete`),
# les rubriques (`nav`) et le panneau de réglages et de résultats (`panneau`) flottent par-dessus. Sous 900 px de
# large, tout reprend le flux normal (téléphone).
PANNEAU = "clamp(320px, 28vw, 420px)"
st.markdown(
    f"""
    <style>
      .note {{font-size:.85rem; color:#4A5568;}}
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .stCaption, .stCaption *,
      [data-testid="stWidgetLabel"] p {{color:#4A5568 !important; opacity:1 !important;}}
      .reco {{font-size:.98rem; background:#F7F4EE; border-left:4px solid #14213D; border-radius:4px;
             padding:.6rem .8rem; margin:.1rem 0 .4rem;}}
      .footer {{font-size:.72rem; color:#4A5568; margin-top:.6rem; border-top:1px solid rgba(43,52,69,.2); padding-top:.5rem;}}
      .marque {{display:flex; align-items:center; gap:10px;}}
      .marque img {{width:44px; height:44px;}}
      .marque b {{font-family:Rubik, sans-serif; font-size:1.45rem; color:#14213D; line-height:1;}}
      .marque span {{display:block; font-size:.74rem; color:#4A5568; margin-top:3px;}}
      .hero {{font-family:Rubik, sans-serif; font-weight:700; font-size:3.4rem; line-height:1; color:#2E7D4F;}}
      .hero-lib {{font-size:.92rem; color:#2B3445; margin:.3rem 0 .2rem;}}
      div[data-testid="stMetricValue"] {{font-size:1.3rem;}}
      [data-testid="stMetricLabel"] p {{font-size:.78rem; white-space:normal; line-height:1.25;}}
      [data-testid="stMetricLabel"] div {{overflow:visible;}}
      @media (min-width: 900px) {{
        header[data-testid="stHeader"] {{background:transparent; pointer-events:none;}}
        /* Streamlit Cloud : la barre d'outils couvre toute la largeur en haut ; seuls ses boutons restent cliquables */
        header[data-testid="stHeader"] [data-testid="stToolbar"], header[data-testid="stHeader"] [data-testid="stToolbar"] > div {{pointer-events:none;}}
        header[data-testid="stHeader"] button, header[data-testid="stHeader"] a {{pointer-events:auto;}}
        .block-container {{padding:0 !important; max-width:100% !important;}}
        .st-key-fond {{position:fixed; inset:0; z-index:0;}}
        .st-key-fond iframe {{position:fixed; inset:0; width:100vw !important; height:100vh !important;}}
        .st-key-entete, .st-key-nav, .st-key-panneau {{position:fixed; z-index:20; background:rgba(255,255,255,.93);
          backdrop-filter:blur(12px); box-shadow:0 8px 28px rgba(20,33,61,.18); box-sizing:border-box;}}
        .st-key-entete {{top:16px; left:16px; width:320px; padding:14px 16px; border-radius:16px; gap:.5rem;}}
        .st-key-entete [data-testid="stHeading"] {{position:absolute; width:1px; height:1px; overflow:hidden;
          clip:rect(0 0 0 0); white-space:nowrap;}}
        .st-key-panneau h2, .st-key-panneau h3 {{font-size:1.05rem !important; padding:.4rem 0 .1rem !important;}}
        .st-key-panneau h4 {{font-size:.98rem !important;}}
        .st-key-nav {{top:16px; left:352px; padding:4px 8px; border-radius:999px; width:auto;}}
        .st-key-panneau {{top:60px; right:16px; bottom:16px; width:{PANNEAU}; padding:16px 18px; border-radius:16px;
          overflow-y:auto; overflow-x:hidden;}}
      }}
      @media (max-width: 899px) {{
        .st-key-fond iframe {{height:60vh !important;}}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- données (en cache)


@st.cache_data(show_spinner="Chargement des données…", max_entries=8)
def load_data(directory: str) -> dict:
    return ind.load_snapshot(directory)


@st.cache_data(show_spinner=False)
def carreaux_geojson(directory: str, modes: tuple) -> dict:
    """Carreaux colorés par classe pour la carte de l'onglet ①, prêts pour folium.GeoJson."""
    cdata = ind.map_carreaux(load_data(directory)["gold_carreau_access"], modes)
    features = []
    for poly, fill, tip in zip(cdata["polygon"], cdata["fill"], cdata["tooltip"]):
        color, opacity = rgba(fill)
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [poly + [poly[0]]]},
                         "properties": {"color": color, "opacity": round(opacity, 2), "tip": tip}})
    return {"type": "FeatureCollection", "features": features}


@st.cache_resource(show_spinner="Préparation de l'optimiseur…", max_entries=4)
def _optim_inputs(directory: str):
    from optim import plan as optim_plan

    return optim_plan, optim_plan.load_inputs("snapshot", directory)


def optimiser():
    """Importe optim.plan (développé à part). Renvoie (module, inputs) ou (None, message).
    Seul le succès est mis en cache : si le module arrive en cours de route, un rechargement suffit."""
    try:
        return _optim_inputs(data_dir)
    except Exception as e:  # module absent ou entrées manquantes : message propre, pas de trace
        return None, f"{type(e).__name__} : {e}"


def table_couts(couts: tuple | None) -> pd.DataFrame | None:
    """Coûts unitaires saisis dans l'app ((type, investissement, fonctionnement annuel), ...) ; sinon ceux des données."""
    if not couts:
        return data["gold_costs"]
    return pd.DataFrame(couts, columns=["type", "capex_eur", "opex_annuel_eur"])


def avec_couts(inputs, couts: tuple | None):
    """Entrées de l'optimiseur avec les coûts saisis, sans modifier celles en cache."""
    if not couts:
        return inputs
    df = table_couts(couts)
    if isinstance(inputs, dict):  # faux optimiseur des tests
        return {**inputs, "gold_costs": df}
    return dataclasses.replace(inputs, costs=df)


@st.cache_data(show_spinner="Calcul du plan…")
def run_plan(directory: str, budget: int, equity: float, types: tuple, horizon: int, min_p: float,
             couts: tuple | None = None) -> pd.DataFrame:
    mod, inputs = _optim_inputs(directory)
    return mod.greedy_plan(avec_couts(inputs, couts), budget, equity_weight=equity, types=types, horizon_years=horizon,
                           min_p_usage=min_p)


@st.cache_data(show_spinner=False)
def plan_kpis(directory: str, budget: int, equity: float, types: tuple, horizon: int, min_p: float,
              couts: tuple | None = None) -> dict:
    """Indicateurs du plan (mêmes paramètres que run_plan), QPV par type compris."""
    plan = run_plan(directory, budget, equity, types, horizon, min_p, couts)
    d = load_data(directory)
    kp = ind.plan_indicators(plan, d["gold_candidates"], d["gold_carreau_access"], d["silver_carreaux"], d["gold_carreau_qpv"])
    kp["qpv_par_type"] = ind.qpv_par_type(plan, d["gold_candidates"], d["gold_carreau_access"], d["gold_carreau_qpv"])
    return kp


@st.cache_data(show_spinner=False)
def plan_zones(directory: str, budget: int, equity: float, types: tuple, horizon: int, min_p: float,
               couts: tuple | None = None) -> list[dict]:
    plan = run_plan(directory, budget, equity, types, horizon, min_p, couts)
    d = load_data(directory)
    return ind.walk_zones(plan, d["gold_candidates"], d["gold_carreau_access"])


@st.cache_data(show_spinner=False)
def qpv_reperes(directory: str) -> dict:
    d = load_data(directory)
    return ind.qpv_reperes(d["gold_carreau_access"], d["gold_carreau_qpv"])


# --------------------------------------------------------------------------- cartes

FOND_CLAIR = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
RUES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ATTR_CLAIR = "Esri, HERE, Garmin, © contributeurs OpenStreetMap"
ATTR_RUES = "© contributeurs OpenStreetMap"
ATTR_SATELLITE = "Esri, Maxar, Earthstar Geographics"


def base_map(df: pd.DataFrame, zoom: int = 12) -> tuple[folium.Map, list[float]]:
    """Carte Leaflet : fond clair par défaut, plan des rues OpenStreetMap ou vue satellite (en haut à droite)."""
    centre = [(df["lat"].min() + df["lat"].max()) / 2, (df["lon"].min() + df["lon"].max()) / 2]
    m = folium.Map(location=centre, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(FOND_CLAIR, name="Fond clair", attr=ATTR_CLAIR, max_zoom=16).add_to(m)
    folium.TileLayer(RUES, name="Plan des rues", attr=ATTR_RUES, show=False).add_to(m)
    folium.TileLayer(SATELLITE, name="Vue satellite", attr=ATTR_SATELLITE, show=False).add_to(m)
    return m, centre


def show_map(m: folium.Map, centre: list[float] | None = None, zoom: int | None = 12, height: int = 560,
             key: str = "carte") -> None:
    """Sans centre : la carte garde son propre cadrage (fit_bounds). Bouton plein écran et adresse recherchée sur
    toutes les cartes."""
    adr = st.session_state.get("adresse")
    if adr and not adr.get("erreur"):
        folium.Marker([adr["lat"], adr["lon"]], tooltip=adr["label"], z_index_offset=100_000,
                      icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
    Fullscreen(position="topleft", title="Plein écran", title_cancel="Quitter le plein écran").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=height, use_container_width=True, returned_objects=[], key=key,
              center=centre, zoom=zoom if centre else None)


class Fond(MacroElement):
    """Carte de fond plein écran : hauteur de la fenêtre, zoom et couches en bas à droite à gauche du panneau,
    cadrage sur l'emprise (ou sur un point) en laissant de la place à l'en-tête et au panneau flottants."""

    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
          .folium-map, .leaflet-container {height: 100vh !important;}
          @media (min-width: 900px) {
            .leaflet-right {right: calc({{ this.panneau }} + 24px);}
            .leaflet-bottom.leaflet-left {left: 8px;}
          }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function () {
            var carte = {{ this._parent.get_name() }};
            carte.zoomControl.setPosition("bottomright");
            var cadrer = function () {
                carte.invalidateSize();
                var large = window.innerWidth >= 900;
                var droite = large ? Math.min(Math.max(320, 0.28 * window.innerWidth), 420) + 40 : 20;
                var opts = {paddingTopLeft: [large ? 350 : 20, large ? 80 : 20], paddingBottomRight: [droite, 30],
                            maxZoom: 15{{ this.option }}};
                {% if this.point %}
                carte.setView({{ this.point }}, {{ this.zoom }}, {animate: false});
                {% else %}
                carte.fitBounds({{ this.bounds }}, opts);
                {% endif %}
            };
            cadrer();
            [300, 1000].forEach(function (ms) { setTimeout(cadrer, ms); });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, bounds: list[list[float]], point: list[float] | None = None, zoom: int = 15,
                 animer: bool = True):
        super().__init__()
        self._name = "Fond"
        self.bounds = json.dumps(bounds)
        self.point = json.dumps(point) if point else None
        self.zoom = zoom
        self.option = "" if animer else ", animate: false"
        self.panneau = PANNEAU


def emprise(df: pd.DataFrame) -> list[list[float]]:
    return [[float(df["lat"].min()), float(df["lon"].min())], [float(df["lat"].max()), float(df["lon"].max())]]


def carte_fond(m: folium.Map, bounds: list[list[float]], point: list[float] | None = None, zoom: int = 15,
               animer: bool = True, key: str = "carte_fond") -> None:
    """Affiche `m` en fond plein écran (conteneur `fond`), avec l'adresse recherchée et le choix du fond de carte."""
    adr = st.session_state.get("adresse")
    if adr and not adr.get("erreur"):
        folium.Marker([adr["lat"], adr["lon"]], tooltip=adr["label"], z_index_offset=100_000,
                      icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
    folium.LayerControl(collapsed=True, position="bottomright").add_to(m)
    Fond(bounds, point, zoom, animer).add_to(m)
    with st.container(key="fond"):
        st_folium(m, height=760, use_container_width=True, returned_objects=[], key=key)


def chiffre_hero(valeur: str, libelle: str) -> None:
    """Le chiffre clé de la rubrique, en très grand en tête du panneau."""
    st.markdown(f'<div class="hero">{valeur}</div><div class="hero-lib">{libelle}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Recherche de l'adresse…", ttl=3600)
def chercher_adresse(q: str, lat: float, lon: float) -> dict | None:
    """Géocodage par la Base Adresse Nationale (api-adresse.data.gouv.fr), au plus près du territoire affiché."""
    url = "https://api-adresse.data.gouv.fr/search/?" + urlencode(
        {"q": q, "limit": 1, "lat": f"{lat:.4f}", "lon": f"{lon:.4f}"})
    try:
        with urlopen(url, timeout=6) as r:
            features = json.load(r).get("features") or []
    except Exception:  # service injoignable : message, pas de trace
        return {"erreur": True}
    if not features:
        return None
    lon_a, lat_a = features[0]["geometry"]["coordinates"]
    return {"label": features[0]["properties"].get("label", q), "lat": float(lat_a), "lon": float(lon_a)}


def barre_adresse() -> None:
    q = st.text_input("Rechercher une adresse", key="adresse_q", placeholder="🔍 Rechercher une adresse",
                      label_visibility="collapsed",
                      help="L'adresse est localisée par la Base Adresse Nationale (api-adresse.data.gouv.fr), "
                           "puis affichée sur les cartes. Rien n'est conservé.")
    q = (q or "").strip()
    if not q:
        st.session_state["adresse"] = None
        return
    st.session_state["adresse"] = chercher_adresse(q, float(access["lat"].mean()), float(access["lon"].mean()))


def encart_adresse() -> pd.Series | None:
    """Résultat de la recherche d'adresse : options à 5 min de son carreau. Renvoie le carreau (ou None)."""
    adr = st.session_state.get("adresse")
    if not st.session_state.get("adresse_q", "").strip():
        return None
    if adr is None:
        st.warning("Adresse introuvable dans la Base Adresse Nationale.")
        return None
    if adr.get("erreur"):
        st.warning("Recherche d'adresse indisponible (service api-adresse.data.gouv.fr injoignable).")
        return None
    c = ind.carreau_proche(access, adr["lat"], adr["lon"])
    if c is None:
        st.info(f"📍 **{adr['label']}** : hors des carreaux habités de {VILLE['titre']} (zone sans habitant recensé "
                "ou autre territoire : changez de territoire en haut à droite).")
        return None
    classe = int(c["classe"]) if pd.notna(c.get("classe")) else None
    st.markdown(f'<div class="reco">📍 <b>{adr["label"]}</b> : à 5 minutes à pied, {ind.options_carreau(c)}'
                + (f" · <b>{ind.CLASS_LABELS[classe]}</b>" if classe else "") + "</div>", unsafe_allow_html=True)
    return c


def rgba(color: list[int]) -> tuple[str, float]:
    """[r, g, b, a] (0-255) -> ('#rrggbb', opacité 0-1)."""
    return "#{:02x}{:02x}{:02x}".format(*color[:3]), (color[3] / 255 if len(color) > 3 else 1.0)


def forme_svg(t: str, size: float, texte: str = "", contour: str | None = None) -> str:
    """Pictogramme d'un type d'équipement : couleur et forme propres (Citiz carré, bus rond, Vélhop losange,
    tram triangle), numéro facultatif au centre, contour coloré facultatif."""
    s, c = float(size), ind.TYPE_COLORS.get(t, "#555555")
    sw = 3.5 if contour else (1.5 if s < 14 else 2)
    m, h = sw / 2 + 0.5, s / 2
    style = f'fill="{c}" stroke="{contour or "#ffffff"}" stroke-width="{sw}"'
    forme = ind.TYPE_SHAPES.get(t, "rond")
    if forme == "carré":
        el = f'<rect x="{m}" y="{m}" width="{s - 2 * m}" height="{s - 2 * m}" rx="{s * 0.12:.1f}" {style}/>'
    elif forme == "losange":
        el = f'<polygon points="{h},{m} {s - m},{h} {h},{s - m} {m},{h}" {style}/>'
    elif forme == "triangle":
        el = f'<polygon points="{h},{m} {s - m},{s - m} {m},{s - m}" {style}/>'
    else:
        el = f'<circle cx="{h}" cy="{h}" r="{h - m}" {style}/>'
    txt = (f'<text x="{h}" y="{h + 0.5}" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" '
           f'font-weight="700" font-size="{s * 0.42:.0f}" fill="#ffffff">{texte}</text>') if texte else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s:.0f}" height="{s:.0f}" viewBox="0 0 {s} {s}" '
            f'style="display:block;overflow:visible;filter:drop-shadow(0 1px 2px rgba(0,0,0,.45))">{el}{txt}</svg>')


class Legende(MacroElement):
    """Légende dessinée dans la carte (contrôle Leaflet en bas à gauche)."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var legende = L.control({position: "bottomleft"});
            legende.onAdd = function () {
                var div = L.DomUtil.create("div", "legende-carte");
                div.innerHTML = {{ this.contenu }};
                return div;
            };
            legende.addTo({{ this._parent.get_name() }});
        })();
        {% endmacro %}
        """
    )

    def __init__(self, titre: str, items: list[tuple[str, str]]):
        """items : (pictogramme HTML, libellé)."""
        super().__init__()
        self._name = "Legende"
        lignes = "".join(f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                         f'<span style="flex:none;width:16px;display:flex;justify-content:center">{pic}</span>{lib}</div>'
                         for pic, lib in items)
        html = (f'<div style="background:rgba(255,255,255,.94);color:{TEXTE};font:12px/1.35 sans-serif;padding:6px 9px;'
                f'border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);max-width:250px">'
                f'<div style="font-weight:700;margin-bottom:3px">{titre}</div>{lignes}</div>')
        self.contenu = json.dumps(html)


class Cadrage(MacroElement):
    """Cadre la carte sur une emprise une fois l'iframe dimensionnée (fit_bounds seul s'exécute trop tôt
    dans streamlit-folium et garde un zoom trop large)."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var carte = {{ this._parent.get_name() }};
            var cadrer = function () {
                carte.invalidateSize();
                carte.fitBounds({{ this.bounds }}, {padding: [25, 25], maxZoom: 15{{ this.option }}});
            };
            cadrer();
            [300, 1000].forEach(function (ms) { setTimeout(cadrer, ms); });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, bounds: list[list[float]], animer: bool = True):
        super().__init__()
        self._name = "Cadrage"
        self.bounds = json.dumps(bounds)
        self.option = "" if animer else ", animate: false"  # cartes de France : contours lourds, pas d'animation


def carre(couleur: str) -> str:
    return (f'<span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:{couleur};'
            'border:1px solid rgba(0,0,0,.2)"></span>')


def legende_types(types, extra: list[tuple[str, str]] | None = None, titre: str = "Équipements") -> Legende:
    items = [(forme_svg(t, 14), f"{ind.TYPE_LABELS[t]} ({ind.TYPE_SHAPES[t]})") for t in types]
    return Legende(titre, items + (extra or []))


@st.cache_data(show_spinner=False)
def demographie(dep: str, horizon: int) -> dict:
    """Projections INSEE Omphale du département (optim/projections.py) ; vide si indisponibles."""
    try:
        from optim.projections import contexte
        return contexte(dep, horizon)
    except Exception:  # fichier de référence absent : l'application reste utilisable
        return {}


def departement(carreaux: pd.DataFrame | None) -> str:
    if carreaux is None or "commune_code" not in carreaux:
        return "67"
    return str(carreaux["commune_code"].astype(str).str[:2].mode().iat[0])


def note(html: str) -> None:
    st.markdown(f'<p class="note">{html}</p>', unsafe_allow_html=True)


def pied_panneau() -> None:
    """Bas du panneau : méthode, limites et sources."""
    with st.expander("Méthode et limites"):
        st.markdown(
            f"""
- **5 minutes = 400 m sur le réseau piéton réel** (OpenStreetMap, 4,8 km/h), pas à vol d'oiseau.
- **Accessibilité** : options à 5 min par carreau INSEE de 200 m ; les arrêts sont pondérés par leur fréquence
  (`min(passages / 200, 3)`), chaque station Citiz ou {VILLE['velo_nom']} compte 1.
- **Habitants nouvellement desservis** : seuls les habitants qui n'ont pas encore d'option du même type à 5 min
  comptent (bus : ni bus ni tram). On ne renforce pas les zones déjà couvertes. Chiffre affiché : personnes
  distinctes (un habitant qui gagne le bus et Citiz compte une fois), dénominateur de toutes les parts.
- **Modèle prédictif** (Citiz) : AUC 0,87 sur un réseau Citiz jamais vu à l'entraînement. Son score mesure la
  ressemblance aux stations Citiz actuelles, pas l'usage réel ni une probabilité de réservation : d'où le curseur équité.
- **CO₂** : 2,6 t CO₂e/an par voiture Citiz ajoutée (borne basse ADEME 2022), arrondi ; rien pour bus et vélo.
- **Vie privée** : carreaux de moins de 11 ménages grisés, sans valeur ; indicateurs sociaux seulement agrégés.

Pour aller plus loin : [limites]({REPO}/docs/LIMITES.md) · [éthique]({REPO}/docs/ETHIQUE.md) ·
[impact carbone]({REPO}/docs/IMPACT.md) · [contrat de données]({REPO}/docs/DATA_CONTRACT.md) ·
[France entière]({REPO}/docs/FRANCE.md)
"""
        )
    gtfs = "GTFS des réseaux (transport.data.gouv.fr)" if EST_FRANCE else "GTFS du réseau urbain"
    calculs = ("Python local, même code que les notebooks Fabric (docs/FRANCE.md)" if EST_FRANCE
               else "Microsoft Fabric, région France Central")
    st.markdown(
        f'<div class="footer">Sources : INSEE Filosofi 2021 (carreaux 200 m) · {gtfs} (arrêts et fréquences) · '
        f"flux GBFS Citiz et {VILLE['velo_nom']} · OpenStreetMap (réseau piéton) · QPV 2024-2030 (ANCT) · ADEME (CO₂). "
        f"Calculs : {calculs}. Fonds de carte : Esri, HERE, Garmin, © contributeurs "
        "OpenStreetMap (fond clair) · © contributeurs OpenStreetMap (plan des rues) · Esri, Maxar, Earthstar Geographics "
        "(vue satellite).</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- en-tête

data = {} if EST_FRANCE else load_data(data_dir)
access = data.get("gold_carreau_access")


def changer_territoire(cle: str, rubrique: str | None = None) -> None:
    """Rappel des boutons « Ouvrir » : change le territoire (et la rubrique) avant la prochaine exécution."""
    st.session_state["territoire"] = cle
    if rubrique:
        st.session_state["rubrique"] = rubrique


with st.container(key="entete"):
    st.markdown(f'<div class="marque"><img src="data:image/png;base64,{LOGO}" alt=""><div><b>{NOM}</b>'
                "<span>Où ouvrir la prochaine option de mobilité</span></div></div>", unsafe_allow_html=True)
    if EST_FRANCE:
        st.title("France métropolitaine : où investir en premier pour la mobilité à 5 minutes ?")
    elif VILLE["est_epci"]:
        st.title(f"{VILLE['titre']} : où ouvrir les prochaines options de mobilité ?")
    else:
        st.title(f"Où ouvrir les prochaines options de mobilité à {VILLE['titre']} ?")
    if len(TERRITOIRES) > 1:
        st.selectbox(
            "Territoire", list(TERRITOIRES), format_func=lambda k: TERRITOIRES[k]["label"], key="territoire",
            placeholder="Nom ou code de l'intercommunalité…", label_visibility="collapsed",
            help="Tapez un nom (« Lyon ») ou un code SIREN d'EPCI (« 200046977 »). Même outil, même méthode partout : "
                 "5 minutes = 400 m sur le réseau piéton réel. Le lien de la page garde le territoire choisi.")
    if territoire_inconnu:
        st.warning(f"Territoire « {territoire_inconnu} » introuvable : {TERRITOIRES[TERRITOIRE]['label']} est affiché.")
    if EST_FRANCE:
        st.caption("Toutes les intercommunalités de France métropolitaine · 5 min à pied = 400 m sur le réseau piéton réel")
    elif access is not None:
        barre_adresse()
if st.query_params.get("territoire") != TERRITOIRE:
    st.query_params["territoire"] = TERRITOIRE  # lien partageable

if not EST_FRANCE and access is None:
    st.info(
        "Les données ne sont pas encore disponibles sur ce poste. Exportez le snapshot depuis le Lakehouse Fabric :\n\n"
        "```\npython app/export_snapshot.py\n```\n\n"
        f"Dossier attendu : `{data_dir}`. Voir `app/README.md`."
    )
    st.stop()

RUBRIQUES = ["① Accès à 5 minutes", "② Habitants desservis", "③ Où investir ?"]
# Territoire sans Citiz ou sans vélo en libre-service aujourd'hui (EPCI de la France entière) : messages adaptés.
SANS_CITIZ = not EST_FRANCE and float(access["n_citiz"].fillna(0).sum()) == 0
SANS_VELO = not EST_FRANCE and (not VILLE["a_velo"] or float(access["n_velhop"].fillna(0).sum()) == 0)


# --------------------------------------------------------------------------- onglet 1 : accessibilité


@st.fragment
def rubrique_acces() -> None:
    k = ind.access_kpis(access)
    with st.container(key="panneau"):
        chiffre_hero(ind.fmt_pct(k["part_aucune"]),
                     f"des habitants n'ont <b>aucune option</b> à 5 minutes à pied "
                     f"({ind.fmt_int(k['population'] * k['part_aucune'])} personnes)")
        c1, c2 = st.columns(2)
        c1.metric("Habitants (carreaux INSEE)", ind.fmt_int(k["population"]), border=True)
        c2.metric("Au moins une option à 5 min", ind.fmt_pct(k["part_au_moins_une"]), border=True)
        c1.metric("Aucune option à 5 min", ind.fmt_pct(k["part_aucune"]), border=True)
        c2.metric("Une station Citiz à 5 min", ind.fmt_pct(k["part_citiz"]), border=True)
        absents = [nom for nom, sans in (("Citiz", SANS_CITIZ), ("vélo en libre-service", SANS_VELO)) if sans]
        if absents:
            note(f"Pas encore de {' ni de '.join(absents)} sur ce territoire : les options comptées ici sont "
                 "les autres modes. La rubrique ③ montre où les premières stations desserviraient le plus d'habitants.")
        modes = st.pills(
            "Modes pris en compte", options=list(ind.MODES), default=list(ind.MODES), selection_mode="multi",
            format_func=lambda m: ind.MODE_LABELS[m], key="modes",
        ) or []
        adresse = encart_adresse()
        if not modes:
            st.info("Cochez au moins un mode de transport pour afficher la carte d'accessibilité.")
        elif set(modes) != set(ind.MODES):
            part = (access["pop"] * ind.has_option(access, modes)).sum() / access["pop"].sum()
            note(f"Avec les modes choisis : <b>{ind.fmt_pct(part)}</b> des habitants ont au moins une option à 5 min. "
                 "Classes 2 à 5 recalculées (quartiles du nombre d'options des modes choisis).")
        note("Les couleurs décrivent une <b>quantité d'options de transport</b> à 5 minutes à pied "
             "(arrêts pondérés par leur fréquence), pas un jugement sur un quartier. Les carreaux gris comptent moins de "
             "11 ménages : leurs valeurs ne sont pas affichées (secret statistique INSEE), mais restent dans les totaux.")
        demo = demographie(departement(data["silver_carreaux"]), 5)
        if demo:
            note(f'<b>Demain</b> (projections INSEE Omphale 2022, {demo["departement"]}) : '
                 f'{"+" if demo["evolution_2040"] >= 0 else ""}{ind.fmt_pct(demo["evolution_2040"])} d\'habitants d\'ici 2040, '
                 f'et la part des 75 ans et plus passerait de {ind.fmt_pct(demo["part_75p_2018"])} à '
                 f'{ind.fmt_pct(demo["part_75p_2070"])} en 2070. Pour une personne âgée, 5 minutes à pied couvrent moins '
                 "que 400 m : l'accès réel sera plus faible que sur cette carte.")
        pied_panneau()

    m, _ = base_map(access)
    if modes:
        folium.GeoJson(
            carreaux_geojson(data_dir, tuple(sorted(modes))), name="Offre à 5 min",
            style_function=lambda f: {"fillColor": f["properties"]["color"], "fillOpacity": f["properties"]["opacity"],
                                      "color": "#ffffff", "weight": 0.3},
            tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False),
        ).add_to(m)
        Legende("Options à 5 min à pied",
                [(carre(ind.CLASS_COLORS[c]), ind.CLASS_LABELS[c]) for c in range(1, 6)]
                + [(carre(ind.PROTECTED_COLOR), ind.PROTECTED_LABEL)]).add_to(m)
    point = [st.session_state["adresse"]["lat"], st.session_state["adresse"]["lon"]] if adresse is not None else None
    carte_fond(m, emprise(access), point, key="carte_acces")


# --------------------------------------------------------------------------- onglet 2 : population desservie


@st.fragment
def rubrique_stations() -> None:
    stations = data["gold_station_population"]
    if stations is None:
        st.info("La table `gold_station_population` n'est pas encore dans le snapshot.")
        return
    types_presents = [t for t in ("tram", "bus", "citiz", "velhop") if t in set(stations["target_type"])]
    with st.container(key="panneau"):
        sel = st.pills("Équipements affichés", options=types_presents, default=types_presents, selection_mode="multi",
                       format_func=lambda t: ind.TYPE_LABELS.get(t, t), key="types_st") or []
        s = stations[stations["target_type"].isin(sel)].copy()
        s["taille"] = (s["pop_desservie"].clip(lower=0) ** 0.5 / 3).clip(6, 28)  # pixels
        s["tooltip"] = [f"<b>{n}</b><br/>{ind.TYPE_LABELS.get(t, t)}<br/>{ind.fmt_int(p)} habitants à 5 min à pied"
                        for n, t, p in zip(s["name"], s["target_type"], s["pop_desservie"])]
        top = ind.top_stations(s, n=10)
        if len(top):
            chiffre_hero(ind.fmt_int(top["pop_desservie"].iloc[0]),
                         f"habitants à 5 min de <b>{top['name'].iloc[0]}</b>, l'équipement qui en dessert le plus")
        encart_adresse()
        if s.empty:
            st.info("Choisissez au moins un type d'équipement.")
        st.subheader("Les 10 qui desservent le plus d'habitants")
        st.dataframe(
            pd.DataFrame({"Nom": top["name"], "Type": top["target_type"].map(ind.TYPE_LABELS),
                          "Habitants à 5 min": top["pop_desservie"].round().astype(int)}),
            hide_index=True, width="stretch",
            column_config={"Habitants à 5 min": st.column_config.ProgressColumn(
                format="%d", min_value=0, max_value=int(max(top["pop_desservie"].max(), 1)) if len(top) else 1)},
        )
        note(f"{ind.fmt_int(len(s))} arrêts et stations affichés. Taille du pictogramme = habitants à 5 minutes à pied. "
             "Un habitant proche de plusieurs arrêts compte pour chacun : on ne peut pas additionner ces chiffres.")
        pied_panneau()

    m, _ = base_map(access)
    # les plus petits par-dessus, pour qu'ils restent visibles
    for i, r in enumerate(s.sort_values("pop_desservie", ascending=False).itertuples()):
        sz = float(r.taille)
        folium.Marker([r.lat, r.lon], tooltip=r.tooltip, z_index_offset=i,
                      icon=folium.DivIcon(html=forme_svg(r.target_type, sz), icon_size=(sz, sz),
                                          icon_anchor=(sz / 2, sz / 2))).add_to(m)
    if sel:
        legende_types(sel, titre="Taille = habitants à 5 min").add_to(m)
    carte_fond(m, emprise(s if not s.empty else access), key="carte_stations")


# --------------------------------------------------------------------------- onglet 3 : plan d'investissement


def plan_map(plan: pd.DataFrame, zones: list[dict], height: int = 520, key: str = "plan",
             differents: set | None = None, fond: bool = False) -> None:
    """Carte du plan : emplacements numérotés par priorité (n° 1 au-dessus), forme et couleur par type, et en
    pointillés la zone réellement atteignable à 5 minutes à pied (réseau piéton, pas un cercle). Cadrage
    automatique sur le plan. `differents` : emplacements absents de l'autre plan (contour doré). `fond` : carte
    plein écran derrière les panneaux, sinon carte dans le panneau."""
    differents = differents or set()
    m, _ = base_map(plan)
    lats, lons = list(plan["lat"]), list(plan["lon"])
    for z in zones:
        color = ind.TYPE_COLORS.get(z["type"], "#555555")
        for ring in z["rings"]:
            folium.Polygon(ring, color=color, weight=2.5, dash_array="7 6", fill=True, fill_opacity=0.10,
                           tooltip=f"Priorité n° {z['rank']} : zone à 5 min à pied").add_to(m)
            lats += [p[0] for p in ring]
            lons += [p[1] for p in ring]
    n = len(plan)
    for r in plan.itertuples():
        diff = r.candidate_id in differents
        size = 34 if diff else 30
        folium.Marker(
            [r.lat, r.lon], z_index_offset=1000 * (n - int(r.rank) + 1),
            icon=folium.DivIcon(html=forme_svg(r.type, size, str(int(r.rank)), CONTOUR_DIFF if diff else None),
                                icon_size=(size, size), icon_anchor=(size / 2, size / 2)),
            tooltip=f"Priorité n° {int(r.rank)} · {ind.TYPE_LABELS.get(r.type, r.type)} · "
                    f"{ind.fmt_int(r.new_pop)} habitants nouvellement desservis · {ind.fmt_eur_plein(r.cost_eur)}"
                    + (" · absent de l'autre plan" if diff else ""),
        ).add_to(m)
    extra = [('<span style="display:inline-block;width:14px;border-top:2px dashed #555"></span>', "Zone à 5 min à pied")]
    if differents:
        extra.append((f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                      f'border:3px solid {CONTOUR_DIFF}"></span>', "Absent de l'autre plan"))
    legende_types([t for t in TYPES_PLAN if t in set(plan["type"])], extra, titre="Numéro = priorité").add_to(m)
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    if fond:
        carte_fond(m, bounds, key=key)
        return
    Cadrage(bounds).add_to(m)
    show_map(m, None, height=height, key=key)


def ecart_budget(params: dict, habitants: float) -> str | None:
    """Écart avec le calcul précédent quand seul le budget a changé (« +1 234 habitants vs 400 k€ »)."""
    etat = st.session_state.setdefault("dernier_calcul", {})
    if etat.get("params") != params:
        etat["prec"] = (etat.get("params"), etat.get("habitants"))
        etat["params"], etat["habitants"] = params, habitants
    p_prec, h_prec = etat.get("prec", (None, None))
    if not p_prec or h_prec is None:
        return None
    if {k: v for k, v in p_prec.items() if k != "budget"} != {k: v for k, v in params.items() if k != "budget"}:
        return None
    d = round(habitants) - round(h_prec)
    return f"{'+' if d >= 0 else '-'}{ind.fmt_int(abs(d))} habitants vs {ind.fmt_eur(p_prec['budget'])}"


def kpis_sociaux(kp: dict) -> None:
    rep = qpv_reperes(data_dir)
    s1, s2, s3 = st.columns(3)
    s1.metric("Part en quartier prioritaire (QPV)", ind.fmt_pct(kp["part_qpv"]), border=True,
              help="Part des habitants nouvellement desservis qui vivent dans un QPV 2024-2030."
                   + ("" if data["gold_carreau_qpv"] is not None else " Table gold_carreau_qpv pas encore disponible."))
    if rep["population"] is not None:
        s1.caption(f"Repères : {ind.fmt_pct(rep['population'])} de toute la population vit en QPV"
                   + (f" ; {ind.fmt_pct(rep['citiz'])} des habitants qui ont aujourd'hui Citiz à 5 min"
                      if rep["citiz"] is not None else "") + ".")
    s2.metric("Dans les zones sans option ou à offre limitée", ind.fmt_pct(kp["part_classes_1_2"]), border=True,
              help="Part des habitants nouvellement desservis qui vivent dans un carreau de classe 1 ou 2 "
                   "de la carte d'accessibilité.")
    nv = kp["niveau_vie_gagnes"]
    s3.metric("Niveau de vie moyen des habitants nouvellement desservis",
              f"{ind.fmt_int(round(nv, -2))}\u00a0€/an" if nv else "—", border=True)
    reperes = [("habitants nouvellement desservis", nv), ("ensemble de la population", kp["niveau_vie_population"]),
               ("zones avec Citiz aujourd'hui", kp["niveau_vie_citiz"])]
    reperes = [f"{nom} {ind.fmt_int(round(v, -2))}\u00a0€/an" for nom, v in reperes if v]
    if len(reperes) > 1:
        s3.caption("Niveau de vie moyen (agrégé) : " + " · ".join(reperes) + ". Jamais calculé au carreau isolé.")

    par_type = kp.get("qpv_par_type") or {}
    if par_type:
        with st.expander("Part en QPV par type d'équipement du plan"):
            for t, part in par_type.items():
                nom = ind.TYPE_PLURIELS[t]
                st.markdown(f"- **{nom[0].upper() + nom[1:]}** du plan : "
                            + (f"{ind.fmt_pct(part)} des habitants nouvellement desservis vivent en QPV"
                               if part is not None else "— (trop peu de carreaux pour être affiché)"))
            st.caption("Habitants nouvellement desservis par chaque type ; un habitant gagné à la fois par le bus et "
                       "par Citiz compte dans les deux lignes.")


def fond_vide() -> None:
    """Pas de plan à montrer : la carte de fond garde le territoire."""
    m, _ = base_map(access)
    carte_fond(m, emprise(access), key="plan")


def rendu_plan(mod, inputs, budget: int, equity: float, types: tuple, horizon: int, min_p: float,
               couts: tuple | None = None, haut=None, bas=None) -> None:
    """`haut` : conteneur en tête du panneau (au-dessus des réglages) pour le chiffre clé et les indicateurs ;
    `bas` : conteneur du panneau sous les réglages (indicateurs sociaux, tableau). La carte du plan va en fond."""
    haut = haut or st.container()
    args = (data_dir, budget, equity, types, horizon, min_p, couts)
    try:
        plan = run_plan(*args)
        kp = plan_kpis(*args)
    except Exception as e:
        haut.warning(f"L'optimiseur n'a pas pu calculer ce plan ({type(e).__name__}). Essayez d'autres paramètres.")
        fond_vide()
        return
    if plan.empty:
        haut.info(ind.message_plan_vide(ind.couts_unitaires(table_couts(couts), horizon, types), budget, horizon, BUDGETS))
        fond_vide()
        return

    with haut:
        chiffre_hero(ind.fmt_int(kp["habitants_uniques"]), "habitants <b>nouvellement desservis</b> à 5 minutes à pied")
        st.markdown(f'<div class="reco">{ind.recommandation(kp, budget, horizon)}</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m3, m4 = m1, m2
    m1.metric("Habitants nouvellement desservis", ind.fmt_int(kp["habitants_uniques"]), border=True,
              delta=ecart_budget({"budget": budget, "equite": equity, "types": types, "horizon": horizon, "min_p": min_p,
                                  "couts": couts},
                                 kp["habitants_uniques"]),
              help="Personnes distinctes qui gagnent au moins une option à 5 min à pied : un habitant qui gagne à la "
                   f"fois le bus et Citiz compte une fois (en comptant chaque type : {ind.fmt_int(kp['habitants_gagnes'])}).")
    m2.metric("Coût du plan", ind.fmt_eur_plein(kp["cout_eur"]), border=True,
              help=f"Investissement + {horizon} ans de fonctionnement, sur un budget de {ind.fmt_eur_plein(budget)}.")
    m3.metric("Coût par habitant desservi", f"{ind.fmt_int(kp['cout_par_habitant'])}\u00a0€" if kp["cout_par_habitant"] else "—",
              border=True, help="Coût du plan / habitants nouvellement desservis (personnes distinctes).")
    m4.metric("CO₂ évité (ordre de grandeur)", f"≈ {kp['co2_t_an']}\u00a0t/an" if kp["voitures_citiz"] else "—", border=True,
              help=f"{kp['voitures_citiz']} voiture(s) Citiz ajoutée(s), hypothèse prudente d'une voiture par station ; "
                   "2,6 t CO₂e par an et par voiture (borne basse ADEME 2022, docs/IMPACT.md). "
                   "Aucune estimation pour les arrêts de bus et stations vélo.")
    with haut:
        demo = demographie(departement(data["silver_carreaux"]), horizon)
        if demo:
            note(f'À l\'horizon du plan ({demo["annee_horizon"]}), avec l\'évolution projetée de la population '
                 f'({demo["departement"]} : {"+" if demo["evolution_horizon"] >= 0 else ""}{ind.fmt_pct(demo["evolution_horizon"])}, INSEE Omphale) : '
                 f'<b>≈ {ind.fmt_int(round(kp["habitants_uniques"] * (1 + demo["evolution_horizon"]), -2))}</b> '
                 "habitants nouvellement desservis. Projection départementale : elle ne change pas l'ordre des emplacements.")
        adresse_et_plan(plan)
    plan_map(plan, plan_zones(*args), key="plan", fond=True)

    with bas or st.container():
        note("Pointillés : zone réellement atteignable en 5 minutes à pied depuis chaque emplacement (réseau piéton "
             "OpenStreetMap), pas un cercle. Fond : clair, plan des rues ou vue satellite (bouton en bas à droite).")
        kpis_sociaux(kp)
        st.subheader("Priorités d'investissement")
        table = ind.plan_table(plan, data["gold_predictions"])
        st.dataframe(ind.plan_table_fr(table), hide_index=True, width="stretch", height=min(38 * (len(table) + 1), 420),
                     column_config={ind.P_USAGE_LABEL: st.column_config.TextColumn(help=ind.P_USAGE_HELP)})
        note("Chaque emplacement est une proposition à vérifier sur le terrain (foncier, voirie, accord des riverains).")
        st.download_button("Télécharger le plan (CSV)", table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"plan_{budget}eur_equite{equity:.1f}.csv", mime="text/csv")
        try:
            stats = mod.summarize(plan, inputs)
            with st.expander("Détail de l'optimiseur"):
                st.json(stats, expanded=False)
        except Exception:
            pass


def adresse_et_plan(plan: pd.DataFrame) -> None:
    """Ce que le plan change pour l'adresse recherchée."""
    adr = st.session_state.get("adresse")
    if not adr or adr.get("erreur"):
        return
    c = ind.carreau_proche(access, adr["lat"], adr["lon"])
    if c is None:
        return
    g = ind.gained_carreaux(plan, data["gold_candidates"], access)
    g = g[g["carreau_id"] == c["carreau_id"]].sort_values("rank")
    if g.empty:
        note(f"📍 {adr['label']} : ce plan ne lui ajoute pas d'option (aujourd'hui : {ind.options_carreau(c)}).")
    else:
        gains = " ; ".join(f"{ind.TYPE_SINGULIERS[t]} (emplacement n° {int(r)})" for t, r in zip(g["type"], g["rank"]))
        note(f"📍 <b>{adr['label']}</b> : ce plan lui apporte à 5 minutes à pied : <b>{gains}</b> "
             f"(aujourd'hui : {ind.options_carreau(c)}).")


def liste_emplacements(plan: pd.DataFrame, ids: set) -> str:
    lignes = [f"- n° {int(r.rank)} · {ind.TYPE_LABELS.get(r.type, r.type)} · {ind.fmt_int(r.new_pop)} habitants"
              for r in plan[plan["candidate_id"].isin(ids)].sort_values("rank").itertuples()]
    return "\n".join(lignes) if lignes else "Aucun : mêmes emplacements."


def rendu_comparaison(budget: int, equity: float, types: tuple, horizon: int, min_p: float,
                      couts: tuple | None = None, haut=None, bas=None) -> None:
    """Deux plans au même budget. Panneau : conclusion et indicateurs des deux plans ; carte de fond : le plan choisi
    (contour doré = emplacement absent de l'autre plan)."""
    haut, bas = haut or st.container(), bas or st.container()
    eq = equity if equity > 0 else 1.0
    a_eff, a_eq = (data_dir, budget, 0.0, types, horizon, min_p, couts), (data_dir, budget, eq, types, horizon, min_p, couts)
    try:
        plan_eff, plan_eq = run_plan(*a_eff), run_plan(*a_eq)
        ke, kq = plan_kpis(*a_eff), plan_kpis(*a_eq)
        z_eff, z_eq = plan_zones(*a_eff), plan_zones(*a_eq)
        conclusion = ind.conclusion_equite(ke, kq)
    except Exception as e:
        haut.warning(f"La comparaison n'a pas pu être calculée ({type(e).__name__}). "
                     "Essayez un autre budget ou d'autres équipements.")
        fond_vide()
        return
    if plan_eff.empty and plan_eq.empty:
        haut.info(ind.message_plan_vide(ind.couts_unitaires(table_couts(couts), horizon, types), budget, horizon, BUDGETS))
        fond_vide()
        return

    seul_eff, seul_eq = ind.plan_diff(plan_eff, plan_eq)
    t_eff, t_eq = "Chaque habitant compte pareil (curseur 0,0)", f"Équité (curseur {ind.fmt_dec(eq)})"
    with haut:
        st.markdown(f"#### Même budget ({ind.fmt_eur_plein(budget)}), deux priorités")
        st.info(conclusion)
        ca, cb = st.columns(2)
        for col, titre, kp in ((ca, t_eff, ke), (cb, t_eq, kq)):
            with col:
                st.markdown(f"**{titre}**")
                st.metric("Nouvellement desservis", ind.fmt_int(kp["habitants_uniques"]), border=True,
                          help="Habitants nouvellement desservis (personnes distinctes).")
                st.metric("Classes 1 et 2", ind.fmt_pct(kp["part_classes_1_2"]), border=True)
                st.metric("En QPV", ind.fmt_pct(kp["part_qpv"]), border=True)
                st.caption(ind.compte_types(kp["n_par_type"])
                           + (f" · CO₂ ≈ {kp['co2_t_an']} t/an" if kp["voitures_citiz"] else ""))
        vue = st.segmented_control("Plan affiché sur la carte", ["efficacite", "equite"], default="equite",
                                   required=True, key="plan_affiche",
                                   format_func={"efficacite": "Chaque habitant pareil", "equite": "Équité"}.get)
    pl, zones, diff = (plan_eff, z_eff, seul_eff) if vue == "efficacite" else (plan_eq, z_eq, seul_eq)
    if pl.empty:
        fond_vide()
    else:
        plan_map(pl, zones, key="plan", differents=diff, fond=True)
    with bas:
        st.markdown("##### Emplacements qui diffèrent entre les deux plans")
        st.markdown("**Seulement dans le plan « chaque habitant compte pareil »**")
        st.markdown(liste_emplacements(plan_eff, seul_eff))
        st.markdown("**Seulement dans le plan équité**")
        st.markdown(liste_emplacements(plan_eq, seul_eq))
        note("Contour doré sur la carte : emplacement absent de l'autre plan. Le curseur équité est figé pendant la "
             f"comparaison (plan équité calculé au curseur {ind.fmt_dec(eq)}).")


def reglages_couts(types) -> tuple | None:
    """Coûts unitaires modifiables (valeurs par défaut : data/ref/couts.csv, sources dans docs/COUTS.md).
    Renvoie None si rien n'a changé, pour garder le plan de référence en cache."""
    ref = data["gold_costs"]
    if ref is None or ref.empty:
        return None
    st.markdown("**Coûts unitaires (HT)**")
    st.caption("Ordres de grandeur sourcés (docs/COUTS.md), à remplacer par vos devis.")
    couts = []
    for r in ref.itertuples():
        bas, haut = getattr(r, "capex_bas", None), getattr(r, "capex_haut", None)
        fourchette = (f" Fourchette trouvée : {ind.fmt_eur_plein(bas)} à {ind.fmt_eur_plein(haut)}."
                      if pd.notna(bas) and pd.notna(haut) else "")
        label = ind.TYPE_LABELS.get(r.type, r.type)
        capex = st.number_input(f"{label} : investissement (€)", min_value=0, max_value=1_000_000, step=500,
                                value=int(r.capex_eur), key=f"capex_{r.type}", disabled=r.type not in types,
                                help=f"Valeur retenue : {ind.fmt_eur_plein(r.capex_eur)}.{fourchette}")
        opex = st.number_input(f"{label} : fonctionnement (€/an)", min_value=0, max_value=500_000, step=100,
                               value=int(r.opex_annuel_eur), key=f"opex_{r.type}", disabled=r.type not in types,
                               help=f"Valeur retenue : {ind.fmt_eur_plein(r.opex_annuel_eur)} par an.")
        couts.append((r.type, float(capex), float(opex)))
    defaut = [(r.type, float(r.capex_eur), float(r.opex_annuel_eur)) for r in ref.itertuples()]
    if couts == defaut:
        return None
    st.caption("Coûts modifiés : le plan est recalculé avec vos valeurs.")
    return tuple(couts)


@st.cache_data(show_spinner=False)
def premieres_citiz(directory: str, n: int = 5) -> pd.DataFrame:
    """Territoire sans Citiz : les n emplacements Citiz candidats qui desservent le plus d'habitants, choisis un à un
    sans compter deux fois un même carreau (habitants nouvellement desservis)."""
    d = load_data(directory)
    cand = d["gold_candidates"]
    if cand is None or "type" not in cand:
        return pd.DataFrame()
    cand = cand[cand["type"] == "citiz"]
    pop = d["gold_carreau_access"].set_index("carreau_id")["pop"].fillna(0)
    cov = {r.candidate_id: [c for c in ind._as_list(r.covered_carreaux) if c in pop.index] for r in cand.itertuples()}
    rows, vus = [], set()
    for _ in range(n):
        gains = {k: float(pop.reindex([c for c in v if c not in vus]).sum()) for k, v in cov.items()}
        if not gains or max(gains.values()) <= 0:
            break
        best = max(gains, key=gains.get)
        vus.update(cov.pop(best))
        r = cand[cand["candidate_id"] == best].iloc[0]
        rows.append({"rank": len(rows) + 1, "candidate_id": best, "type": "citiz", "lat": r["lat"], "lon": r["lon"],
                     "new_pop": gains[best]})
    out = pd.DataFrame(rows)
    if not out.empty and d["gold_predictions"] is not None and "p_usage" in d["gold_predictions"]:
        out = out.merge(d["gold_predictions"][["candidate_id", "p_usage"]], on="candidate_id", how="left")
    return out


def premieres_stations_citiz(horizon: int) -> None:
    st.info("Citiz n'est pas encore présent ici : voici où une première station desservirait le plus d'habitants.")
    top = premieres_citiz(data_dir)
    if top.empty:
        note("Aucun emplacement Citiz candidat dans les données de ce territoire.")
        return
    cout = ind.couts_unitaires(data["gold_costs"], horizon, ("citiz",)).get("citiz", 0.0)
    with st.expander("Premières stations Citiz possibles", expanded=True):
        a, b = st.container(), st.container()
        with a:
            plan_map(top.assign(cost_eur=cout), ind.walk_zones(top, data["gold_candidates"], access), height=300,
                     key="premieres_citiz")
        with b:
            p_usage = top["p_usage"] if "p_usage" in top else pd.Series(float("nan"), index=top.index)
            st.dataframe(pd.DataFrame({
                "N°": top["rank"], "Habitants nouvellement desservis": top["new_pop"].map(ind.fmt_int),
                ind.P_USAGE_LABEL: p_usage.map(lambda v: "—" if pd.isna(v) else ind.fmt_dec(v, 2))}),
                hide_index=True, width="stretch",
                column_config={ind.P_USAGE_LABEL: st.column_config.TextColumn(help=ind.P_USAGE_HELP)})
            note(f"Stations choisies une à une, sans compter deux fois un même habitant. Coût d'une station sur "
                 f"{horizon} an{'s' if horizon > 1 else ''} : {ind.fmt_eur_plein(cout)}. Le plan ci-dessous combine "
                 "Citiz, bus et vélo sous votre budget.")


@st.fragment
def rubrique_plan() -> None:
    mod, inputs_or_err = optimiser()
    if mod is None or data["gold_candidates"] is None:
        st.info(
            "L'optimiseur n'est pas encore disponible sur ce poste (module `optim/plan.py` ou table "
            "`gold_candidates` manquants). Les onglets ① et ② restent utilisables."
        )
        with st.expander("Détail technique"):
            st.code(inputs_or_err if mod is None else "gold_candidates absent du snapshot")
        return
    panneau = st.container(key="panneau")
    with panneau:
        haut = st.container()
        st.subheader("Vos paramètres")
        budget = st.select_slider("Budget", options=BUDGETS, value=500_000, format_func=ind.fmt_eur, key="budget")
        horizon = st.slider("Horizon (années de fonctionnement comptées)", 1, 10, 5, key="horizon")
        types = st.pills("Équipements", options=list(TYPES_PLAN), default=list(TYPES_PLAN), selection_mode="multi",
                         format_func=lambda t: ind.TYPE_LABELS[t], key="types_plan") or []
        compare = bool(st.session_state.get("comparer", False))
        equity = st.select_slider(
            "Curseur équité", options=EQUITE, value=0.0, format_func=ind.fmt_dec, key="equite", disabled=compare,
            help="0,0 : chaque habitant nouvellement desservi compte pareil. C'est aussi un choix, pas une position "
                 "neutre : il favorise les quartiers denses. 1,0 : un habitant compte jusqu'à deux fois plus selon le "
                 "besoin de son carreau (peu ou pas d'options à 5 min, part de ménages pauvres de l'IRIS). "
                 "Détail : optim/README.md.")
        st.caption("◀ 0,0 : chaque habitant compte pareil · 1,0 : priorité aux zones à potentiel ▶")
        with st.expander("Réglages avancés"):
            min_p = st.select_slider(
                f"{ind.P_USAGE_LABEL} : minimum", options=MIN_P, value=0.0, key="min_p",
                format_func=lambda v: ind.fmt_dec(v, 2),
                help=ind.P_USAGE_HELP + " Les emplacements Citiz sous ce score sont écartés (modèle, AUC 0,87).")
            couts = reglages_couts(types)
        st.toggle("Comparer avec et sans équité", key="comparer",
                  help="Calcule côte à côte le plan au curseur 0,0 et le plan au curseur choisi (1,0 si le curseur "
                       "est à 0,0). Le curseur est figé pendant la comparaison.")

        bas = st.container()
        pied_panneau()
    with haut:
        if SANS_CITIZ:
            premieres_stations_citiz(horizon)
        if SANS_VELO and "velhop" in types:
            note("Pas de vélo en libre-service sur ce territoire aujourd'hui : une station vélo proposée supposerait de "
                 "lancer le service (coût de lancement non compté dans le coût unitaire).")
    if not types:
        haut.info("Choisissez au moins un type d'équipement.")
        fond_vide()
    elif compare:
        rendu_comparaison(budget, equity, tuple(types), horizon, min_p, couts, haut=haut, bas=bas)
    else:
        rendu_plan(mod, inputs_or_err, budget, equity, tuple(types), horizon, min_p, couts, haut=haut, bas=bas)


# --------------------------------------------------------------------------- vue France métropolitaine

COMMANDE_NATIONALE = """python -m national.donnees
python -m national.offre all
python -m national.walk --download --extract --all
python -m national.acces --all
python -m national.candidats --roads --all
python -m national.modele
python -m national.territoires --all"""


def signature_france(directory: Path) -> tuple:
    """(fichier, date de modification) des fichiers nationaux : clé de cache, pour relire un fichier reproduit."""
    return tuple((f, (directory / f).stat().st_mtime if (directory / f).exists() else None)
                 for f in (fr.EPCI_ACCESS, fr.EPCI_GEO, fr.PLAN_FRANCE))


@st.cache_data(show_spinner="Chargement des données nationales…", max_entries=2)
def donnees_france(directory: str, signature: tuple) -> dict:
    return fr.charger_france(directory)


@st.cache_data(show_spinner="Préparation de la carte de France…", max_entries=2)
def carte_epci(directory: str, signature: tuple) -> dict:
    d = donnees_france(directory, signature)
    return fr.geojson_epci(d["epci_access"], d["epci_geo"])


INDICATEURS_CARTE = ["part_sans_option", "potentiel_500k"]  # « part avec Citiz » : dans le texte, pas sur la carte


def choix_indicateur() -> str:
    return st.segmented_control("Indicateur de la carte", INDICATEURS_CARTE, default="part_sans_option",
                                format_func=lambda k: fr.INDICATEURS[k][0], required=True,
                                key="indicateur_france") or "part_sans_option"


def carte_france(directory: str, signature: tuple, cle: str) -> None:
    """Carte de fond : intercommunalités colorées selon l'indicateur choisi dans le panneau."""
    try:
        carte = carte_epci(directory, signature)
    except Exception as e:  # contours illisibles : message propre, le reste de la page reste utilisable
        st.info(f"La carte des EPCI n'a pas pu être préparée ({type(e).__name__}) : vérifiez `{fr.EPCI_GEO}`.")
        return
    (lat0, lon0), (lat1, lon1) = carte["bounds"]
    m, _ = base_map(pd.DataFrame({"lat": [lat0, lat1], "lon": [lon0, lon1]}), zoom=6)
    folium.GeoJson(
        carte["geojson"], name=fr.INDICATEURS[cle][0],
        style_function=lambda f, c=f"c_{cle}": {"fillColor": f["properties"][c], "fillOpacity": 0.72,
                                               "color": "#ffffff", "weight": 0.4},
        highlight_function=lambda f: {"weight": 2, "color": TEXTE},
        tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False),
    ).add_to(m)
    Legende(fr.INDICATEURS[cle][0], [(carre(ind.CLASS_COLORS[c]), lib) for c, lib in carte["legendes"][cle]]
            + [(carre(ind.PROTECTED_COLOR), "Pas de donnée")]).add_to(m)
    carte_fond(m, carte["bounds"], animer=False, key="carte_france")


def classement_france(eacc: pd.DataFrame) -> None:
    st.subheader("Où investir en premier ?")
    top = fr.classement(eacc, "tous", n=10)
    if top.empty:
        st.info("Aucune intercommunalité à classer.")
        return
    avec_detail = {k for k in TERRITOIRES if k != fr.FRANCE}
    st.dataframe(fr.classement_fr(top, avec_detail), hide_index=True, width="stretch",
                 height=min(36 * (len(top) + 1), 760),
                 column_config={"Habitants nouvellement desservis (500 k€)": st.column_config.TextColumn(
                     help="Habitants qui gagnent au moins une option à 5 min à pied avec un plan de 500 000 € "
                          "(investissement et 5 ans de fonctionnement) dans l'EPCI, mêmes coûts et même optimiseur "
                          "que Strasbourg.")})
    ouvrables = [c for c in top["epci_code"] if c in avec_detail]
    if not ouvrables:
        note("Le détail par carreau de ces intercommunalités n'est pas encore produit (snapshots "
             "<code>data/snapshot/epci/&lt;code&gt;/</code>).")
        return
    noms = dict(zip(top["epci_code"], top["epci_nom"]))
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    choix = a.selectbox("Ouvrir une intercommunalité du classement", ouvrables, key="ouvrir_epci",
                        format_func=lambda c: f"{noms[c]} ({c})")
    b.button("Ouvrir ce territoire", key="bouton_ouvrir", type="primary", width="stretch",
             on_click=changer_territoire, args=(choix, RUBRIQUES[2]))
    note("Ouvre la rubrique ③ « Où investir ? » de l'intercommunalité : emplacements proposés, carte et coûts. "
         "Le lien de la page (<code>?territoire=&lt;code&gt;</code>) se partage tel quel.")


def vue_france() -> None:
    directory = fr.dossier_france()
    sig = signature_france(directory)
    d = donnees_france(str(directory), sig)
    eacc = d["epci_access"]
    manquantes = fr.colonnes_manquantes(eacc)
    if eacc is None or manquantes:
        detail = (f"Fichier attendu : `{directory / fr.EPCI_ACCESS}`." if eacc is None else
                  f"Colonnes manquantes dans `{fr.EPCI_ACCESS}` : {', '.join(manquantes)}.")
        st.info(
            "Les données nationales ne sont pas encore disponibles sur ce poste. Produisez-les avec la chaîne "
            f"nationale (docs/FRANCE.md) :\n\n```\n{COMMANDE_NATIONALE}\n```\n\n{detail} Dossier : variable "
            "`DATA_FRANCE`. En attendant, chaque territoire exporté reste consultable dans le sélecteur."
        )
        return
    k = fr.kpis_france(eacc)
    with st.container(key="panneau"):
        chiffre_hero(ind.fmt_pct(k["part_sans_option"]),
                     f"des habitants de France métropolitaine n'ont <b>aucune option</b> à 5 minutes à pied")
        c1, c2 = st.columns(2)
        c1.metric("Habitants (carreaux INSEE)", ind.fmt_int(k["population"]), border=True,
                  help=f"{ind.fmt_int(k['n_epci'])} intercommunalités (EPCI) de France métropolitaine.")
        c2.metric("Une station Citiz à 5 min", ind.fmt_pct(k["part_citiz"]), border=True,
                  help=f"Citiz est présent dans {ind.fmt_int(k['n_epci_citiz'])} intercommunalités sur "
                       f"{ind.fmt_int(k['n_epci'])}.")
        note(f"Si chaque intercommunalité investissait 500 000 € (investissement et 5 ans de fonctionnement) : "
             f"<b>{ind.fmt_int(k['potentiel_500k'])}</b> habitants nouvellement desservis au total, même méthode et même "
             "optimiseur que Strasbourg dans chaque EPCI.")
        cle = None
        if d["epci_geo"] is None:
            st.info(f"Contours des EPCI pas encore produits (`{directory / fr.EPCI_GEO}`) : la carte s'affichera "
                    "dès qu'ils seront là.")
        else:
            cle = choix_indicateur()
        classement_france(eacc)
        pied_panneau()
    if cle:
        carte_france(str(directory), sig, cle)

if EST_FRANCE:
    vue_france()
else:
    st.session_state.setdefault("rubrique", RUBRIQUES[0])  # pas de valeur par défaut : le classement France la choisit
    with st.container(key="nav"):
        rubrique = st.segmented_control("Rubrique", RUBRIQUES, required=True, key="rubrique",
                                        label_visibility="collapsed") or RUBRIQUES[0]
    {RUBRIQUES[0]: rubrique_acces, RUBRIQUES[1]: rubrique_stations, RUBRIQUES[2]: rubrique_plan}[rubrique]()
