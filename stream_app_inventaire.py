# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
from io import BytesIO
import unicodedata
import random, math
from collections import defaultdict
from urllib.parse import urlencode
import calendar

DB_FILE = "inventaire_temp.db"

# ──────────────────────────────────────────────────────────────────────────────
# 💡 LOGIQUE CONSERVÉE (adaptée de gestion_inventaire_pluscourt.py)
# ──────────────────────────────────────────────────────────────────────────────

from datetime import date, timedelta
import math
import pandas as pd

def add_suggested_order_dates(df: pd.DataFrame, nb_mois: int) -> pd.DataFrame:
    """
    Calcule df['date_suggeree_commande'] à partir de:
      - stock_disponible
      - besoin_total sur nb_mois
      - délai fournisseur (delai_choisi > delai_1 > delai > delai_livraison_jours)
      - buffer proportionnel au délai (plus de lousse en vert)
      - petit jitter par pièce pour éviter que tout tombe le même jour
    """
    if df.empty or "besoin_total" not in df.columns:
        df["date_suggeree_commande"] = ""
        return df

    today = date.today()

    def pick_lead(r):
        # ordre de préférence pour trouver un délai
        for k in ("delai_choisi", "delai_1", "delai", "delai_livraison_jours"):
            if k in r and pd.notna(r[k]):
                try:
                    v = int(round(float(r[k])))
                    if v > 0:
                        return v
                except Exception:
                    pass
        return 14  # fallback

    def daily_demand(r):
        besoin = float(r.get("besoin_total", 0) or 0)
        mois = max(1, int(nb_mois))
        return 0.0 if besoin <= 0 else besoin / (mois * 30.0)

    # buffer proportionnel au délai (plutôt que 0/3/7 jours fixes)
    # -> vert: 70% du délai, jaune: 30%, rouge: 0%
    def buffer_days_frac(dlead: int, statut: str) -> int:
        s = str(statut or "")
        if s.startswith("🔴"):
            beta = 0.00
        elif s.startswith("🟡"):
            beta = 0.30
        else:
            beta = 0.70
        # borne le buffer (évite des buffers énormes sur très gros délais)
        return int(math.ceil(min(30, beta * max(1, dlead))))

    # petit jitter déterministe par pièce pour éviter des regroupements artificiels
    def jitter_days_for(r, base: int) -> int:
        pid = str(r.get("id_piece", ""))
        if not pid:
            return 0
        h = sum(ord(c) for c in pid)  # déterministe
        return h % max(1, base)

    out = []
    for _, row in df.iterrows():
        stock = float(row.get("stock_disponible", 0) or 0)
        dlead = pick_lead(row)
        ddem  = daily_demand(row)
        statut = row.get("statut_stock", "")
        buf   = buffer_days_frac(dlead, statut)

        if ddem <= 0:
            out.append("")  # pas de demande -> pas de date
            continue

        # point de commande visé = demande pendant (délai + buffer)
        reorder_op = ddem * (dlead + buf)

        # jours avant d'atteindre ce point
        days_until_order = math.ceil((stock - reorder_op) / ddem)

        # si <= 0 -> commander maintenant; sinon appliquer un léger jitter par statut
        if days_until_order <= 0:
            od = today
        else:
            # jitter par statut pour lisser les dates proches
            if str(statut).startswith("🔴"):
                j = 0
            elif str(statut).startswith("🟡"):
                j = jitter_days_for(row, 2)     # 0..1 jour
            else:
                j = jitter_days_for(row, 3)     # 0..2 jours
            od = today + timedelta(days=days_until_order + j)

        out.append(od.strftime("%Y-%m-%d"))

    df["date_suggeree_commande"] = out
    return df


def open_conn():
    # IMPORTANT: ne PAS remplacer cette ligne par open_conn() !
    connect = sqlite3.connect  # alias anti-remplacement global
    conn = connect(DB_FILE)
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn


# ── Setup DB performance (indexes + WAL) ───────────────────
def ensure_db_indexes():
    conn = open_conn()
    cur = conn.cursor()

    def has_table(name: str) -> bool:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (name,))
        return cur.fetchone() is not None

    def safe_index(sql: str):
        try:
            cur.execute(sql)
        except sqlite3.OperationalError:
            pass
        except Exception:
            pass

    # Index seulement si la table existe
    if has_table("pieces"):
        safe_index("CREATE INDEX IF NOT EXISTS idx_pieces_id  ON pieces(id_piece)")
        safe_index("CREATE INDEX IF NOT EXISTS idx_pieces_nom ON pieces(nom_piece)")

    if has_table("fournisseurs"):
        safe_index("CREATE INDEX IF NOT EXISTS idx_fourn_id   ON fournisseurs(id_piece)")
        safe_index("CREATE INDEX IF NOT EXISTS idx_fourn_nom  ON fournisseurs(nom_fournisseur)")
        safe_index("CREATE UNIQUE INDEX IF NOT EXISTS uniq_fourn ON fournisseurs(id_piece, nom_fournisseur)")
        safe_index("CREATE INDEX IF NOT EXISTS idx_fourn_prix_delai ON fournisseurs(id_piece, prix_unitaire, delai_livraison_jours)")

    if has_table("stocks"):
        safe_index("CREATE INDEX IF NOT EXISTS idx_stocks_id  ON stocks(id_piece)")

    if has_table("achats"):
        safe_index("CREATE INDEX IF NOT EXISTS idx_achats_piece ON achats(id_piece)")
        safe_index("CREATE INDEX IF NOT EXISTS idx_achats_date  ON achats(date_achat)")

    if has_table("historique_commandes"):
        safe_index("CREATE INDEX IF NOT EXISTS idx_hist_date ON historique_commandes(date_commande)")
        safe_index("CREATE INDEX IF NOT EXISTS idx_hist_prod_date ON historique_commandes(id_produit, date_commande)")

    try:
        cur.execute("""
            CREATE VIEW IF NOT EXISTS v_cheapest AS
            SELECT id_piece, MIN(prix_unitaire) AS prix_min
            FROM fournisseurs GROUP BY id_piece;
        """)
        cur.execute("""
            CREATE VIEW IF NOT EXISTS v_fastest AS
            SELECT f1.id_piece, f1.nom_fournisseur, f1.delai_livraison_jours
            FROM fournisseurs f1
            JOIN (
            SELECT id_piece, MIN(delai_livraison_jours) AS dmin
            FROM fournisseurs GROUP BY id_piece
            ) m ON f1.id_piece=m.id_piece AND f1.delai_livraison_jours=m.dmin;
        """)
    except Exception:
        pass

    # Mode WAL = meilleure stabilité en lecture/écriture
    try:
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass

    conn.commit()
    conn.close()


# >>> AJOUT : tables commandes d'achat + lignes
def ensure_po_tables():
    conn = open_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS commandes_achat (
            id_commande TEXT PRIMARY KEY,
            fournisseur TEXT,
            date_commande TEXT,
            statut TEXT DEFAULT 'en_transit',  -- 'en_transit' | 'reçue'
            montant_total REAL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lignes_commande_achat (
            id_commande TEXT,
            id_piece TEXT,
            quantite INTEGER,
            prix_unitaire REAL,
            delai_jours INTEGER,
            date_reception_prevue TEXT,
            PRIMARY KEY (id_commande, id_piece),
            FOREIGN KEY (id_commande) REFERENCES commandes_achat(id_commande),
            FOREIGN KEY (id_piece) REFERENCES pieces(id_piece)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lca_piece ON lignes_commande_achat(id_piece)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lca_recep ON lignes_commande_achat(date_reception_prevue)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ca_statut ON commandes_achat(statut, date_commande)")
    conn.commit()
    conn.close()

def ensure_production_tables():
    conn = open_conn(); cur = conn.cursor()
    # Heures standard par produit (temps de fabrication d'une unité)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS product_std (
            id_produit TEXT PRIMARY KEY,
            heures_unite REAL DEFAULT 1
        )""")
    # Calendrier de prod (plan lissé)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS production_calendar (
            annee INTEGER,
            mois  INTEGER,            -- 1..12
            id_produit TEXT,
            qte INTEGER,
            heures REAL,
            PRIMARY KEY (annee, mois, id_produit)
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prodcal_mois ON production_calendar(annee, mois)")
    conn.commit(); conn.close()


# >>> AJOUT : réception auto des lignes dont la date prévue est passée
def auto_reception_commandes():
    today = datetime.today().strftime("%Y-%m-%d")
    conn = open_conn()
    cur = conn.cursor()

    # lignes dues (commandes en transit, date prévue <= aujourd'hui)
    cur.execute("""
        SELECT l.id_commande, l.id_piece, l.quantite, l.prix_unitaire
        FROM lignes_commande_achat l
        JOIN commandes_achat c ON c.id_commande = l.id_commande
        WHERE c.statut = 'en_transit' AND date(l.date_reception_prevue) <= date(?)
    """, (today,))
    dues = cur.fetchall()

    # réception = +stock
    for id_cmd, id_piece, qte, prix in dues:
        cur.execute("UPDATE stocks SET stock_disponible = COALESCE(stock_disponible,0) + ? WHERE id_piece = ?", (qte, id_piece))

    # passer les commandes concernées en 'reçue'
    if dues:
        cur.execute("""
            UPDATE commandes_achat
            SET statut = 'reçue'
            WHERE statut = 'en_transit'
              AND id_commande IN (SELECT DISTINCT id_commande
                                   FROM lignes_commande_achat
                                   WHERE date(date_reception_prevue) <= date(?))
        """, (today,))

    conn.commit()
    conn.close()


from datetime import datetime

def get_on_order_summary():
    """Retourne un DF (id_piece, en_attente, prochaine_reception) pour les commandes en transit."""
    today = datetime.today().strftime("%Y-%m-%d")
    conn = open_conn()
    try:
        df = pd.read_sql(
            """
            SELECT l.id_piece,
                   SUM(COALESCE(l.quantite,0)) AS en_attente,
                   MIN(l.date_reception_prevue) AS prochaine_reception
            FROM lignes_commande_achat l
            JOIN commandes_achat c ON c.id_commande = l.id_commande
            WHERE c.statut = 'en_transit'
              AND date(l.date_reception_prevue) >= date(?)
            GROUP BY l.id_piece
            """,
            conn,
            params=(today,),
        )
    except Exception:
        conn.close()
        return pd.DataFrame(columns=["id_piece","en_attente","prochaine_reception"])
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id_piece","en_attente","prochaine_reception"])
    df["en_attente"] = df["en_attente"].fillna(0).astype(int)
    df["prochaine_reception"] = df["prochaine_reception"].fillna("")
    return df


def _load_std_hours() -> dict:
    """map id_produit -> heures_unite (fallback 1.0 si manquant)"""
    conn = open_conn()
    try:
        df = pd.read_sql("SELECT id_produit, heures_unite FROM product_std", conn)
    finally:
        conn.close()
    m = dict(df.values) if not df.empty else {}
    return m

def _save_std_hours(df_std: pd.DataFrame):
    conn = open_conn(); cur = conn.cursor()
    cur.execute("DELETE FROM product_std")
    cur.executemany("INSERT INTO product_std (id_produit, heures_unite) VALUES (?,?)",
                    [(r["id_produit"], float(r["heures_unite"])) for _, r in df_std.iterrows()])
    conn.commit(); conn.close()

def _monthly_capacity(year:int, default_hours:float=160.0) -> pd.DataFrame:
    """Capacité initiale (heures) par mois, éditable ensuite."""
    return pd.DataFrame({"annee": [year]*12,
                         "mois": list(range(1,13)),
                         "heures_disponibles": [default_hours]*12})

def _baseline_annual_demand(year:int, growth_pct:float=0.0) -> pd.DataFrame:
    """
    Annualise la demande par produit à partir de l'historique (moyenne mensuelle * 12),
    puis applique un % de croissance.
    """
    conn = open_conn()
    try:
        hist = pd.read_sql("SELECT id_produit, quantite, date_commande FROM historique_commandes", conn)
    finally:
        conn.close()
    if hist.empty:
        return pd.DataFrame(columns=["id_produit","annual_units"])
    hist["date_commande"] = pd.to_datetime(hist["date_commande"], errors="coerce")
    hist = hist.dropna(subset=["date_commande"])
    hist["mois"] = hist["date_commande"].dt.to_period("M")
    par_mois = hist.groupby(["id_produit","mois"]).quantite.sum().reset_index()
    moy = par_mois.groupby("id_produit").quantite.mean()
    df = moy.reset_index().rename(columns={"quantite":"annual_units"})
    df["annual_units"] = (df["annual_units"]*12.0*(1.0+growth_pct/100.0)).round().astype(int)
    return df

def _level_load_schedule(annual_units: dict, std_hours: dict, cap_by_month: dict) -> pd.DataFrame:
    """
    Lissage simple capacité→demande :
      - pour m=1..12, on répartit la capacité du mois entre produits, en priorité ceux avec
        beaucoup de "heures restantes" (remaining_units * heures_unite).
      - greedy : on alloue le max d'unités entières possible sans dépasser la capacité.
    Retourne df(mois,id_produit,qte,heures).
    """
    rem_units = {p:int(max(0,annual_units.get(p,0))) for p in annual_units.keys()}
    # si des produits sans std_hours: 1.0h par défaut
    h_un = {p: float(std_hours.get(p, 1.0)) for p in rem_units.keys()}

    rows = []
    for m in range(1,13):
        cap = float(cap_by_month.get(m, 0.0))
        if cap <= 0 or sum(rem_units.values()) == 0:
            continue

        # ordre par poids restant (heures) décroissant
        order = sorted(rem_units.keys(), key=lambda p: rem_units[p]*h_un[p], reverse=True)

        # greedy : pour chaque produit, on prend le maximum d'unités entières qui rentre
        for p in order:
            if rem_units[p] <= 0 or cap <= 0:
                continue
            hu = max(h_un[p], 1e-6)
            max_units = min(rem_units[p], int(cap // hu))
            if max_units > 0:
                rows.append({"mois": m, "id_produit": p,
                             "qte": max_units, "heures": max_units*hu})
                rem_units[p] -= max_units
                cap -= max_units*hu

        # si il reste un peu de capacité et des unités restantes, on distribue 1 par 1
        if cap >= 1e-6 and sum(rem_units.values()) > 0:
            # priorité toujours au plus "lourd"
            for p in order:
                if rem_units[p] <= 0:
                    continue
                if h_un[p] <= cap + 1e-6:  # on peut caser 1 unité
                    rows.append({"mois": m, "id_produit": p,
                                 "qte": 1, "heures": h_un[p]})
                    rem_units[p] -= 1
                    cap -= h_un[p]
                if cap < 1e-6:
                    break

    return pd.DataFrame(rows)

def _product_to_pieces_map() -> dict:
    """map id_produit -> set([id_piece, ...]) via sous_assemblages/pieces."""
    conn = open_conn()
    try:
        df = pd.read_sql("""
            SELECT sa.id_produit, pi.id_piece
            FROM sous_assemblages sa
            JOIN pieces pi ON pi.id_sous_assemblage = sa.id_sous_assemblage
        """, conn)
    finally:
        conn.close()
    m = {}
    for _, r in df.iterrows():
        m.setdefault(r["id_produit"], set()).add(r["id_piece"])
    return m


# ── FTS: index plein-texte (sécurisé si FTS5 absent) ─────────────────────────
SUPPORTS_FTS = True

def ensure_fts():
    import sqlite3
    conn = open_conn()
    cur = conn.cursor()

    # sortir si 'pieces' n'existe pas encore
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pieces' LIMIT 1;")
    if cur.fetchone() is None:
        conn.close()
        return

    # essayer de créer la table virtuelle FTS5
    try:
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pieces_fts USING fts5(
                id_piece, nom_piece, contenu, content=''
            )
        """)
    except sqlite3.OperationalError:
        # build SQLite sans FTS5 -> on désactive proprement
        globals()["SUPPORTS_FTS"] = False
        conn.close()
        return
    except Exception:
        globals()["SUPPORTS_FTS"] = False
        conn.close()
        return

    # vider l'index correctement pour une table FTS5 "contentless"
    try:
        # commande spéciale FTS5
        cur.execute("INSERT INTO pieces_fts(pieces_fts) VALUES ('delete-all');")
    except sqlite3.OperationalError:
        # fallback: drop puis recreate
        cur.execute("DROP TABLE IF EXISTS pieces_fts;")
        cur.execute("""
            CREATE VIRTUAL TABLE pieces_fts USING fts5(
                id_piece, nom_piece, contenu, content=''
            )
        """)

    # (re)peupler l'index
    # Vérifie si la colonne pays_origine existe dans 'pieces'
    cur.execute("PRAGMA table_info(pieces)")
    cols = [r[1] for r in cur.fetchall()]
    has_pays = "pays_origine" in cols

    if has_pays:
        cur.execute("""
            INSERT INTO pieces_fts (id_piece, nom_piece, contenu)
            SELECT
                COALESCE(id_piece,''),
                COALESCE(nom_piece,''),
                COALESCE(id_piece,'') || ' ' ||
                COALESCE(nom_piece,'') || ' ' ||
                COALESCE(pays_origine,'')
            FROM pieces
        """)
    else:
        cur.execute("""
            INSERT INTO pieces_fts (id_piece, nom_piece, contenu)
            SELECT
                COALESCE(id_piece,''),
                COALESCE(nom_piece,''),
                COALESCE(id_piece,'') || ' ' ||
                COALESCE(nom_piece,'')
            FROM pieces
        """)

    conn.commit()
    conn.close()



def search_ids_from_fts(q: str) -> list[str]:
    if not q or not q.strip() or not SUPPORTS_FTS:
        return []
    conn = open_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id_piece FROM pieces_fts WHERE pieces_fts MATCH ? LIMIT 1000;",
        (q.strip(),)
    )
    ids = [r[0] for r in cur.fetchall()]
    conn.close()
    return ids


if "db_init_done" not in st.session_state:
    ensure_db_indexes()
    ensure_po_tables()
    ensure_production_tables()     # <<< AJOUT
    st.session_state["db_init_done"] = True


# --- Migration: s'assure que la colonne pays_origine existe dans 'pieces' ---
def ensure_pays_origine_column():
    conn = open_conn()
    cur = conn.cursor()

    # sortir si 'pieces' n'existe pas
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pieces' LIMIT 1;")
    if cur.fetchone() is None:
        conn.close()
        return

    # ajouter la colonne si absente
    cur.execute("PRAGMA table_info(pieces)")
    cols = [row[1] for row in cur.fetchall()]
    if "pays_origine" not in cols:
        # (option A) simple, puis remplir existants:
        cur.execute("ALTER TABLE pieces ADD COLUMN pays_origine TEXT")
        # remplir uniquement les lignes existantes qui sont NULL
        cur.execute("UPDATE pieces SET pays_origine='Canada' WHERE pays_origine IS NULL")

        # --- Option B à la place de A: mettre un DEFAULT directement ---
        # cur.execute("ALTER TABLE pieces ADD COLUMN pays_origine TEXT DEFAULT 'Canada'")
        # (dans ce cas, pas besoin du UPDATE ci-dessus)

        conn.commit()
    conn.close()


# Appel au démarrage
ensure_pays_origine_column()

try:
    ensure_fts()
except Exception:
    pass

# Réception automatique (à chaque exécution de l’app)
auto_reception_commandes()



@st.cache_data(show_spinner=False, ttl=600)
def _get_recette_with_qpp():
    """Recette produit -> pièces, avec qte_par_produit si la colonne existe, sinon 1."""
    conn = open_conn()
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(pieces)")
        cols = [r[1] for r in cur.fetchall()]
    except Exception:
        cols = []
    has_qpp = "qte_par_produit" in cols

    sql = f"""
        SELECT p.id_produit,
               pi.id_piece,
               pi.nom_piece,
               {"COALESCE(pi.qte_par_produit,1)" if has_qpp else "1"} AS qte_par_produit
        FROM produits p
        JOIN sous_assemblages sa ON p.id_produit = sa.id_produit
        JOIN pieces pi ON sa.id_sous_assemblage = pi.id_sous_assemblage
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

@st.cache_data(show_spinner=False, ttl=300)
def _get_stock_df():
    """Stock actuel par pièce (id_piece, stock_disponible) ou DF vide si table absente."""
    conn = open_conn()
    try:
        has = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'", conn
        )
        if len(has):
            df = pd.read_sql("SELECT id_piece, stock_disponible FROM stocks", conn)
        else:
            df = pd.DataFrame(columns=["id_piece", "stock_disponible"])
    except Exception:
        df = pd.DataFrame(columns=["id_piece", "stock_disponible"])
    conn.close()
    return df


# --- Helper: map pays d'origine -> grande région ---
def pays_to_region(pays: str) -> str:
    if not pays:
        return "Autre"
    s = str(pays).strip().lower()
    # normalisation simple (sans accents)
    import unicodedata
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    amerique = {
        "canada", "etats-unis", "united states", "usa", "us", "mexique", "mexico"
    }
    europe = {
        "france","allemagne","italie","royaume-uni","espagne","pays-bas","suede","suisse",
        "belgique","autriche","irlande","portugal","norvege","danemark","finlande","grece",
        "pologne","hongrie","roumanie","tchequie","republique tcheque","pays de galles",
        "ecosse","angleterre"
    }
    asie = {
        "chine","inde","japon","coree du sud","coree-du-sud","coree","taiwan","vietnam",
        "thailande","malaisie","indonesie","philippines","singapour","bangladesh","pakistan"
    }

    if s in amerique:
        return "Amérique"
    if s in europe:
        return "Europe"
    if s in asie:
        return "Asie"
    return "Autre"



def _fmt_money(v):
    try:
        if pd.isna(v): return ""
        return f"${v:,.2f}"
    except Exception:
        return ""

def _fmt_int(v):
    try:
        if pd.isna(v): return ""
        return f"{int(v):,}"
    except Exception:
        return ""

def style_table(df, money_cols=None, int_cols=None, color_func=None):
    """Retourne un Styler avec formats + éventuelle coloration par ligne."""
    if df is None or df.empty:
        return df
    styler = df.style
    # formats $ (2 décimales)
    if money_cols:
        styler = styler.format({c: _fmt_money for c in money_cols if c in df.columns})
    # formats int
    if int_cols:
        styler = styler.format({c: _fmt_int for c in int_cols if c in df.columns})
    # coloration (tes color_row / color_row2)
    if color_func:
        styler = styler.apply(color_func, axis=1)
    return styler

def _strip_accents(s: str) -> str:
    s = "" if s is None else str(s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def make_search_mask(df: pd.DataFrame, query: str, columns: list[str], mode: str = "AND") -> pd.Series:
    """
    Filtre df en cherchant les 'mots' de `query` dans `columns`.
    - Insensible aux accents et à la casse
    - `mode`: "AND" (tous les mots) ou "OR" (au moins un)
    Retourne une Series booléenne indexée comme df.
    """
    # Colonnes existantes uniquement
    cols = [c for c in columns if c in df.columns]

    # Cas sans recherche
    if not query or df.empty or not cols:
        return pd.Series(True, index=df.index)

    # Normaliser et concaténer par ligne
    # (équiv. de applymap, sans warning)
    normed = df[cols].apply(lambda s: s.astype(str).map(_strip_accents).str.lower())
    blob = normed.agg(" | ".join, axis=1)

    # Mots recherchés
    tokens = [_strip_accents(t).lower() for t in str(query).split() if t.strip()]
    if not tokens:
        return pd.Series(True, index=df.index)

    # AND / OR
    if mode == "AND":
        mask = pd.Series(True, index=df.index)
        for t in tokens:
            mask &= blob.str.contains(t, na=False)
    else:
        mask = pd.Series(False, index=df.index)
        for t in tokens:
            mask |= blob.str.contains(t, na=False)

    return mask

@st.cache_data(show_spinner=True, ttl=600)
def _get_historique_cached():
    conn = open_conn()
    df = pd.read_sql("SELECT id_produit, quantite, date_commande FROM historique_commandes", conn)
    conn.close()
    return df

@st.cache_data(show_spinner=False, ttl=600)
def _get_fournisseurs_cached():
    conn = open_conn()
    df = pd.read_sql("SELECT * FROM fournisseurs", conn)
    conn.close()
    return df

    
def calculer_previsions_historique(nb_mois=3, hausse_pct=0):
    conn = open_conn()

    # 1) moyenne mensuelle par produit
    historique = _get_historique_cached()
    if not historique.empty:
        historique["date_commande"] = pd.to_datetime(historique["date_commande"])
        historique["mois"] = historique["date_commande"].dt.to_period("M")
        par_mois = historique.groupby(["id_produit", "mois"]).quantite.sum().reset_index()
        moyenne_mensuelle = par_mois.groupby("id_produit").quantite.mean()
    else:
        moyenne_mensuelle = pd.Series(dtype=float)

    # 2) recette produit→pièces
    recette = pd.read_sql("""
        SELECT p.id_produit, pi.id_piece, pi.nom_piece
        FROM produits p
        JOIN sous_assemblages sa ON p.id_produit = sa.id_produit
        JOIN pieces pi ON sa.id_sous_assemblage = pi.id_sous_assemblage
    """, conn)

    # 3) stock actuel par pièce
    stock_df = _get_stock_df()
    stock = stock_df.set_index("id_piece") if not stock_df.empty else pd.DataFrame().set_index(pd.Index([]))


    besoins = {}
    hausse = hausse_pct / 100.0

    for _, row in recette.iterrows():
        id_prod = row["id_produit"]
        id_piece = row["id_piece"]
        nom_piece = row["nom_piece"]

        base_prod = float(moyenne_mensuelle.get(id_prod, 0.0)) * nb_mois
        if base_prod == 0:
            continue
        besoin_ajuste = int(round(base_prod * (1 + hausse)))

        stock_dispo = int(stock["stock_disponible"].get(id_piece, 0)) if not stock.empty else 0

        if id_piece not in besoins:
            besoins[id_piece] = {
                "nom_piece": nom_piece,
                "besoin_total": 0,
                "stock_disponible": stock_dispo
            }
        besoins[id_piece]["besoin_total"] += besoin_ajuste

    conn.close()
    return besoins


def calculer_previsions(df_previsions: pd.DataFrame, hausse_pct: float = 0):
    """
    df_previsions : DataFrame ['id_produit','quantite_prevue']
    Retour : dict { id_piece: {'nom_piece','besoin_total','stock_disponible'} }
    """
    recette = _get_recette_with_qpp()
    stock   = _get_stock_df().set_index("id_piece") if not _get_stock_df().empty else pd.DataFrame().set_index(pd.Index([]))

    df_prev = df_previsions.copy()
    df_prev["quantite_prevue"] = pd.to_numeric(df_prev["quantite_prevue"], errors="coerce").fillna(0)
    facteur_hausse = 1.0 + float(hausse_pct) / 100.0

    besoins = {}
    for _, row in df_prev.iterrows():
        id_prod  = row["id_produit"]
        qty_prod = float(row["quantite_prevue"]) * facteur_hausse
        if qty_prod <= 0:
            continue

        sous = recette[recette["id_produit"] == id_prod]
        if sous.empty:
            continue

        for _, r in sous.iterrows():
            id_piece = str(r["id_piece"])
            nom_piece = r.get("nom_piece", "")
            qpp = float(r.get("qte_par_produit", 1) if pd.notna(r.get("qte_par_produit", 1)) else 1)

            besoin_piece = int(round(qty_prod * qpp))
            stock_dispo = (
                int(stock["stock_disponible"].get(id_piece, 0))
                if not stock.empty and id_piece in stock.index
                else 0
            )

            if id_piece not in besoins:
                besoins[id_piece] = {"nom_piece": nom_piece, "besoin_total": 0, "stock_disponible": stock_dispo}
            besoins[id_piece]["besoin_total"] += besoin_piece

    return besoins




def _score_equilibre(df):
    # normalisation simple prix/délai
    if df.empty:
        return df.assign(score=1.0)
    max_p = df["prix_unitaire"].max() or 1.0
    max_d = df["delai_livraison_jours"].max() or 1.0
    out = df.copy()
    out["score"] = 0.5 * out["prix_unitaire"] / max_p + 0.5 * out["delai_livraison_jours"] / max_d
    return out


def choisir_fournisseur_multiple(besoins: dict, strategie: str) -> pd.DataFrame:
    fournisseurs = _get_fournisseurs_cached()

    choix = {}
    for id_piece, groupe in fournisseurs.groupby("id_piece"):
        if strategie == "prix":
            tri = groupe.sort_values("prix_unitaire")
        elif strategie == "delai":
            tri = groupe.sort_values("delai_livraison_jours")
        else:
            tri = _score_equilibre(groupe).sort_values("score")

        tri = tri.drop_duplicates(subset="nom_fournisseur").head(3)
        choix[id_piece] = tri.reset_index(drop=True)

    resultats = []
    for id_piece, infos in besoins.items():
        ligne = {
            "id_piece": id_piece,
            "nom_piece": infos["nom_piece"],
            "besoin_total": infos["besoin_total"],
            "stock_disponible": infos.get("stock_disponible", 0),
        }
        for i in range(3):
            if id_piece in choix and i < len(choix[id_piece]):
                f = choix[id_piece].iloc[i]
                ligne[f"fournisseur_{i+1}"] = f["nom_fournisseur"]
                ligne[f"prix_{i+1}"] = float(f["prix_unitaire"])
                ligne[f"delai_{i+1}"] = int(f["delai_livraison_jours"])
            else:
                ligne[f"fournisseur_{i+1}"] = None
                ligne[f"prix_{i+1}"] = None
                ligne[f"delai_{i+1}"] = None
        resultats.append(ligne)

        df = pd.DataFrame(resultats)

    # Harmoniser les types pour Arrow/Streamlit
    num_cols = ["besoin_total", "stock_disponible",
                "prix_1", "prix_2", "prix_3",
                "delai_1", "delai_2", "delai_3"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    str_cols = ["id_piece", "nom_piece", "fournisseur_1", "fournisseur_2", "fournisseur_3"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # -- Calculs d'affichage et enrichissements --
    if not df.empty:
        # statut pour la coloration (tient compte de la sévérité)
        try:
            sev = float(st.session_state.get("severity", 1.0))
        except Exception:
            sev = 1.0

        def statut_row(r):
            try:
                besoin = int(round(float(r["besoin_total"]) * sev))
                stock  = int(round(float(r["stock_disponible"])))
            except Exception:
                return "—"
            if stock == 0:
                return "🔴 Rupture"
            elif stock < besoin:
                return "🟡 À commander"
            else:
                return "🟢 OK"

        df["statut_stock"] = df.apply(statut_row, axis=1)

        # coût “fournisseur 1” estimé (basé sur besoin_total brut)
        def cout_1(r):
            try:
                return float(r["prix_1"]) * int(r["besoin_total"])
            except Exception:
                return None
        df["cout_estime_fournisseur1"] = df.apply(cout_1, axis=1)

        # dates suggérées (une seule fois)
        try:
            nbm = int(st.session_state.get("nb_mois", 3))
        except Exception:
            nbm = 3
        df = add_suggested_order_dates(df, nbm)

        # "en_attente" & "prochaine_reception" à partir des commandes en transit
        try:
            onord = get_on_order_summary()
            if not onord.empty:
                df = df.merge(onord, on="id_piece", how="left")
        except Exception:
            pass
        if "en_attente" not in df.columns:
            df["en_attente"] = 0
        if "prochaine_reception" not in df.columns:
            df["prochaine_reception"] = ""
        df["en_attente"] = df["en_attente"].fillna(0).astype(int)
        df["prochaine_reception"] = df["prochaine_reception"].fillna("")

        # stock projeté = stock dispo + en attente
        df["stock_projeté"] = (df["stock_disponible"].fillna(0).astype(int)
                               + df["en_attente"].fillna(0).astype(int))

    # --- Publier les besoins pour l’onglet 🛒 (format {id_piece: {"besoin_total": int}}) ---
    try:
        d = (
            df[["id_piece", "besoin_total"]]
            .dropna(subset=["id_piece", "besoin_total"])
            .astype({"besoin_total": "int64"})
            .set_index("id_piece")["besoin_total"]
            .to_dict()
        )
        st.session_state["besoins_courants"] = {k: {"besoin_total": int(v)} for k, v in d.items()}
    except Exception:
        st.session_state["besoins_courants"] = {}
    # ---------------------------------------------------------------------------------------

    return df



def choisir_fournisseur_unique(
    besoins: dict,
    strategie: str,
    filtre_fournisseur: str = "",
    filtre_prix_max: float | None = None,
    filtre_delai_max: int | None = None
) -> pd.DataFrame:
    """
    Retourne, pour chaque pièce demandée, le meilleur fournisseur selon la stratégie.

    - Accepte besoins sous les formes:
        {"PC-0001": 12, ...}
        {"PC-0001": {"besoin_total": 12}, ...}
        {"PC-0001": {"besoin_total": 12, "nom_piece": "XYZ"}, ...}
    - Si nom_piece est manquant, on le récupère depuis la table pieces.
    """

    fournisseurs = _get_fournisseurs_cached()

    # --- map id_piece -> nom_piece (pour combler les infos manquantes)
    try:
        _df_pn = pd.read_sql("SELECT id_piece, nom_piece FROM pieces", open_conn())
        _nom_map = dict(_df_pn.values)  # {id_piece: nom_piece}
    except Exception:
        _nom_map = {}

    def _coerce_record(id_piece: str, infos) -> tuple[int, str]:
        """Retourne (quantite, nom_piece) en normalisant 'infos'."""
        if isinstance(infos, dict):
            q = int(infos.get("besoin_total", 0) or 0)
            nom = infos.get("nom_piece")
        else:
            q = int(infos or 0)
            nom = None
        if nom is None:
            nom = _nom_map.get(id_piece, id_piece)
        return q, nom

    resultats = []
    for id_piece, infos in (besoins or {}).items():
        quantite, nom_piece = _coerce_record(id_piece, infos)
        if quantite <= 0:
            continue

        fournisseurs_piece = fournisseurs[fournisseurs["id_piece"] == id_piece]

        if filtre_fournisseur:
            fournisseurs_piece = fournisseurs_piece[
                fournisseurs_piece["nom_fournisseur"] == filtre_fournisseur
            ]
        if filtre_prix_max is not None:
            fournisseurs_piece = fournisseurs_piece[
                fournisseurs_piece["prix_unitaire"] <= float(filtre_prix_max)
            ]
        if filtre_delai_max is not None:
            fournisseurs_piece = fournisseurs_piece[
                fournisseurs_piece["delai_livraison_jours"] <= int(filtre_delai_max)
            ]

        if fournisseurs_piece.empty:
            # aucun fournisseur ne passe les filtres → on ignore cette pièce
            continue

        if strategie == "prix":
            choix = fournisseurs_piece.sort_values("prix_unitaire", kind="mergesort")
        elif strategie == "delai":
            choix = fournisseurs_piece.sort_values("delai_livraison_jours", kind="mergesort")
        else:
            # 'equilibre' = scoring prix/délai (ta fonction existante)
            choix = _score_equilibre(fournisseurs_piece).sort_values("score", kind="mergesort")

        best = choix.iloc[0]
        resultats.append({
            "id_piece": id_piece,
            "nom_piece": nom_piece,
            "besoin_total": int(quantite),
            "fournisseur": str(best["nom_fournisseur"]),
            "prix": float(best["prix_unitaire"]),
            "delai": int(best["delai_livraison_jours"]),
            "cout_total": round(float(best["prix_unitaire"]) * int(quantite), 2),
        })

    return pd.DataFrame(resultats)


def choisir_fournisseur_top3(
    besoins: dict,
    strategie: str,
    filtre_fournisseur: str = "",
    filtre_prix_max: float | None = None,
    filtre_delai_max: int | None = None
) -> pd.DataFrame:
    """
    Retourne, pour chaque pièce demandée, jusqu'à 3 options de fournisseurs
    (fournisseur_1..3, prix_1..3, delai_1..3) + stock_disponible, besoin_total, nom_piece.
    Accepte besoins au format: {"PC-0001": 12} ou {"PC-0001": {"besoin_total": 12, "nom_piece": "..."}}
    """
    fournisseurs = _get_fournisseurs_cached().copy()

    # nom des pièces (pour combler si absent)
    try:
        df_pieces = pd.read_sql("SELECT id_piece, nom_piece FROM pieces", open_conn())
        nom_map = dict(df_pieces.values)
    except Exception:
        nom_map = {}

    def _coerce_record(id_piece: str, infos) -> tuple[int, str]:
        if isinstance(infos, dict):
            q = int(infos.get("besoin_total", 0) or 0)
            nom = infos.get("nom_piece") or nom_map.get(id_piece, id_piece)
        else:
            q = int(infos or 0)
            nom = nom_map.get(id_piece, id_piece)
        return q, nom

    rows = []
    for id_piece, infos in (besoins or {}).items():
        quantite, nom_piece = _coerce_record(id_piece, infos)
        if quantite <= 0:
            continue

        dfp = fournisseurs[fournisseurs["id_piece"] == id_piece]
        if filtre_fournisseur:
            dfp = dfp[dfp["nom_fournisseur"] == filtre_fournisseur]
        if filtre_prix_max is not None:
            dfp = dfp[dfp["prix_unitaire"] <= float(filtre_prix_max)]
        if filtre_delai_max is not None:
            dfp = dfp[dfp["delai_livraison_jours"] <= int(filtre_delai_max)]
        if dfp.empty:
            continue

        if strategie == "prix":
            dfp = dfp.sort_values("prix_unitaire", kind="mergesort")
        elif strategie == "delai":
            dfp = dfp.sort_values("delai_livraison_jours", kind="mergesort")
        else:
            dfp = _score_equilibre(dfp).sort_values("score", kind="mergesort")

        top = dfp.head(3).reset_index(drop=True)
        row = {"id_piece": id_piece, "nom_piece": nom_piece, "besoin_total": int(quantite)}
        for i in range(len(top)):
            row[f"fournisseur_{i+1}"] = str(top.loc[i, "nom_fournisseur"])
            row[f"prix_{i+1}"] = float(top.loc[i, "prix_unitaire"])
            row[f"delai_{i+1}"] = int(top.loc[i, "delai_livraison_jours"])
        for i in range(len(top), 3):
            row[f"fournisseur_{i+1}"] = None
            row[f"prix_{i+1}"] = None
            row[f"delai_{i+1}"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ajouter stock_disponible
    try:
        stk = _get_stock_df()[["id_piece", "stock_disponible"]]
        df = df.merge(stk, on="id_piece", how="left")
    except Exception:
        df["stock_disponible"] = 0
    df["stock_disponible"] = df["stock_disponible"].fillna(0).astype(int)


    # (si pas déjà fait plus haut) enrichir avec “en_attente” / “prochaine_reception”
    try:
        onord = get_on_order_summary()
        if not onord.empty:
            df = df.merge(onord, on="id_piece", how="left")
    except Exception:
        pass
    if "en_attente" not in df.columns:
        df["en_attente"] = 0
    if "prochaine_reception" not in df.columns:
        df["prochaine_reception"] = ""
    df["en_attente"] = df["en_attente"].fillna(0).astype(int)
    df["prochaine_reception"] = df["prochaine_reception"].fillna("")

    # >>> AJOUT : stock projeté
    df["stock_projeté"] = (df["stock_disponible"].fillna(0).astype(int)
                           + df["en_attente"].fillna(0).astype(int))

    return df



def build_commande_excel(df_commande: pd.DataFrame, resume_fourn: pd.DataFrame | None = None) -> bytes:
    """Crée un .xlsx avec:
       - 'Commande (toutes)' : toutes les lignes
       - 1 onglet / fournisseur
       - 'Résumé fournisseurs' : totaux $ + stats
       Retourne les bytes prêts pour st.download_button.
    """
    buf = BytesIO()
    try:
        # ── Version formatée (XlsxWriter)
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            wb = writer.book
            fmt_hdr = wb.add_format({"bold": True})
            fmt_money = wb.add_format({"num_format": "$#,##0.00"})
            fmt_int = wb.add_format({"num_format": "#,##0"})

            # Colonnes dans l’ordre
            cols = [c for c in ["id_piece","nom_piece","besoin_total","fournisseur","prix","delai","cout_total"]
                    if c in df_commande.columns]
            df_main = df_commande[cols].copy()

            # Feuille principale
            sh_name = "Commande (toutes)"
            df_main.to_excel(writer, sheet_name=sh_name, index=False)
            ws = writer.sheets[sh_name]
            ws.set_row(0, None, fmt_hdr)
            # Largeurs & formats colonnes
            widths = {"A":18,"B":28,"C":14,"D":24,"E":12,"F":12,"G":16}
            for col, w in widths.items():
                ws.set_column(f"{col}:{col}", w)
            ws.set_column("C:C", 14, fmt_int)    # besoin_total
            ws.set_column("E:E", 12, fmt_money)  # prix
            ws.set_column("F:F", 12, fmt_int)    # delai
            ws.set_column("G:G", 16, fmt_money)  # cout_total
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, df_main.shape[0], df_main.shape[1]-1)

            # 1 onglet par fournisseur
            for supplier, grp in df_main.groupby("fournisseur", dropna=False):
                name = str(supplier) if pd.notna(supplier) else "Sans fournisseur"
                safe = name.replace("/", "-")[:31]  # Excel limite 31 chars
                grp.to_excel(writer, sheet_name=safe, index=False)
                wss = writer.sheets[safe]
                wss.set_row(0, None, fmt_hdr)
                for col, w in widths.items():
                    wss.set_column(f"{col}:{col}", w)
                wss.set_column("C:C", 14, fmt_int)
                wss.set_column("E:E", 12, fmt_money)
                wss.set_column("F:F", 12, fmt_int)
                wss.set_column("G:G", 16, fmt_money)
                wss.freeze_panes(1, 0)
                wss.autofilter(0, 0, grp.shape[0], grp.shape[1]-1)

            # Résumé fournisseurs
            if resume_fourn is None:
                tmp = df_commande.copy()
                for c in ["besoin_total","prix","delai","cout_total"]:
                    if c in tmp.columns:
                        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                resume_fourn = (
                    tmp.groupby("fournisseur", as_index=False)
                       .agg(
                           lignes=("id_piece", "count"),
                           quantite=("besoin_total", "sum"),
                           cout_total=("cout_total", "sum"),
                           prix_moyen_unitaire=("prix", "mean"),
                           delai_moyen_jours=("delai", "mean"),
                       )
                       .sort_values("cout_total", ascending=False, na_position="last")
                )

            sheet_r = "Résumé fournisseurs"
            resume_fourn.to_excel(writer, sheet_name=sheet_r, index=False)
            wsr = writer.sheets[sheet_r]
            wsr.set_row(0, None, fmt_hdr)
            # Formats rapides
            cols_r = resume_fourn.columns.tolist()
            for i in range(len(cols_r)):
                wsr.set_column(i, i, 22)
            if "cout_total" in cols_r:
                j = cols_r.index("cout_total"); wsr.set_column(j, j, 16, fmt_money)
            if "prix_moyen_unitaire" in cols_r:
                j = cols_r.index("prix_moyen_unitaire"); wsr.set_column(j, j, 18, fmt_money)
            for nm in ["quantite","lignes","delai_moyen_jours"]:
                if nm in cols_r:
                    j = cols_r.index(nm); wsr.set_column(j, j, 16, fmt_int)

    except Exception:
        # ── Fallback simple (OpenPyXL) si XlsxWriter absent : sans mise en forme avancée
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_commande.to_excel(writer, sheet_name="Commande (toutes)", index=False)
            if resume_fourn is not None and not resume_fourn.empty:
                resume_fourn.to_excel(writer, sheet_name="Résumé fournisseurs", index=False)

    buf.seek(0)
    return buf.getvalue()


def _safe_col(df, col, default=None):
    try:
        if df is not None and not df.empty and col in df.columns:
            v = df.iloc[0][col]
            return None if pd.isna(v) else v
    except Exception:
        pass
    return default

def get_piece_details(id_piece: str) -> dict:
    """
    Retourne un dictionnaire de détails : description, numero_serie, stock, dernier achat
    (prix/date/fournisseur) si table 'achats' existe, sinon fournisseur le moins cher +
    délai. Robuste aux colonnes manquantes.
    """
    conn = open_conn()

    # 1) Infos 'pieces'
    try:
        df_p = pd.read_sql("SELECT * FROM pieces WHERE id_piece = ? LIMIT 1", conn, params=[id_piece])
    except Exception:
        df_p = pd.DataFrame()

    # 2) Stock
    try:
        df_s = pd.read_sql("SELECT stock_disponible FROM stocks WHERE id_piece = ? LIMIT 1", conn, params=[id_piece])
    except Exception:
        df_s = pd.DataFrame()

    # 3) Dernier achat (si table 'achats' existe)
    try:
        df_a = pd.read_sql(
            "SELECT fournisseur, prix_unitaire, date_achat "
            "FROM achats WHERE id_piece = ? ORDER BY date_achat DESC LIMIT 1",
            conn, params=[id_piece]
        )
    except Exception:
        df_a = pd.DataFrame()

    # 4) Fournisseurs (fallback + infos)
    try:
        # si une colonne date_maj existe on priorise la plus récente, sinon juste moins cher
        df_f = pd.read_sql(
            "SELECT nom_fournisseur, prix_unitaire, delai_livraison_jours, "
            "COALESCE(date_maj, '') AS date_maj "
            "FROM fournisseurs WHERE id_piece = ?",
            conn, params=[id_piece]
        )
    except Exception:
        df_f = pd.DataFrame()

    conn.close()

    details = {
        "id_piece": id_piece,
        "nom_piece": _safe_col(df_p, "nom_piece", ""),
        "description": _safe_col(df_p, "description", "—"),
        "numero_serie": _safe_col(df_p, "numero_serie", "—"),
        "stock_disponible": _safe_col(df_s, "stock_disponible", 0),
        "pays_origine": _safe_col(df_p, "pays_origine", "—"),
        # Dernier achat si dispo
        "dernier_achat_fournisseur": _safe_col(df_a, "fournisseur"),
        "dernier_achat_prix": _safe_col(df_a, "prix_unitaire"),
        "dernier_achat_date": _safe_col(df_a, "date_achat"),
        # Fallback “moins cher / plus rapide”
        "fournisseur_moins_cher": None,
        "prix_moins_cher": None,
        "fournisseur_plus_rapide": None,
        "delai_plus_rapide": None,
    }

    if df_f is not None and not df_f.empty:
        try:
            cheap = df_f.sort_values(["prix_unitaire", "delai_livraison_jours"], ascending=[True, True]).iloc[0]
            details["fournisseur_moins_cher"] = cheap.get("nom_fournisseur")
            details["prix_moins_cher"] = cheap.get("prix_unitaire")
        except Exception:
            pass
        try:
            fast = df_f.sort_values(["delai_livraison_jours", "prix_unitaire"], ascending=[True, True]).iloc[0]
            details["fournisseur_plus_rapide"] = fast.get("nom_fournisseur")
            details["delai_plus_rapide"] = fast.get("delai_livraison_jours")
        except Exception:
            pass

    return details

def render_piece_popover(id_piece: str, nom_piece: str = ""):
    """Petit bouton '🔍 Détails' qui ouvre une popover avec les infos de la pièce."""
    label = f"🔍 {id_piece}" if not nom_piece else f"🔍 {id_piece} — {nom_piece}"
    with st.popover(label):
        d = get_piece_details(id_piece)
        left, right = st.columns([2, 1])

        with left:
            st.markdown(f"**{d.get('id_piece','')} — {d.get('nom_piece','')}**")
            st.write(d.get("description") or "—")
            st.write(f"**N° série** : {d.get('numero_serie') or '—'}")
            st.write(f"**Stock** : {int(d.get('stock_disponible') or 0)}")

        with right:
            # Dernier achat connu
            if d.get("dernier_achat_prix") is not None:
                st.write("**Dernier achat**")
                st.write(f"{d.get('dernier_achat_fournisseur','?')}")
                st.write(f"{_fmt_money(float(d['dernier_achat_prix']))}")
                st.write(f"{d.get('dernier_achat_date','')}")
            else:
                st.write("**Dernier achat**")
                st.write("— (non disponible)")

            # Fallbacks fournisseur
            st.write("**Fourn. moins cher**")
            if d.get("prix_moins_cher") is not None:
                st.write(f"{d.get('fournisseur_moins_cher','?')} · {_fmt_money(float(d['prix_moins_cher']))}")
            else:
                st.write("—")

            st.write("**Plus rapide**")
            if d.get("delai_plus_rapide") is not None:
                st.write(f"{d.get('fournisseur_plus_rapide','?')} · {int(d['delai_plus_rapide'])} j")
            else:
                st.write("—")


def show_piece_modal_if_requested():
    """Affiche la fiche pièce si ?piece=ID est dans l’URL.
       Compatible anciennes versions (sans st.modal).
       SÉCURITÉ: n’affiche qu’UNE SEULE FOIS par exécution,
       même si la fonction est appelée plusieurs fois.
    """
    # --- Guard "1 seule fois par exécution" (via variable module globale) ---
    g = globals()
    if "_piece_modal_rendered_this_run" not in g:
        g["_piece_modal_rendered_this_run"] = False

    # 1) Lire le paramètre d’URL ?piece=...
    piece = None
    try:
        qp = st.query_params  # Streamlit récent
        piece_val = qp.get("piece", None)
        piece = piece_val[0] if isinstance(piece_val, list) else piece_val
    except Exception:
        qp = st.experimental_get_query_params()  # anciens Streamlit
        piece = qp.get("piece", [None])[0] if "piece" in qp else None

    # Si pas de param, on remet le flag à False et on sort
    if not piece:
        g["_piece_modal_rendered_this_run"] = False
        return

    # Si déjà affichée dans CETTE exécution, on sort
    if g["_piece_modal_rendered_this_run"]:
        return

    # Marquer IMMÉDIATEMENT comme affichée pour bloquer les autres appels du même run
    g["_piece_modal_rendered_this_run"] = True

    # 2) Récupérer les données
    d = get_piece_details(piece)
    title = f"📦 Détails — {d.get('id_piece','')}"

    # 3) Rendu du contenu (partagé modal/faux-modal)
    def _render_content():
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**{d.get('id_piece','')} — {d.get('nom_piece','')}**")
            st.write(d.get("description") or "—")
            st.write(f"**N° série** : {d.get('numero_serie') or '—'}")
            st.write(f"**Stock** : {int(d.get('stock_disponible') or 0)}")
            st.write(f"**Origine** : {d.get('pays_origine') or '—'}")
        with right:
            st.write("**Dernier achat**")
            if d.get("dernier_achat_prix") is not None:
                st.write(f"{d.get('dernier_achat_fournisseur','?')}")
                try:
                    st.write(_fmt_money(float(d["dernier_achat_prix"])))
                except Exception:
                    st.write(str(d.get("dernier_achat_prix")))
                st.write(str(d.get("dernier_achat_date","")))
            else:
                st.write("— (non disponible)")
            st.write("**Fourn. moins cher**")
            if d.get("prix_moins_cher") is not None:
                try:
                    st.write(f"{d.get('fournisseur_moins_cher','?')} · {_fmt_money(float(d['prix_moins_cher']))}")
                except Exception:
                    st.write(f"{d.get('fournisseur_moins_cher','?')} · {d.get('prix_moins_cher')}")
            else:
                st.write("—")
            st.write("**Plus rapide**")
            if d.get("delai_plus_rapide") is not None:
                st.write(f"{d.get('fournisseur_plus_rapide','?')} · {int(d['delai_plus_rapide'])} j")
            else:
                st.write("—")

        # Bouton Fermer : nettoie l’URL puis rerun
        # (key unique par pièce + suffixe pour éviter collision improbable)
        if st.button("Fermer", key=f"close_piece_{piece}_unique"):
            try:
                st.query_params.clear()
            except Exception:
                st.experimental_set_query_params()
            try:
                st.rerun()
            except Exception:
                pass

    # 4) Modal si dispo, sinon panneau "faux-modal"
    if hasattr(st, "modal"):
        with st.modal(title, key=f"modal_{piece}"):
            _render_content()
    else:
        st.markdown(f"### {title}")
        try:
            with st.container(border=True):
                _render_content()
        except Exception:
            _render_content()
        st.divider()


def _moyenne_mensuelle_par_produit(conn) -> pd.Series:
    """Retourne une Series: index=id_produit, valeur=moyenne mensuelle (historique)."""
    hist = pd.read_sql("SELECT id_produit, quantite, date_commande FROM historique_commandes", conn)
    if hist.empty:
        return pd.Series(dtype=float)
    hist["date_commande"] = pd.to_datetime(hist["date_commande"])
    hist["mois"] = hist["date_commande"].dt.to_period("M")
    par_mois = hist.groupby(["id_produit", "mois"]).quantite.sum().reset_index()
    return par_mois.groupby("id_produit").quantite.mean()

def _season_factor(month_idx_1to12: int, mode: str) -> float:
    """Renvoie un multiplicateur saison (aucune / légère / forte)."""
    if mode == "Aucune":
        return 1.0
    amp = 0.10 if mode == "Légère (±10%)" else 0.30  # Forte (±30%)
    # sinusoïde sur 12 mois, pic autour de mois 6-7
    x = 2 * math.pi * (month_idx_1to12 - 1) / 12.0
    return 1.0 + amp * math.sin(x)

def _min_price_by_piece(conn) -> pd.Series:
    """Renvoie une Series min prix par id_piece pour estimer le coût."""
    f = pd.read_sql("SELECT id_piece, MIN(prix_unitaire) AS min_price FROM fournisseurs GROUP BY id_piece", conn)
    if f.empty:
        return pd.Series(dtype=float)
    return f.set_index("id_piece")["min_price"]

def simulate_annee(
    nb_runs: int = 100,
    croissance_pct: float = 0.0,
    variabilite_pct: float = 15.0,
    saison_mode: str = "Aucune",
    seed: int | None = None
) -> pd.DataFrame:
    """
    Simule 12 mois de demande produits -> usage pièces.
    Retourne un DF: id_piece, nom_piece, usage_moyen, usage_p95, stock_initial, besoin_achat_estime, cout_estime_cheapest.
    """
    if seed is not None:
        random.seed(seed)

    conn = open_conn()

    # Recette produit -> pièces
    recette = pd.read_sql("""
        SELECT p.id_produit, pi.id_piece, pi.nom_piece
        FROM produits p
        JOIN sous_assemblages sa ON p.id_produit = sa.id_produit
        JOIN pieces pi ON sa.id_sous_assemblage = pi.id_sous_assemblage
    """, conn)

    # Base: moyennes mensuelles produits (historique)
    moy = _moyenne_mensuelle_par_produit(conn)

    # Stock initial par pièce
    stocks = pd.read_sql("SELECT id_piece, stock_disponible FROM stocks", conn)
    stocks = stocks.set_index("id_piece")["stock_disponible"] if not stocks.empty else pd.Series(dtype=int)

    # Min prix par pièce (pour coût estimé)
    min_price = _min_price_by_piece(conn)

    conn.close()

    # Préparer listes des pièces
    pieces_info = recette[["id_piece", "nom_piece"]].drop_duplicates().set_index("id_piece")
    piece_ids = pieces_info.index.tolist()

    # Accumulateur des runs
    usages_runs = {pid: [] for pid in piece_ids}

    # Pré-index pour accélérer: mapping produit -> liste (id_piece, nom_piece)
    map_prod_to_pieces = {}
    for pid, grp in recette.groupby("id_produit"):
        map_prod_to_pieces[pid] = grp[["id_piece", "nom_piece"]].values.tolist()

    # Boucles de simulation
    croissance = 1.0 + (croissance_pct / 100.0)
    for _ in range(max(1, nb_runs)):
        usage_annuel_piece = {pid: 0 for pid in piece_ids}

        for m in range(1, 13):  # 12 mois
            s = _season_factor(m, saison_mode)
            # pour chaque produit: génère une demande mensuelle aléatoire
            for id_prod in set(recette["id_produit"]):
                base = float(moy.get(id_prod, 0.0))
                if base <= 0:
                    continue
                mean_m = base * croissance * s
                sigma = max(0.01, mean_m * (variabilite_pct / 100.0))
                dem = random.gauss(mu=mean_m, sigma=sigma)
                q = max(0, int(round(dem)))

                if q == 0:
                    continue
                # convertir produit -> pièces
                for (id_piece, nom_piece) in map_prod_to_pieces.get(id_prod, []):
                    usage_annuel_piece[id_piece] += q

        # enregistrer ce run
        for pid, total in usage_annuel_piece.items():
            usages_runs[pid].append(total)

    # Construire le DataFrame résultat
    rows = []
    for pid in piece_ids:
        runs = usages_runs[pid] or [0]
        sr = pd.Series(runs, dtype="float64")
        usage_moy = float(sr.mean())
        usage_p95 = float(sr.quantile(0.95)) if len(sr) > 1 else usage_moy
        stock_init = int(stocks.get(pid, 0)) if not stocks.empty else 0
        besoin_achat = max(0, usage_moy - stock_init)
        price = float(min_price.get(pid, float("nan"))) if not min_price.empty else float("nan")
        cout_estime = price * usage_moy if not pd.isna(price) else float("nan")
        rows.append({
            "id_piece": pid,
            "nom_piece": pieces_info.loc[pid, "nom_piece"] if pid in pieces_info.index else "",
            "usage_moyen": round(usage_moy, 1),
            "usage_p95": round(usage_p95, 1),
            "stock_initial": stock_init,
            "besoin_achat_estime": round(besoin_achat, 1),
            "prix_unitaire_min": None if pd.isna(price) else price,
            "cout_estime_cheapest": None if pd.isna(cout_estime) else round(cout_estime, 2),
        })

    df = pd.DataFrame(rows)

    # types propres pour Streamlit/Arrow
    for c in ["usage_moyen", "usage_p95", "stock_initial", "besoin_achat_estime", "prix_unitaire_min", "cout_estime_cheapest"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["id_piece", "nom_piece"]] = df[["id_piece", "nom_piece"]].astype("string")

    return df.sort_values("usage_moyen", ascending=False)


def _z_from_service(service_pct: float) -> float:
    """
    Approx du quantile normal pour niveau de service (fill rate ~ no-rupture pendant le délai).
    On mappe aux valeurs courantes.
    """
    table = {
        85: 1.04, 90: 1.28, 92: 1.41, 95: 1.645, 96: 1.751,
        97: 1.881, 98: 2.054, 99: 2.326
    }
    # prend la clé la plus proche
    nearest = min(table.keys(), key=lambda k: abs(k - float(service_pct)))
    return table[nearest]

def _season_factors_array(mode: str) -> list[float]:
    # 12 facteurs mensuels
    if mode == "Aucune":
        return [1.0]*12
    amp = 0.10 if mode.startswith("Légère") else 0.30
    return [1.0 + amp*math.sin(2*math.pi*m/12.0) for m in range(12)]

def _supplier_choice_for_piece(conn, id_piece: str, priority: str) -> tuple[float, int]:
    """
    priority: 'prix' (moins cher), 'delai' (plus rapide), 'equilibre'
    Retourne (prix_unitaire, delai_livraison_mois arrondi haut).
    """
    df = pd.read_sql("SELECT nom_fournisseur, prix_unitaire, delai_livraison_jours FROM fournisseurs WHERE id_piece = ?", conn, params=[id_piece])
    if df.empty:
        return (float("nan"), 1)
    if priority == "prix":
        df = df.sort_values("prix_unitaire")
    elif priority == "delai":
        df = df.sort_values("delai_livraison_jours")
    else:
        # équilibre simple prix/délai
        max_p = df["prix_unitaire"].max() or 1.0
        max_d = df["delai_livraison_jours"].max() or 1.0
        df = df.assign(score=0.5*df["prix_unitaire"]/max_p + 0.5*df["delai_livraison_jours"]/max_d).sort_values("score")
    row = df.iloc[0]
    lead_m = max(1, math.ceil(float(row["delai_livraison_jours"])/30.0))
    return (float(row["prix_unitaire"]), int(lead_m))

def _piece_mu_sigma(conn, croissance_pct: float, variabilite_pct: float) -> pd.DataFrame:
    """
    Renvoie un DF par pièce avec mu_mensuel et sigma_mensuel estimés depuis l'historique produit.
    """
    recette = pd.read_sql("""
        SELECT p.id_produit, pi.id_piece, pi.nom_piece
        FROM produits p
        JOIN sous_assemblages sa ON p.id_produit = sa.id_produit
        JOIN pieces pi ON sa.id_sous_assemblage = pi.id_sous_assemblage
    """, conn)
    hist = pd.read_sql("SELECT id_produit, quantite, date_commande FROM historique_commandes", conn)
    if hist.empty:
        return pd.DataFrame(columns=["id_piece","nom_piece","mu","sigma"])

    hist["date_commande"] = pd.to_datetime(hist["date_commande"])
    hist["mois"] = hist["date_commande"].dt.to_period("M")
    par_mois = hist.groupby(["id_produit","mois"]).quantite.sum().reset_index()
    moy_prod = par_mois.groupby("id_produit").quantite.mean()

    # mu pièce = somme des mu produits qui la composent
    mu_piece = recette.groupby(["id_piece","nom_piece"])["id_produit"].apply(lambda s: sum(moy_prod.get(pid,0.0) for pid in set(s))).reset_index()
    mu_piece = mu_piece.rename(columns={"id_produit":"mu"})
    croissance = 1.0 + (croissance_pct/100.0)
    mu_piece["mu"] = mu_piece["mu"] * croissance

    # sigma ~ variabilité% de mu (approx)
    mu_piece["sigma"] = mu_piece["mu"] * (variabilite_pct/100.0)
    return mu_piece

def simulate_policy_annee(
    service_level: float = 97.0,
    couverture_mois_S: float = 1.0,
    supplier_priority: str = "prix",
    holding_rate_pct_year: float = 20.0,
    variabilite_pct: float = 15.0,
    croissance_pct: float = 0.0,
    saison_mode: str = "Aucune",
    nb_runs: int = 100,
) -> pd.DataFrame:
    """
    Simule un an avec politique Min-Max (s,S):
      s = ROP = mu*L + z*sigma*sqrt(L)
      S = s + couverture_mois_S * mu
    Commande quand position <= s pour remonter à S (arrive après L mois).
    Mesures: fill rate, ruptures, inv. moyen, coût achat, coût possession, coût total.
    """
    z = _z_from_service(service_level)
    season = _season_factors_array(saison_mode)
    hold_monthly = (holding_rate_pct_year/100.0) / 12.0

    conn = open_conn()
    mu_df = _piece_mu_sigma(conn, croissance_pct, variabilite_pct)
    stocks = pd.read_sql("SELECT id_piece, stock_disponible FROM stocks", conn)
    stocks = stocks.set_index("id_piece")["stock_disponible"] if not stocks.empty else pd.Series(dtype=int)

    # Préparer prix & lead time du fournisseur choisi
    price_lead = {}
    for pid in mu_df["id_piece"]:
        price_lead[pid] = _supplier_choice_for_piece(conn, pid, supplier_priority)
    conn.close()

    rows = []
    # Pour chaque pièce, simuler nb_runs et agréger
    for _, r in mu_df.iterrows():
        pid = r["id_piece"]; nomp = r["nom_piece"]; mu = float(r["mu"]); sigma = max(0.01, float(r["sigma"]))
        price, L = price_lead.get(pid, (float("nan"), 1))
        if math.isnan(price):  # pas de fournisseur → ignorer
            continue

        s = mu*L + z*sigma*math.sqrt(L)          # reorder point
        S = s + max(0.0, couverture_mois_S)*mu    # niveau haut

        # agrégats
        total_demand = 0.0
        total_served = 0.0
        sum_end_inv = 0.0
        achat_total = 0.0
        # simulation runs
        for _run in range(max(1, int(nb_runs))):
            on_hand = int(stocks.get(pid, 0))
            on_order = defaultdict(int)  # arrival_month -> qty
            inv_pos = on_hand
            month_end_inv_sum = 0.0

            for m in range(12):
                # réception des commandes du mois m
                if on_order[m] > 0:
                    on_hand += on_order[m]
                    inv_pos += on_order[m]
                    on_order[m] = 0

                # demande du mois m
                demand = max(0, int(round(random.gauss(mu*season[m], sigma))))
                total_demand += demand

                served = min(on_hand, demand)
                total_served += served
                on_hand -= served

                # décision commande
                if inv_pos <= s:
                    # quantité pour remonter à S
                    qty = max(0, int(round(S - inv_pos)))
                    if qty > 0:
                        arrival = m + L
                        if arrival < 12:  # si l'arrivée tombe dans l'année
                            on_order[arrival] += qty
                        inv_pos += qty
                        achat_total += qty * price

                month_end_inv_sum += on_hand

            sum_end_inv += month_end_inv_sum/12.0  # inv moyen sur l'année pour ce run

        # métriques
        fill_rate = (total_served / total_demand) if total_demand > 0 else 1.0
        inv_moyen = sum_end_inv / max(1, int(nb_runs))
        cout_possession = inv_moyen * price * 12 * hold_monthly  # (inv moyen * prix * 12 mois * taux mensuel)
        cout_total = achat_total + cout_possession

        rows.append({
            "id_piece": pid,
            "nom_piece": nomp,
            "mu_mensuel": round(mu, 2),
            "sigma_mensuel": round(sigma, 2),
            "lead_time_mois": L,
            "ROP_s": round(s, 1),
            "S_cible": round(S, 1),
            "stock_initial": int(stocks.get(pid, 0)),
            "fill_rate": round(100*fill_rate, 1),
            "ruptures_estimees": int(round((1 - fill_rate)*total_demand)),
            "inv_moyen": round(inv_moyen, 1),
            "prix_unitaire": round(price, 2),
            "achat_total_$": round(achat_total, 2),
            "cout_possession_$": round(cout_possession, 2),
            "cout_total_$": round(cout_total, 2),
        })

    df = pd.DataFrame(rows).sort_values(["fill_rate","cout_total_$"], ascending=[True, True])
    # types propres
    for c in ["lead_time_mois","stock_initial"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def auto_tune_couverture_par_piece(
    fill_rate_cible_pct: float = 97.0,
    service_level_pct: float = 97.0,
    supplier_priority: str = "prix",
    holding_rate_pct_year: float = 20.0,
    variabilite_pct: float = 15.0,
    croissance_pct: float = 0.0,
    saison_mode: str = "Aucune",
    nb_runs: int = 60,
    cov_min: float = 0.0,
    cov_max: float = 3.0,
    cov_step: float = 0.25,
    reseed_each_step: bool = True,
) -> pd.DataFrame:
    """
    Pour un fill rate cible, trouve par pièce la couverture (mois) minimale qui l’atteint.
    On parcourt un grille de couvertures [cov_min, cov_max] avec step 'cov_step'.
    Retourne un DF avec la couverture conseillée + métriques et coûts.
    """

    # Grille de couvertures (0.00, 0.25, 0.50, …)
    nsteps = int(round((cov_max - cov_min) / cov_step)) + 1
    cov_values = [round(cov_min + i * cov_step, 2) for i in range(max(1, nsteps))]

    best_row = {}        # pid -> (row)  première couverture qui atteint la cible
    best_cov = {}        # pid -> couverture correspondante
    last_row = {}        # pid -> (row)  dernière ligne vue (fallback si jamais atteint)
    pids_seen = set()

    for cov in cov_values:
        # facultatif: re-seed pour que chaque step soit indépendant
        if reseed_each_step:
            random.seed(None)

        df_cov = simulate_policy_annee(
            service_level=float(service_level_pct),
            couverture_mois_S=float(cov),
            supplier_priority=supplier_priority,
            holding_rate_pct_year=float(holding_rate_pct_year),
            variabilite_pct=float(variabilite_pct),
            croissance_pct=float(croissance_pct),
            saison_mode=saison_mode,
            nb_runs=int(nb_runs),
        )

        if df_cov is None or df_cov.empty:
            continue

        df_cov = df_cov.copy()
        df_cov["couverture_mois_S"] = cov

        for _, r in df_cov.iterrows():
            pid = r["id_piece"]
            pids_seen.add(pid)
            last_row[pid] = r  # garde la dernière métrique vue (au cas où on n'atteint jamais la cible)

            if pid not in best_row:
                # Atteint la cible ?
                try:
                    if float(r["fill_rate"]) >= float(fill_rate_cible_pct):
                        best_row[pid] = r
                        best_cov[pid] = cov
                except Exception:
                    pass

        # Petite optimisation: si on a une "best_row" pour tous les pids déjà vus, on peut s'arrêter
        if len(best_row) == len(pids_seen):
            break

    # Construire le résultat final
    out_rows = []
    for pid in pids_seen:
        r = best_row.get(pid, None)
        achieved = True
        if r is None:
            r = last_row.get(pid)  # fallback: la dernière (souvent cov_max)
            achieved = False

        out_rows.append({
            "id_piece": r.get("id_piece", None),
            "nom_piece": r.get("nom_piece", None),
            "lead_time_mois": r.get("lead_time_mois", None),
            "prix_unitaire": r.get("prix_unitaire", None),
            # Couverture conseillée
            "couverture_min_suggeree": best_cov.get(pid, None) if achieved else f">= {cov_values[-1]}",
            "atteint_fill_rate_cible": "Oui" if achieved else "Non",
            "fill_rate_obtenu_%": r.get("fill_rate", None),
            "ROP_s": r.get("ROP_s", None),
            "S_cible": r.get("S_cible", None),
            "stock_initial": r.get("stock_initial", None),
            "inv_moyen": r.get("inv_moyen", None),
            "achat_total_$": r.get("achat_total_$", None),
            "cout_possession_$": r.get("cout_possession_$", None),
            "cout_total_$": r.get("cout_total_$", None),
            "couverture_testee": r.get("couverture_mois_S", None),  # utile si Non atteint
        })

    df_out = pd.DataFrame(out_rows)

    # Types propres pour affichage
    num_cols_money = ["prix_unitaire", "achat_total_$", "cout_possession_$", "cout_total_$"]
    num_cols_int    = ["lead_time_mois", "stock_initial"]
    num_cols_float  = ["fill_rate_obtenu_%", "ROP_s", "S_cible", "inv_moyen"]
    for c in num_cols_money + num_cols_int + num_cols_float:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    return df_out.sort_values(
        by=["atteint_fill_rate_cible", "couverture_min_suggeree", "cout_total_$"],
        ascending=[False, True, True]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 🖥️ UI STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prévision & Achats — Inventaire", layout="wide")

st.title("📦 Prévisions et choix fournisseurs")
st.caption("Base : `inventaire_temp.db` | UI Streamlit (logique conservée)")

# ── Sidebar: paramètres globaux
with st.sidebar:
    st.header("Paramètres")
    strategie = st.radio(
        "Stratégie de sélection", ["prix", "delai", "equilibre"],
        index=0, horizontal=False
    )
    nb_mois = st.number_input("Durée de prévision (mois)", min_value=1, max_value=24, value=3, step=1)
    st.session_state["nb_mois"] = int(nb_mois)

    hausse_pct = st.number_input("Hausse de la demande (%)", min_value=-100.0, max_value=500.0, value=0.0, step=1.0)

    severity = st.slider("Sévérité (coloration)", 0.6, 1.4, 1.0, 0.05,
                        help="0.6 = plus indulgent (plus de vert), 1.4 = plus strict (plus de jaune/rouge)")
    st.session_state["severity"] = float(severity)



    st.markdown("---")
    recherche = st.text_input("Recherche (ID, nom, fournisseur, etc.)")
    mode_recherche = st.radio("Mode de recherche", ["Tous les mots (AND)", "Au moins un (OR)"], index=0)
    st.markdown("---")
    only_alerts = st.checkbox("Afficher seulement 🔴 Rupture / 🟡 À commander (masquer 🟢 OK)", value=False)
    mode_flag = "AND" if mode_recherche.startswith("Tous") else "OR"
    st.markdown("---")
    max_fiches = st.slider("Nb de fiches détails à l’écran", 5, 100, 20, 5)

    with st.expander("🔎 Diagnostic recherche"):
        cols_dbg = [
            "id_piece","nom_piece",
            "fournisseur_1","fournisseur_2","fournisseur_3",
            "prix_1","prix_2","prix_3",
            "delai_1","delai_2","delai_3",
            "statut_stock","fournisseur","prix","delai"
        ]
        tokens = [_strip_accents(t).lower() for t in str(recherche).split() if t.strip()]
        st.write({
            "mots_recherchés": tokens,
            "colonnes_ciblées": cols_dbg
        })


with st.sidebar.expander("🌎 Gestion des pays d'origine"):
    try:
        conn = open_conn()
        df_pays = pd.read_sql(
            "SELECT id_piece, nom_piece, COALESCE(pays_origine,'Canada') AS pays_origine FROM pieces",
            conn
        )
        conn.close()
    except Exception as e:
        st.error(f"Impossible de charger les pièces pour l’édition d’origine: {e}")
        df_pays = pd.DataFrame()

    if not df_pays.empty:
        # Liste de pays suggérés (modifie/augmente au besoin)
        pays_options = [
            "Canada","États-Unis","Mexique","France","Allemagne","Italie",
            "Royaume-Uni","Chine","Inde","Japon","Corée du Sud","Autre"
        ]
        edited_pays = st.data_editor(
            df_pays,
            use_container_width=True,
            hide_index=True,
            column_config={
                "id_piece":  st.column_config.TextColumn("ID pièce", disabled=True),
                "nom_piece": st.column_config.TextColumn("Nom",      disabled=True),
                "pays_origine": st.column_config.SelectboxColumn(
                    "Pays d'origine", options=pays_options, required=True
                ),
            },
            key="editor_pays_origine"
        )
        if st.button("💾 Enregistrer les origines"):
            try:
                conn = open_conn()
                cur = conn.cursor()
                for _, r in edited_pays.iterrows():
                    cur.execute(
                        "UPDATE pieces SET pays_origine=? WHERE id_piece=?",
                        (str(r["pays_origine"]).strip(), str(r["id_piece"]).strip())
                    )
                conn.commit(); conn.close()
                st.success("Origines mises à jour ✅")
            except Exception as e:
                st.error(f"Erreur lors de l’enregistrement: {e}")
    else:
        st.info("Aucune pièce trouvée dans la table `pieces`.")


# ── Choix source des prévisions
onglets = st.tabs([
    "🧮 Prévision historique",
    "✍️ Prévision manuelle (par produit)",
    "🧾 Bon de commande",
    "🔮 Simulation annuelle",
    "📅 Calendrier de production",   # <<< AJOUT
    "🛒 Passer une commande"  # <<< AJOUT
])


# ============ ONGLET 1 : Prévision historique ============
with onglets[0]:
    st.subheader("Prévision via historique de commandes")
    besoins_hist = calculer_previsions_historique(nb_mois=nb_mois, hausse_pct=hausse_pct)
    df_resultat_hist = choisir_fournisseur_multiple(besoins_hist, strategie)

    # (Optionnel) Préfiltrer via FTS si champ recherche rempli
    if recherche and recherche.strip():
        try:
            ids_fts = search_ids_from_fts(recherche)
            if ids_fts:
                df_resultat_hist = df_resultat_hist[df_resultat_hist["id_piece"].astype(str).isin(ids_fts)]
        except Exception:
            pass


    # Filtre de recherche
    df_aff = df_resultat_hist.copy()
    cols_rech = [
        "id_piece", "nom_piece",
        "fournisseur_1", "fournisseur_2", "fournisseur_3",
        "prix_1", "prix_2", "prix_3",
        "delai_1", "delai_2", "delai_3",
        "statut_stock",
    ]
    mask = make_search_mask(df_aff, recherche, cols_rech, mode=mode_flag)
    df_aff = df_aff[mask]

    # Filtre statut 🔴/🟡
    if only_alerts and "statut_stock" in df_aff.columns:
        df_aff = df_aff[df_aff["statut_stock"].isin(["🔴 Rupture", "🟡 À commander"])]


    # Compteur de lignes
    st.write(f"**Lignes**: {0 if df_aff is None else len(df_aff)}")

    if df_aff.empty:
        st.warning("Aucun besoin généré (historique peut-être vide ou paramètres trop restrictifs).")
    else:
        # --- Style par statut (couleurs foncées lisibles) ---
        DARK_RED    = "#B71C1C"  # rouge foncé
        DARK_GREEN  = "#1B5E20"  # vert foncé
        DARK_YELLOW = "#F57F17"  # jaune foncé

        def color_row(row):
            bg = ""
            fg = "black"
            if row.get("statut_stock") == "🔴 Rupture":
                bg, fg = DARK_RED, "white"
            elif row.get("statut_stock") == "🟡 À commander":
                bg, fg = DARK_YELLOW, "black"
            elif row.get("statut_stock") == "🟢 OK":
                bg, fg = DARK_GREEN, "white"
            style = f"background-color: {bg}; color: {fg}; font-weight: 600"
            return [style] * len(row)

        # Casts numériques sûrs (pour tri/affichage)
        for c in ["besoin_total","stock_disponible","prix_1","prix_2","prix_3","delai_1","delai_2","delai_3"]:
            if c in df_aff.columns:
                df_aff[c] = pd.to_numeric(df_aff[c], errors="coerce")

        money_cols1 = ["prix_1", "prix_2", "prix_3", "cout_estime_fournisseur1"]
        int_cols1   = ["besoin_total", "stock_disponible", "delai_1", "delai_2", "delai_3"]

        # 🔗 Colonne cliquable vers ?piece=<ID>
        df_aff = df_aff.copy()
        df_aff.insert(0, "details_link", df_aff["id_piece"].astype(str).apply(lambda pid: f"?piece={pid}"))

        # Tableau coloré + lien "Détails"
        if hasattr(st.column_config, "LinkColumn"):
            st.dataframe(
                style_table(df_aff, money_cols1, int_cols1, color_row),
                use_container_width=True,
                column_order=["details_link"] + [c for c in df_aff.columns if c != "details_link"],
                column_config={
                    "details_link": st.column_config.LinkColumn(
                        "🔍 Détails",
                        help="Voir la fiche rapide de la pièce",
                        display_text="🔍 Détails"
                    )
                },
            )
        else:
            # Fallback si version Streamlit sans LinkColumn
            st.dataframe(style_table(df_aff, money_cols1, int_cols1, color_row), use_container_width=True)
            pid_sel = st.selectbox(
                "🔍 Voir les détails d’une pièce :",
                ["—"] + df_aff["id_piece"].astype(str).tolist(),
                key="details_onglet1"
            )
            if pid_sel and pid_sel != "—":
                st.experimental_set_query_params(piece=pid_sel)

        # Ouvre la modale si l’URL contient ?piece=...
        show_piece_modal_if_requested()

        # 📤 Export CSV (garde ce que tu avais)
        st.download_button(
            "📤 Exporter (CSV)",
            data=df_aff.to_csv(index=False).encode("utf-8"),
            file_name=f"previsions_historique_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )



# ============ ONGLET 2 : Prévision manuelle ============
with onglets[1]:
    st.subheader("Saisir des prévisions par produit")
    conn = open_conn()
    produits = pd.read_sql("SELECT id_produit, nom_produit FROM produits", conn)
    conn.close()

    if produits.empty:
        st.warning("Aucun produit dans la base.")
    else:
        df_prev = produits.copy()
        df_prev["quantite_prevue"] = 0
        st.markdown("Modifie la colonne **quantite_prevue** selon tes besoins, puis applique.")
        edited = st.data_editor(
            df_prev,
            num_rows="fixed",
            use_container_width=True,
            key="editor_prev"
        )

        if st.button("Appliquer les prévisions manuelles"):
            try:
                edited["quantite_prevue"] = pd.to_numeric(edited["quantite_prevue"], errors="coerce").fillna(0).astype(int)
            except Exception:
                st.error("Valeurs invalides dans la colonne 'quantite_prevue'.")
            else:
                besoins_man = calculer_previsions(edited[["id_produit", "quantite_prevue"]], hausse_pct=hausse_pct)
                df_resultat_man = choisir_fournisseur_multiple(besoins_man, strategie)

                # (Optionnel) Préfiltrer via FTS
                if recherche and recherche.strip():
                    try:
                        ids_fts = search_ids_from_fts(recherche)
                        if ids_fts:
                            df_resultat_man = df_resultat_man[df_resultat_man["id_piece"].astype(str).isin(ids_fts)]
                    except Exception:
                        pass


                # Filtrer par recherche
                df_aff2 = df_resultat_man.copy()
                cols_rech = [
                    "id_piece", "nom_piece",
                    "fournisseur_1", "fournisseur_2", "fournisseur_3",
                    "prix_1", "prix_2", "prix_3",
                    "delai_1", "delai_2", "delai_3",
                    "statut_stock",
                ]
                mask2 = make_search_mask(df_aff2, recherche, cols_rech, mode=mode_flag)
                df_aff2 = df_aff2[mask2]

                # Filtre statut 🔴/🟡
                if only_alerts and "statut_stock" in df_aff2.columns:
                    df_aff2 = df_aff2[df_aff2["statut_stock"].isin(["🔴 Rupture", "🟡 À commander"])]


                if df_aff2.empty:
                    st.warning("Aucune ligne à afficher (prévisions = 0 partout ?).")
                else:
                    # --- Couleurs foncées lisibles (locales à l’onglet 2) ---
                    DARK_RED    = "#B71C1C"
                    DARK_GREEN  = "#1B5E20"
                    DARK_YELLOW = "#F57F17"

                    def color_row2(row):
                        bg = ""; fg = "black"
                        if row.get("statut_stock") == "🔴 Rupture":
                            bg, fg = DARK_RED, "white"
                        elif row.get("statut_stock") == "🟡 À commander":
                            bg, fg = DARK_YELLOW, "black"
                        elif row.get("statut_stock") == "🟢 OK":
                            bg, fg = DARK_GREEN, "white"
                        return [f"background-color: {bg}; color: {fg}; font-weight: 600"] * len(row)

                    # Casts numériques sûrs
                    for c in ["besoin_total","stock_disponible","prix_1","prix_2","prix_3","delai_1","delai_2","delai_3"]:
                        if c in df_aff2.columns:
                            df_aff2[c] = pd.to_numeric(df_aff2[c], errors="coerce")

                    # === AJOUT 1 : manque à commander
                    df_aff2["manque_a_commander"] = np.maximum(
                        0,
                        pd.to_numeric(df_aff2.get("besoin_total"), errors="coerce").fillna(0)
                        - pd.to_numeric(df_aff2.get("stock_disponible"), errors="coerce").fillna(0)
                    ).astype(int)

                    # === AJOUT 2 : qte_par_produit (recette)
                    recette_qpp = _get_recette_with_qpp()[["id_piece", "qte_par_produit"]].drop_duplicates()
                    df_aff2 = df_aff2.merge(recette_qpp, on="id_piece", how="left")
                    df_aff2["qte_par_produit"] = pd.to_numeric(df_aff2["qte_par_produit"], errors="coerce").fillna(1).astype(int)

                    # colonnes à styler
                    money_cols2 = [c for c in ["prix_1","prix_2","prix_3","cout_estime_fournisseur1"] if c in df_aff2.columns]
                    int_cols2   = [c for c in ["besoin_total","stock_disponible","manque_a_commander","qte_par_produit","delai_1","delai_2","delai_3"] if c in df_aff2.columns]

                    # 🔗 colonne cliquable vers ?piece=<ID>
                    df_aff2 = df_aff2.copy()
                    df_aff2.insert(0, "details_link", df_aff2["id_piece"].astype(str).apply(lambda pid: f"?piece={pid}"))

                    if hasattr(st.column_config, "LinkColumn"):
                        st.dataframe(
                            style_table(df_aff2, money_cols2, int_cols2, color_row2),
                            use_container_width=True,
                            column_order=["details_link"] + [c for c in df_aff2.columns if c != "details_link"],
                            column_config={
                                "details_link": st.column_config.LinkColumn(
                                    "🔍 Détails", help="Voir la fiche rapide de la pièce", display_text="🔍 Détails"
                                )
                            },
                        )
                    else:
                        # Fallback si ta version de Streamlit n'a pas LinkColumn
                        st.dataframe(style_table(df_aff2, money_cols2, int_cols2, color_row2), use_container_width=True)
                        pid_sel = st.selectbox(
                            "🔍 Voir les détails d’une pièce :",
                            ["—"] + df_aff2["id_piece"].astype(str).tolist(),
                            key="details_onglet2"
                        )
                        if pid_sel and pid_sel != "—":
                            st.experimental_set_query_params(piece=pid_sel)

                    # 👉 Ouvre la modale si ?piece=... est dans l’URL
                    show_piece_modal_if_requested()

                    # Résumé & export
                    total_pieces2  = int(pd.to_numeric(df_aff2["besoin_total"], errors="coerce").fillna(0).sum())
                    total_manque2  = int(pd.to_numeric(df_aff2.get("manque_a_commander"), errors="coerce").fillna(0).sum()) if "manque_a_commander" in df_aff2.columns else 0
                    prix_valides2  = pd.to_numeric(df_aff2.get("prix_1"), errors="coerce").dropna() if "prix_1" in df_aff2.columns else pd.Series(dtype=float)
                    cout_moyen2    = round(prix_valides2.mean(), 2) if not prix_valides2.empty else None
                    st.info(
                        f"🧾 **Total pièces**: {total_pieces2}  |  "
                        f"❗ **Manque à commander**: {total_manque2}  |  "
                        f"💰 **Prix moyen fournisseur 1**: {cout_moyen2 if cout_moyen2 is not None else '-'} $"
                    )

                    st.download_button(
                        "📤 Exporter (CSV)",
                        data=df_aff2.to_csv(index=False).encode("utf-8"),
                        file_name=f"previsions_manuelles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )


# ============ ONGLET 3 : Bon de commande ============
with onglets[2]:
    st.subheader("Bon de commande suggéré (1 fournisseur par pièce)")
    st.caption("Applique les filtres puis génère une commande basée sur l’historique (ou saisis d’abord des prévisions manuelles dans l’onglet précédent).")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        filtre_fournisseur = st.text_input("Fournisseur spécifique (optionnel)")
    with colf2:
        filtre_prix_max = st.number_input("Prix unitaire max (optionnel)", min_value=0.0, value=0.0, step=1.0)
        prix_max_effectif = filtre_prix_max if filtre_prix_max > 0 else None
    with colf3:
        filtre_delai_max = st.number_input("Délai max (jours, optionnel)", min_value=0, value=0, step=1)
        delai_max_effectif = int(filtre_delai_max) if filtre_delai_max > 0 else None

    show_group_by_supplier = st.checkbox("Afficher le résumé par fournisseur (option)", value=False)

    st.markdown("---")
    colt1, colt2, colt3 = st.columns([1,1,1])
    with colt1:
        tampon_jours = st.number_input(
            "Tampon sécurité (jours de conso)", min_value=0, max_value=60, value=7, step=1
        )
    with colt2:
        mode_agenda = st.selectbox(
            "Mode de revue",
            ["Continu (ROP)", "Hebdo (lundi)", "Fixe (1er et 15)"],
            index=0
        )
    with colt3:
        # si tu as déjà ce widget dans la section (s,S) de l’onglet 4, c’est juste pour récupérer la valeur ici
        couverture_mois_S = float(st.session_state.get("couverture_mois_S", 1.0))

    # Source besoins : on repart de l'historique pour la commande
    besoins_cmd = calculer_previsions_historique(nb_mois=nb_mois, hausse_pct=hausse_pct)
    df_commande = choisir_fournisseur_unique(
        besoins_cmd, strategie,
        filtre_fournisseur=filtre_fournisseur.strip(),
        filtre_prix_max=prix_max_effectif,
        filtre_delai_max=delai_max_effectif
    )

    # 👇 Injecter le stock réel (évite 0 partout)
    try:
        df_stk = _get_stock_df()[["id_piece", "stock_disponible"]]
        # on supprime une éventuelle colonne fantôme avant le merge
        df_commande = df_commande.drop(columns=["stock_disponible"], errors="ignore") \
                                .merge(df_stk, on="id_piece", how="left")
    except Exception:
        pass


    # (Optionnel) Préfiltrer via FTS
    if recherche and recherche.strip():
        try:
            ids_fts = search_ids_from_fts(recherche)
            if ids_fts:
                df_commande = df_commande[df_commande["id_piece"].astype(str).isin(ids_fts)]
        except Exception:
            pass


    # ----- Affichage -----
    if df_commande.empty:
        st.warning("Aucune ligne de commande (filtres trop restrictifs, ou besoins nuls).")
    else:
        # conversions sûres pour agrégations/affichage
        for c in ["besoin_total", "prix", "delai", "cout_total"]:
            if c in df_commande.columns:
                df_commande[c] = pd.to_numeric(df_commande[c], errors="coerce")

            # Assurer la présence/cast du stock
        if "stock_disponible" in df_commande.columns:
            df_commande["stock_disponible"] = pd.to_numeric(df_commande["stock_disponible"], errors="coerce").fillna(0)
        else:
            df_commande["stock_disponible"] = 0

        # ➕ manque à commander = max(0, besoin_total - stock_disponible)
        stock_col  = pd.to_numeric(df_commande["stock_disponible"], errors="coerce").fillna(0)
        besoin_col = pd.to_numeric(df_commande["besoin_total"], errors="coerce").fillna(0)
        df_commande["manque_a_commander"] = np.maximum(0, besoin_col - stock_col).astype(int)
   

        total_cout = round(float(df_commande["cout_total"].sum()), 2)
        delai_moy = round(float(df_commande["delai"].mean()), 1)

        st.success(f"💰 **Coût total estimé** : {total_cout} $   |   ⏱️ **Délai moyen estimé** : {delai_moy} jours")

        # ℹ️ Résumé manque total (toutes pièces confondues)
        total_manque_cmd = int(pd.to_numeric(df_commande["manque_a_commander"], errors="coerce").fillna(0).sum())
        st.info(f"❗ **Manque à commander (total)** : {total_manque_cmd} pièces")


        AUJ = datetime.now().date()

        def _prochaine_date_selon_mode(base_date: datetime.date, mode: str) -> datetime.date:
            d = base_date
            if mode == "Continu (ROP)":
                return d
            if mode == "Hebdo (lundi)":
                # prochain lundi (0=lundi)
                return d + timedelta(days=(7 - d.weekday()) % 7)
            if mode == "Fixe (1er et 15)":
                if d.day <= 1:
                    return d.replace(day=1)
                if d.day <= 15:
                    return d.replace(day=15)
                # sinon 1er du mois suivant
                month = d.month + 1 if d.month < 12 else 1
                year  = d.year + 1 if month == 1 else d.year
                return datetime(year, month, 1).date()
            return d

        def _safe_num(x, default=0.0):
            try:
                v = float(x)
                if np.isnan(v):
                    return float(default)
                return v
            except Exception:
                return float(default)

        # On suppose que df_commande contient au moins: id_piece, fournisseur, prix, delai (jours), besoin_total (période), stock_disponible (si dispo)
        # On va deviner une conso mensuelle si tu n'as pas de colonne dédiée:
        NB_MOIS_HIST = max(1, int(globals().get("nb_mois", 6)))  # nb_mois utilisé dans tes prévisions historiques

        rows = []
        for _, r in df_commande.iterrows():
            pid = str(r.get("id_piece",""))
            four = str(r.get("fournisseur",""))
            prix = _safe_num(r.get("prix"), 0)
            delai_j = int(_safe_num(r.get("delai"), 0))
            stock  = _safe_num(r.get("stock_disponible"), 0)
            besoin_periode = _safe_num(r.get("besoin_total"), 0)

            # Estimation conso (pièces / jour)
            # priorité: si tu as une colonne 'conso_mensuelle_estimee' -> utilise-la; sinon derive de besoin_total / nb_mois
            conso_mois = _safe_num(r.get("conso_mensuelle_estimee"), besoin_periode / NB_MOIS_HIST)
            conso_jour = max(conso_mois / 30.0, 1e-6)  # évite division par 0

            # s (ROP en unités) ≈ demande pendant délai + tampon en jours
            s_units = conso_jour * (delai_j + tampon_jours)

            # jours avant d’atteindre s
            jours_avant_rop = (stock - s_units) / conso_jour  # peut être < 0 si déjà sous s
            if np.isinf(jours_avant_rop) or np.isnan(jours_avant_rop):
                jours_avant_rop = -9999  # force URGENT si données incohérentes

            # Décalage selon le niveau de couverture (plus de lousse si stock >> s)
            ratio_cov = (stock / max(1e-6, s_units)) if s_units > 0 else 0.0
            dec_lousse = 5 if ratio_cov >= 1.5 else (2 if ratio_cov >= 1.2 else 0)
            jours_base = int(np.floor(jours_avant_rop)) - dec_lousse
            jitter = (sum(ord(c) for c in pid) % (3 if ratio_cov >= 1.2 else 1))
            date_dernier_commande = AUJ if jours_base <= 0 else AUJ + timedelta(days=jours_base)
            date_suggeree = _prochaine_date_selon_mode(date_dernier_commande + timedelta(days=jitter), mode_agenda)
            date_reception = date_suggeree + timedelta(days=delai_j)



            # Quantité s,S (facultatif) -> vise S global en mois si dispo
            # Projection du stock à la réception: stock - conso durant le délai
            stock_proj_recep = stock - conso_jour * delai_j
            S_units = conso_mois * couverture_mois_S
            q_suggeree = max(0.0, S_units - stock_proj_recep)  # remonte au niveau S

            urgence = "👍 OK"
            if jours_avant_rop <= 0:
                urgence = "🚨 URGENT"
            elif jours_avant_rop <= 7:
                urgence = "⚠️ < 7 jours"

            rows.append({
                "id_piece": pid,
                "fournisseur": four,
                "prix_unitaire": round(prix, 2),
                "delai_jours": delai_j,
                "stock_actuel": round(stock, 2),
                "conso_mois_estimee": round(conso_mois, 2),
                "ROP_s_unites": round(s_units, 2),
                "jours_avant_rop": round(jours_avant_rop, 1),
                "date_commander_au_plus_tard": date_dernier_commande,
                "date_commande_recommandee": date_suggeree,
                "date_reception_estimee": date_reception,
                "q_suggeree_sS": int(np.ceil(q_suggeree)),
                "statut_urgence": urgence,
                "cout_total_estime": round(prix * int(np.ceil(q_suggeree)), 2)
            })

        df_quand = pd.DataFrame(rows)

        st.markdown("#### 🗓️ Calendrier de réappro (propositions)")
        if df_quand.empty:
            st.info("Pas de calendrier à afficher.")
        else:
            # Tri: urgent d’abord, puis par date recommandée
            ordre_urgence = {"🚨 URGENT": 0, "⚠️ < 7 jours": 1, "👍 OK": 2}
            df_quand["ord"] = df_quand["statut_urgence"].map(ordre_urgence).fillna(9)
            df_quand = df_quand.sort_values(["ord","date_commande_recommandee","id_piece"]).drop(columns=["ord"])

            # Vue détaillée
            money_cols_q = ["prix_unitaire","cout_total_estime"]
            int_cols_q   = ["delai_jours","stock_actuel","q_suggeree_sS","ROP_s_unites"]
            st.dataframe(style_table(df_quand, money_cols_q, int_cols_q), use_container_width=True)

            # Vue agrégée par date (et fournisseur)
            st.markdown("##### 📅 Agrégat par date de commande recommandée")
            agg = (df_quand
                .groupby(["date_commande_recommandee"], as_index=False)
                .agg(lignes=("id_piece","count"),
                        pieces=("q_suggeree_sS","sum"),
                        cout=("cout_total_estime","sum")))
            st.dataframe(style_table(agg, ["cout"], ["lignes","pieces"]), use_container_width=True)


            def build_ics_from_agg(df_dates: pd.DataFrame, titre_prefix="Passer commande"):
                # Crée un ICS minimal (EVENT par date)
                lines = ["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//Probewell Prévision//FR"]
                for _, r in df_dates.iterrows():
                    d: datetime.date = r["date_commande_recommandee"]
                    dt = datetime(d.year, d.month, d.day, 9, 0, 0)  # 09:00 local
                    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    dtstart = dt.strftime("%Y%m%dT%H%M%S")
                    summary = f"{titre_prefix}: {int(r['lignes'])} lignes (~{round(float(r['cout']),2)}$)"
                    uid = f"{dtstart}-{int(r['lignes'])}@probewell"
                    lines += [
                        "BEGIN:VEVENT",
                        f"DTSTAMP:{dtstamp}",
                        f"DTSTART:{dtstart}",
                        f"SUMMARY:{summary}",
                        "END:VEVENT"
                    ]
                lines.append("END:VCALENDAR")
                return "\r\n".join(lines).encode("utf-8")

            if not agg.empty:
                ics_bytes = build_ics_from_agg(agg)
                st.download_button(
                    "📆 Exporter l’agenda (.ics)",
                    data=ics_bytes,
                    file_name=f"agenda_commandes_{datetime.now().strftime('%Y%m%d_%H%M')}.ics",
                    mime="text/calendar"
                )


        # --- 🌍 Répartition par région d'origine (bon de commande courant) ---
        mode_region_cmd = st.radio(
            "Base du calcul",
            ["Par coût (recommandé)", "Par unités"],
            index=0, horizontal=True, key="kpi_region_mode_cmd"
        )

        try:
            conn = open_conn()
            df_pays = pd.read_sql("SELECT id_piece, COALESCE(pays_origine,'Canada') AS pays_origine FROM pieces", conn)
            conn.close()
            df_cmd_reg = df_commande.merge(df_pays, on="id_piece", how="left")
            df_cmd_reg["pays_origine"] = df_cmd_reg["pays_origine"].fillna("Canada")
        except Exception as e:
            st.error(f"Impossible de calculer la répartition régionale du bon de commande : {e}")
            df_cmd_reg = df_commande.copy()
            df_cmd_reg["pays_origine"] = "Canada"

        df_cmd_reg["region"] = df_cmd_reg["pays_origine"].apply(pays_to_region)

        col_metric_cmd = "cout_total" if "coût" in mode_region_cmd.lower() else "besoin_total"
        vals_cmd = pd.to_numeric(df_cmd_reg[col_metric_cmd], errors="coerce").fillna(0)

        totaux_cmd = df_cmd_reg.assign(val=vals_cmd).groupby("region", as_index=False)["val"].sum()
        ordre = ["Amérique","Europe","Asie","Autre"]
        totaux_cmd = totaux_cmd.set_index("region").reindex(ordre, fill_value=0.0)["val"]
        total_all_cmd = float(totaux_cmd.sum())

        pct_cmd = (totaux_cmd / total_all_cmd * 100.0) if total_all_cmd > 0 else totaux_cmd*0
        pct_cmd_amerique = float(pct_cmd.get("Amérique", 0.0))

        # Affichage des 3 régions + focus Amérique
        r1, r2, r3 = st.columns(3)
        r1.metric("🇺🇸🇨🇦🇲🇽 Amérique", f"{pct_cmd.get('Amérique',0):.1f}%")
        r2.metric("🇪🇺 Europe",        f"{pct_cmd.get('Europe',0):.1f}%")
        r3.metric("🌏 Asie",           f"{pct_cmd.get('Asie',0):.1f}%")

        st.progress(min(max(int(round(pct_cmd_amerique)), 0), 100))
        st.info(f"🎯 **% Amérique (Canada + USA + Mexique)** : {pct_cmd_amerique:.1f}%")


        # 👇 rendre la variable dispo même si la case n'est pas cochée
        resume_fourn = None

        # ✅ Résumé optionnel par fournisseur (s’affiche seulement si la case est cochée)
        if show_group_by_supplier:
            resume_fourn = (
                df_commande.groupby("fournisseur", as_index=False)
                        .agg(
                            lignes=("id_piece", "count"),
                            quantite=("besoin_total", "sum"),
                            manque_total=("manque_a_commander", "sum"),
                            cout_total=("cout_total", "sum"),
                            prix_moyen_unitaire=("prix", "mean"),
                            delai_moyen_jours=("delai", "mean")
                        )
                        .sort_values("cout_total", ascending=False, na_position="last")
            )

            st.markdown("#### 📊 Résumé par fournisseur")

            money_cols_res = [c for c in ["cout_total","prix_moyen_unitaire"] if c in resume_fourn.columns]
            int_cols_res   = [c for c in ["quantite","manque_total","lignes","delai_moyen_jours"] if c in resume_fourn.columns]


            st.dataframe(style_table(resume_fourn, money_cols_res, int_cols_res), use_container_width=True)

            st.download_button(
                "📤 Exporter le résumé par fournisseur (CSV)",
                data=resume_fourn.to_csv(index=False).encode("utf-8"),
                file_name="resume_par_fournisseur.csv",
                mime="text/csv"
            )


        # ───────────────────────────────────────────────────────────────────────────
        # 🤝 Focus fournisseur (négociation)
        # ───────────────────────────────────────────────────────────────────────────
        st.markdown("#### 🤝 Focus fournisseur (négociation)")

        four_list = sorted(df_commande.get("fournisseur", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if not four_list:
            st.info("Aucun fournisseur présent dans le bon de commande.")
        else:
            colfA, colfB, colfC, colfD = st.columns([2,1,1,1])
            with colfA:
                four_sel = st.selectbox("Fournisseur", options=["—"] + four_list, index=0, key="focus_four")
            with colfB:
                annualiser = st.checkbox("Projeter en annuel", value=True, key="focus_four_ann")
            with colfC:
                incl_ok = st.checkbox("Inclure pièces OK (manque=0)", value=False, key="focus_four_ok")
            with colfD:
                base_depense = st.selectbox("Base dépense", ["prix × quantités", "cout_total (si présent)"], index=0, key="focus_four_base")

            if four_sel and four_sel != "—":
                q_col = "manque_a_commander" if "manque_a_commander" in df_commande.columns else "besoin_total"

                dff = df_commande[df_commande["fournisseur"].astype(str) == four_sel].copy()

                # Casts sûrs
                for c in [q_col, "prix", "cout_total", "delai", "stock_disponible"]:
                    if c in dff.columns:
                        dff[c] = pd.to_numeric(dff[c], errors="coerce").fillna(0)

                # Si "prix" absent mais cout_total et quantité dispo, approx prix
                if "prix" not in dff.columns and "cout_total" in dff.columns and q_col in dff.columns:
                    q_nonzero = dff[q_col].replace(0, np.nan)
                    dff["prix"] = (dff["cout_total"] / q_nonzero).fillna(0)

                # Filtrer pièces OK si on ne veut que le manque
                if not incl_ok and q_col in dff.columns:
                    dff = dff[dff[q_col] > 0]

                # Quantité période
                dff["qte_achat"] = dff[q_col].astype(float)

                # Annualisation
                nb_mois_conf = int(globals().get("nb_mois", 12))
                facteur = 12.0 / max(1, nb_mois_conf)
                dff["qte_achat_proj"] = (dff["qte_achat"] * (facteur if annualiser else 1)).round(0).astype(int)

                # Dépense estimée
                if base_depense.startswith("cout_total") and "cout_total" in dff.columns:
                    dff["depense_estimee"]   = dff["cout_total"].astype(float)
                    dff["depense_proj_annee"] = (dff["depense_estimee"] * (facteur if annualiser else 1)).round(2)
                else:
                    prix = pd.to_numeric(dff.get("prix", 0), errors="coerce").fillna(0)
                    dff["depense_estimee"]   = (prix * dff["qte_achat"]).round(2)
                    dff["depense_proj_annee"] = (prix * dff["qte_achat_proj"]).round(2)

                # KPIs
                total_lignes    = len(dff)
                total_qty       = int(dff["qte_achat"].sum())
                total_spend     = float(dff["depense_estimee"].sum())
                avg_price       = float(pd.to_numeric(dff.get("prix", 0), errors="coerce").replace(0, np.nan).mean() or 0)
                avg_delai       = float(pd.to_numeric(dff.get("delai", 0), errors="coerce").replace(0, np.nan).mean() or 0)
                total_qty_ann   = int(dff["qte_achat_proj"].sum()) if annualiser else None
                total_spend_ann = float(dff["depense_proj_annee"].sum()) if annualiser else None

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Lignes concernées", f"{total_lignes}")
                k2.metric("Qté à acheter", f"{total_qty:,}")
                k3.metric("Prix moyen", f"{avg_price:.2f} $")
                k4.metric("Délai moyen", f"{avg_delai:.1f} j")

                if annualiser:
                    st.info(f"📅 **Projection annuelle** — Qté: {total_qty_ann:,} | Dépense: {total_spend_ann:,.2f} $")
                    st.caption(f"Sur la période — Dépense: {total_spend:,.2f} $")
                else:
                    st.info(f"💰 **Dépense estimée (période)** : {total_spend:,.2f} $")

                # Tableau focus
                money_cols_focus = [c for c in ["prix","depense_estimee","depense_proj_annee","cout_total"] if c in dff.columns]
                int_cols_focus   = [c for c in ["qte_achat","qte_achat_proj","stock_disponible","delai"] if c in dff.columns]
                st.dataframe(style_table(dff, money_cols_focus, int_cols_focus), use_container_width=True)

                st.download_button(
                    "📤 Exporter le focus fournisseur (CSV)",
                    data=dff.to_csv(index=False).encode("utf-8"),
                    file_name=f"focus_fournisseur_{four_sel}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        # ───────────────────────────────────────────────────────────────────────────


        # tableau détaillé (avec lien cliquable vers la modale)
        money_cols_cmd = [c for c in ["prix","cout_total"] if c in df_commande.columns]
        int_cols_cmd   = [c for c in ["besoin_total","stock_disponible","manque_a_commander","delai"] if c in df_commande.columns]


        df_commande = df_commande.copy()
        df_commande.insert(0, "details_link", df_commande["id_piece"].astype(str).apply(lambda pid: f"?piece={pid}"))

        if hasattr(st.column_config, "LinkColumn"):
            st.dataframe(
                style_table(df_commande, money_cols_cmd, int_cols_cmd),
                use_container_width=True,
                column_order=["details_link"] + [c for c in df_commande.columns if c != "details_link"],
                column_config={
                    "details_link": st.column_config.LinkColumn(
                        "🔍 Détails", help="Voir la fiche rapide de la pièce", display_text="🔍 Détails"
                    )
                },
            )
        else:
            # Fallback si LinkColumn indisponible
            st.dataframe(style_table(df_commande, money_cols_cmd, int_cols_cmd), use_container_width=True)
            pid_sel = st.selectbox(
                "🔍 Voir les détails d’une pièce :",
                ["—"] + df_commande["id_piece"].astype(str).tolist(),
                key="details_onglet3"
            )
            if pid_sel and pid_sel != "—":
                st.experimental_set_query_params(piece=pid_sel)

        # 👉 Ouvre la modale si ?piece=... est dans l’URL
        show_piece_modal_if_requested()

        # ⬇⬇⬇ Boutons d'export côte à côte (Excel + CSV) ⬇⬇⬇
        col_xlsx, col_csv = st.columns(2)

        with col_xlsx:
            # nécessite la fonction build_commande_excel(...) et `from io import BytesIO` en haut du fichier
            excel_bytes = build_commande_excel(df_commande, resume_fourn)
            st.download_button(
                "📥 Exporter le bon de commande (Excel .xlsx)",
                data=excel_bytes,
                file_name=f"bon_de_commande_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col_csv:
            st.download_button(
                "📤 Exporter le bon de commande (CSV)",
                data=df_commande.to_csv(index=False).encode("utf-8"),
                file_name=f"bon_de_commande_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )



# ============ ONGLET 4 : Simulation annuelle ============
with onglets[3]:
    st.subheader("🔮 Simulation d'une année — usage total des pièces")

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        nb_runs = st.number_input("Itérations (Monte Carlo)", min_value=10, max_value=1000, value=200, step=10)
    with col2:
        croissance_pct = st.number_input("Croissance annuelle (%)", min_value=-100.0, max_value=300.0, value=0.0, step=1.0)
    with col3:
        variabilite_pct = st.number_input("Variabilité mensuelle (%)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
    with col4:
        saison_mode = st.selectbox("Saisonnalité", ["Aucune", "Légère (±10%)", "Forte (±30%)"], index=0)


    randomize_sim = st.checkbox("Aléatoire à chaque exécution (simulation annuelle)", value=True)
    run = st.button("Lancer la simulation")


    if run:
        with st.spinner("Simulation en cours…"):
            seed_val = None if randomize_sim else 42  # None = nouvelle graine à chaque exécution
            df_sim = simulate_annee(
                nb_runs=int(nb_runs),
                croissance_pct=float(croissance_pct),
                variabilite_pct=float(variabilite_pct),
                saison_mode=saison_mode,
                seed=seed_val
            )


        # Filtre recherche réutilisé si tu en as un global
        try:
            cols_rech_sim = ["id_piece","nom_piece"]
            mask_sim = make_search_mask(df_sim, recherche, cols_rech_sim, mode=mode_flag)  # si mode_flag existe
            df_sim = df_sim[mask_sim]
        except Exception:
            pass

        # Affichage formaté
        money_cols_sim = ["prix_unitaire_min", "cout_estime_cheapest"]
        int_cols_sim   = ["stock_initial"]
        st.dataframe(
            style_table(df_sim, money_cols_sim, int_cols_sim),
            use_container_width=True
        )

        # Résumé
        total_usage = int(round(df_sim["usage_moyen"].sum()))
        total_besoin = int(round(df_sim["besoin_achat_estime"].sum()))
        cout_total = pd.to_numeric(df_sim["cout_estime_cheapest"], errors="coerce").sum()
        st.info(f"🧾 **Usage annuel total (moyen)**: {total_usage:,} pièces | "
                f"📦 **Besoin d'achat estimé**: {total_besoin:,} pièces | "
                f"💰 **Coût estimé (min prix)**: {_fmt_money(cout_total)}")
        

        # --- 🌍 Répartition par région d'origine (simulation) ---
        st.markdown("### 🌍 Répartition par région d'origine")

        mode_region_sim = st.radio(
            "Base du calcul",
            ["Par coût (recommandé)", "Par unités"],
            index=0, horizontal=True, key="kpi_region_mode_sim"
        )

        # Joindre l'origine + calcul des régions
        try:
            conn = open_conn()
            df_pays = pd.read_sql("SELECT id_piece, COALESCE(pays_origine,'Canada') AS pays_origine FROM pieces", conn)
            conn.close()
            df_sim_reg = df_sim.merge(df_pays, on="id_piece", how="left")
            df_sim_reg["pays_origine"] = df_sim_reg["pays_origine"].fillna("Canada")
        except Exception as e:
            st.error(f"Impossible de calculer la répartition régionale (jointure): {e}")
            df_sim_reg = df_sim.copy()
            df_sim_reg["pays_origine"] = "Canada"

        df_sim_reg["region"] = df_sim_reg["pays_origine"].apply(pays_to_region)

        # Colonne métrique selon le mode
        col_metric = "cout_estime_cheapest" if "coût" in mode_region_sim.lower() else "usage_moyen"
        vals = pd.to_numeric(df_sim_reg[col_metric], errors="coerce").fillna(0)

        totaux = df_sim_reg.assign(val=vals).groupby("region", as_index=False)["val"].sum()
        ordre = ["Amérique","Europe","Asie","Autre"]
        totaux = totaux.set_index("region").reindex(ordre, fill_value=0.0)["val"]
        total_all = float(totaux.sum())

        pct = (totaux / total_all * 100.0) if total_all > 0 else totaux*0
        pct_amerique = float(pct.get("Amérique", 0.0))

        # Affichage des 3 régions + focus Amérique
        c1, c2, c3 = st.columns(3)
        c1.metric("🇺🇸🇨🇦🇲🇽 Amérique", f"{pct.get('Amérique',0):.1f}%")
        c2.metric("🇪🇺 Europe",        f"{pct.get('Europe',0):.1f}%")
        c3.metric("🌏 Asie",           f"{pct.get('Asie',0):.1f}%")

        st.progress(min(max(int(round(pct_amerique)), 0), 100))
        st.info(f"🎯 **% Amérique (Canada + USA + Mexique)** : {pct_amerique:.1f}%")

        # Export
        st.download_button(
            "📤 Exporter la simulation (CSV)",
            data=df_sim.to_csv(index=False).encode("utf-8"),
            file_name=f"simulation_annuelle_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # --- Réapprovisionnement (Min–Max s, S) ---
    st.markdown("---")
    st.subheader("♻️ Simulation avec réapprovisionnement (Min–Max s, S)")

    colp1, colp2, colp3, colp4 = st.columns([1,1,1,1])
    with colp1:
        service_level = st.slider("Niveau de service cible (%)", min_value=85, max_value=99, value=97, step=1)
    with colp2:
        couverture_mois_S = st.number_input(
        "Couverture pour S (mois)", min_value=0.0, max_value=6.0, value=1.0,
        step=0.5, format="%.1f", key="couverture_mois_S"
    )
    with colp3:
        supplier_priority = st.selectbox("Priorité fournisseur", ["prix", "delai", "equilibre"], index=0)
    with colp4:
        holding_rate = st.number_input("Taux possession (%/an)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    nb_runs_policy = st.slider("Itérations (Monte Carlo)", min_value=20, max_value=500, value=100, step=20)
    randomize_policy = st.checkbox("Aléatoire à chaque exécution (s,S)", value=True)
    run_policy = st.button("Lancer la simulation de réapprovisionnement")

    if run_policy:
        # graine aléatoire (pour voir l'effet des itérations)
        if randomize_policy:
            random.seed(None)   # nouvelle séquence à chaque exécution
        else:
            random.seed(42)     # résultats reproductibles si paramètres identiques

        with st.spinner("Simulation (s,S) en cours…"):
            df_policy = simulate_policy_annee(
                service_level=float(service_level),
                couverture_mois_S=float(couverture_mois_S),
                supplier_priority=supplier_priority,
                holding_rate_pct_year=float(holding_rate),
                variabilite_pct=float(variabilite_pct),
                croissance_pct=float(croissance_pct),
                saison_mode=saison_mode,
                nb_runs=int(nb_runs_policy),
            )


        # (optionnel) filtre recherche global si tu l'as
        try:
            mask_pol = make_search_mask(df_policy, recherche, ["id_piece","nom_piece"], mode=mode_flag)
            df_policy = df_policy[mask_pol]
        except Exception:
            pass

        # Affichage formaté
        money_cols_pol = ["prix_unitaire","achat_total_$","cout_possession_$","cout_total_$"]
        int_cols_pol   = ["lead_time_mois","stock_initial","ruptures_estimees"]
        st.dataframe(style_table(df_policy, money_cols_pol, int_cols_pol), use_container_width=True)

        # Résumé global
        fill_rate_moy = df_policy["fill_rate"].mean() if not df_policy.empty else 100.0
        cout_total_global = pd.to_numeric(df_policy["cout_total_$"], errors="coerce").sum()
        inv_moy_global = pd.to_numeric(df_policy["inv_moyen"], errors="coerce").mean()
        st.success(
            f"🎯 Fill rate moyen: {fill_rate_moy:.1f}% | "
            f"🧺 Inventaire moyen (moyenne des pièces): {inv_moy_global:.1f} u. | "
            f"💰 Coût total annuel estimé: {_fmt_money(cout_total_global)}"
        )

        # Export CSV
        st.download_button(
            "📤 Exporter la simulation (s,S) (CSV)",
            data=df_policy.to_csv(index=False).encode("utf-8"),
            file_name=f"simulation_policy_sS_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # --- Auto-tuner : trouver la couverture minimale par pièce ---
    st.markdown("----")
    st.subheader("🧠 Auto-tuner : couverture minimale par pièce pour atteindre le fill rate cible")

    colt1, colt2, colt3 = st.columns([1,1,1])
    with colt1:
        fill_rate_cible = st.slider("Fill rate cible (%)", min_value=90, max_value=99, value=int(service_level), step=1)
    with colt2:
        cov_range = st.slider("Plage de couverture (mois)", min_value=0.0, max_value=6.0, value=(0.0, 3.0), step=0.25)
    with colt3:
        nb_runs_tune = st.slider("Itérations Monte Carlo (tuning)", min_value=20, max_value=300, value=60, step=20)

    randomize_tune = st.checkbox("Aléatoire à chaque exécution (tuning)", value=True)
    btn_tune = st.button("🔍 Lancer l’auto-tuning")

    if btn_tune:
        if randomize_tune:
            random.seed(None)
        else:
            random.seed(42)

        with st.spinner("Calcul de la couverture minimale par pièce…"):
            df_tuned = auto_tune_couverture_par_piece(
                fill_rate_cible_pct=float(fill_rate_cible),
                service_level_pct=float(service_level),             # même service que ta section s,S
                supplier_priority=supplier_priority,
                holding_rate_pct_year=float(holding_rate),
                variabilite_pct=float(variabilite_pct),
                croissance_pct=float(croissance_pct),
                saison_mode=saison_mode,
                nb_runs=int(nb_runs_tune),
                cov_min=float(cov_range[0]),
                cov_max=float(cov_range[1]),
                cov_step=0.25,
                reseed_each_step=randomize_tune,
            )

        # (optionnel) filtre recherche global
        try:
            mask_tune = make_search_mask(df_tuned, recherche, ["id_piece","nom_piece"], mode=mode_flag)
            df_tuned = df_tuned[mask_tune]
        except Exception:
            pass

        # Affichage formaté
        money_cols_tune = ["prix_unitaire","achat_total_$","cout_possession_$","cout_total_$"]
        int_cols_tune   = ["lead_time_mois","stock_initial"]
        st.dataframe(
            style_table(df_tuned, money_cols_tune, int_cols_tune),
            use_container_width=True
        )

        # Résumé global (sur les pièces qui atteignent la cible)
        try:
            atteints = df_tuned[df_tuned["atteint_fill_rate_cible"] == "Oui"]
            fill_rate_moy_tune = pd.to_numeric(atteints["fill_rate_obtenu_%"], errors="coerce").mean()
            cout_total_glob_tune = pd.to_numeric(df_tuned["cout_total_$"], errors="coerce").sum()
            st.success(
                f"✅ Pièces atteignant la cible: {len(atteints)}/{len(df_tuned)} | "
                f"🎯 Fill rate moyen (atteints): {fill_rate_moy_tune:.1f}% | "
                f"💰 Coût total estimé (toutes pièces): {_fmt_money(cout_total_glob_tune)}"
            )
        except Exception:
            pass

        # Export CSV du tuning
        st.download_button(
            "📤 Exporter l’auto-tuning (CSV)",
            data=df_tuned.to_csv(index=False).encode("utf-8"),
            file_name=f"auto_tuning_couverture_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )


# ============ ONGLET : 📅 Calendrier de production ============
with onglets[4]:   # attention: l'indice doit correspondre à ta liste réelle
    st.subheader("📅 Calendrier de production (lissage annuel)")
    colA, colB, colC = st.columns(3)
    with colA:
        annee = st.number_input("Année planifiée", min_value=2024, value=datetime.today().year, step=1, key="prodcal_annee")
    with colB:
        croissance = st.number_input("Croissance vs historique (%)", value=0.0, step=1.0, key="prodcal_growth")
    with colC:
        cap_default = st.number_input("Capacité standard par mois (heures)", min_value=0.0, value=160.0, step=10.0, key="prodcal_capdef")

    # 1) Demande annuelle par produit (baseline historique)
    df_dmd = _baseline_annual_demand(int(annee), float(croissance))
    if df_dmd.empty:
        st.info("Aucun historique : renseigne la demande annuelle manuellement dans le tableau ci-dessous.")
        # créer un squelette avec tous les produits
        conn = open_conn(); df_prod = pd.read_sql("SELECT id_produit, nom_produit FROM produits", conn); conn.close()
        df_dmd = df_prod.copy(); df_dmd["annual_units"] = 0

    # 2) Heures standard par produit (éditable)
    std_map = _load_std_hours()
    conn = open_conn(); df_prod = pd.read_sql("SELECT id_produit, nom_produit FROM produits", conn); conn.close()
    if std_map:
        df_std = pd.DataFrame({"id_produit": list(std_map.keys()),
                               "heures_unite": list(std_map.values())}).merge(df_prod, on="id_produit", how="right")
    else:
        df_std = df_prod.copy(); df_std["heures_unite"] = 1.0
    st.markdown("#### Heures standard par produit (éditable)")
    df_std_edit = st.data_editor(df_std[["id_produit","nom_produit","heures_unite"]],
                                 use_container_width=True, key="prodcal_std_edit")
    if st.button("💾 Sauvegarder les heures standard", key="prodcal_save_std"):
        _save_std_hours(df_std_edit[["id_produit","heures_unite"]])
        st.success("Heures standard sauvegardées.")

    # 3) Capacité mensuelle (éditable)
    st.markdown("#### Capacité mensuelle (heures, éditable)")
    df_cap = _monthly_capacity(int(annee), float(cap_default))
    df_cap = st.data_editor(df_cap, use_container_width=True, key="prodcal_cap_edit")

    # 4) Demande annuelle (éditable)
    st.markdown("#### Demande annuelle par produit (éditable)")
    df_dmd = df_dmd.merge(df_prod, on="id_produit", how="right")
    df_dmd = df_dmd[["id_produit","nom_produit","annual_units"]].fillna({"annual_units":0}).astype({"annual_units":"int64"})
    df_dmd_edit = st.data_editor(df_dmd, use_container_width=True, key="prodcal_dmd_edit")

    # 5) Lancer le lissage
    if st.button("▶️ Générer le calendrier lissé", key="prodcal_run"):
        std_map2 = {r["id_produit"]: float(r["heures_unite"]) for _, r in df_std_edit.iterrows()}
        dmd_map = {r["id_produit"]: int(r["annual_units"]) for _, r in df_dmd_edit.iterrows()}
        cap_map = {int(r["mois"]): float(r["heures_disponibles"]) for _, r in df_cap.iterrows()}

        plan = _level_load_schedule(dmd_map, std_map2, cap_map)  # (mois,id_produit,qte,heures)

        if plan.empty:
            st.warning("Plan vide : vérifie que la demande > 0 et que la capacité > 0.")
        else:
            # Enregistrer dans production_calendar (optionnel ; sinon juste afficher)
            conn = open_conn(); cur = conn.cursor()
            cur.execute("DELETE FROM production_calendar WHERE annee=?", (int(annee),))
            cur.executemany("""INSERT INTO production_calendar (annee, mois, id_produit, qte, heures)
                               VALUES (?,?,?,?,?)""",
                            [(int(annee), int(r["mois"]), r["id_produit"], int(r["qte"]), float(r["heures"]))
                             for _, r in plan.iterrows()])
            conn.commit(); conn.close()

            # Affichage pivot (mois en colonnes)
            pivot = plan.pivot_table(index="id_produit",
                                     columns="mois",
                                     values="qte", aggfunc="sum", fill_value=0).sort_index()
            st.markdown("#### 📊 Calendrier (unités par mois)")
            st.dataframe(pivot, use_container_width=True)

            # Résumé charge vs capacité
            charge = plan.groupby("mois")["heures"].sum().reindex(range(1,13), fill_value=0.0)
            cap = pd.Series(cap_map).reindex(range(1,13), fill_value=0.0)
            df_rc = pd.DataFrame({"heures_planifiées": charge, "heures_disponibles": cap})
            st.markdown("#### ⚖️ Charge vs Capacité (heures)")
            st.dataframe(df_rc, use_container_width=True)

            # 6) ⇨ Préparer l'achat pièces pour les X prochains mois
            st.markdown("#### ⇨ Générer les besoins pièces à partir du plan")
            horizon = st.slider("Horizon (mois) pour alimenter 🛒", 1, 6, 3, 1, key="prodcal_horizon")
            # map produit->pieces (1:1 par défaut; si tu ajoutes des quantités de nomenclature, on les intégrera ici)
            p2pcs = _product_to_pieces_map()
            # besoins pièces cumulés sur l'horizon
            df_hor = plan[plan["mois"].between(1, int(horizon))]
            besoins_pieces = {}
            for _, r in df_hor.iterrows():
                pcs = p2pcs.get(r["id_produit"], set())
                for pid in pcs:
                    besoins_pieces[pid] = besoins_pieces.get(pid, 0) + int(r["qte"])  # qty=1 par défaut

            if st.button("📦 Alimenter l’onglet 🛒 avec ces besoins", key="prodcal_push"):
                # format attendu : {id_piece: {"besoin_total": int}}
                st.session_state["besoins_courants"] = {k: {"besoin_total": int(v)} for k, v in besoins_pieces.items()}
                st.success(f"Besoins pour {len(besoins_pieces)} pièces envoyés à l’onglet 🛒.")



# ============ ONGLET 6 : Passer une commande ============
with onglets[5]:
    st.subheader("🛒 Passer une commande (manuelle)")

    st.caption("Sélectionne les pièces à commander, choisis le fournisseur pour chaque ligne, ajuste les quantités, puis clique sur *Créer la/les commande(s)*. Les réceptions s'appliquent automatiquement à la date prévue.")

    # --- Paramètres simples (recalcul fournisseurs & besoins)
    colA, colB, colC = st.columns(3)
    with colA:
        strategie = st.selectbox("Stratégie par défaut", ["prix","delai","equilibre"], index=0, key="po_strategie")
    with colB:
        filtre_prix_max = st.number_input("Prix unitaire max (optionnel)", min_value=0.0, value=0.0, step=1.0, key="po_prix_max")
        prix_max_effectif = None if filtre_prix_max <= 0 else filtre_prix_max
    with colC:
        filtre_delai_max = st.number_input("Délai max (jours, optionnel)", min_value=0, value=0, step=1, key="po_delai_max")
        delai_max_effectif = None if filtre_delai_max <= 0 else filtre_delai_max

    # Besoins utilisés = même logique que l'onglet actif (historique ou manuel)
    besoins_dict = st.session_state.get("besoins_courants", {})  # si tu stockes déjà les besoins
    
    # Fallback : si aucun besoin n'a été publié par 1/2, proposer 50 pièces à quantité 0
    if not besoins_dict:
        conn = open_conn()
        base = pd.read_sql("SELECT id_piece, nom_piece FROM pieces ORDER BY id_piece LIMIT 50", conn)
        conn.close()
        besoins_dict = {
            row["id_piece"]: {"besoin_total": 0, "nom_piece": row["nom_piece"], "stock_disponible": 0}
            for _, row in base.iterrows()
        }


    df_commande = choisir_fournisseur_top3(
        besoins=besoins_dict,
        strategie=strategie,
        filtre_fournisseur=None,
        filtre_prix_max=prix_max_effectif,
        filtre_delai_max=delai_max_effectif
)


    if df_commande.empty:
        st.info("Aucune ligne candidate. Génère des besoins dans les onglets 1 ou 2, ou enlève les filtres.")
    else:
        # On prépare un petit éditeur avec : à_commander, choix fournisseur (1/2/3), quantite
        # Quantité par défaut : max(besoin - stock, 0)
        df_edit = df_commande[[
            "id_piece","nom_piece",
            "besoin_total","stock_disponible","en_attente","stock_projeté","prochaine_reception",
            "fournisseur_1","prix_1","delai_1",
            "fournisseur_2","prix_2","delai_2",
            "fournisseur_3","prix_3","delai_3"
        ]].copy()


        import numpy as np
        qdef = np.maximum((df_edit["besoin_total"] - df_edit["stock_disponible"]).fillna(0).astype(float), 0).astype(int)
        df_edit.insert(1, "a_commander", (qdef > 0))  # coché par défaut si besoin > stock
        df_edit.insert(2, "choix_fournisseur", 1)     # 1=colonne _1, 2=_2, 3=_3
        df_edit.insert(3, "quantite", qdef)

        # Affichage éditable
        st.markdown("#### Lignes à valider")
        cfg = {
            "a_commander": st.column_config.CheckboxColumn("À commander"),
            "choix_fournisseur": st.column_config.NumberColumn("Choix fournisseur (1/2/3)", min_value=1, max_value=3, step=1),
            "quantite": st.column_config.NumberColumn("Quantité", min_value=0, step=1),
        }
        df_sel = st.data_editor(
            df_edit,
            column_config=cfg,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="po_editor"
        )

        # Création des commandes groupées par fournisseur choisi
        if st.button("🧾 Créer la/les commande(s)", key="po_creer"):
            today = datetime.today().strftime("%Y-%m-%d")
            lignes_valides = []

            for _, r in df_sel.iterrows():
                if not bool(r.get("a_commander", False)):
                    continue
                q = int(r.get("quantite", 0) or 0)
                if q <= 0:
                    continue
                ch = int(r.get("choix_fournisseur", 1) or 1)
                if ch == 1:
                    four, prix, delai = r.get("fournisseur_1"), r.get("prix_1"), r.get("delai_1")
                elif ch == 2:
                    four, prix, delai = r.get("fournisseur_2"), r.get("prix_2"), r.get("delai_2")
                else:
                    four, prix, delai = r.get("fournisseur_3"), r.get("prix_3"), r.get("delai_3")
                if pd.isna(four) or pd.isna(prix) or pd.isna(delai):
                    continue
                lignes_valides.append({
                    "id_piece": r["id_piece"],
                    "fournisseur": str(four),
                    "quantite": int(q),
                    "prix_unitaire": float(prix),
                    "delai_jours": int(delai),
                    "date_reception_prevue": (datetime.today() + timedelta(days=int(delai))).strftime("%Y-%m-%d")
                })

            if not lignes_valides:
                st.warning("Aucune ligne valide sélectionnée.")
            else:
                # Grouper par fournisseur -> 1 commande par fournisseur
                from collections import defaultdict
                groupes = defaultdict(list)
                for ln in lignes_valides:
                    groupes[ln["fournisseur"]].append(ln)

                conn = open_conn()
                cur = conn.cursor()
                created = []
                for four, lignes in groupes.items():
                    poid = f"PO-{datetime.today().strftime('%Y%m%d-%H%M%S')}-{abs(hash(four))%1000:03d}"
                    total = sum(ln["quantite"] * ln["prix_unitaire"] for ln in lignes)
                    cur.execute("INSERT INTO commandes_achat (id_commande, fournisseur, date_commande, statut, montant_total) VALUES (?, ?, ?, 'en_transit', ?)",
                                (poid, four, today, total))
                    for ln in lignes:
                        cur.execute("""
                            INSERT INTO lignes_commande_achat
                            (id_commande, id_piece, quantite, prix_unitaire, delai_jours, date_reception_prevue)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (poid, ln["id_piece"], ln["quantite"], ln["prix_unitaire"], ln["delai_jours"], ln["date_reception_prevue"]))
                    created.append((poid, four, len(lignes), total))
                conn.commit(); conn.close()

                st.success(f"{len(created)} commande(s) créée(s) ✅")
                for poid, four, n, tot in created:
                    st.write(f"- **{poid}** → {four} · {n} ligne(s) · total estimé : ${tot:,.2f}")

            # Afficher le carnet courant
            st.markdown("#### 📚 Carnet des commandes")
            try:
                conn = open_conn()
                df_po = pd.read_sql("""
                    SELECT c.id_commande, c.fournisseur, c.date_commande, c.statut, c.montant_total,
                           MIN(l.date_reception_prevue) AS reception_min, MAX(l.date_reception_prevue) AS reception_max,
                           COUNT(*) AS lignes
                    FROM commandes_achat c
                    LEFT JOIN lignes_commande_achat l ON l.id_commande = c.id_commande
                    GROUP BY c.id_commande, c.fournisseur, c.date_commande, c.statut, c.montant_total
                    ORDER BY c.date_commande DESC
                """, conn)
                conn.close()
                st.dataframe(df_po, use_container_width=True)
            except Exception as e:
                st.error(f"Impossible d'afficher le carnet des commandes : {e}")
