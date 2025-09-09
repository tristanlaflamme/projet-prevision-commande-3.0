import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
import math

DB_FILE = "inventaire_temp.db"
random.seed(42)

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Désactiver temporairement pour pouvoir DROP dans n'importe quel ordre
cur.execute("PRAGMA foreign_keys=OFF;")

for t in [
    "fournisseurs",
    "stocks",
    "mouvements_stock",
    "historique_commandes",
    "pieces",
    "sous_assemblages",
    "produits",
    "fabricants"
]:
    cur.execute(f"DROP TABLE IF EXISTS {t}")

# Réactiver avant de (re)créer les tables pour que les FK soient bien appliquées
cur.execute("PRAGMA foreign_keys=ON;")


# Tables
cur.execute("""CREATE TABLE produits (
    id_produit TEXT PRIMARY KEY,
    nom_produit TEXT,
    version TEXT,
    description TEXT
)""")

cur.execute("""CREATE TABLE sous_assemblages (
    id_sous_assemblage TEXT PRIMARY KEY,
    id_produit TEXT,
    nom_sous_assemblage TEXT,
    FOREIGN KEY (id_produit) REFERENCES produits(id_produit)
)""")

cur.execute("""CREATE TABLE fabricants (
    id_fabricant TEXT PRIMARY KEY,
    nom_fabricant TEXT,
    pays TEXT,
    delai_fabrication_jours INTEGER
)""")

cur.execute("""CREATE TABLE pieces (
    id_piece TEXT PRIMARY KEY,
    id_sous_assemblage TEXT,
    nom_piece TEXT,
    id_fabricant TEXT,
    FOREIGN KEY (id_sous_assemblage) REFERENCES sous_assemblages(id_sous_assemblage),
    FOREIGN KEY (id_fabricant) REFERENCES fabricants(id_fabricant)
)""")

cur.execute("""CREATE TABLE stocks (
    id_piece TEXT PRIMARY KEY,
    stock_disponible INTEGER,
    seuil_minimum INTEGER,
    derniere_maj TEXT
)""")

cur.execute("""CREATE TABLE fournisseurs (
    id_fournisseur TEXT,
    id_piece TEXT,
    nom_fournisseur TEXT,
    prix_unitaire REAL,
    delai_livraison_jours INTEGER
)""")

cur.execute("""CREATE TABLE historique_commandes (
    id_commande TEXT PRIMARY KEY,
    id_produit TEXT,
    quantite INTEGER,
    date_commande TEXT
)""")

# Produits
produits = [
    (f"P{i:03}", f"Produit {i}", f"{1+i%3}.0", f"Description produit {i}")
    for i in range(1, 18)  # 17 produits × 3 sous-assemblages = 51 pièces
]

cur.executemany("INSERT INTO produits VALUES (?, ?, ?, ?)", produits)

# Sous-assemblages
sous_assemblages = []
for pid, *_ in produits:
    for j in range(3):
        sid = f"SA-{pid}-{j+1}"
        sous_assemblages.append((sid, pid, f"SA {j+1} de {pid}"))
cur.executemany("INSERT INTO sous_assemblages VALUES (?, ?, ?)", sous_assemblages)

# Fabricants
fabricants = [
    ("F001", "FabCanada", "Canada", 25),
    ("F002", "FabUSA", "USA", 20),
    ("F003", "FabDE", "Allemagne", 30)
]
cur.executemany("INSERT INTO fabricants VALUES (?, ?, ?, ?)", fabricants)

# Pièces : générer plusieurs pièces par sous-assemblage pour atteindre ~1000 pièces
PIECES_PAR_SA = 20   # 17 produits × 3 SA × 20 ≈ 1020 pièces

pieces = []
i_global = 0
for (id_sous_assemblage, id_produit, *_) in sous_assemblages:
    for k in range(PIECES_PAR_SA):
        i_global += 1
        id_piece = f"PC-{i_global:04d}"
        nom_piece = f"Pièce {i_global}"
        id_fabricant = random.choice(["F001", "F002", "F003"])
        pieces.append((id_piece, id_sous_assemblage, nom_piece, id_fabricant))
cur.executemany("INSERT INTO pieces VALUES (?, ?, ?, ?)", pieces)


# Fournisseurs (2-3 par pièce)
fournisseur_noms = ["FastShip", "CheapCo", "BalanceInc"]
fournisseurs = []
fid = 1
for piece in pieces:
    id_piece = piece[0]
    n_fourn = random.choice([2, 3])
    noms = random.sample(fournisseur_noms, n_fourn)
    for nom in noms:
        fournisseurs.append((f"F{fid:03d}", id_piece, nom,
                             round(random.uniform(10, 100), 2),
                             random.randint(5, 30)))
        fid += 1
cur.executemany("INSERT INTO fournisseurs VALUES (?, ?, ?, ?, ?)", fournisseurs)


# Historique commandes
NB_CMD = 2000  # au lieu de 100
commandes = []
for i in range(NB_CMD):
    pid = random.choice([p[0] for p in produits])
    quantite = random.randint(1, 20)
    date_cmd = datetime.today() - timedelta(days=random.randint(30, 365))
    commandes.append((f"CMD-{i+1:04d}", pid, quantite, date_cmd.strftime("%Y-%m-%d")))
cur.executemany("INSERT INTO historique_commandes VALUES (?, ?, ?, ?)", commandes)

# --- Recalibrage des stocks à partir de la demande réelle sur NB_MOIS_BASE mois ---
NB_MOIS_BASE = 3  # aligne avec la durée de prévision par défaut dans l'app

# 1) moyenne mensuelle par produit (depuis l'historique que tu viens d'insérer)
hist = pd.read_sql("SELECT id_produit, quantite, date_commande FROM historique_commandes", conn)
if not hist.empty:
    hist["date_commande"] = pd.to_datetime(hist["date_commande"])
    hist["mois"] = hist["date_commande"].dt.to_period("M")
    par_mois = hist.groupby(["id_produit", "mois"]).quantite.sum().reset_index()
    moy_prod = par_mois.groupby("id_produit").quantite.mean()  # Series: id_produit -> moy mensuelle
else:
    moy_prod = pd.Series(dtype=float)

# 2) mapping pièce -> produit (chaque pièce appartient à un seul produit via son sous-assemblage)
prod_map = pd.read_sql("""
    SELECT pi.id_piece, sa.id_produit
    FROM pieces pi
    JOIN sous_assemblages sa ON sa.id_sous_assemblage = pi.id_sous_assemblage
""", conn).set_index("id_piece")["id_produit"]

# 3) délai minimal par pièce (utile pour un seuil intelligent)
lt_df = pd.read_sql("""
    SELECT id_piece, MIN(delai_livraison_jours) AS lt
    FROM fournisseurs
    GROUP BY id_piece
""", conn).set_index("id_piece")

# 4) besoin 3 mois par pièce (selon historique réel)
need_3m = {pid: float(moy_prod.get(prod_map.get(pid, ""), 0.0)) * NB_MOIS_BASE for pid in prod_map.index}

# 5) cibles de répartition des couleurs
GREEN_SHARE  = 0.40   # ~40% vert (stock >= besoin)
YELLOW_SHARE = 0.45   # ~45% jaune (0 < stock < besoin)
# le reste ~15% rouge (stock = 0)

today = datetime.today().strftime("%Y-%m-%d")
stocks = []

for pid in prod_map.index:
    besoin = need_3m.get(pid, 0.0)

    # seuil d'alerte: max de (20% du besoin 3 mois) et (besoin pendant le délai mini)
    L = int(lt_df.loc[pid, "lt"]) if pid in lt_df.index else random.randint(7, 45)
    seuil_delai = (besoin / NB_MOIS_BASE) * (L / 30.0)   # besoin pendant le délai
    seuil_min = max(1, int(math.ceil(max(0.2 * besoin, seuil_delai))))

    r = random.random()
    if besoin <= 0:
        # Pas de demande dans l'historique -> petit mix aléatoire (donne un peu de vert/jaune/rouge aussi)
        if r < 0.25:
            stock = random.randint(5, 20)    # vert
        elif r < 0.75:
            stock = random.randint(1, 7)     # jaune
        else:
            stock = 0                        # rouge
    else:
        if r < GREEN_SHARE:
            # 🟢 suffisant: entre 100% et 140% du besoin 3 mois
            stock = int(round(besoin * random.uniform(1.0, 1.4)))
        elif r < GREEN_SHARE + YELLOW_SHARE:
            # 🟡 partiel: entre 25% et 90% du besoin 3 mois
            stock = int(round(besoin * random.uniform(0.25, 0.9)))
        else:
            # 🔴 rupture
            stock = 0

    stocks.append((pid, stock, seuil_min, today))

cur.executemany("INSERT INTO stocks VALUES (?, ?, ?, ?)", stocks)
# --- fin recalibrage stocks ---


cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_prod_date ON historique_commandes(id_produit, date_commande)")

cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_fourn ON fournisseurs(id_piece, nom_fournisseur)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_fourn_prix_delai ON fournisseurs(id_piece, prix_unitaire, delai_livraison_jours)")

conn.commit()
conn.close()
print("✅ Base de données 'inventaire_temp.db' créée avec succès !")

