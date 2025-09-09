# demarreur.py
# Lance stream_app_inventaire.py via Streamlit, et prépare la BD si absente.

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# 👉 adapte ici si ton fichier a un autre nom (ex.: "streamlit_app_inventaire.py")
CANDIDATES = ["stream_app_inventaire.py", "streamlit_app_inventaire.py"]

def find_app():
    for name in CANDIDATES:
        p = BASE_DIR / name
        if p.exists():
            return p
    return None

def ensure_db():
    db = BASE_DIR / "inventaire_temp.db"
    if db.exists():
        return
    setup = BASE_DIR / "setup_inventaire_v10.py"
    if setup.exists():
        print("⚙️  BD absente → exécution de setup_inventaire_v10.py…")
        try:
            subprocess.run([sys.executable, str(setup)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de la création de la BD: {e}")
    else:
        print("⚠️  BD absente et setup_inventaire_v10.py introuvable. Je continue quand même.")

def ensure_streamlit():
    try:
        import streamlit  # noqa
        return
    except ImportError:
        print("📦 Streamlit non installé → installation…")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "streamlit"], check=False)

def main():
    app = find_app()
    if not app:
        print(f"❌ Fichier Streamlit introuvable ({' ou '.join(CANDIDATES)}) dans :\n{BASE_DIR}")
        input("\nAppuie sur Entrée pour fermer…")
        return

    ensure_db()
    ensure_streamlit()

    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    print(f"- Démarrage : {' '.join(cmd)}")
    print("- L’URL locale s’affichera dans la console (http://localhost:8501)")
    print("- Appuyer sur ctrl + c pour arrêter")

    # Lance Streamlit et attend sa fermeture
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n Arrêt. Appuie sur Enter pour fermer…")
        try:
            input()
        except Exception:
            pass

if __name__ == "__main__":
    main()

