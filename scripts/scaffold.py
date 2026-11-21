#!/usr/bin/env python3
"""Copy the Flask hybrid RAG template into a new project folder."""
import argparse, shutil, sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "template"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", required=True, help="target project directory")
    p.add_argument("--name", default="Hybrid RAG Assistant", help="app display name")
    p.add_argument("--port", default="5050", help="port (default 5050)")
    a = p.parse_args()

    dest = Path(a.dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"Refusing to overwrite non-empty directory: {dest}")
    shutil.copytree(TEMPLATE, dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env"))

    env = (dest / ".env.example").read_text()
    env = env.replace("APP_NAME=Hybrid RAG Assistant", f"APP_NAME={a.name}")
    env = env.replace("PORT=5050", f"PORT={a.port}")
    (dest / ".env").write_text(env)
    (dest / "data").mkdir(exist_ok=True)

    print(f"Created {dest}\nNext:\n  1. edit .env (AWS_REGION, credentials or profile)\n"
          f"  2. pip install -r requirements.txt\n  3. python app.py  ->  http://localhost:{a.port}")


if __name__ == "__main__":
    main()
