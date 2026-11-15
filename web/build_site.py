"""Assemble the static site (for GitHub Pages or any web server) into ``_site/``.

    python web/build_site.py && python -m http.server -d _site 8000

``web/index.html`` holds the page body; this wraps it in a full HTML document.
"""

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "_site"
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='14' fill='%233d36a8'/%3E%3Ccircle cx='11' cy='13' r='4' fill='%23d8c4a6'/%3E%3Ccircle cx='20' cy='12' r='3' fill='%23d8c4a6'/%3E%3Ccircle cx='17' cy='21' r='4' fill='%23c2336b'/%3E%3C/svg%3E">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    body = (ROOT / "index.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(HEAD + body + "\n</body>\n</html>\n", encoding="utf-8")
    for name in ("mndsim.js", "app.js", "trial-worker.js"):
        shutil.copy(ROOT / name, OUT / name)
    (OUT / ".nojekyll").write_text("")
    print(f"site written to {OUT}")


if __name__ == "__main__":
    main()
