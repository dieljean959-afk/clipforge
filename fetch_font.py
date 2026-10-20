import urllib.request
from pathlib import Path

BASE = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf"
DEST = Path("app/fonts")
DEST.mkdir(parents=True, exist_ok=True)
for name in (
    "Montserrat-ExtraBold.ttf",
    "Montserrat-Bold.ttf",
    "Montserrat-SemiBold.ttf",
):
    try:
        urllib.request.urlretrieve(f"{BASE}/{name}", DEST / name)
        print("ok", name)
    except Exception as e:
        print("ERR", name, type(e).__name__, str(e)[:120])
