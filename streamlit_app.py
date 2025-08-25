import os, sqlite3, json
from contextlib import closing
from cryptography.fernet import Fernet

DB_PATH = os.getenv("APP_DB_PATH", "data/app.sqlite")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY CHECK (id=1),
        sim_api_key BLOB,
        wallets TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS notes (
        wallet TEXT PRIMARY KEY,
        note TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        ts TEXT,
        wallet TEXT,
        data_json TEXT
    )""")
    c.commit()
    return c

def _get_fernet():
    key = os.getenv("APP_FERNET_KEY")
    if not key:
        keyfile = "data/fernet.key"
        if os.path.exists(keyfile):
            key = open(keyfile,"rb").read()
        else:
            key = Fernet.generate_key()
            os.makedirs("data", exist_ok=True)
            open(keyfile,"wb").write(key)
    if isinstance(key, str): key = key.encode()
    return Fernet(key)

def save_settings(sim_api_key: str|None, wallets: list[str]|None):
    with closing(_conn()) as c:
        enc = None
        if sim_api_key:
            f = _get_fernet()
            enc = f.encrypt(sim_api_key.encode())
        c.execute("INSERT OR REPLACE INTO settings (id, sim_api_key, wallets) VALUES (1, ?, ?)",
                  (enc, json.dumps(wallets or [])))
        c.commit()

def load_settings():
    with closing(_conn()) as c:
        row = c.execute("SELECT sim_api_key, wallets FROM settings WHERE id=1").fetchone()
        if not row: return {"sim_api_key": None, "wallets": []}
        enc, wallets = row
        dec = None
        if enc:
            try:
                f = _get_fernet()
                dec = f.decrypt(enc).decode()
            except Exception:
                dec = None
        return {"sim_api_key": dec, "wallets": (json.loads(wallets) if wallets else [])}

def save_note(wallet: str, note: str):
    with closing(_conn()) as c:
        c.execute("INSERT OR REPLACE INTO notes (wallet, note) VALUES (?, ?)", (wallet, note))
        c.commit()

def load_note(wallet: str) -> str:
    with closing(_conn()) as c:
        row = c.execute("SELECT note FROM notes WHERE wallet=?", (wallet,)).fetchone()
        return row[0] if row else ""

def save_snapshot(wallet: str, data: dict, ts: str):
    with closing(_conn()) as c:
        c.execute("INSERT INTO snapshots (ts, wallet, data_json) VALUES (?, ?, ?)",
                  (ts, wallet, json.dumps(data)))
        c.commit()
