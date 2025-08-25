# snapshot.py
import os
import re
import json
import requests
import pandas as pd
from datetime import datetime
from typing import List

# =========================
# Config
# =========================
BASE_SIM = "https://api.sim.dune.com/v1/evm"

CHAIN_OPTIONS = {
    "Ethereum": 1,
    "Optimism": 10,
    "Arbitrum": 42161,
    "Polygon": 137,
    "Base": 8453,
    "Bnb Smart Chain": 56,
    "Avalanche": 43114,
    "Fantom": 250,
}

# Plasma & XPL overrides
PLASMA_SYMBOL_NAME = "PlasmaUSD"; PLASMA_DECIMALS = 6; PLASMA_PRICE_USD = 1.0
XPL_SYMBOL_NAME    = "XPL";       XPL_DECIMALS    = 18; XPL_PRICE_USD    = 1.0

# =========================
# Helpers
# =========================
def norm_addrs(lines: List[str]) -> List[str]:
    """Normalize & deduplicate Ethereum addresses."""
    out = []
    for x in lines:
        x = x.strip()
        if re.match(r"^0x[a-fA-F0-9]{40}$", x):
            out.append(x)
    seen = set(); res = []
    for a in out:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            res.append(a)
    return res

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def fetch_balances(addr: str, api_key: str, chain_ids: str) -> dict:
    url = f"{BASE_SIM}/balances/{addr}"
    if chain_ids:
        url += f"?chain_ids={chain_ids}"
    r = requests.get(url, headers={"X-Sim-Api-Key": api_key}, timeout=25)
    r.raise_for_status()
    return r.json() or {}

def build_df(wallets: List[str], chains: List[str], api_key: str) -> pd.DataFrame:
    chain_ids = ",".join(str(CHAIN_OPTIONS[c]) for c in chains if c in CHAIN_OPTIONS)
    rows = []
    for addr in wallets:
        try:
            data = fetch_balances(addr, api_key, chain_ids)
        except Exception as e:
            print(f"[warn] {addr}: {e}")
            continue

        balances = (data.get("balances") or data.get("tokens") or [])
        for b in balances:
            sym = (b.get("symbol") or "").strip().upper()
            raw_amount = b.get("amount"); dec = b.get("decimals")
            human = to_float(raw_amount, 0.0)
            if dec is not None:
                try:
                    human = to_float(raw_amount, 0.0) / (10 ** int(dec))
                except Exception:
                    pass

            # Special overrides
            if sym == PLASMA_SYMBOL_NAME.upper():
                human = to_float(raw_amount, 0.0) / (10 ** PLASMA_DECIMALS)
                price = PLASMA_PRICE_USD; value = human * price
            elif sym == XPL_SYMBOL_NAME.upper():
                human = to_float(raw_amount, 0.0) / (10 ** XPL_DECIMALS)
                price = XPL_PRICE_USD; value = human * price
            else:
                price = to_float(b.get("price_usd"), None)
                value = to_float(b.get("value_usd"), None)
                if value is None and price is not None:
                    value = human * price

            rows.append({
                "wallet": addr,
                "chain": (b.get("chain") or "").lower(),
                "symbol": sym,
                "amount": human,
                "price_usd": price,
                "value_usd": value,
                "token_address": b.get("address"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ("amount", "price_usd", "value_usd"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# =========================
# Main
# =========================
def main():
    sim_key = os.getenv("SIM_API_KEY", "")
    if not sim_key:
        raise RuntimeError("❌ SIM_API_KEY missing in environment/secrets")

    # Wallets: prefer data/wallets.txt, else GitHub Secret WALLETS
    wallets = []
    if os.path.exists("data/wallets.txt"):
        with open("data/wallets.txt") as f:
            wallets = norm_addrs(f.readlines())
    else:
        raw = os.getenv("WALLETS", "")
        if raw:
            wallets = norm_addrs(raw.splitlines() or raw.split(","))
    if not wallets:
        raise RuntimeError("❌ No wallets provided. Use WALLETS secret or create data/wallets.txt")

    # Chains: prefer data/chains.json, else CHAINS_JSON secret, else defaults
    chains = ["Ethereum", "Optimism", "Arbitrum", "Polygon", "Base"]
    try:
        if os.path.exists("data/chains.json"):
            with open("data/chains.json") as f:
                payload = json.load(f)
                if isinstance(payload.get("chains"), list) and payload["chains"]:
                    chains = payload["chains"]
        else:
            raw_c = os.getenv("CHAINS_JSON", "")
            if raw_c:
                payload = json.loads(raw_c)
                if isinstance(payload.get("chains"), list) and payload["chains"]:
                    chains = payload["chains"]
    except Exception:
        pass

    # Build dataframe
    df = build_df(wallets, chains, sim_key)
    if df.empty:
        print("[warn] No balances found.")
        return

    # Save snapshots
    os.makedirs("backups", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    df.to_csv(f"backups/snapshot_{ts}.csv", index=False)
    df.to_json(f"backups/snapshot_{ts}.json", orient="records")
    print(f"[ok] Snapshot saved to backups/ at {ts}")

if __name__ == "__main__":
    main()
