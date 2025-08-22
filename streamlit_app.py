import os
import re
import requests
import pandas as pd
import streamlit as st

# =========================
# App config
# =========================
st.set_page_config(page_title="Dune SIM Wallet Tracker", layout="wide")
BASE_SIM = "https://api.sim.dune.com/v1/evm"

# SIM-supported common EVM chains
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
PLASMA_SYMBOL_NAME = "PlasmaUSD"
PLASMA_DECIMALS = 6
PLASMA_PRICE_USD = 1.0

XPL_SYMBOL_NAME = "XPL"
XPL_DECIMALS = 18
XPL_PRICE_USD = 1.0

# =========================
# Helpers
# =========================
def get_secret(name: str) -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, "")

def normalize_wallets(raw: str) -> list[str]:
    lines = [x.strip() for x in (raw or "").splitlines()]
    addrs = [x for x in lines if x]
    validish = [a for a in addrs if re.match(r"^0x[a-fA-F0-9]{40}$", a)]
    return list(dict.fromkeys(validish))

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

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Settings")
    SIM_API_KEY = st.text_input("SIM API Key", type="password", value=get_secret("SIM_API_KEY"))
    selected_chains = st.multiselect(
        "Chains to include",
        list(CHAIN_OPTIONS.keys()),
        default=["Ethereum", "Optimism", "Arbitrum", "Polygon", "Base"],
    )
    st.caption("Tip: put your key in .streamlit/secrets.toml or export SIM_API_KEY in terminal.")

# =========================
# Main UI
# =========================
st.title("Dune SIM Wallet Tracker")

wallets_raw = st.text_area(
    "Wallets (one per line)",
    value="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    height=110,
)
wallets = normalize_wallets(wallets_raw)

if st.button("Fetch Balances", type="primary"):
    if not SIM_API_KEY:
        st.error("Missing SIM API key.")
        st.stop()
    if not wallets:
        st.warning("Please enter at least one wallet address.")
        st.stop()

    chain_ids = ",".join(str(CHAIN_OPTIONS[c]) for c in selected_chains if c in CHAIN_OPTIONS)

    rows: list[dict] = []
    for addr in wallets:
        try:
            data = fetch_balances(addr, SIM_API_KEY, chain_ids)
        except Exception as e:
            st.warning(f"{addr}: {e}")
            continue

        balances = (data.get("balances") or data.get("tokens") or [])
        for b in balances:
            sym_raw = (b.get("symbol") or "").strip()
            sym = sym_raw.upper()

            raw_amount = b.get("amount")
            price_sim = b.get("price_usd")
            value_sim = b.get("value_usd")
            token_addr = b.get("address")
            sim_dec = b.get("decimals")

            # Default human amount = raw / 10**decimals (if provided)
            human_amount = to_float(raw_amount, 0.0)
            if sim_dec is not None:
                try:
                    human_amount = to_float(raw_amount, 0.0) / (10 ** int(sim_dec))
                except Exception:
                    pass

            # --- Plasma override ---
            if sym == PLASMA_SYMBOL_NAME.upper():
                human_amount = to_float(raw_amount, 0.0) / (10 ** PLASMA_DECIMALS)
                price_usd = PLASMA_PRICE_USD
                value_usd = human_amount * price_usd

            # --- XPL override ---
            elif sym == XPL_SYMBOL_NAME.upper():
                human_amount = to_float(raw_amount, 0.0) / (10 ** XPL_DECIMALS)
                price_usd = XPL_PRICE_USD
                value_usd = human_amount * price_usd

            else:
                price_usd = to_float(price_sim, None)
                if value_sim is None and price_usd is not None:
                    value_usd = human_amount * price_usd
                else:
                    value_usd = to_float(value_sim, None)

            rows.append({
                "wallet": addr,
                "chain": (b.get("chain") or "").lower(),
                "symbol": sym,
                "amount": human_amount,
                "price_usd": price_usd,
                "value_usd": value_usd,
                "token_address": token_addr,
            })

    if not rows:
        st.info("No balances found.")
    else:
        df = pd.DataFrame(rows)

        for col in ("amount", "price_usd", "value_usd"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Top metrics
        total = float(df["value_usd"].fillna(0).sum()) if "value_usd" in df.columns else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total USD", f"${total:,.2f}")
        c2.metric("Wallets", df["wallet"].nunique() if "wallet" in df.columns else 0)
        c3.metric("Assets", df["symbol"].nunique() if "symbol" in df.columns else 0)

        st.divider()

        # By chain
        if {"chain", "value_usd"}.issubset(df.columns):
            by_chain = (
                df.groupby("chain", as_index=False)["value_usd"]
                  .sum()
                  .sort_values("value_usd", ascending=False)
            )
            st.subheader("By Chain")
            st.dataframe(by_chain, use_container_width=True, height=220)

        # Pretty display
        st.subheader("Balances")
        df_show = df.copy()
        if "amount" in df_show.columns:
            df_show["amount"] = df_show["amount"].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")
        if "price_usd" in df_show.columns:
            df_show["price_usd"] = df_show["price_usd"].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")
        if "value_usd" in df_show.columns:
            df_show["value_usd"] = df_show["value_usd"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

        show_cols = ["wallet", "chain", "symbol", "amount", "price_usd", "value_usd", "token_address"]
        st.dataframe(
            df_show.sort_values(by=["value_usd"], ascending=[False], na_position="last")[show_cols],
            use_container_width=True, height=460
        )

