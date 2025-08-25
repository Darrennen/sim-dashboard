import os
import re
import json
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
# Session state
# =========================
if "balances_df" not in st.session_state:
    st.session_state.balances_df = pd.DataFrame()
if "selected_wallet" not in st.session_state:
    st.session_state.selected_wallet = None
# Per-wallet notes store (persist within session)
if "wallet_notes" not in st.session_state:
    st.session_state.wallet_notes = {}  # {wallet: "note text"}

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

def df_not_empty(obj) -> bool:
    return isinstance(obj, pd.DataFrame) and not obj.empty

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

    st.divider()
    st.subheader("Notes: Import / Export")
    # Export notes
    notes_json = json.dumps(st.session_state.wallet_notes, indent=2)
    st.download_button("Download notes JSON", data=notes_json, file_name="wallet_notes.json", mime="application/json")

    # Import notes (merge)
    uploaded = st.file_uploader("Upload notes JSON to merge", type=["json"])
    if uploaded:
        try:
            new_notes = json.loads(uploaded.read().decode("utf-8"))
            if isinstance(new_notes, dict):
                st.session_state.wallet_notes.update({k: str(v) for k, v in new_notes.items()})
                st.success("Notes imported & merged.")
            else:
                st.error("Invalid notes file (expected a JSON object).")
        except Exception as e:
            st.error(f"Failed to import notes: {e}")

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

colA, colB = st.columns([1, 3])
with colA:
    run = st.button("Fetch Balances", type="primary")
with colB:
    st.caption("Edit notes inline in the Wallets table. Click a wallet to drill into its assets.")

# =========================
# Fetch & build DF
# =========================
if run:
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
        st.stop()

    df = pd.DataFrame(rows)
    for col in ("amount", "price_usd", "value_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # cache for drilldown views
    st.session_state.balances_df = df
    st.session_state.selected_wallet = None  # reset selection on new fetch

# =========================
# Overview + Wallets table (inline note editing) + Drilldown
# =========================
def fmt_money(x, prec=2, prefix="$"):
    return f"{prefix}{x:,.{prec}f}"

if df_not_empty(st.session_state.balances_df):
    df_all = st.session_state.balances_df.copy()

    # ---- Top metrics
    total = float(df_all["value_usd"].fillna(0).sum()) if "value_usd" in df_all.columns else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total USD", fmt_money(total))
    c2.metric("Wallets", df_all["wallet"].nunique() if "wallet" in df_all.columns else 0)
    c3.metric("Assets", df_all["symbol"].nunique() if "symbol" in df_all.columns else 0)

    st.divider()

    # ---- By chain
    if {"chain", "value_usd"}.issubset(df_all.columns):
        by_chain = (
            df_all.groupby("chain", as_index=False)["value_usd"]
                  .sum()
                  .sort_values("value_usd", ascending=False)
        )
        st.subheader("By Chain")
        st.dataframe(by_chain, use_container_width=True, height=220)

    st.divider()

    # ---- Wallets summary (inline editable notes)
    st.subheader("Wallets")

    if {"wallet", "value_usd"}.issubset(df_all.columns):
        wallet_table = (
            df_all.groupby("wallet", as_index=False)["value_usd"]
                  .sum()
                  .rename(columns={"value_usd": "total_usd"})
                  .sort_values("total_usd", ascending=False)
        )
    else:
        wallet_table = pd.DataFrame(columns=["wallet", "total_usd"])

    # Add note column (populate from session notes)
    wallet_table["note"] = wallet_table["wallet"].map(lambda w: st.session_state.wallet_notes.get(w, ""))

    # Show an editable table for the 'note' column (wallet & total_usd are locked)
    edited = st.data_editor(
        wallet_table,
        num_rows="fixed",
        use_container_width=True,
        height=300,
        column_config={
            "wallet": st.column_config.TextColumn("Wallet", disabled=True),
            "total_usd": st.column_config.NumberColumn("Total USD", disabled=True, format="$%.2f"),
            "note": st.column_config.TextColumn("Note", help="Edit notes here; auto-saves on change."),
        },
        key="wallets_editor",
    )

    # Auto-save edited notes back to session_state
    try:
        for _, row in edited.iterrows():
            st.session_state.wallet_notes[row["wallet"]] = str(row.get("note") or "").strip()
    except Exception:
        pass

    # View buttons to drill down each wallet
    st.caption("Click a wallet to view details:")
    for i, row in edited.iterrows():
        cols = st.columns([6, 2, 1])
        cols[0].write(f"{row['wallet']}")
        cols[1].write(fmt_money(float(row["total_usd"]) if pd.notnull(row["total_usd"]) else 0.0))
        if cols[2].button("View", key=f"view_{i}"):
            st.session_state.selected_wallet = row["wallet"]
            st.rerun()

    # ---- Drilldown pane
    if st.session_state.selected_wallet:
        sel = st.session_state.selected_wallet
        st.divider()
        st.subheader(f"Details for {sel}")

        df_sel = df_all[df_all["wallet"] == sel].copy()

        # Chain breakdown for this wallet
        if {"chain", "value_usd"}.issubset(df_sel.columns):
            chain_totals = (
                df_sel.groupby("chain", as_index=False)["value_usd"]
                      .sum()
                      .sort_values("value_usd", ascending=False)
            )
        else:
            chain_totals = pd.DataFrame(columns=["chain", "value_usd"])

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("**By Chain (USD)**")
            st.dataframe(chain_totals, use_container_width=True)

            # Also show the current note (read-only here; edit inline above)
            st.markdown("**Note (read-only here)**")
            st.text_area(
                "Note",
                value=st.session_state.wallet_notes.get(sel, ""),
                height=160,
                key=f"readonly_note_{sel}",
                disabled=True
            )

        with col2:
            st.markdown("**Holdings**")
            show_cols = ["chain", "symbol", "amount", "price_usd", "value_usd", "token_address"]
            show_cols = [c for c in show_cols if c in df_sel.columns]

            df_view = df_sel.copy()
            if "amount" in df_view.columns:
                df_view["amount"] = df_view["amount"].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")
            if "price_usd" in df_view.columns:
                df_view["price_usd"] = df_view["price_usd"].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")
            if "value_usd" in df_view.columns:
                df_view["value_usd"] = df_view["value_usd"].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

            st.dataframe(
                df_view.sort_values("value_usd", ascending=False, na_position="last")[show_cols],
                use_container_width=True, height=420
            )

        # Back button
        if st.button("Back to all wallets"):
            st.session_state.selected_wallet = None
            st.rerun()

else:
    st.info("Enter wallets and click Fetch Balances to begin.")
