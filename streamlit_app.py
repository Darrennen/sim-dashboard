import os
import re
import json
import sqlite3
import requests
import pandas as pd
import datetime as dt
from contextlib import closing
import streamlit as st

# =========================
# App config
# =========================
st.set_page_config(
    page_title="Portfolio Tracker - Your go-to for Ethereum and EVM", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for DeBank-like styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        padding-top: 2rem;
        background-color: #f8fafc;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Header styling */
    .dashboard-header {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Wallet section styling */
    .wallet-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Token list styling */
    .token-row {
        display: flex;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    
    /* Chain filter buttons */
    .chain-filter {
        display: inline-block;
        background: #f1f5f9;
        color: #374151;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.875rem;
        border: 1px solid #e2e8f0;
    }
    
    .chain-filter.active {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

BASE_SIM = "https://api.sim.dune.com/v1/evm"

# SIM-supported common EVM chains (expanded to match DeBank coverage)
CHAIN_OPTIONS = {
    "Ethereum": 1, "Optimism": 10, "Arbitrum": 42161, "Polygon": 137,
    "Base": 8453, "BNB Smart Chain": 56, "Avalanche": 43114, "Fantom": 250,
    "Gnosis Chain": 100, "Moonbeam": 1284, "Blast": 81457, "Scroll": 534352,
    "Linea": 59144, "zkSync Era": 324, "Mantle": 5000, "Celo": 42220,
    "Aurora": 1313161554, "Metis": 1088, "Cronos": 25, "Moonriver": 1285
}

# Plasma & XPL overrides
PLASMA_SYMBOL_NAME = "PlasmaUSD"; PLASMA_DECIMALS = 6; PLASMA_PRICE_USD = 1.0
XPL_SYMBOL_NAME = "XPL";          XPL_DECIMALS    = 18; XPL_PRICE_USD    = 1.0

# =========================
# Session state
# =========================
if "balances_df" not in st.session_state:
    st.session_state.balances_df = pd.DataFrame()
if "selected_wallet" not in st.session_state:
    st.session_state.selected_wallet = None

# =========================
# Database functions (same as before)
# =========================
DB_PATH = os.getenv("APP_DB_PATH", "data/app.sqlite")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        ts TEXT,
        wallet TEXT,
        data_json TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS notes (
        wallet TEXT PRIMARY KEY,
        note TEXT
    )""")
    c.commit()
    return c

def save_snapshot_df(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    ts = dt.datetime.now().isoformat(timespec="seconds")
    with closing(_conn()) as c:
        for w in df["wallet"].dropna().unique():
            wdf = df[df["wallet"] == w]
            rec = {
                "total_value": float(wdf["value_usd"].fillna(0).sum()),
                "rows": wdf.to_dict(orient="records"),
            }
            c.execute("INSERT INTO snapshots (ts, wallet, data_json) VALUES (?, ?, ?)",
                      (ts, w, json.dumps(rec)))
        c.commit()
    return ts

def load_latest_snapshot() -> pd.DataFrame:
    with closing(_conn()) as c:
        rows = c.execute("""
            SELECT s1.ts, s1.wallet, s1.data_json
            FROM snapshots s1
            JOIN (
                SELECT wallet, MAX(ts) AS max_ts
                FROM snapshots
                GROUP BY wallet
            ) s2 ON s1.wallet = s2.wallet AND s1.ts = s2.max_ts
        """).fetchall()
    all_rows = []
    for _, _, data_json in rows:
        try:
            payload = json.loads(data_json)
            all_rows.extend(payload.get("rows", []))
        except Exception:
            pass
    return pd.DataFrame(all_rows)

def save_note(wallet: str, note: str):
    with closing(_conn()) as c:
        c.execute("INSERT OR REPLACE INTO notes (wallet, note) VALUES (?, ?)", (wallet, note))
        c.commit()

def load_note(wallet: str) -> str:
    with closing(_conn()) as c:
        row = c.execute("SELECT note FROM notes WHERE wallet=?", (wallet,)).fetchone()
        return row[0] if row else ""

def delete_wallet_data(wallet: str):
    """Delete all data for a specific wallet from both snapshots and notes"""
    with closing(_conn()) as c:
        # Delete from snapshots table
        c.execute("DELETE FROM snapshots WHERE wallet=?", (wallet,))
        # Delete from notes table
        c.execute("DELETE FROM notes WHERE wallet=?", (wallet,))
        c.commit()

# =========================
# Helper functions (same as before)
# =========================
def get_sim_key() -> str:
    try:
        return st.secrets["SIM_API_KEY"]
    except Exception:
        return os.getenv("SIM_API_KEY", "")

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
    r = requests.get(url, headers={"X-Sim-Api-Key": api_key}, timeout=30)
    r.raise_for_status()
    return r.json() or {}

def fetch_balances_with_fallback(addr: str, api_key: str, chain_ids: str) -> dict:
    """Fetch balances with chain-by-chain fallback if batch request fails"""
    # First try all chains together
    try:
        return fetch_balances(addr, api_key, chain_ids)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            st.warning(f"Batch request failed for {addr[:8]}..., trying individual chains...")
            
            # If batch fails, try each chain individually
            combined_data = {"balances": []}
            chain_id_list = [int(x.strip()) for x in chain_ids.split(",") if x.strip()]
            
            for chain_id in chain_id_list:
                try:
                    chain_data = fetch_balances(addr, api_key, str(chain_id))
                    if chain_data.get("balances"):
                        # Add chain info to each balance
                        for balance in chain_data["balances"]:
                            balance["chain"] = str(chain_id)
                        combined_data["balances"].extend(chain_data["balances"])
                except Exception as chain_error:
                    chain_name = next((name for name, id in CHAIN_OPTIONS.items() if id == chain_id), str(chain_id))
                    st.warning(f"Chain {chain_name} failed for {addr[:8]}...: {chain_error}")
                    continue
            
            return combined_data
        else:
            raise e

def fetch_defi_positions_with_fallback(addr: str, api_key: str, chain_ids: str) -> dict:
    """Fetch DeFi positions with fallback handling"""
    try:
        return fetch_defi_positions(addr, api_key, chain_ids)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            # Try individual chains for DeFi positions too
            combined_data = {"positions": []}
            chain_id_list = [int(x.strip()) for x in chain_ids.split(",") if x.strip()]
            
            for chain_id in chain_id_list:
                try:
                    chain_data = fetch_defi_positions(addr, api_key, str(chain_id))
                    if chain_data.get("positions"):
                        combined_data["positions"].extend(chain_data["positions"])
                except Exception:
                    continue  # Silently continue for DeFi positions
            
            return combined_data
        else:
            return {}  # Return empty for DeFi positions errors

def parse_defi_positions(positions_data: dict, addr: str) -> list[dict]:
    """Parse DeFi positions data into standardized format"""
    rows = []
    
    # Handle different response structures
    positions = positions_data.get("positions", [])
    if not positions and "data" in positions_data:
        positions = positions_data["data"]
    
    for position in positions:
        try:
            # Extract common fields
            protocol = position.get("protocol_name", "Unknown")
            position_type = position.get("type", "defi")
            chain = position.get("chain", "ethereum").lower()
            
            # Handle different position structures
            tokens = position.get("tokens", [])
            if not tokens and "detail" in position:
                tokens = position["detail"].get("supply_token_list", []) + position["detail"].get("borrow_token_list", [])
            
            for token in tokens:
                symbol = (token.get("symbol") or "").upper()
                amount = float(token.get("amount", 0))
                price_usd = float(token.get("price", 0)) if token.get("price") else None
                value_usd = float(token.get("value", 0)) if token.get("value") else None
                
                if value_usd is None and price_usd is not None:
                    value_usd = amount * price_usd
                
                if amount > 0 or (value_usd and value_usd > 0):
                    rows.append({
                        "wallet": addr,
                        "chain": chain,
                        "symbol": f"{symbol}({protocol})" if protocol != "Unknown" else symbol,
                        "amount": amount,
                        "price_usd": price_usd,
                        "value_usd": value_usd,
                        "token_address": token.get("address", ""),
                        "protocol": protocol,
                        "position_type": position_type
                    })
        except Exception as e:
            continue
    
    return rows

def df_not_empty(obj) -> bool:
    return isinstance(obj, pd.DataFrame) and not obj.empty

def fmt_money(x): 
    return f"${x:,.2f}" if pd.notnull(x) else "$0.00"

def fmt_large_number(x):
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    elif x >= 1e3:
        return f"${x/1e3:.2f}K"
    else:
        return f"${x:.2f}"

# =========================
# Main UI
# =========================

# Header section
st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Portfolio Tracker</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #6b7280; font-size: 1rem;">Your go-to portfolio tracker for Ethereum and EVM</p>', unsafe_allow_html=True)

# Settings row
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    wallets_raw = st.text_area(
        "Wallet Addresses",
        value="",
        height=100,
        placeholder="Enter wallet addresses (one per line)"
    )

with col2:
    selected_chains = st.multiselect(
        "Select Chains",
        list(CHAIN_OPTIONS.keys()),
        default=["Ethereum", "Optimism", "Arbitrum", "Polygon", "Base", "BNB Smart Chain"],
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    fetch_btn = st.button("Fetch Balances", type="primary", use_container_width=True)
    
    SIM_API_KEY = get_sim_key()
    api_status = "Connected" if SIM_API_KEY else "Not Connected"
    status_color = "#10b981" if SIM_API_KEY else "#ef4444"
    st.markdown(f'<p style="color: {status_color}; font-size: 0.875rem;">API: {api_status}</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Preload last snapshot
if not df_not_empty(st.session_state.balances_df):
    cached = load_latest_snapshot()
    if not cached.empty:
        st.session_state.balances_df = cached.copy()

wallets = normalize_wallets(wallets_raw)

# =========================
# Fetch & build DF
# =========================
if fetch_btn:
    if not SIM_API_KEY:
        st.error("Missing SIM API key. Please add it in App Settings → Secrets.")
        st.stop()
    if not wallets:
        st.warning("Please enter at least one wallet address.")
        st.stop()

    chain_ids = ",".join(str(CHAIN_OPTIONS[c]) for c in selected_chains if c in CHAIN_OPTIONS)

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    rows: list[dict] = []
    total_wallets = len(wallets)
    
    for i, addr in enumerate(wallets):
        status_text.text(f"Fetching wallet {i+1}/{total_wallets}: {addr[:8]}...")
        progress_bar.progress((i + 1) / total_wallets)
        
        try:
            # Fetch regular token balances with fallback
            data = fetch_balances_with_fallback(addr, SIM_API_KEY, chain_ids)
            
            # Also try to fetch DeFi positions with fallback
            defi_data = fetch_defi_positions_with_fallback(addr, SIM_API_KEY, chain_ids)
            
        except Exception as e:
            st.error(f"Failed to fetch data for {addr}: {e}")
            continue

        # Process regular balances
        balances = (data.get("balances") or data.get("tokens") or [])
        for b in balances:
            sym_raw = (b.get("symbol") or "").strip()
            sym = sym_raw.upper()

            raw_amount = b.get("amount")
            price_sim = b.get("price_usd")
            value_sim = b.get("value_usd")
            token_addr = b.get("address")
            sim_dec = b.get("decimals")

            human_amount = to_float(raw_amount, 0.0)
            if sim_dec is not None:
                try:
                    human_amount = to_float(raw_amount, 0.0) / (10 ** int(sim_dec))
                except Exception:
                    pass

            if sym == PLASMA_SYMBOL_NAME.upper():
                human_amount = to_float(raw_amount, 0.0) / (10 ** PLASMA_DECIMALS)
                price_usd = PLASMA_PRICE_USD
                value_usd = human_amount * price_usd
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
                "protocol": "wallet",
                "position_type": "balance"
            })
        
        # Process DeFi positions
        defi_rows = parse_defi_positions(defi_data, addr)
        rows.extend(defi_rows)
    
    progress_bar.empty()
    status_text.empty()

    if not rows:
        st.info("No balances found.")
        st.stop()

    df = pd.DataFrame(rows)
    for col in ("amount", "price_usd", "value_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove rows with zero or null values
    df = df[(df["value_usd"].fillna(0) > 0) | (df["amount"].fillna(0) > 0)]

    st.session_state.balances_df = df
    st.session_state.selected_wallet = None
    ts_saved = save_snapshot_df(df)
    if ts_saved:
        st.success(f"Portfolio updated at {ts_saved}")
        
        # Show summary of fetched data
        total_found = df["value_usd"].fillna(0).sum()
        defi_positions = len(df[df.get("protocol", "") != "wallet"])
        regular_tokens = len(df[df.get("protocol", "") == "wallet"])
        
        st.info(f"Found: ${total_found:,.2f} total value | {regular_tokens} wallet tokens | {defi_positions} DeFi positions")

# =========================
# Dashboard Display
# =========================
if df_not_empty(st.session_state.balances_df):
    df_all = st.session_state.balances_df.copy()

    # Overview metrics
    total = float(df_all["value_usd"].fillna(0).sum()) if "value_usd" in df_all.columns else 0.0
    num_wallets = df_all["wallet"].nunique() if "wallet" in df_all.columns else 0
    num_tokens = df_all["symbol"].nunique() if "symbol" in df_all.columns else 0
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Balance", fmt_large_number(total))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Wallets", num_wallets)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tokens", num_tokens)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        chains_count = df_all["chain"].nunique() if "chain" in df_all.columns else 0
        st.metric("Chains", chains_count)
        st.markdown('</div>', unsafe_allow_html=True)

    # Wallets section
    st.markdown('<div class="wallet-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Wallets</h2>', unsafe_allow_html=True)
    
    if {"wallet", "value_usd"}.issubset(df_all.columns):
        wallet_table = (
            df_all.groupby("wallet", as_index=False)["value_usd"]
                  .sum()
                  .rename(columns={"value_usd": "total_usd"})
                  .sort_values("total_usd", ascending=False)
        )
        
        # Wallet list with enhanced styling
        for i, row in wallet_table.iterrows():
            wallet_addr = row['wallet']
            total_value = row["total_usd"]
            
            # Wallet container
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{wallet_addr[:8]}...{wallet_addr[-6:]}**")
                    st.caption(wallet_addr)
                
                with col2:
                    st.markdown(f"**{fmt_money(total_value)}**")
                
                with col3:
                    current_note = load_note(wallet_addr)
                    comment_key = f"comment_{wallet_addr}_{i}"
                    
                    new_comment = st.text_input(
                        "Note",
                        value=current_note,
                        key=comment_key,
                        placeholder="Add a note...",
                        label_visibility="collapsed"
                    )
                    
                    if new_comment != current_note:
                        if st.button("💾", key=f"save_{wallet_addr}_{i}", help="Save note"):
                            save_note(wallet_addr, new_comment)
                            st.success("Saved!")
                            st.rerun()
                
                with col4:
                    # Create two buttons side by side
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        if st.button("View", key=f"view_{i}", use_container_width=True):
                            st.session_state.selected_wallet = wallet_addr
                            st.rerun()
                    
                    with sub_col2:
                        if st.button("🗑️", key=f"delete_{i}", help="Delete wallet", use_container_width=True):
                            # Store the wallet to delete in session state for confirmation
                            st.session_state[f"confirm_delete_{wallet_addr}"] = True
                            st.rerun()
                
                # Show confirmation dialog if delete was clicked
                if st.session_state.get(f"confirm_delete_{wallet_addr}", False):
                    st.warning(f"⚠️ Are you sure you want to delete wallet {wallet_addr[:8]}...{wallet_addr[-6:]}?")
                    col_yes, col_no, col_spacer = st.columns([1, 1, 3])
                    
                    with col_yes:
                        if st.button("Yes, Delete", key=f"confirm_yes_{i}", type="secondary"):
                            try:
                                # Delete from database
                                delete_wallet_data(wallet_addr)
                                
                                # Remove from current session data
                                if df_not_empty(st.session_state.balances_df):
                                    st.session_state.balances_df = st.session_state.balances_df[
                                        st.session_state.balances_df["wallet"] != wallet_addr
                                    ]
                                
                                # Clear any selection of this wallet
                                if st.session_state.selected_wallet == wallet_addr:
                                    st.session_state.selected_wallet = None
                                
                                # Clear confirmation state
                                if f"confirm_delete_{wallet_addr}" in st.session_state:
                                    del st.session_state[f"confirm_delete_{wallet_addr}"]
                                
                                st.success(f"Wallet {wallet_addr[:8]}...{wallet_addr[-6:]} deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting wallet: {e}")
                    
                    with col_no:
                        if st.button("Cancel", key=f"confirm_no_{i}"):
                            # Clear confirmation state
                            if f"confirm_delete_{wallet_addr}" in st.session_state:
                                del st.session_state[f"confirm_delete_{wallet_addr}"]
                            st.rerun()
                
                st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Token Holdings section
    if st.session_state.selected_wallet:
        sel = st.session_state.selected_wallet
        df_sel = df_all[df_all["wallet"] == sel].copy()
        
        st.markdown('<div class="wallet-section">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="section-title">Token Holdings - {sel[:8]}...{sel[-6:]}</h2>', unsafe_allow_html=True)
        
        if not df_sel.empty:
            # Token list with protocol information
            df_tokens = df_sel.groupby(['symbol', 'chain', 'protocol']).agg({
                'amount': 'sum',
                'value_usd': 'sum',
                'price_usd': 'first',
                'position_type': 'first'
            }).reset_index().sort_values('value_usd', ascending=False)
            
            for _, token in df_tokens.iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    # Show protocol badge for DeFi positions
                    if token.get('protocol', 'wallet') != 'wallet':
                        st.markdown(f"**{token['symbol']}** `{token['protocol']}`")
                        st.caption(f"on {token['chain'].title()} • {token.get('position_type', 'defi')}")
                    else:
                        st.markdown(f"**{token['symbol']}**")
                        st.caption(f"on {token['chain'].title()}")
                
                with col2:
                    st.markdown(f"{token['amount']:,.6f}")
                
                with col3:
                    price = token['price_usd']
                    st.markdown(f"${price:,.4f}" if pd.notnull(price) else "N/A")
                
                with col4:
                    st.markdown(f"**{fmt_money(token['value_usd'])}**")
                
                st.divider()
        
        if st.button("← Back to All Wallets"):
            st.session_state.selected_wallet = None
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chain distribution
    if {"chain", "value_usd"}.issubset(df_all.columns) and not st.session_state.selected_wallet:
        by_chain = (
            df_all.groupby("chain", as_index=False)["value_usd"]
                  .sum()
                  .sort_values("value_usd", ascending=False)
        )
        
        st.markdown('<div class="wallet-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Chain Distribution</h2>', unsafe_allow_html=True)
        
        for _, chain_data in by_chain.iterrows():
            chain_name = chain_data['chain'].title()
            chain_value = chain_data['value_usd']
            percentage = (chain_value / total * 100) if total > 0 else 0
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"**{chain_name}**")
            with col2:
                st.markdown(f"{fmt_money(chain_value)}")
            with col3:
                st.markdown(f"{percentage:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Debug section for troubleshooting
    if st.checkbox("🔍 Show Debug Info (compare with DeBank)", value=False):
        st.markdown('<div class="wallet-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Debug Information</h2>', unsafe_allow_html=True)
        
        # Show protocol breakdown
        if "protocol" in df_all.columns:
            protocol_breakdown = df_all.groupby("protocol")["value_usd"].sum().sort_values(ascending=False)
            st.markdown("**By Protocol:**")
            for protocol, value in protocol_breakdown.items():
                st.markdown(f"- {protocol}: {fmt_money(value)}")
        
        # Show all unique symbols found
        st.markdown("**All Tokens Found:**")
        if "symbol" in df_all.columns:
            symbols_with_values = df_all.groupby("symbol")["value_usd"].sum().sort_values(ascending=False)
            for symbol, value in symbols_with_values.head(20).items():
                st.markdown(f"- {symbol}: {fmt_money(value)}")
        
        # Raw data table
        if st.checkbox("Show Raw Data Table"):
            st.dataframe(df_all.sort_values("value_usd", ascending=False).head(50))
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Enter wallet addresses and click 'Fetch Balances' to view your portfolio.")
