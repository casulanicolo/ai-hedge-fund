"""
src/portfolio/manager.py — Gestione posizioni aperte

Comandi:
    python -m src.portfolio open  TICKER DIRECTION ENTRY SL TP SIZE_USD [NOTE]
    python -m src.portfolio close TICKER CLOSE_PRICE [NOTE]
    python -m src.portfolio status
    python -m src.portfolio history

Esempi:
    python -m src.portfolio open  AAPL SHORT 254.98 266.21 232.53 1000
    python -m src.portfolio open  MSFT LONG  370.00 355.00 400.00 2000 "segnale forte"
    python -m src.portfolio close AAPL 240.00
    python -m src.portfolio status
"""

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "hedge_fund.db"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def ensure_table() -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS positions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker        TEXT    NOT NULL,
        direction     TEXT    NOT NULL CHECK(direction IN ('LONG','SHORT')),
        entry_price   REAL    NOT NULL,
        stop_loss     REAL    NOT NULL,
        take_profit   REAL    NOT NULL,
        size_usd      REAL    NOT NULL,
        status        TEXT    NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','CLOSED')),
        opened_at     TEXT    NOT NULL,
        closed_at     TEXT,
        close_price   REAL,
        pnl_usd       REAL,
        pnl_pct       REAL,
        note          TEXT
    );
    """
    with _conn() as c:
        c.execute(sql)
        c.commit()


# ---------------------------------------------------------------------------
# Operazioni principali
# ---------------------------------------------------------------------------

def open_position(
    ticker: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    size_usd: float,
    note: str = "",
) -> int:
    """Apre una nuova posizione. Ritorna l'id assegnato."""
    ensure_table()
    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"direction deve essere LONG o SHORT, ricevuto: {direction}")

    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO positions
               (ticker, direction, entry_price, stop_loss, take_profit, size_usd, opened_at, note)
               VALUES (?,?,?,?,?,?,?,?)""",
            (ticker.upper(), direction, entry_price, stop_loss, take_profit, size_usd, now, note),
        )
        c.commit()
        pos_id = cur.lastrowid

    rr = _calc_rr(direction, entry_price, stop_loss, take_profit)
    print(f"\n✅ Posizione aperta — id={pos_id}")
    print(f"   {direction} {ticker.upper()} | entry={entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}")
    print(f"   Size: ${size_usd:,.2f} | R/R: 1:{rr:.1f}")
    return pos_id


def close_position(ticker: str, close_price: float, note: str = "") -> None:
    """Chiude tutte le posizioni OPEN per il ticker specificato."""
    ensure_table()
    ticker = ticker.upper()
    now = datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM positions WHERE ticker=? AND status='OPEN'", (ticker,)
        ).fetchall()

        if not rows:
            print(f"⚠️  Nessuna posizione aperta trovata per {ticker}.")
            return

        for row in rows:
            pnl_pct, pnl_usd = _calc_pnl(
                row["direction"], row["entry_price"], close_price, row["size_usd"]
            )
            c.execute(
                """UPDATE positions
                   SET status='CLOSED', closed_at=?, close_price=?, pnl_usd=?, pnl_pct=?, note=?
                   WHERE id=?""",
                (now, close_price, pnl_usd, pnl_pct, note or row["note"], row["id"]),
            )
            sign = "+" if pnl_usd >= 0 else ""
            emoji = "🟢" if pnl_usd >= 0 else "🔴"
            print(f"\n{emoji} Posizione chiusa — id={row['id']}")
            print(f"   {row['direction']} {ticker} | entry={row['entry_price']:.2f} → close={close_price:.2f}")
            print(f"   P&L: {sign}${pnl_usd:,.2f} ({sign}{pnl_pct:.2f}%)")
        c.commit()


def get_open_positions() -> list[dict]:
    """Ritorna lista di dict con tutte le posizioni OPEN."""
    ensure_table()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY opened_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_positions() -> list[dict]:
    """Ritorna tutte le posizioni (aperte + chiuse)."""
    ensure_table()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM positions ORDER BY opened_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def print_status() -> None:
    """Stampa le posizioni aperte con P&L unrealised stimato (prezzo entry come proxy)."""
    ensure_table()
    positions = get_open_positions()

    if not positions:
        print("\n📭 Nessuna posizione aperta. Portafoglio in cash.")
        return

    print(f"\n{'='*72}")
    print(f"  POSIZIONI APERTE ({len(positions)})")
    print(f"{'='*72}")
    print(f"  {'ID':>4}  {'TICKER':<6}  {'DIR':<5}  {'ENTRY':>8}  {'SL':>8}  {'TP':>8}  {'SIZE $':>9}  {'R/R':>5}  {'APERTA IL'}")
    print(f"  {'-'*68}")
    for p in positions:
        rr  = _calc_rr(p["direction"], p["entry_price"], p["stop_loss"], p["take_profit"])
        dt  = p["opened_at"][:16].replace("T", " ")
        print(
            f"  {p['id']:>4}  {p['ticker']:<6}  {p['direction']:<5}  "
            f"{p['entry_price']:>8.2f}  {p['stop_loss']:>8.2f}  {p['take_profit']:>8.2f}  "
            f"{p['size_usd']:>9,.0f}  {rr:>5.1f}  {dt}"
        )
        if p.get("note"):
            print(f"         📝 {p['note']}")
    total_usd = sum(p["size_usd"] for p in positions)
    print(f"  {'-'*68}")
    print(f"  Esposizione totale: ${total_usd:,.2f}\n")


def print_history() -> None:
    """Stampa storico completo posizioni chiuse."""
    ensure_table()
    positions = [p for p in get_all_positions() if p["status"] == "CLOSED"]

    if not positions:
        print("\n📭 Nessuna posizione chiusa nel database.")
        return

    total_pnl = sum(p.get("pnl_usd") or 0 for p in positions)
    winners   = sum(1 for p in positions if (p.get("pnl_usd") or 0) > 0)
    losers    = sum(1 for p in positions if (p.get("pnl_usd") or 0) < 0)

    print(f"\n{'='*80}")
    print(f"  STORICO POSIZIONI CHIUSE ({len(positions)})")
    print(f"{'='*80}")
    print(f"  {'ID':>4}  {'TICKER':<6}  {'DIR':<5}  {'ENTRY':>8}  {'CLOSE':>8}  {'SIZE $':>9}  {'P&L $':>9}  {'P&L %':>7}")
    print(f"  {'-'*76}")
    for p in positions:
        pnl_usd = p.get("pnl_usd") or 0
        pnl_pct = p.get("pnl_pct") or 0
        sign    = "+" if pnl_usd >= 0 else ""
        emoji   = "🟢" if pnl_usd >= 0 else "🔴"
        print(
            f"  {emoji}{p['id']:>3}  {p['ticker']:<6}  {p['direction']:<5}  "
            f"{p['entry_price']:>8.2f}  {p.get('close_price') or 0:>8.2f}  "
            f"{p['size_usd']:>9,.0f}  {sign}{pnl_usd:>8,.2f}  {sign}{pnl_pct:>6.2f}%"
        )
    sign_tot = "+" if total_pnl >= 0 else ""
    print(f"  {'-'*76}")
    print(f"  P&L totale: {sign_tot}${total_pnl:,.2f}  |  Win: {winners}  Loss: {losers}")
    wr = winners / len(positions) * 100 if positions else 0
    print(f"  Win rate: {wr:.0f}%\n")


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _calc_pnl(direction: str, entry: float, close: float, size_usd: float):
    """Ritorna (pnl_pct, pnl_usd)."""
    if direction == "LONG":
        pnl_pct = (close - entry) / entry * 100
    else:
        pnl_pct = (entry - close) / entry * 100
    pnl_usd = size_usd * pnl_pct / 100
    return round(pnl_pct, 4), round(pnl_usd, 2)


def _calc_rr(direction: str, entry: float, sl: float, tp: float) -> float:
    """Calcola il rapporto Risk/Reward."""
    try:
        if direction == "LONG":
            risk   = abs(entry - sl)
            reward = abs(tp - entry)
        else:
            risk   = abs(sl - entry)
            reward = abs(entry - tp)
        return round(reward / risk, 2) if risk > 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Athanor Alpha — Portfolio Manager",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # open
    p_open = sub.add_parser("open", help="Apre una nuova posizione")
    p_open.add_argument("ticker",      type=str,   help="Es. AAPL")
    p_open.add_argument("direction",   type=str,   help="LONG o SHORT")
    p_open.add_argument("entry_price", type=float, help="Prezzo di entrata")
    p_open.add_argument("stop_loss",   type=float, help="Stop loss")
    p_open.add_argument("take_profit", type=float, help="Take profit")
    p_open.add_argument("size_usd",    type=float, help="Dimensione in USD")
    p_open.add_argument("note",        nargs="?",  default="", help="Nota opzionale")

    # close
    p_close = sub.add_parser("close", help="Chiude una posizione")
    p_close.add_argument("ticker",      type=str,   help="Es. AAPL")
    p_close.add_argument("close_price", type=float, help="Prezzo di chiusura")
    p_close.add_argument("note",        nargs="?",  default="", help="Nota opzionale")

    # status
    sub.add_parser("status",  help="Mostra posizioni aperte")
    sub.add_parser("history", help="Mostra storico posizioni chiuse")

    args = parser.parse_args()

    if args.command == "open":
        open_position(
            args.ticker, args.direction, args.entry_price,
            args.stop_loss, args.take_profit, args.size_usd, args.note,
        )
    elif args.command == "close":
        close_position(args.ticker, args.close_price, args.note)
    elif args.command == "status":
        print_status()
    elif args.command == "history":
        print_history()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
