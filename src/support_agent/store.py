"""The demo store's database: customers, orders, actions (refunds/cancellations) and escalation tickets.

Every function that reads or changes an order takes the logged-in customer's ID, so the agent can
never see or touch another customer's orders, whatever the conversation says.
"""

import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "store.db"

CUSTOMERS = [
    ("C001", "Sara Ahmed", "en"),
    ("C002", "Omar Hassan", "ar"),
    ("C003", "Mona Adel", "en"),
]

# (order_id, customer_id, status, items, total EGP, days since ordered, days since delivered or None)
ORDERS = [
    ("ORD-1001", "C001", "delivered", "Wireless headphones", 2400.0, 12, 9),
    ("ORD-1002", "C001", "shipped", "Phone case, USB-C cable", 650.0, 2, None),
    ("ORD-1003", "C001", "processing", "Smart watch", 5200.0, 0, None),
    ("ORD-2001", "C002", "delivered", "Laptop stand", 900.0, 50, 45),
    ("ORD-2002", "C002", "delivered", "Bluetooth speaker", 1800.0, 7, 4),
    ("ORD-3001", "C003", "processing", "Air fryer", 3900.0, 1, None),
]


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB, today: date | None = None) -> None:
    """(Re)create the demo database. Dates are relative to today so the scenarios stay valid."""
    today = today or date.today()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS customers; DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS actions; DROP TABLE IF EXISTS tickets;
            CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT, language TEXT);
            CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id TEXT, status TEXT, items TEXT,
                                 total REAL, ordered_on TEXT, delivered_on TEXT);
            CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, customer_id TEXT,
                                  type TEXT, amount REAL, status TEXT, reviewer TEXT, note TEXT,
                                  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, summary TEXT,
                                  priority TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            """
        )
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    oid,
                    cid,
                    status,
                    items,
                    total,
                    str(today - timedelta(days=ordered)),
                    None if delivered is None else str(today - timedelta(days=delivered)),
                )
                for oid, cid, status, items, total, ordered, delivered in ORDERS
            ],
        )


def get_customer(conn, customer_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return dict(row) if row else None


def list_orders(conn, customer_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM orders WHERE customer_id = ? ORDER BY ordered_on DESC", (customer_id,))
    return [dict(r) for r in rows]


def get_order(conn, order_id: str, customer_id: str) -> dict | None:
    """Returns None both when the order doesn't exist and when it belongs to someone else,
    so the agent can't be used to probe other customers' order numbers."""
    order_id = order_id.strip().upper().lstrip("#")
    if re.fullmatch(r"\d+", order_id):  # "1001" -> "ORD-1001"
        order_id = f"ORD-{order_id}"
    row = conn.execute("SELECT * FROM orders WHERE id = ? AND customer_id = ?", (order_id, customer_id)).fetchone()
    return dict(row) if row else None


def record_action(conn, order: dict, action_type: str, status: str, reviewer: str = "", note: str = "") -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO actions (order_id, customer_id, type, amount, status, reviewer, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (order["id"], order["customer_id"], action_type, order["total"], status, reviewer, note),
        )
        if status == "completed":
            new_status = "refunded" if action_type == "refund" else "cancelled"
            conn.execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order["id"]))
    return cur.lastrowid


def create_ticket(conn, customer_id: str, summary: str, priority: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO tickets (customer_id, summary, priority) VALUES (?, ?, ?)", (customer_id, summary, priority)
        )
    return cur.lastrowid


def recent(conn, table: str, limit: int = 20) -> list[dict]:
    assert table in {"actions", "tickets"}
    return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))]
