import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Optional

DB_PATH        = "./data/support.db"
SEED_DATA_PATH = "./data/seed_data.json"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    today = datetime.now()
    mapping = {
        "DAYS_AGO_1":   today - timedelta(days=1),
        "DAYS_AGO_2":   today - timedelta(days=2),
        "DAYS_AGO_3":   today - timedelta(days=3),
        "DAYS_AGO_4":   today - timedelta(days=4),
        "DAYS_AGO_5":   today - timedelta(days=5),
        "DAYS_AGO_8":   today - timedelta(days=8),
        "DAYS_AGO_10":  today - timedelta(days=10),
        "DAYS_AGO_13":  today - timedelta(days=13),
        "DAYS_AGO_15":  today - timedelta(days=15),
        "DAYS_AHEAD_2": today + timedelta(days=2),
        "DAYS_AHEAD_3": today + timedelta(days=3),
    }
    resolved = mapping.get(value)
    return resolved.strftime("%Y-%m-%d") if resolved else value


def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    if os.path.exists(DB_PATH):
        conn = get_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        conn.close()
        if count > 0:
            print("[Database] Already initialized. Skipping.")
            return

    if not os.path.exists(SEED_DATA_PATH):
        print(f"[Database] Seed file not found: {SEED_DATA_PATH}")
        return

    with open(SEED_DATA_PATH, "r") as f:
        seed = json.load(f)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            email      TEXT UNIQUE NOT NULL,
            phone      TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS orders (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id         TEXT UNIQUE NOT NULL,
            customer_id      INTEGER REFERENCES customers(id),
            status           TEXT NOT NULL,
            total            REAL NOT NULL,
            shipping_address TEXT,
            carrier          TEXT,
            tracking_number  TEXT,
            order_date       TEXT,
            ship_date        TEXT,
            delivery_date    TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS order_items (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT REFERENCES orders(order_id),
            product  TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price    REAL NOT NULL
        );
    """)

    for c in seed["customers"]:
        cursor.execute(
            "INSERT OR IGNORE INTO customers (id, name, email, phone) VALUES (?,?,?,?)",
            (c["id"], c["name"], c["email"], c["phone"])
        )

    for o in seed["orders"]:
        cursor.execute(
            """INSERT OR IGNORE INTO orders
               (order_id, customer_id, status, total, shipping_address,
                carrier, tracking_number, order_date, ship_date, delivery_date)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                o["order_id"], o["customer_id"], o["status"], o["total"],
                o["shipping_address"], o["carrier"], o["tracking_number"],
                _resolve_date(o["order_date"]),
                _resolve_date(o["ship_date"]),
                _resolve_date(o["delivery_date"]),
            )
        )

    for i in seed["order_items"]:
        cursor.execute(
            "INSERT OR IGNORE INTO order_items (order_id, product, quantity, price) VALUES (?,?,?,?)",
            (i["order_id"], i["product"], i["quantity"], i["price"])
        )

    conn.commit()
    conn.close()
    print("[Database] Initialized from seed_data.json.")


def get_order(order_id: str) -> Optional[dict]:
    conn = get_connection()

    order = conn.execute(
        """SELECT o.*, c.name as customer_name, c.email as customer_email
           FROM orders o
           LEFT JOIN customers c ON o.customer_id = c.id
           WHERE o.order_id = ?""",
        (order_id.upper(),)
    ).fetchone()

    if not order:
        conn.close()
        return None

    items = conn.execute(
        "SELECT product, quantity, price FROM order_items WHERE order_id = ?",
        (order_id.upper(),)
    ).fetchall()

    conn.close()

    return {
        "order_id":         order["order_id"],
        "status":           order["status"],
        "total":            order["total"],
        "customer_name":    order["customer_name"],
        "shipping_address": order["shipping_address"],
        "carrier":          order["carrier"],
        "tracking_number":  order["tracking_number"],
        "order_date":       order["order_date"],
        "ship_date":        order["ship_date"],
        "delivery_date":    order["delivery_date"],
        "items": [
            {"product": i["product"], "quantity": i["quantity"], "price": i["price"]}
            for i in items
        ],
    }


def update_order_status(order_id: str, new_status: str) -> bool:
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE orders SET status = ? WHERE order_id = ?",
        (new_status, order_id.upper())
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0