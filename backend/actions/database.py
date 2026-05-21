"""
actions/database.py — SQLite database setup and seed data.

Creates a real database with:
- customers table
- orders table
- order_items table

Runs once on startup, seeds with realistic demo data.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = "./data/support.db"


def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    Creates tables and seeds demo data if DB doesn't exist yet.
    Called once on server startup.
    """
    # Create data directory if needed
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # If DB already exists and has data, skip
    if os.path.exists(DB_PATH):
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        conn.close()
        if count > 0:
            print("[Database] Already initialized. Skipping.")
            return

    conn = get_connection()
    cursor = conn.cursor()

    # ── Create Tables ─────────────────────────────────────────────────────

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            phone         TEXT,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS orders (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id        TEXT UNIQUE NOT NULL,
            customer_id     INTEGER REFERENCES customers(id),
            status          TEXT NOT NULL,
            total           REAL NOT NULL,
            shipping_address TEXT,
            carrier         TEXT,
            tracking_number TEXT,
            order_date      TEXT,
            ship_date       TEXT,
            delivery_date   TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS order_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id    TEXT REFERENCES orders(order_id),
            product     TEXT NOT NULL,
            quantity    INTEGER NOT NULL,
            price       REAL NOT NULL
        );
    """)

    # ── Seed Demo Data ────────────────────────────────────────────────────

    today = datetime.now()

    customers = [
        (1, "Ahmed Hassan",   "ahmed@example.com",   "+20-100-123-4567"),
        (2, "Sara Mohamed",   "sara@example.com",    "+20-101-234-5678"),
        (3, "Omar Khalil",    "omar@example.com",    "+20-102-345-6789"),
        (4, "Nour Ibrahim",   "nour@example.com",    "+20-103-456-7890"),
        (5, "Hana Tarek",     "hana@example.com",    "+20-104-567-8901"),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO customers (id, name, email, phone) VALUES (?,?,?,?)",
        customers
    )

    orders = [
        # (order_id, customer_id, status, total, address, carrier, tracking, order_date, ship_date, delivery_date)
        (
            "ORD-123", 5, "shipped", 79.99,
            "123 Main St, Cairo, Egypt",
            "FedEx", "FX480248927",
            (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            (today - timedelta(days=3)).strftime("%Y-%m-%d"),
            (today + timedelta(days=2)).strftime("%Y-%m-%d"),
        ),
        (
            "ORD-456", 5, "processing", 149.99,
            "123 Main St, Cairo, Egypt",
            None, None,
            (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            None, None,
        ),
        (
            "ORD-789", 5, "delivered", 49.99,
            "123 Main St, Cairo, Egypt",
            "UPS", "UP123456789",
            (today - timedelta(days=10)).strftime("%Y-%m-%d"),
            (today - timedelta(days=8)).strftime("%Y-%m-%d"),
            (today - timedelta(days=5)).strftime("%Y-%m-%d"),
        ),
        (
            "ORD-999", 5, "cancelled", 29.99,
            "123 Main St, Cairo, Egypt",
            None, None,
            (today - timedelta(days=2)).strftime("%Y-%m-%d"),
            None, None,
        ),
        (
            "ORD-111", 1, "shipped", 199.99,
            "456 Nile St, Alexandria, Egypt",
            "DHL", "DH987654321",
            (today - timedelta(days=4)).strftime("%Y-%m-%d"),
            (today - timedelta(days=2)).strftime("%Y-%m-%d"),
            (today + timedelta(days=3)).strftime("%Y-%m-%d"),
        ),
        (
            "ORD-222", 2, "delivered", 89.99,
            "789 Pyramids Ave, Giza, Egypt",
            "FedEx", "FX111222333",
            (today - timedelta(days=15)).strftime("%Y-%m-%d"),
            (today - timedelta(days=13)).strftime("%Y-%m-%d"),
            (today - timedelta(days=10)).strftime("%Y-%m-%d"),
        ),
    ]

    cursor.executemany(
        """INSERT OR IGNORE INTO orders
           (order_id, customer_id, status, total, shipping_address,
            carrier, tracking_number, order_date, ship_date, delivery_date)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        orders
    )

    items = [
        ("ORD-123", "Wireless Headphones",  1, 79.99),
        ("ORD-456", "Laptop Stand",         1, 49.99),
        ("ORD-456", "USB-C Hub",            2, 50.00),
        ("ORD-789", "Phone Case",           1, 49.99),
        ("ORD-999", "Screen Protector",     2, 14.99),
        ("ORD-111", "Mechanical Keyboard",  1, 199.99),
        ("ORD-222", "Wireless Mouse",       1, 89.99),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO order_items (order_id, product, quantity, price) VALUES (?,?,?,?)",
        items
    )

    conn.commit()
    conn.close()
    print("[Database] Initialized with demo data.")


# ── Query Functions ───────────────────────────────────────────────────────────

def get_order(order_id: str) -> Optional[dict]:
    """Fetches full order details including items."""
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
        "order_id":        order["order_id"],
        "status":          order["status"],
        "total":           order["total"],
        "customer_name":   order["customer_name"],
        "shipping_address": order["shipping_address"],
        "carrier":         order["carrier"],
        "tracking_number": order["tracking_number"],
        "order_date":      order["order_date"],
        "ship_date":       order["ship_date"],
        "delivery_date":   order["delivery_date"],
        "items": [
            {"product": i["product"], "quantity": i["quantity"], "price": i["price"]}
            for i in items
        ],
    }


def update_order_status(order_id: str, new_status: str) -> bool:
    """Updates order status. Returns True if updated."""
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE orders SET status = ? WHERE order_id = ?",
        (new_status, order_id.upper())
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0