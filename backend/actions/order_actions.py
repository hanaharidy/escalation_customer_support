"""
actions/order_actions.py — Order actions connected to real SQLite database.

"""

from datetime import datetime, timedelta
from typing import Optional
from backend.actions.database import get_order, update_order_status


def lookup_order(order_id: str) -> dict:
    """
    Looks up real order from SQLite database.
    """
    if not order_id:
        return {"success": False, "error": "No order ID provided."}

    order = get_order(order_id.upper())

    if not order:
        return {
            "success": False,
            "error": f"Order {order_id.upper()} not found in our system. Please check your order ID.",
        }

    # Build response based on real status
    result = {
        "success":          True,
        "order_id":         order["order_id"],
        "status":           order["status"],
        "total":            order["total"],
        "items":            order["items"],
        "shipping_address": order["shipping_address"],
        "order_date":       order["order_date"],
    }

    if order["status"] == "shipped":
        result["carrier"]         = order["carrier"]
        result["tracking_number"] = order["tracking_number"]
        result["ship_date"]       = order["ship_date"]
        result["delivery_date"]   = order["delivery_date"]
        result["message"]         = f"Your order is on its way via {order['carrier']}. Tracking: {order['tracking_number']}."

    elif order["status"] == "delivered":
        result["delivery_date"] = order["delivery_date"]
        result["message"]       = f"Your order was delivered on {order['delivery_date']}."

    elif order["status"] == "processing":
        result["message"] = "Your order is being processed and will ship within 1-2 business days."

    elif order["status"] == "cancelled":
        result["message"] = "This order has been cancelled. If you didn't request this, please contact support immediately."

    return result


def process_refund(order_id: str, reason: str) -> dict:
    """
    Processes refund — reads from DB to get real order total.
    """
    if not order_id:
        return {"success": False, "error": "No order ID provided."}

    order = get_order(order_id.upper())

    if not order:
        return {
            "success": False,
            "error": f"Order {order_id.upper()} not found.",
        }

    # Can't refund cancelled or already processing refund
    if order["status"] == "cancelled":
        return {
            "success": False,
            "error": "This order is already cancelled.",
        }

    # Generate refund ID
    import random
    refund_id = f"REF-{random.randint(10000, 99999)}"

    # Update order status in DB
    update_order_status(order_id.upper(), "refund_initiated")

    return {
        "success":         True,
        "refund_id":       refund_id,
        "order_id":        order_id.upper(),
        "amount":          order["total"],
        "reason":          reason,
        "status":          "initiated",
        "processing_time": "5-7 business days",
        "refund_method":   "Original payment method",
        "message":         f"Refund {refund_id} initiated for ${order['total']:.2f}. You'll receive it within 5-7 business days.",
    }


def check_return_eligibility(order_id: str) -> dict:
    """
    Checks return eligibility based on real order date from DB.
    """
    if not order_id:
        return {"success": False, "error": "No order ID provided."}

    order = get_order(order_id.upper())

    if not order:
        return {
            "success": False,
            "error": f"Order {order_id.upper()} not found.",
        }

    if not order["order_date"]:
        return {"success": False, "error": "Order date not available."}

    order_date     = datetime.strptime(order["order_date"], "%Y-%m-%d")
    days_since     = (datetime.now() - order_date).days
    eligible       = days_since <= 30
    days_remaining = max(0, 30 - days_since)

    return {
        "success":        True,
        "order_id":       order_id.upper(),
        "eligible":       eligible,
        "days_since":     days_since,
        "days_remaining": days_remaining,
        "order_status":   order["status"],
        "message": (
            f"Your order is eligible for return. You have {days_remaining} days remaining."
            if eligible else
            f"Return window has expired ({days_since} days since order)."
        ),
    }


# ── Tool Registry ─────────────────────────────────────────────────────────────

ACTION_TOOLS = {
    "order_tracking": lookup_order,
    "refund":         process_refund,
    "product_issue":  check_return_eligibility,
}


def get_available_actions() -> list:
    return list(ACTION_TOOLS.keys())