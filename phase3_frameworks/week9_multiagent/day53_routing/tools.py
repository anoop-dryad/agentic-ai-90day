"""Tools for three specialist domains."""

from langchain_core.tools import tool

# ---------- billing domain ----------

_INVOICES = {
    "INV-001": {"amount_eur": 49.00, "status": "paid",   "date": "2026-06-01"},
    "INV-002": {"amount_eur": 149.00, "status": "unpaid", "date": "2026-07-01"},
}


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by its ID.

    Args:
        invoice_id: Invoice identifier, e.g. 'INV-001'.
    """
    key = invoice_id.strip().upper()
    if key not in _INVOICES:
        return {"error": f"Invoice '{invoice_id}' not found.",
                "known_ids": sorted(_INVOICES)}
    return {"invoice_id": key, **_INVOICES[key]}


@tool
def get_account_balance(customer_id: str) -> dict:
    """Get the outstanding balance for a customer account.

    Args:
        customer_id: Customer identifier, e.g. 'CUST-42'.
    """
    return {"customer_id": customer_id, "outstanding_eur": 149.00, "currency": "EUR"}


# ---------- technical support domain ----------

_KNOWN_ERRORS = {
    "e401": "Authentication failed. Check your API key is set and not expired.",
    "e429": "Rate limit exceeded. Wait 60 seconds and retry with backoff.",
    "e503": "Service temporarily unavailable. This is transient — retry shortly.",
}


@tool
def lookup_error_code(code: str) -> dict:
    """Look up what a system error code means and how to fix it.

    Args:
        code: The error code, e.g. 'E429'.
    """
    key = code.strip().lower()
    if key not in _KNOWN_ERRORS:
        return {"error": f"Unknown error code '{code}'.",
                "known_codes": sorted(c.upper() for c in _KNOWN_ERRORS)}
    return {"code": code.upper(), "explanation": _KNOWN_ERRORS[key]}


@tool
def check_service_status(service_name: str) -> dict:
    """Check whether a service is currently operational.

    Args:
        service_name: Service to check, e.g. 'api' or 'dashboard'.
    """
    return {"service": service_name, "status": "operational", "uptime_pct": 99.94}


BILLING_TOOLS = [lookup_invoice, get_account_balance]
TECH_TOOLS = [lookup_error_code, check_service_status]