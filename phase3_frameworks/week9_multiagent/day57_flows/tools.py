"""Same domains as Day 54, CrewAI tool syntax."""

from crewai.tools import tool

_INVOICES = {
    "INV-001": {"amount_eur": 49.00, "status": "paid", "date": "2026-06-01"},
    "INV-002": {"amount_eur": 149.00, "status": "unpaid", "date": "2026-07-01"},
}

_KNOWN_ERRORS = {
    "e401": "Authentication failed. Check your API key is set and not expired.",
    "e429": "Rate limit exceeded. Wait 60 seconds and retry with backoff.",
    "e503": "Service temporarily unavailable. Transient — retry shortly.",
}


@tool("Lookup Invoice")
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by ID, e.g. 'INV-001'. Returns status, amount, date."""
    key = invoice_id.strip().upper()
    if key not in _INVOICES:
        return f"ERROR: Invoice '{invoice_id}' not found. Known: {sorted(_INVOICES)}"
    inv = _INVOICES[key]
    return f"{key}: {inv['amount_eur']} EUR, status={inv['status']}, issued {inv['date']}"


@tool("Get Account Balance")
def get_account_balance(customer_id: str) -> str:
    """Get outstanding balance for a customer ID, e.g. 'CUST-42'."""
    return f"{customer_id}: outstanding 149.00 EUR"


@tool("Lookup Error Code")
def lookup_error_code(code: str) -> str:
    """Look up what a system error code means, e.g. 'E429'."""
    key = code.strip().lower()
    if key not in _KNOWN_ERRORS:
        return f"ERROR: Unknown code '{code}'. Known: {sorted(c.upper() for c in _KNOWN_ERRORS)}"
    return f"{code.upper()}: {_KNOWN_ERRORS[key]}"


@tool("Check Service Status")
def check_service_status(service_name: str) -> str:
    """Check whether a service is operational, e.g. 'api'."""
    return f"{service_name}: operational, 99.94% uptime"