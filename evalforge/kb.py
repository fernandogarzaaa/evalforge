"""Knowledge base of the synthetic "Acme Cloud" product.

It plays two roles: the synthetic traffic generator uses it to write good and bad
responses, and the labelers use it as *source data* to verify factual answers
(Phase 3.1: "for factual questions, verify against source data if available").
In a real deployment this is your docs/help-centre index.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KBEntry:
    intent: str
    feature: str
    question: str
    answer: str
    key_facts: tuple[str, ...]
    traps: tuple[str, ...] = ()  # known-wrong claims (hallucination traps)
    subjective: bool = False  # no single golden answer -> rubric
    rubric: tuple[str, ...] = field(default_factory=tuple)
    keywords: str = ""  # search keywords, as a help-centre article would carry


KB: tuple[KBEntry, ...] = (
    KBEntry(
        "reset_password",
        "support_bot",
        "How do I reset my password?",
        "Open Settings > Security and click Reset password; the reset link expires after 30 minutes.",
        ("Settings > Security", "30 minutes"),
        ("contact your bank", "expires after 7 days"),
    ),
    KBEntry(
        "billing_refund",
        "support_bot",
        "How do I get a refund?",
        "Refunds are available within 14 days of purchase from Billing > Invoices > Request refund.",
        ("14 days", "Billing > Invoices"),
        ("90 days", "refunds are not possible"),
    ),
    KBEntry(
        "cancel_subscription",
        "support_bot",
        "How do I cancel my subscription?",
        "Go to Billing > Plan and choose Cancel plan; access continues until the end of the billing period.",
        ("Billing > Plan", "end of the billing period"),
        ("immediately deleted", "call our hotline"),
    ),
    KBEntry(
        "outage_status",
        "support_bot",
        "Is Acme Cloud down right now?",
        "Live incident status is published at status.acme.example and you can subscribe to updates there.",
        ("status.acme.example",),
        ("there are never outages", "restart your router"),
    ),
    KBEntry(
        "upgrade_plan",
        "support_bot",
        "How do I upgrade my plan?",
        "Upgrade from Billing > Plan > Change plan; the price difference is prorated for the current period.",
        ("Billing > Plan", "prorated"),
        ("full price again", "only sales can upgrade"),
    ),
    KBEntry(
        "api_rate_limit",
        "support_bot",
        "What are the API rate limits?",
        "The Pro plan allows 600 requests per minute and returns HTTP 429 with a Retry-After header when exceeded.",
        ("600 requests per minute", "429"),
        ("unlimited requests", "HTTP 500"),
    ),
    KBEntry(
        "sql_join",
        "sql_helper",
        "How do I join orders with customers?",
        "Use SELECT ... FROM orders JOIN customers ON orders.customer_id = customers.id.",
        ("JOIN customers", "orders.customer_id = customers.id"),
        ("CROSS JOIN", "orders.id = customers.id"),
    ),
    KBEntry(
        "sql_aggregate",
        "sql_helper",
        "How do I count orders per customer?",
        "Use SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id.",
        ("COUNT(*)", "GROUP BY customer_id"),
        ("ORDER BY COUNT", "DISTINCT *"),
    ),
    KBEntry(
        "sql_window",
        "sql_helper",
        "How do I rank orders by amount within each customer?",
        "Use RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC).",
        ("RANK() OVER", "PARTITION BY customer_id"),
        ("GROUP BY amount", "ROWNUM"),
    ),
    KBEntry(
        "python_bug",
        "code_assistant",
        "Why does my Python loop skip items when I remove from the list?",
        "Removing items while iterating shifts the indices; iterate over a copy such as list(items) or build a new list.",
        ("copy", "new list"),
        ("Python bug in the interpreter", "use global"),
    ),
    KBEntry(
        "regex_help",
        "code_assistant",
        "What regex matches an ISO date?",
        r"Use ^\d{4}-\d{2}-\d{2}$ and validate the month and day ranges separately.",
        (r"\d{4}-\d{2}-\d{2}",),
        (r"\w+@\w+", "[0-9]*"),
    ),
    KBEntry(
        "api_usage",
        "code_assistant",
        "How do I authenticate to the Acme API from Python?",
        "Send the API key in the Authorization: Bearer header, for example with requests.get(url, headers=...).",
        ("Authorization: Bearer",),
        ("put the key in the URL", "basic auth with your password"),
    ),
    KBEntry(
        "summarize_meeting",
        "doc_summarizer",
        "Summarize these meeting notes.",
        "A good summary lists decisions, owners and deadlines in a few bullet points.",
        (),
        (),
        True,
        (
            "lists the decisions made",
            "names an owner for each action item",
            "keeps deadlines exact",
            "is under 120 words",
        ),
    ),
    KBEntry(
        "summarize_contract",
        "doc_summarizer",
        "Summarize this contract clause.",
        "A good summary states the obligation, the parties and any termination or liability terms.",
        (),
        (),
        True,
        (
            "identifies the parties",
            "states the obligation",
            "mentions termination or liability terms",
            "adds no legal advice",
        ),
    ),
)

KEYWORDS = {
    "reset_password": "forgot change password reset link login locked out",
    "billing_refund": "refund money back invoice charge annual policy",
    "cancel_subscription": "cancel stop paying subscription plan data access end",
    "outage_status": "down outage not loading slow broken issues incident status",
    "upgrade_plan": "upgrade move plan starter pro enterprise cost renewal mid-month",
    "api_rate_limit": "api rate limit 429 errors throttling back off requests",
    "sql_join": "sql join combine orders customers names inner left",
    "sql_aggregate": "sql count number orders per customer group by grouped total duplicates",
    "sql_window": "sql rank top order per customer window functions row_number amount",
    "python_bug": "python loop skip remove removing elements list iterating modifying",
    "regex_help": "regex regular expression iso date yyyy-mm-dd validate",
    "api_usage": "api authenticate python key example calling store send",
    "summarize_meeting": "summarize meeting notes standup tl;dr decisions owners",
    "summarize_contract": "summarize contract clause plain english agreement terminate",
}
KB = tuple(KBEntry(**{**e.__dict__, "keywords": KEYWORDS.get(e.intent, "")}) for e in KB)
KB_BY_INTENT = {e.intent: e for e in KB}
