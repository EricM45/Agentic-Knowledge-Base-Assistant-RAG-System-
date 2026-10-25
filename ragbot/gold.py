"""Questions with known answers in the sample Helios Labs knowledge base."""

GOLD: list[dict] = [
    {
        "question": "What is the standard warranty on the Helios Arc inverter?",
        "answer_contains": ["12"],
        "source_contains": "products.md",
    },
    {
        "question": "How many PTO days does a new employee receive?",
        "answer_contains": ["18", "paid time off"],
        "source_contains": "employee_policies.md",
    },
    {
        "question": "How much vacation does a new hire get in the first year?",
        "answer_contains": ["18", "paid time off"],
        "source_contains": "employee_policies.md",
    },
    {
        "question": "What should I do if an Arc inverter shows error E17?",
        "answer_contains": ["ground fault"],
        "source_contains": "support_faq.md",
    },
    {
        "question": "What is the monthly price of Gridline monitoring?",
        "answer_contains": ["8", "month"],
        "source_contains": "products.md",
    },
    {
        "question": "Who is the chief executive of Helios Labs?",
        "answer_contains": ["Amina Shah"],
        "source_contains": "company_overview.md",
    },
    {
        "question": "How much energy does the Helios Lumen battery store?",
        "answer_contains": ["13.5"],
        "source_contains": "products.md",
    },
    {
        "question": "How much paid parental leave does a primary caregiver receive?",
        "answer_contains": ["16 weeks"],
        "source_contains": "employee_policies.md",
    },
    {
        "question": "When does Helios schedule the site survey after a customer signs a proposal?",
        "answer_contains": ["5 business days"],
        "source_contains": "installation.md",
    },
    {
        "question": "Where do customers sign in to Helios Gridline?",
        "answer_contains": ["app.helioslabs.example"],
        "source_contains": "support_faq.md",
    },
    {
        "question": "Why will the Helios Lumen battery not charge in cold weather?",
        "answer_contains": ["32"],
        "source_contains": "support_faq.md",
    },
    {
        "question": "What is the capital of France?",
        "refuse": True,
    },
    {
        "question": "How many days are in a calendar year?",
        "refuse": True,
    },
]

SEMANTIC: list[dict] = [
    {
        "question": "How big is the home battery?",
        "answer_contains": ["13.5"],
        "source_contains": "products.md",
    },
    {
        "question": "How much time off do new people get?",
        "answer_contains": ["18"],
        "source_contains": "employee_policies.md",
    },
    {
        "question": "How long is the inverter covered?",
        "answer_contains": ["12"],
        "source_contains": "products.md",
    },
]

FOLLOWUPS: list[dict] = [
    {
        "history": [
            {"role": "user", "content": "What is the standard warranty on the Helios Arc inverter?"},
            {"role": "assistant", "content": "The standard warranty is 12 years."},
        ],
        "question": "How long can it be extended?",
        "answer_contains": ["25"],
        "source_contains": "products.md",
    },
]

SUGGESTIONS = [item["question"] for item in GOLD if not item.get("refuse")][:5]
