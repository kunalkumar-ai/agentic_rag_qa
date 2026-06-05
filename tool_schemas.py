TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search the 10-K filings for relevant content. "
                "Call once per focused sub-query. For multi-company or multi-year questions, "
                "call multiple times — once per company/year combination. "
                "Returns labeled source passages for the specified companies and years."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused question or topic to retrieve context for.",
                    },
                    "companies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Companies to search. Valid values: 'tesla', 'gm', 'ford'.",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filing years to search. Valid values: '2022', '2023', '2024', '2025'.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of source passages to return. Default 5. Use higher values (7-9) for multi-year queries.",
                        "default": 5,
                    },
                },
                "required": ["query", "companies", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_index",
            "description": (
                "Return the list of available companies and years in the corpus. "
                "Call this first when the question is ambiguous about which companies or years to search."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
