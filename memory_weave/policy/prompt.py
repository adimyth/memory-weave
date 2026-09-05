"""Versioned framework prompt text that explains safe use of the memory tools."""

MEMORY_USE_POLICY_VERSION = "v1"

MEMORY_USE_POLICY = " ".join(
    (
        "You have long-term memory available through tools.",
        "Before acting on anything that could depend on the user's preferences, earlier decisions, or previous sessions, call `memory_search` with one to three specific phrases.",  # noqa: E501
        "Do not search for general knowledge or for facts already visible in this conversation.",
        "When results come back, check their status, source, and date before relying on them; a provisional or old record may be wrong, and you can ask the user.",  # noqa: E501
        "When the user states a preference, a fact about themselves, or a decision, save it with `memory_write` and quote their words as evidence.",  # noqa: E501
        "Save decisions you make together as episodic records with the reason.",
        "Do not save guesses as facts.",
    )
)

AUTO_MEMORY_USE_POLICY = " ".join(
    (
        "Relevant memories may also appear automatically before you answer, marked as recalled memory.",
        "Treat them exactly like search results: check their status, source, and date, and use `memory_search` yourself for anything more specific.",  # noqa: E501
    )
)
