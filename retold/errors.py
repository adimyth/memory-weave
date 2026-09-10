"""Exceptions that a host is expected to handle by name."""

from __future__ import annotations


class LocalModelUnavailable(RuntimeError):
    """A local model could not be imported or loaded.

    Raised by the embedder, the NLI judge, and the reranker when ``sentence-transformers`` is not installed
    or the model files are not in the local cache. It is the one failure a host fixes by installing the
    ``local-models`` extra or allowing the first download; every other ``RuntimeError`` means something else.
    """


class UnsupportedEvidenceError(ValueError):
    """The quote does not support the claim, so the write would be stored as an expiring inference.

    Raised by the facade's ``remember`` for a ``user_statement`` or ``tool_result`` claim the evidence check
    downgraded. Pass ``allow_inference=True`` to accept the provisional record deliberately.
    """

    def __init__(self, claim: str, evidence: str, note: str | None) -> None:
        self.claim = claim
        self.evidence = evidence
        self.note = note
        reason = note or "the evidence does not support the claim"
        super().__init__(
            f"The quote {evidence!r} does not support {claim!r} ({reason}), so Retold would store it as an "
            "expiring inference. Phrase the content the way the quote says it, or pass allow_inference=True."
        )
