"""Store and recall one private memory with Retold's lightweight local profile and no API key.

The first run downloads MiniLM and the NLI evidence model. Later runs reuse the local model cache.
"""

from pathlib import Path

from retold import Retold


def main(database: str | Path = "memory.sqlite") -> None:
    with Retold.open(database, profile="lite") as retold:
        with retold.session(user_id="aditya") as memory:
            memory.remember(
                "I prefer concise answers.",
                evidence="I prefer concise answers.",
                attribute="answer_style",
            )
            result = memory.search("What kind of answers do I prefer?")
            print(result.text)


if __name__ == "__main__":
    main()
