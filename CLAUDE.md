# Commit messages

All commits created by Claude Code in this repository must follow [Conventional Commits](https://www.conventionalcommits.org/).

Use the header format `<type>[optional scope][!]: <description>`. Use a short, imperative description that states the change. Scopes are optional and lowercase. Use lowercase types; prefer `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `style`, or `revert`.

Use `feat` for a new capability and `fix` for a bug fix. For a breaking change, add `!` immediately before the colon and include a `BREAKING CHANGE: <description>` footer when more context is needed. A body and other trailer-style footers are optional and must follow one blank line after the header or body.

Before committing, inspect the staged diff and choose the narrowest accurate type and scope. Do not use generic messages such as `update files`, `changes`, or `wip`.
