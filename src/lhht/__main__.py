"""Allow ``python -m lhht`` to behave like the installed CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
