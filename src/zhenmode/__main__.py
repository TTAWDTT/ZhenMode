"""Allow python -m zhenmode to use the same installed interface."""
from zhenmode.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
