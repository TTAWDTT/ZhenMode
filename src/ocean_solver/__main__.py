"""Allow python -m ocean_solver to use the same installed interface."""
from ocean_solver.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
