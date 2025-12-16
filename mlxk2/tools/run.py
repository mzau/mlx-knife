"""mlx-run: thin wrapper that forces the 'run' subcommand as $0."""

import sys
from mlxk2.cli import main as mlxk_main


def main() -> None:
    # Insert the 'run' subcommand so `mlx-run <model> ...` maps to `mlxk run ...`
    sys.argv.insert(1, "run")
    mlxk_main()


if __name__ == "__main__":
    main()
