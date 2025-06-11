def main():  # pragma: no cover
    """Thin wrapper that allows `python -m open_kevin.cli` to invoke training."""
    from .train import main as _train_main

    _train_main()


if __name__ == "__main__":  # pragma: no cover
    main() 