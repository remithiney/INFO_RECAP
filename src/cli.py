import argparse
from pathlib import Path

from src.config import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_CHARS,
    DEFAULT_TIMEOUT,
    DEFAULT_FASTTEXT_MODEL_PATH,
    UNIVERSES_DIR,
)
from src.pipeline import resolve_cache_path, resolve_errors_csv, resolve_out_csv, run_ingest


def _resolve_universe_path(args):
    if args.universe_file:
        return Path(args.universe_file)
    if args.universe:
        name = args.universe
        if name.endswith('.txt'):
            return UNIVERSES_DIR / name
        return UNIVERSES_DIR / f"{name}.txt"
    raise ValueError('Missing universe selection. Use --universe or --universe-file.')


def build_parser():
    parser = argparse.ArgumentParser(description='INFO_RECAP pipeline')
    sub = parser.add_subparsers(dest='command', required=True)

    ingest = sub.add_parser('ingest', help='Ingest RSS for a universe')
    ingest.add_argument('--universe', help='Universe name (file in config/universes)')
    ingest.add_argument('--universe-file', help='Path to universe txt file')
    ingest.add_argument('--min-chars', type=int, default=DEFAULT_MIN_CHARS)
    ingest.add_argument('--max-chars', type=int, default=DEFAULT_MAX_CHARS)
    ingest.add_argument('--long-action', choices=['skip', 'truncate'], default='skip')
    ingest.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS)
    ingest.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
    ingest.add_argument('--detect-lang', action='store_true')
    ingest.add_argument('--fasttext-model', default=DEFAULT_FASTTEXT_MODEL_PATH)
    ingest.add_argument('--out-csv', help='CSV output path')
    ingest.add_argument('--errors-csv', help='CSV errors output path')
    ingest.add_argument('--cache-file', help='Cache file path (etag/modified)')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'ingest':
        universe_path = _resolve_universe_path(args)
        out_csv = resolve_out_csv(universe_path, args.out_csv)
        errors_csv = resolve_errors_csv(universe_path, args.errors_csv)
        cache_path = resolve_cache_path(args.cache_file)
        run_ingest(
            universe_path,
            args.min_chars,
            args.max_chars,
            args.long_action,
            args.max_workers,
            args.timeout,
            args.detect_lang,
            args.fasttext_model,
            out_csv,
            errors_csv,
            cache_path,
        )


if __name__ == '__main__':
    main()
