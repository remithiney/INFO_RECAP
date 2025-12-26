import csv
from datetime import datetime
from pathlib import Path

from src.config import CACHE_FILE, DATA_DIR
from src.ingest import ingest_from_universe_file
from src.nlp import detect_languages
from src.utils import Progress


def _save_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'title',
        'content',
        'date_iso',
        'authors',
        'feed_source',
        'link',
        'lang',
        'lang_score',
    ]

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None

    if pd is not None:
        df = pd.DataFrame(records, columns=columns)
        df.to_csv(path, index=False)
        return

    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _save_errors(errors, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['url', 'error', 'entries']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for error in errors:
            writer.writerow(error)


def _print_stats(stats):
    print('ingest stats')
    for key, value in stats.items():
        print(f'- {key}: {value}')


def run_ingest(
    universe_path,
    min_chars,
    max_chars,
    long_action,
    max_workers,
    timeout,
    detect_lang,
    fasttext_model_path,
    out_csv,
    errors_csv,
    cache_path,
):
    progress = Progress()
    try:
        records, stats, errors = ingest_from_universe_file(
            universe_path,
            min_chars,
            max_chars,
            long_action,
            max_workers,
            timeout,
            cache_path,
            progress,
        )
        if detect_lang:
            detect_languages(records, fasttext_model_path, progress)
        progress.set_stage('save outputs', 1)
        if out_csv:
            _save_csv(records, out_csv)
        if errors_csv and errors:
            _save_errors(errors, errors_csv)
        progress.update(1)
    finally:
        progress.close()
    _print_stats(stats)
    return records, stats, errors


def resolve_out_csv(universe_path, out_csv):
    if out_csv:
        return Path(out_csv)
    universe_path = Path(universe_path)
    name = universe_path.stem or 'universe'
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return DATA_DIR / f'ingest_{name}_{stamp}.csv'


def resolve_errors_csv(universe_path, errors_csv):
    if errors_csv:
        return Path(errors_csv)
    universe_path = Path(universe_path)
    name = universe_path.stem or 'universe'
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return DATA_DIR / f'ingest_{name}_{stamp}_errors.csv'


def resolve_cache_path(cache_path):
    if cache_path:
        return Path(cache_path)
    return CACHE_FILE
