from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / 'config'
UNIVERSES_DIR = CONFIG_DIR / 'universes'
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 5000
DEFAULT_MAX_WORKERS = 6
DEFAULT_TIMEOUT = 10
CACHE_FILE = DATA_DIR / 'rss_cache.json'
DEFAULT_FASTTEXT_MODEL_PATH = ''
