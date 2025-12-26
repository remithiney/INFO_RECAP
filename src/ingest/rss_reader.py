import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

from .html_cleaner import clean_html


def load_feed_urls(path):
    urls = []
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            urls.append(line)
    return urls


def _is_valid_url(value):
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in ('http', 'https') and bool(parsed.netloc)


def _dedupe_urls(urls, stats):
    seen = set()
    deduped = []
    for url in urls:
        if not _is_valid_url(url):
            stats['urls_invalid'] += 1
            continue
        if url in seen:
            stats['urls_duplicate'] += 1
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _parse_date(value):
    if not value:
        return ''
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return ''
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _extract_authors(entry):
    authors = []
    for author in entry.get('authors', []) or []:
        name = author.get('name')
        if name:
            authors.append(name)
    if not authors:
        single = entry.get('author')
        if single:
            authors.append(single)
    return ', '.join(authors)


def _extract_content(entry):
    content = ''
    if entry.get('content'):
        content = entry['content'][0].get('value', '')
    if not content:
        content = entry.get('summary', '') or entry.get('description', '')
    return clean_html(content)


def _extract_title(entry):
    title = entry.get('title') or ''
    return clean_html(title)


def _require_feedparser():
    try:
        import feedparser  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Missing dependency: feedparser. Install it to use RSS ingestion.'
        ) from exc
    return feedparser


def _load_cache(cache_path):
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache_path, cache):
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding='utf-8')


def _fetch_feed(feedparser, url, cache_entry, timeout):
    kwargs = {}
    if cache_entry:
        if cache_entry.get('etag'):
            kwargs['etag'] = cache_entry['etag']
        if cache_entry.get('modified'):
            kwargs['modified'] = cache_entry['modified']
    if timeout:
        kwargs['request_timeout'] = timeout
    try:
        return feedparser.parse(url, **kwargs)
    except TypeError as exc:
        if 'request_timeout' in kwargs and 'request_timeout' in str(exc):
            kwargs.pop('request_timeout', None)
            return feedparser.parse(url, **kwargs)
        raise


def _truncate(text, max_chars):
    if max_chars and len(text) > max_chars:
        return text[:max_chars].strip()
    return text


def _hash_key(title, source, date_iso):
    raw = f'{title}|{source}|{date_iso}'.lower().strip()
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def ingest_from_universe_file(
    universe_path,
    min_chars,
    max_chars,
    long_action,
    max_workers,
    timeout,
    cache_path,
    progress,
):
    stats = {
        'urls_invalid': 0,
        'urls_duplicate': 0,
        'feeds_total': 0,
        'feeds_ok': 0,
        'feeds_error': 0,
        'items_total': 0,
        'items_kept': 0,
        'items_skip_empty': 0,
        'items_skip_short': 0,
        'items_skip_long': 0,
        'items_truncated': 0,
        'items_duplicate': 0,
    }

    urls = load_feed_urls(universe_path)
    progress.set_stage('load urls', 1)
    progress.update(1)

    urls = _dedupe_urls(urls, stats)
    feedparser = _require_feedparser()

    cache = _load_cache(cache_path)
    cache_lock = threading.Lock()

    feeds = []
    errors = []
    stats['feeds_total'] = len(urls)
    progress.set_stage('fetch feeds', max(len(urls), 1))

    workers = max(1, min(max_workers or 1, len(urls) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for url in urls:
            future = executor.submit(
                _fetch_feed, feedparser, url, cache.get(url, {}), timeout
            )
            future_map[future] = url

        for future in as_completed(future_map):
            url = future_map[future]
            try:
                feed = future.result()
            except Exception as exc:
                errors.append({'url': url, 'error': str(exc), 'entries': 0})
                stats['feeds_error'] += 1
                progress.update(1)
                continue

            entries_count = len(getattr(feed, 'entries', []) or [])
            if getattr(feed, 'bozo', 0):
                exc = getattr(feed, 'bozo_exception', '')
                errors.append({'url': url, 'error': str(exc), 'entries': entries_count})
                if entries_count == 0:
                    stats['feeds_error'] += 1
                else:
                    stats['feeds_ok'] += 1
            else:
                stats['feeds_ok'] += 1

            feeds.append((url, feed))
            progress.update(1)

            etag = getattr(feed, 'etag', None)
            modified = getattr(feed, 'modified', None)
            if etag or modified:
                with cache_lock:
                    cache[url] = {'etag': etag, 'modified': modified}

    _save_cache(cache_path, cache)

    total_entries = sum(len(feed.entries) for _, feed in feeds)
    stats['items_total'] = total_entries
    progress.set_stage('process entries', max(total_entries, 1))

    seen = set()
    records = []
    for feed_url, feed in feeds:
        feed_title = feed.feed.get('title') if hasattr(feed, 'feed') else None
        feed_source = feed_title or feed_url
        for entry in feed.entries:
            content = _extract_content(entry)
            if not content:
                stats['items_skip_empty'] += 1
                progress.update(1)
                continue

            if min_chars and len(content) < min_chars:
                stats['items_skip_short'] += 1
                progress.update(1)
                continue

            if max_chars and len(content) > max_chars:
                if long_action == 'truncate':
                    content = _truncate(content, max_chars)
                    stats['items_truncated'] += 1
                else:
                    stats['items_skip_long'] += 1
                    progress.update(1)
                    continue

            title = _extract_title(entry)
            date_iso = _parse_date(
                entry.get('published') or entry.get('updated') or entry.get('pubDate')
            )
            key = _hash_key(title, feed_source, date_iso)
            if key in seen:
                stats['items_duplicate'] += 1
                progress.update(1)
                continue
            seen.add(key)

            record = {
                'title': title,
                'content': content,
                'date_iso': date_iso,
                'authors': _extract_authors(entry),
                'feed_source': feed_source,
                'link': entry.get('link') or entry.get('id') or '',
            }
            records.append(record)
            stats['items_kept'] += 1
            progress.update(1)

    return records, stats, errors
