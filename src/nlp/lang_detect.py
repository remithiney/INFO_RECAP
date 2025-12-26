def _require_fasttext():
    try:
        import fasttext  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Missing dependency: fasttext. Install it to use language detection.'
        ) from exc
    return fasttext


def detect_languages(records, model_path, progress):
    if not model_path:
        raise ValueError('Missing fastText model path.')
    fasttext = _require_fasttext()
    model = fasttext.load_model(model_path)

    progress.set_stage('detect language', max(len(records), 1))
    for record in records:
        title = (record.get('title') or '').strip()
        if not title:
            record['lang'] = ''
            record['lang_score'] = 0.0
            progress.update(1)
            continue
        labels, scores = model.predict(title, k=1)
        label = labels[0] if labels else ''
        score = float(scores[0]) if scores else 0.0
        record['lang'] = label.replace('__label__', '')
        record['lang_score'] = score
        progress.update(1)
    return records
