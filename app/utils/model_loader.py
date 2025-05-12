import os
import tempfile
import requests

_model_cache = {}

def download_model_if_not_exists(url: str):
    if url in _model_cache:
        return _model_cache[url]

    response = requests.get(url)
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.write(response.content)
    tmp.close()

    _model_cache[url] = tmp.name
    return tmp.name
