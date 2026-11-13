"""Discovery of xBD-like disaster events, both from GitHub and local folders.

The remote catalog is queried on demand; it never hardcodes an event list.
"""
from __future__ import annotations
import json
import re
import urllib.request
from pathlib import Path

REPO_OWNER = 'Kaceli1909'
REPO_NAME = 'TF_Grupo04'
REPO_URL = f'https://github.com/{REPO_OWNER}/{REPO_NAME}'
DATA_API = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/data'
_VALID = re.compile(r'^[a-zA-Z0-9._-]+$')


def valid_event(event):
    return isinstance(event,str) and bool(_VALID.fullmatch(event)) and event not in ('.','..')


def github_events(timeout=20):
    """List current directories under GitHub data/; future events appear on refresh.

    Avoid fabricating events; report HTTP/rate limit/network errors to users.
    """
    request=urllib.request.Request(DATA_API, headers={
        'User-Agent':'CIMLAB-AI-event-catalog',
        'Accept':'application/vnd.github+json',
        'X-GitHub-Api-Version':'2022-11-28'
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            items=json.load(response)
    except Exception as e:
        raise RuntimeError('No se pudo consultar GitHub. Revisa Internet, acceso al repositorio o límite de la API. Detalle: '+str(e)) from e
    if not isinstance(items,list):
        raise RuntimeError('GitHub no devolvió una lista de eventos en data/.')
    return sorted(item['name'] for item in items if item.get('type')=='dir' and valid_event(item.get('name')))


def has_xbd_files(path):
    """Detect image-pair convention; a folder can contain tier1/test/hold subfolders."""
    path=Path(path)
    if not path.is_dir(): return False
    has_pre=any(any(path.rglob('*_pre_disaster'+ext)) for ext in ('.tif','.tiff','.png','.jpg','.jpeg'))
    has_json=any(path.rglob('*_post_disaster.json'))
    return has_pre and has_json


def local_event_roots(path):
    """Resolve repository/data, collection of event directories or a single xBD event.

    A folder with no event subfolders but with xBD pairs is a valid single event.
    """
    path=Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError('No existe la carpeta local: '+str(path))
    roots=[]
    base=path/'data' if (path/'data').is_dir() else path
    for child in sorted(base.iterdir()):
        if child.is_dir() and has_xbd_files(child):
            roots.append(child)
    if not roots and has_xbd_files(base):
        roots=[base]
    # Local folder may contain TIFF+JSON in sibling directories, without xBD event named parent.
    if not roots:
        raise FileNotFoundError('No se encontraron imágenes PRE/POST con nombres xBD (*_pre_disaster.tif/png/jpg) en esta carpeta.')
    return roots


def local_events(path):
    roots=local_event_roots(path)
    return {r.name:r for r in roots}


def selected_roots(path,selected=None):
    catalog=local_events(path)
    if not selected:return list(catalog.values())
    missing=[name for name in selected if name not in catalog]
    if missing:
        raise FileNotFoundError('Eventos no disponibles en disco: '+', '.join(missing)+'. Descárgalos primero o cambia de carpeta.')
    return [catalog[name] for name in selected]
