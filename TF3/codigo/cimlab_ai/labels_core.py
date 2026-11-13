"""Explicit binary ontology for pooled multi-disaster xBD building labels.

Positive means ANY VISIBLE BUILDING DAMAGE (minor, major, destroyed).
Unknown taxonomies never become negatives by default.
"""
from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path

TASK_ID = 'any_damage_vs_no_damage_v1'
TASK_NAME = 'Dañado (1) / No dañado (0)'
MAPPING_FILE = 'cimlab_labels.json'


def normalize(value):
    s=unicodedata.normalize('NFKD',str(value or '').strip().lower())
    s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+','-',s).strip('-')

# Explicitly xBD's four official damage levels; no inferences about other variables.
STANDARD = {'no-damage':0, 'minor-damage':1,'major-damage':1,'destroyed':1}
# Known explicit equivalents for visible damage status of buildings.
EQUIVALENT = {
    'sin-dano':0,'undamaged':0,'sin-danos':0,'no-visible-damage':0,
    'dano-menor':1,'dano-leve':1,'dano-mayor':1,'dano-severo':1,
    'damaged':1,'danado':1,'danada':1,'dano':1,
    'minor-damage':1,'major-damage':1,
    'destruido':1,'destroyed':1,'collapsed':1,'colapsado':1,
    'completely-destroyed':1,'fully-destroyed':1,'partially-destroyed':1,
}
IGNORE = {'un-classified','unclassified','un-labeled','unlabeled','unknown',''}


class IncompatibleLabelsError(ValueError):
    pass


def event_mapping(event_root):
    path=Path(event_root)/MAPPING_FILE
    if not path.exists():return {}
    try:
        raw=json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise IncompatibleLabelsError(f'{path}: JSON de equivalencias inválido: {exc}') from exc
    if raw.get('task')!=TASK_ID or raw.get('confirmed_building_damage') is not True:
        raise IncompatibleLabelsError(f'{path}: la equivalencia debe confirmar task={TASK_ID} y confirmed_building_damage=true.')
    mapping=raw.get('mapping',{})
    if not isinstance(mapping,dict) or not mapping:
        raise IncompatibleLabelsError(f'{path}: mapping debe contener etiquetas y su categoría binaria.')
    out={}
    for key,val in mapping.items():
        normalized=normalize(key)
        if not normalized or type(val) is not int or val not in (0,1,-1):
            raise IncompatibleLabelsError(f'{path}: equiv. inválida para {key!r}: usa 0, 1 o -1 para excluir.')
        # Never override official xBD semantics silently.
        if (normalized in STANDARD and STANDARD[normalized]!=val) or (normalized in EQUIVALENT and EQUIVALENT[normalized]!=val):
            raise IncompatibleLabelsError(f'{path}: no se puede cambiar la definición oficial xBD de {key!r}.')
        out[normalized]=val
    return out


def classify(original, mapping=None):
    label=normalize(original)
    if label in STANDARD:return STANDARD[label]
    if label in IGNORE:return None
    if label in EQUIVALENT:return EQUIVALENT[label]
    if mapping and label in mapping:return mapping[label]
    return 'INCOMPATIBLE'


def save_event_mapping(root, mapping):
    """For explicit, human-reviewed classification; never auto-guess unknown labels."""
    root=Path(root)
    current=event_mapping(root)
    combined=dict(current)
    for key,val in mapping.items():
        norm=normalize(key)
        if not norm or type(val) is not int or val not in (0,1,-1):
            raise ValueError('Correspondencia inválida: '+str(key))
        if (norm in STANDARD and val!=STANDARD[norm]) or (norm in EQUIVALENT and val!=EQUIVALENT[norm]):
            raise ValueError('No se permite cambiar la clase xBD de '+str(key))
        combined[norm]=val
    path=root/MAPPING_FILE
    payload={'task':TASK_ID,'confirmed_building_damage':True,
        'note':'Asignación manual: 1=dañado (menor, mayor, destruido); 0=sin daño; -1=excluir. Revisar procedencia.',
        'mapping':combined}
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    return path


def require_task(workspace):
    work=Path(workspace)
    path=work/'dataset_meta.json'
    if not path.exists():
        raise RuntimeError('El conjunto de trabajo no tiene metadatos de la clasificación binaria. '
                           'Puede ser un entrenamiento anterior sin metadatos verificables. '
                           'Crea una carpeta de trabajo NUEVA en Preparar para evitar mezclar modelos y etiquetas.')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('task')!=TASK_ID:
        raise RuntimeError(f'Objetivo de clasificación incompatible ({data.get("task")}). '
                           'No se pueden reutilizar pesos/resultados con otra definición de positivo.')
    return data
