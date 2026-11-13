"""CIMLAB AI v0.5.3 - Lectura RGB de TIFF int16 sin perder fidelidad.

No inventa detalles visuales ni cambia las etiquetas; no usa test para configurar modelos.
"""
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

TIFF_HEADERS=(b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+')


def describe(path):
    p=Path(path)
    if not p.is_file():
        return {'path':str(p),'size':None,'signature':'','type':'MISSING'}
    with p.open('rb') as f:
        h=f.read(256)
    kind=('LFS_POINTER' if b'version https://git-lfs.github.com/spec/v1' in h[:160]
          else 'TIFF' if h.startswith(TIFF_HEADERS)
          else 'JPEG' if h.startswith(b'\xff\xd8\xff')
          else 'PNG' if h.startswith(b'\x89PNG\r\n\x1a\n')
          else 'HTML' if h.lstrip().lower().startswith((b'<html',b'<!doctype'))
          else 'ZIP' if h.startswith(b'PK\x03\x04')
          else 'UNKNOWN')
    return {'path':str(p),'size':p.stat().st_size,'signature':h[:24].hex(' '),'type':kind}


def _to_rgb_array(arr):
    """Conversión radiométrica reproducible de TIFF RGB a 8 bits.

    xBD almacena algunas imágenes RGB de 8 bits en contenedores TIFF int16:
    los valores reales siguen siendo 0..255. Para ellas convertir de int16
    a uint8 es EXACTO y NO cambia la iluminación de PRE ni POST.

    En imágenes con rangos distintos no se adivina una normalización:
    fallamos explícitamente y reportamos el rango de píxeles.
    """
    arr=np.asarray(arr)
    if arr.ndim==2:
        arr=np.repeat(arr[:,:,None],3,axis=2)
    elif arr.ndim==3 and arr.shape[-1] in (1,3,4):
        if arr.shape[-1]==1:arr=np.repeat(arr,3,axis=2)
        if arr.shape[-1]==4:arr=arr[:,:,:3]
    elif arr.ndim==3 and arr.shape[0] in (1,3,4):
        arr=np.moveaxis(arr,0,-1)
        if arr.shape[-1]==1:arr=np.repeat(arr,3,axis=2)
        if arr.shape[-1]==4:arr=arr[:,:,:3]
    else:
        raise ValueError(f'TIFF con forma {arr.shape}, incompatible con RGB; revisar bandas')
    if arr.size==0:
        raise ValueError('TIFF vacío')
    if arr.dtype==np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype,np.integer):
        mn=int(arr.min());mx=int(arr.max())
        # Política exacta: sin escalados por imagen. Conserva colores y pares PRE/POST.
        if 0<=mn and mx<=255:
            return np.ascontiguousarray(arr.astype(np.uint8))
        # Mantener semántica previa de uint16 de rango completo fijo cuando proceda.
        if arr.dtype==np.uint16 and 0<=mn and mx>255:
            return np.ascontiguousarray(np.rint(arr.astype(np.float64)*(255.0/65535.0)).clip(0,255).astype(np.uint8))
        raise ValueError(
            f'TIFF {arr.dtype} con píxeles [{mn}, {mx}]; no es RGB de 8 bits almacenado como int16. '
            'No se aplicó normalización automática para no alterar el significado de los datos. '
            'Envíe DIAGNOSTICO_TIFF_v053.json para definir la conversión correcta.')
    if np.issubdtype(arr.dtype,np.floating):
        if not np.isfinite(arr).all():
            raise ValueError('TIFF flotante con NaN/inf; revisar datos')
        mn=float(arr.min());mx=float(arr.max())
        if 0<=mn and mx<=1:
            return np.ascontiguousarray(np.rint(arr*255).clip(0,255).astype(np.uint8))
        if 0<=mn and mx<=255:
            return np.ascontiguousarray(np.rint(arr).clip(0,255).astype(np.uint8))
        raise ValueError(f'TIFF flotante con píxeles [{mn:.5g}, {mx:.5g}]; rango desconocido')
    raise ValueError(f'TIFF con tipo de píxel {arr.dtype}, requiere política explícita')


def pixel_stats(path):
    """Extrae tipo, forma y rango sin modificar imagen; útil para diagnóstico."""
    import tifffile
    a=tifffile.imread(path)
    ar=np.asarray(a)
    return {'dtype':str(ar.dtype),'shape':list(ar.shape),
            'min':float(np.min(ar)) if ar.size else None,
            'max':float(np.max(ar)) if ar.size else None}

def open_rgb(path):
    p=Path(path)
    detail=describe(p)
    if detail['type']=='MISSING':raise FileNotFoundError(str(p))
    if detail['type']=='LFS_POINTER':
        raise RuntimeError(f'{p.name}: es un puntero Git LFS, no una imagen. Descarga los datos mediante Git LFS.')
    if detail['type']=='HTML':
        raise RuntimeError(f'{p.name}: contiene HTML, no una imagen (descarga inválida).')
    if detail['type']=='UNKNOWN':
        raise RuntimeError(f'{p.name}: contenido NO reconocido como imagen TIFF/JPEG/PNG. Tamaño={detail["size"]}, primeros bytes={detail["signature"]}. Git LFS puede haber guardado un contenido incorrecto; no se puede recortar.')
    try:
        with Image.open(p) as image:
            image.load()
            return image.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError) as e:
        if detail['type'] != 'TIFF':
            raise RuntimeError(f'{p.name}: formato {detail["type"]} ilegible por Pillow ({type(e).__name__}: {e})') from e
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError(f'{p.name}: cabecera TIFF válida pero Pillow no puede abrirla. Faltaría solo el lector opcional tifffile; desde .venv ejecuta: python -m pip install tifffile. Causa Pillow: {type(e).__name__}: {e}') from exc
        try:
            arr=tifffile.imread(p)
            return Image.fromarray(_to_rgb_array(arr),'RGB')
        except Exception as exc:
            raise RuntimeError(f'{p.name}: TIFF válido por cabecera, pero Pillow y tifffile fallaron. tifffile: {type(exc).__name__}: {exc}; Pillow: {type(e).__name__}: {e}') from exc
