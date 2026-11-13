"""CIMLAB AI v0.3 — comparador visual PRE/POST sin alterar los datos de aprendizaje.

La armonización es **solo de visualización**. El mapa de cambio no es un
predictor de daños y no se utiliza para calcular scores ni etiquetas.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def balance_post_illumination(pre: Image.Image, post: Image.Image) -> Image.Image:
    """Armoniza suavemente la luminosidad de POST respecto a PRE (no cambia geometría).

    Transformación afín común a RGB, limitada para evitar falsos detalles.
    Mapea de modo aproximado percentiles 10/90 de luminancia; no utiliza
    etiquetas ni estadísticas de test. NO se exporta al dataset.
    """
    a=np.asarray(pre.convert('RGB'),dtype=np.float32)
    b=np.asarray(post.convert('RGB'),dtype=np.float32)
    if a.shape!=b.shape:
        raise ValueError('PRE/POST de dimensiones distintas')
    weight=np.array([.299,.587,.114],dtype=np.float32)
    ya=a@weight;yb=b@weight
    alo,ahi=np.percentile(ya,[10,90]);blo,bhi=np.percentile(yb,[10,90])
    # En áreas con luminancia casi uniforme o post apagada no forzamos cambios.
    if ahi-alo<8 or bhi-blo<8:
        return post.copy()
    gain=np.clip((ahi-alo)/(bhi-blo),.80,1.25)
    offset=np.clip((alo-blo*gain),-24.0,24.0)
    result=np.clip(b*gain+offset,0,255).astype(np.uint8)
    return Image.fromarray(result,'RGB')


def visual_change_map(pre: Image.Image, post: Image.Image, harmonize=True) -> Image.Image:
    """Mapa de magnitud de diferencias RGB, para revisión humana únicamente.

    Una diferencia destacada puede venir de sombras, vegetación, geometría
    o compresión, no necesariamente de daño. Usa mismos píxeles por posición;
    NO corrige una desalineación espacial.
    """
    if pre.size!=post.size:raise ValueError('PRE/POST con tamaño diferente')
    b=balance_post_illumination(pre,post) if harmonize else post
    a=np.asarray(pre.convert('RGB'),dtype=np.float32)
    bb=np.asarray(b.convert('RGB'),dtype=np.float32)
    delta=np.mean(np.abs(a-bb),axis=2)
    strength=np.clip((delta-10)/85,0.,1.)**.85
    # bajo: azul muy oscuro / alto: naranja; ninguna inferencia de daño
    low=np.array([16,30,48],dtype=np.float32)
    high=np.array([255,155,45],dtype=np.float32)
    out=(low[None,None,:]*(1-strength[:,:,None])+high[None,None,:]*strength[:,:,None])
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8),'RGB')


def view_images(pre: Image.Image,post: Image.Image,mode='original'):
    """Devuelve lista de pares (título, PIL image) para el visor de auditoría."""
    if mode=='original':
        return [('PRE · original',pre.copy()),('POST · original',post.copy())]
    if mode=='balanced':
        return [('PRE · original',pre.copy()),('POST · luz equilibrada',balance_post_illumination(pre,post))]
    if mode=='change':
        return [('PRE · original',pre.copy()),('POST · original',post.copy()),
                ('CAMBIOS · orientativo',visual_change_map(pre,post))]
    raise ValueError('Modo de comparación no reconocido: '+str(mode))
