"""Grupo 04 | T2 | Imágenes PRE/POST y geometrías xBD (México).

Adaptación del código elaborado por el integrante del Grupo 04 para su uso
con carpetas locales o con el dataset montado en Kaggle.

Requiere imágenes PNG originales y JSON de anotaciones en carpetas hermanas
``images`` y ``labels``. Los JSON por sí solos no bastan para mostrar PRE/POST.

Instalación: pip install pandas numpy matplotlib pillow shapely

Ejemplos:
  python analisis_imagenes_geometrias.py --datos "D:/xBD/train"
  python analisis_imagenes_geometrias.py --datos "/kaggle/input"
  python analisis_imagenes_geometrias.py --images "D:/xBD/train/images" --labels "D:/xBD/train/labels"

Las áreas obtenidas de features.xy son píxeles cuadrados (px²), NO m².
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Guardar gráficos sin abrir ventanas; útil en Windows/Kaggle.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Polygon as MplPolygon
from shapely import wkt

PREFIJO = "mexico-earthquake_"
CLASES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
COLORES = {
    "no-damage": "#2ecc71",
    "minor-damage": "#f1c40f",
    "major-damage": "#e67e22",
    "destroyed": "#e74c3c",
    "un-classified": "#95a5a6",
}


def carpetas_de_datos(datos: Path | None, imagenes: Path | None, etiquetas: Path | None):
    """Localiza un único directorio images con su correspondiente labels."""
    if imagenes is not None or etiquetas is not None:
        if imagenes is None or etiquetas is None:
            raise ValueError("Debes indicar --images y --labels juntos.")
        candidatos = [(imagenes, etiquetas)]
    else:
        if datos is None:
            datos = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
        if not datos.exists():
            raise FileNotFoundError(f"La carpeta de datos no existe: {datos}")
        posibles = [datos] if datos.name == "images" else []
        posibles.extend(p for p in datos.rglob("images") if p.is_dir())
        candidatos = [(p, p.parent / "labels") for p in posibles]

    candidatos = [
        (p, lab, len(list(p.glob(f"{PREFIJO}*_pre_disaster.png"))))
        for p, lab in candidatos if p.is_dir() and lab.is_dir()
    ]
    candidatos = [c for c in candidatos if c[2] > 0]
    if not candidatos:
        raise FileNotFoundError(
            "No encontré un par de carpetas 'images' y 'labels' con imágenes "
            "mexico-earthquake_..._pre_disaster.png. Los archivos JSON que "
            "están en GitHub no reemplazan las imágenes PNG originales. "
            "Indica --datos RUTA, o --images RUTA --labels RUTA."
        )
    carpeta_img, carpeta_lab, _ = max(candidatos, key=lambda t: t[2])
    return carpeta_img, carpeta_lab


def indexar(rutas: list[Path], sufijo: str):
    resultado = {}
    for ruta in rutas:
        nombre = ruta.name
        if nombre.startswith(PREFIJO) and nombre.endswith(sufijo):
            sid = nombre[len(PREFIJO):-len(sufijo)]
            if sid in resultado:
                raise ValueError(f"Identificador duplicado: {sid} ({ruta})")
            resultado[sid] = ruta
    return resultado


def cargar_geometria(feature: dict):
    try:
        geometria = wkt.loads(feature.get("wkt", ""))
        valido = bool(geometria.is_valid and not geometria.is_empty)
        return geometria, valido
    except Exception:
        return None, False


def guardar_integridad(pre_imgs, post_imgs, post_jsons, salida):
    ids_pre = indexar(pre_imgs, "_pre_disaster.png")
    ids_post = indexar(post_imgs, "_post_disaster.png")
    ids_json = indexar(post_jsons, "_post_disaster.json")
    todos = sorted(set(ids_pre) | set(ids_post) | set(ids_json))
    filas = []
    for sid in todos:
        pre = sid in ids_pre
        post = sid in ids_post
        json_post = sid in ids_json
        filas.append({
            "escena_id": f"{PREFIJO}{sid}",
            "pre_png": pre,
            "post_png": post,
            "post_json": json_post,
            "par_completo": pre and post and json_post,
        })
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        raise ValueError("No se encontraron escenas válidas del terremoto de México.")
    tabla.to_csv(salida / "tabla_integridad_pares_pre_post.csv", index=False)
    resumen = pd.DataFrame([
        {"Indicador": "Escenas totales identificadas (México)", "Conteo": len(todos)},
        {"Indicador": "Imágenes PRE disponibles (.png)", "Conteo": len(pre_imgs)},
        {"Indicador": "Imágenes POST disponibles (.png)", "Conteo": len(post_imgs)},
        {"Indicador": "Etiquetas POST disponibles (.json)", "Conteo": len(post_jsons)},
        {"Indicador": "Pares PRE/POST + JSON completos (por existencia)",
         "Conteo": int(tabla["par_completo"].sum())},
        {"Indicador": "Archivos faltantes o pares incompletos",
         "Conteo": int((~tabla["par_completo"]).sum())},
    ])
    resumen.to_csv(salida / "resumen_integridad_mexico.csv", index=False)
    print("\n=== 1. VERIFICACIÓN DE PARES PRE/POST ===")
    print(resumen.to_string(index=False))
    return tabla, ids_pre, ids_post, ids_json


def guardar_areas(ids_json, salida):
    registros = []
    invalidas = 0
    for sid, ruta_json in ids_json.items():
        data = json.loads(ruta_json.read_text(encoding="utf-8"))
        for feature in data.get("features", {}).get("xy", []):
            props = feature.get("properties", {})
            geometria, valido = cargar_geometria(feature)
            if not valido:
                invalidas += 1
            registros.append({
                "escena_id": f"{PREFIJO}{sid}",
                "uid": props.get("uid", "sin_uid"),
                "clase_dano": props.get("subtype", "un-classified"),
                "geometria_valida": valido,
                "area_px2": geometria.area if valido else np.nan,
                "perimetro_px": geometria.length if valido else np.nan,
            })
    tabla = pd.DataFrame(registros)
    if tabla.empty:
        raise ValueError("Los JSON no contienen polígonos en features.xy.")
    tabla.to_csv(salida / "edificios_geometrias_mexico.csv", index=False)
    validas = tabla[tabla["geometria_valida"] & tabla["clase_dano"].isin(CLASES)].copy()
    if validas.empty:
        raise ValueError("No se encontraron geometrías válidas con las clases esperadas.")
    areas = (
        validas.groupby("clase_dano")["area_px2"].agg(
            Cantidad="count",
            Porcentaje=lambda x: round(len(x) / len(validas) * 100, 2),
            Area_Media_px2=lambda x: round(x.mean(), 1),
            Area_Mediana_px2=lambda x: round(x.median(), 1),
            Area_Min_px2=lambda x: round(x.min(), 1),
            Area_Max_px2=lambda x: round(x.max(), 1),
        ).reindex([c for c in CLASES if c in validas["clase_dano"].unique()]).reset_index()
    )
    areas.to_csv(salida / "tabla_areas_por_clase.csv", index=False)
    print("\n=== 2. GEOMETRÍAS Y ÁREAS ===")
    print(f"Total de polígonos: {len(tabla):,}; geometrías inválidas: {invalidas:,}")
    print(areas.to_string(index=False))
    return validas


def grafico_areas(validas, salida):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    p98 = validas["area_px2"].quantile(0.98)
    datos = validas[validas["area_px2"] <= p98]
    axes[0].hist(datos["area_px2"], bins=40, color="#2b7bba", edgecolor="white")
    axes[0].set_title("Distribución de áreas de edificaciones (px²) - México xBD")
    axes[0].set_xlabel("Área del polígono (px²)")
    axes[0].set_ylabel("Número de edificaciones")
    clases = [c for c in CLASES[:4] if c in datos["clase_dano"].unique()]
    cajas = [datos.loc[datos["clase_dano"] == c, "area_px2"].to_numpy() for c in clases]
    axes[1].boxplot(cajas, tick_labels=clases, showfliers=False)
    axes[1].set_title("Comparación de área (px²) según nivel de daño")
    axes[1].set_xlabel("Categoría de daño")
    axes[1].set_ylabel("Área del polígono (px²)")
    fig.tight_layout()
    fig.savefig(salida / "grafico_areas_geometrias.png", dpi=200)
    plt.close(fig)


def grafico_ejemplo(validas, ids_pre, ids_post, ids_json, salida):
    ids_completos = set(ids_pre) & set(ids_post) & set(ids_json)
    candidatas = validas[validas["escena_id"].isin({f"{PREFIJO}{x}" for x in ids_completos})]
    if candidatas.empty:
        print("No hay escenas completas para generar la visualización PRE/POST.")
        return
    graves = candidatas[candidatas["clase_dano"].isin(("destroyed", "major-damage"))]
    conteo = graves.groupby("escena_id").size()
    escena = conteo.idxmax() if not conteo.empty else candidatas["escena_id"].iloc[0]
    sid = escena.removeprefix(PREFIJO)
    with Image.open(ids_pre[sid]) as img:
        imagen_pre = img.convert("RGB").copy()
    with Image.open(ids_post[sid]) as img:
        imagen_post = img.convert("RGB").copy()
    anotaciones = json.loads(ids_json[sid].read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(imagen_pre)
    axes[0].set_title(f"ANTES DEL SISMO (PRE)\n{escena}", fontsize=12, fontweight="bold")
    axes[1].imshow(imagen_post)
    axes[1].set_title(f"DESPUÉS DEL SISMO (POST)\n{escena}", fontsize=12, fontweight="bold")
    axes[2].imshow(imagen_post)
    for feat in anotaciones.get("features", {}).get("xy", []):
        categoria = feat.get("properties", {}).get("subtype", "un-classified")
        poly, valido = cargar_geometria(feat)
        if not valido:
            continue
        partes = [poly] if poly.geom_type == "Polygon" else list(poly.geoms) if poly.geom_type == "MultiPolygon" else []
        for parte in partes:
            coords = np.asarray(parte.exterior.coords)
            patch = MplPolygon(coords, closed=True,
                               facecolor=COLORES.get(categoria, "#95a5a6"),
                               edgecolor=COLORES.get(categoria, "#95a5a6"),
                               alpha=0.45, linewidth=1.2)
            axes[2].add_patch(patch)
    axes[2].set_title("ANOTACIONES JSON SUPERPUESTAS\n"
                      "Verde: Sin daño | Amarillo: Menor | Naranja: Mayor | Rojo: Destruida",
                      fontsize=11, fontweight="bold")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(salida / "visualizacion_real_pre_post_poligonos.png", dpi=200)
    plt.close(fig)
    print(f"Escena elegida para visualización: {escena}")


def main():
    parser = argparse.ArgumentParser(description="T2 | Grupo 04 | Análisis de imágenes y geometrías")
    parser.add_argument("--datos", type=Path, help="Carpeta que contiene images y labels (o su carpeta padre)")
    parser.add_argument("--images", type=Path, help="Ruta directa a la carpeta images")
    parser.add_argument("--labels", type=Path, help="Ruta directa a la carpeta labels")
    parser.add_argument("--salida", type=Path, help="Directorio de resultados")
    args = parser.parse_args()
    datos = args.datos
    images_dir, labels_dir = carpetas_de_datos(datos, args.images, args.labels)
    salida = args.salida or (Path("/kaggle/working/resultados_mexico") if Path("/kaggle/working").exists()
                              else Path.cwd() / "resultados_mexico")
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Imágenes: {images_dir}\nEtiquetas: {labels_dir}\nSalida: {salida}")
    pre = sorted(images_dir.glob(f"{PREFIJO}*_pre_disaster.png"))
    post = sorted(images_dir.glob(f"{PREFIJO}*_post_disaster.png"))
    json_post = sorted(labels_dir.glob(f"{PREFIJO}*_post_disaster.json"))
    _, ids_pre, ids_post, ids_json = guardar_integridad(pre, post, json_post, salida)
    validas = guardar_areas(ids_json, salida)
    grafico_areas(validas, salida)
    grafico_ejemplo(validas, ids_pre, ids_post, ids_json, salida)
    print(f"\nAnálisis terminado. Resultados en: {salida}")


if __name__ == "__main__":
    main()
