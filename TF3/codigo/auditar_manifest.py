#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audita un manifest.csv de CIMLAB AI / TF3 sin modificar sus archivos.

Este script utiliza solo librerías estándar. Contrasta la etiqueta binaria con
la original y el split actual con la ubicación original del TIFF. Comprueba
separación de escenas, duplicados por escena y UID, y rutas PRE/POST.

Uso:
    python auditar_manifest.py "RUTA/manifest.csv" --salida resumen.json
    python auditar_manifest.py "RUTA/manifest.csv" --verificar-recortes
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_ORIGEN = {"tier1": "train", "hold": "val", "test": "test"}
CLASES = {"no-damage": 0, "minor-damage": 1,
          "major-damage": 1, "destroyed": 1}
COLUMNAS = {"scene", "uid", "label", "original_label", "split",
            "pre", "post", "crop_pre", "crop_post"}


def info_origen(ruta: str) -> tuple[str, str]:
    """Obtiene evento y split *original* desde /data/<evento>/**/tier1|hold|test.

    Las rutas son absolutas de Windows en el manifiesto local. Sus imágenes
    JPG pueden seguir en crops/train aunque su split fuese corregido después;
    por eso se examina exclusivamente la ruta TIFF original PRE/POST.
    """
    trozos = [p.lower() for p in ruta.replace("\\", "/").split("/") if p]
    ocurrencias = [SPLIT_ORIGEN[p] for p in trozos if p in SPLIT_ORIGEN]
    if len(ocurrencias) != 1:
        raise ValueError("Ruta TIFF sin una única partición tier1/hold/test")
    if "data" not in trozos:
        raise ValueError("Ruta TIFF no contiene data/<evento>")
    pos = trozos.index("data")
    if len(trozos) <= pos + 1:
        raise ValueError("Ruta TIFF no identifica el evento")
    return trozos[pos + 1], ocurrencias[0]


def auditar(ruta: Path, verificar_recortes: bool = False) -> tuple[dict, int]:
    casos = Counter()
    escenas = defaultdict(set)
    vistos = set()
    etiquetas = Counter()
    problemas = []
    total_errores = 0
    total = 0

    with ruta.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        faltantes = COLUMNAS - set(reader.fieldnames or [])
        if faltantes:
            raise ValueError(f"Faltan columnas en manifest.csv: {sorted(faltantes)}")

        for numero, row in enumerate(reader, start=2):
            total += 1
            scene = (row["scene"] or "").strip()
            uid = (row["uid"] or "").strip()
            etiqueta = (row["original_label"] or "").strip().lower()
            etiquetas[etiqueta] += 1
            errores = []
            llave = (scene, uid)
            if not scene or not uid:
                errores.append("Identificador de escena/UID vacío")
            if llave in vistos:
                errores.append("UID duplicado dentro de la escena")
            vistos.add(llave)

            try:
                evento_pre, split_pre = info_origen(row["pre"])
                evento_post, split_post = info_origen(row["post"])
                if (evento_pre, split_pre) != (evento_post, split_post):
                    errores.append("PRE y POST provienen de eventos o splits distintos")
                if row["split"].strip().lower() != split_pre:
                    errores.append("Split del manifest no coincide con el origen de los TIFF")
                if not scene.startswith(evento_pre + "_"):
                    errores.append("El prefijo de scene no coincide con el evento de TIFF")

                if scene in escenas and escenas[scene] and split_pre not in escenas[scene]:
                    errores.append("Escena compartida entre diferentes particiones")
                escenas[scene].add(split_pre)

                if etiqueta not in CLASES:
                    errores.append(f"Etiqueta original no reconocida: {etiqueta!r}")
                elif int(row["label"]) != CLASES[etiqueta]:
                    errores.append("Etiqueta binaria no concuerda con etiqueta original")

                # Conteos por origen, incluso ante un error en el split asignado.
                if etiqueta in CLASES:
                    casos[(evento_pre, split_pre, CLASES[etiqueta])] += 1

            except (ValueError, TypeError, KeyError) as exc:
                errores.append(f"Lectura inconsistente: {exc}")

            if verificar_recortes:
                for campo in ("crop_pre", "crop_post"):
                    if not row[campo] or not Path(row[campo]).is_file():
                        errores.append(f"No existe recorte {campo}")

            if errores:
                total_errores += 1
                if len(problemas) < 30:
                    problemas.append({"fila": numero, "escena": scene,
                                      "uid": uid, "errores": errores})

            if total % 20000 == 0:
                print(f"Auditoría: {total:,} filas revisadas", file=sys.stderr)

    resumen = []
    for (evento, split, clase), cantidad in sorted(casos.items()):
        resumen.append({"evento": evento, "split_original": split,
                        "clase_binaria": clase, "edificios": cantidad})

    resultado = {
        "archivo": ruta.name,
        "filas": total,
        "escenas_unicas": len(escenas),
        "etiquetas_originales": dict(sorted(etiquetas.items())),
        "errores_en_filas": total_errores,
        "muestra_hasta_30_errores": problemas,
        "conteos_por_evento_particion_clase": resumen,
        "verifico_existencia_de_jpg": verificar_recortes,
        "nota": "Auditoría del manifiesto; no decodifica archivos de imagen.",
    }
    return resultado, total_errores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Ruta local al manifest.csv")
    parser.add_argument("--salida", type=Path, help="Ruta JSON donde guardar resumen")
    parser.add_argument("--verificar-recortes", action="store_true",
                        help="Comprobar existencia de JPG en esta computadora")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"No existe el manifiesto: {args.manifest}")
    try:
        informe, errores = auditar(args.manifest, args.verificar_recortes)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    documento = json.dumps(informe, ensure_ascii=False, indent=2)
    print(documento)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(documento + "\n", encoding="utf-8")
        print(f"Resumen guardado: {args.salida}", file=sys.stderr)
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
