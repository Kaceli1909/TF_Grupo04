"""Auditoría reproducible del CSV de edificaciones de xBD (México).

Uso:
    python validar_limpiar_edificaciones.py tabla_edificaciones.csv

Dependencias: pandas

El archivo fuente no se altera. Se separan etiquetas de daño conocidas,
casos sin clasificar y registros con errores de validación.
"""
from pathlib import Path
import argparse
import re
import pandas as pd

COLUMNAS = ["escena", "uid", "clasificacion_dano"]
ETIQUETAS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
TODAS = ETIQUETAS + ["un-classified"]
ESCENA_REGEX = r"mexico-earthquake_\d{8}"
UID_REGEX = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"


def auditar(origen: Path, destino: Path) -> None:
    datos = pd.read_csv(origen, dtype="string", encoding="utf-8-sig")
    if list(datos.columns) != COLUMNAS:
        raise ValueError(
            f"Columnas esperadas: {COLUMNAS}. Recibidas: {list(datos.columns)}"
        )
    # No afecta al archivo original; retira únicamente espacios de bordes.
    for columna in COLUMNAS:
        datos[columna] = datos[columna].str.strip()

    vacios = datos[COLUMNAS].isna() | datos[COLUMNAS].fillna("").eq("")
    escena_invalida = ~datos.escena.fillna("").str.fullmatch(ESCENA_REGEX)
    uid_invalido = ~datos.uid.fillna("").str.fullmatch(UID_REGEX)
    etiqueta_desconocida = ~datos.clasificacion_dano.isin(TODAS).fillna(False)
    uid_duplicado = datos.duplicated(subset=["escena", "uid"], keep=False)
    fila_duplicada = datos.duplicated(keep=False)

    # Un error de datos se separa para investigar, en lugar de borrar registros.
    invalido = vacios.any(axis=1) | escena_invalida | uid_invalido | etiqueta_desconocida | uid_duplicado
    sin_clasificar = ~invalido & datos.clasificacion_dano.eq("un-classified").fillna(False)
    clasificados = ~invalido & datos.clasificacion_dano.isin(ETIQUETAS).fillna(False)

    destino.mkdir(parents=True, exist_ok=True)
    datos.loc[clasificados].to_csv(
        destino / "edificaciones_clasificadas.csv", index=False, encoding="utf-8-sig")
    datos.loc[sin_clasificar].to_csv(
        destino / "edificaciones_sin_clasificar.csv", index=False, encoding="utf-8-sig")
    datos.loc[invalido].to_csv(
        destino / "registros_por_revisar.csv", index=False, encoding="utf-8-sig")

    por_escena = (
        datos.groupby(["escena", "clasificacion_dano"], dropna=False)
        .size().unstack(fill_value=0).reindex(columns=TODAS, fill_value=0)
    )
    por_escena["total"] = por_escena.sum(axis=1)
    por_escena.reset_index().to_csv(
        destino / "conteo_danos_por_escena.csv", index=False, encoding="utf-8-sig")

    contadores = datos.clasificacion_dano.value_counts(dropna=False)
    n = len(datos)
    resumen = [
        "AUDITORÍA DE TABLA_EDIFICACIONES.CSV — xBD / MÉXICO",
        "",
        f"Archivo: {origen.name}",
        f"Registros originales: {n:,}",
        f"Escenas diferentes: {datos.escena.nunique():,}",
        f"Registros con valores vacíos: {int(vacios.any(axis=1).sum()):,}",
        f"Filas duplicadas exactas (excluida primera aparición): {int(datos.duplicated().sum()):,}",
        f"Registros implicados en UIDs duplicados por escena: {int(uid_duplicado.sum()):,}",
        f"Escenas con formato incorrecto: {int(escena_invalida.sum()):,}",
        f"UIDs con formato incorrecto: {int(uid_invalido.sum()):,}",
        f"Etiquetas desconocidas: {int(etiqueta_desconocida.sum()):,}",
        "",
        "CATEGORÍAS (DATOS ORIGINALES)",
    ]
    for nombre in TODAS:
        k = int(contadores.get(nombre, 0))
        resumen.append(f"  {nombre:16s} {k:>6,}  ({k/n*100:.4f} %)" if n else f"  {nombre}: 0")
    resumen += [
        "",
        "SALIDAS",
        f"Edificaciones con etiqueta conocida: {int(clasificados.sum()):,}",
        f"Edificaciones sin clasificar, separadas: {int(sin_clasificar.sum()):,}",
        f"Registros que requieren investigación: {int(invalido.sum()):,}",
        "",
        "LÍMITES DE ESTA AUDITORÍA",
        "- Solo se revisa la consistencia de las tres columnas del CSV.",
        "- No se verifican imágenes, coordenadas, polígonos ni JSON originales.",
        "- No se corrige el desbalance de clases ni se asignan etiquetas nuevas.",
        "- Se recomienda separar conjuntos de entrenamiento/prueba por escena o zona.",
    ]
    (destino / "resumen_auditoria.txt").write_text("\n".join(resumen)+"\n", encoding="utf-8")
    print("\n".join(resumen))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validar y separar etiquetas de xBD")
    parser.add_argument("archivo_csv", type=Path, help="Ruta del CSV original")
    parser.add_argument("--salida", type=Path, default=Path("resultados_validacion"))
    argumentos = parser.parse_args()
    auditar(argumentos.archivo_csv, argumentos.salida)
