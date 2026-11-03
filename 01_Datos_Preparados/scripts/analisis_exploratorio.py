"""Grupo 04 — T2: análisis exploratorio de xBD México.

Ubicar este script en: 01_Datos_Preparados/scripts/analisis_exploratorio.py
Leerá: 01_Datos_Preparados/resultados/tabla_edificaciones.csv
Creará: 01_Datos_Preparados/resultados/EDA/

Dependencias: pip install pandas matplotlib
No modifica el archivo original. Los gráficos se obtienen del CSV real.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ORIGEN = BASE / "resultados" / "tabla_edificaciones.csv"
DESTINO = BASE / "resultados" / "EDA"
CLASES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
NOMBRES = ["Sin daño", "Daño menor", "Daño mayor", "Destruida", "Sin clasificar"]


def guardar_grafico(fig, archivo):
    fig.savefig(DESTINO / archivo, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    if not ORIGEN.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {ORIGEN}")

    tabla = pd.read_csv(ORIGEN, encoding="utf-8-sig", dtype="string")
    esperadas = ["escena", "uid", "clasificacion_dano"]
    faltantes = set(esperadas).difference(tabla.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}")

    etiquetas = tabla["clasificacion_dano"].str.strip()
    nulos = int(tabla[esperadas].isna().sum().sum())
    duplicados = int(tabla.duplicated(subset=["escena", "uid"]).sum())
    desconocidas = int((etiquetas.notna() & ~etiquetas.isin(CLASES)).sum())
    if nulos or duplicados or desconocidas:
        print("ATENCIÓN: el CSV tiene problemas por revisar antes de usarlo en un modelo.")

    DESTINO.mkdir(parents=True, exist_ok=True)
    conteo = etiquetas.value_counts().reindex(CLASES, fill_value=0)
    n = len(tabla)
    distribucion = pd.DataFrame({
        "clasificacion": CLASES,
        "edificaciones": [int(conteo[c]) for c in CLASES],
        "porcentaje_total": [round(int(conteo[c]) * 100 / n, 4) for c in CLASES],
    })
    distribucion.to_csv(DESTINO / "distribucion_eda.csv", index=False, encoding="utf-8-sig")

    por_escena = tabla.groupby("escena", dropna=False).size().rename("n_edificaciones")
    con_dano = tabla.loc[etiquetas.isin(["minor-damage", "major-damage", "destroyed"])]
    por_escena_dano = con_dano.groupby("escena").size().rename("n_con_dano")
    escenas = pd.concat([por_escena, por_escena_dano], axis=1).fillna(0).astype(int)
    escenas = escenas.reset_index().sort_values("escena")
    escenas.to_csv(DESTINO / "edificaciones_por_escena.csv", index=False, encoding="utf-8-sig")

    # GRÁFICO 1: Todas las clases. Escala logarítmica para mostrar 3 y 51 084 a la vez.
    fig, ax = plt.subplots(figsize=(10, 5.3))
    barras = ax.bar(NOMBRES, [int(conteo[c]) for c in CLASES])
    if (conteo > 0).all():
        ax.set_yscale("log")
        ax.set_ylim(.7, int(conteo.max()) * 5)
    ax.set_title("xBD México | Distribución completa de clasificaciones", fontsize=14, weight="bold")
    ax.set_ylabel("Edificaciones · escala logarítmica")
    ax.tick_params(axis="x", labelrotation=10)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    for barra, v in zip(barras, [int(conteo[c]) for c in CLASES]):
        ax.annotate(f"{v:,}".replace(",", " "),
                    (barra.get_x() + barra.get_width()/2, barra.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    fig.text(.5, .015, f"Base: {n:,} registros | Los ejes logarítmicos NO representan distancias proporcionales".replace(",", " "),
             ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.20)
    guardar_grafico(fig, "01_distribucion_danos.png")

    # GRÁFICO 2: Acercamiento a daños + sin clasificar, SIN incluir 'sin daño'.
    clase_minoritaria = CLASES[1:]
    nombres_detalle = NOMBRES[1:]
    cantidades = [int(conteo[c]) for c in clase_minoritaria]
    fig, ax = plt.subplots(figsize=(10, 5.1))
    barras = ax.barh(nombres_detalle, cantidades)
    ax.invert_yaxis()
    ax.set_xlim(0, max(cantidades) * 1.48 if max(cantidades) else 1)
    ax.set_xlabel("Número de edificaciones")
    ax.set_title("Detalle de categorías minoritarias", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    for barra, v in zip(barras, cantidades):
        ax.text(v + max(cantidades) * .025, barra.get_y() + barra.get_height()/2,
                f"{v:,}   ({100*v/n:.3f}% del total)".replace(",", " "),
                va="center", ha="left", fontsize=10)
    fig.text(.5, .015,
             f"Se omite 'Sin daño' (51 084 edificios) SOLO para ampliar la escala | Total del dataset: {n:,}".replace(",", " "),
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.19, bottom=.18)
    guardar_grafico(fig, "02_detalle_categorias_minoritarias.png")

    # GRÁFICO 3: número de edificaciones anotadas por escena.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(escenas["n_edificaciones"], bins=20, edgecolor="black", linewidth=.4)
    mediana = float(escenas["n_edificaciones"].median())
    ax.axvline(mediana, linestyle="--", label=f"Mediana: {mediana:g}")
    ax.set(title="xBD México | Edificaciones registradas por escena",
           xlabel="Edificaciones en cada escena", ylabel="Número de escenas")
    ax.legend()
    fig.tight_layout()
    guardar_grafico(fig, "03_edificaciones_por_escena.png")

    # Reporte cuantitativo, sin alterar ninguna etiqueta o dato.
    texto = (
        "T2 | GRUPO 04 | EDA INICIAL DE xBD MÉXICO\n"
        f"Registros analizados: {n}\n"
        f"Escenas: {len(escenas)}\n"
        f"Nulos: {nulos}\n"
        f"Duplicados por escena y uid: {duplicados}\n"
        f"Etiquetas distintas de las cinco esperadas: {desconocidas}\n"
        f"Sin clasificar: {int(conteo['un-classified'])}\n"
        f"Escenas con al menos una edificación dañada: {int((escenas['n_con_dano'] > 0).sum())}\n"
        f"Mediana de edificaciones por escena: {mediana:g}\n"
        "\nLos registros 'sin clasificar' se mantienen en los gráficos de EDA.\n"
        "No hay variables numéricas físicas para calcular correlaciones válidas.\n"
        "Para análisis geométrico se requieren los JSON de anotaciones originales.\n"
    )
    (DESTINO / "resumen_eda.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Gráficos y resultados en: {DESTINO}")


if __name__ == "__main__":
    main()
