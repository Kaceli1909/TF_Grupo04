#!/usr/bin/env python3
"""Reproduce matrices de confusión y gráfico de FP/FN desde un CSV agregado.

No utiliza predicciones de edificios ni reevalúa modelos. El archivo CSV incluye
las cantidades reconstruidas de las métricas publicadas (ProtoNet: registro).

Uso desde la raíz del repositorio:
    python TF3/codigo/graficar_matrices_confusion.py

Dependencias: numpy, matplotlib (incluidas en el entorno de CIMLAB AI).
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
CSV_PATH = HERE / "resultados/conteos_matriz_confusion_validacion.csv"
OUT = HERE / "resultados/figuras"
OUT.mkdir(parents=True, exist_ok=True)
DISPLAY = {
    "random_forest": "Random Forest",
    "dinov2_xgboost": "DINOv2 + XGBoost",
    "protonet_dino": "ProtoNet DINO",
    "hybrid": "Hybrid",
}


def run():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        key = row["modelo"]
        values = {k: int(row[k]) for k in ("tp", "fp", "fn", "tn")}
        cm = np.array([[values["tn"], values["fp"]],
                       [values["fn"], values["tp"]]])
        pc = cm / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(6, 4.8))
        im = ax.imshow(pc, vmin=0, vmax=100)
        fig.colorbar(im, ax=ax, fraction=.045, pad=.055,
                     label="Porcentaje dentro de la clase real (%)")
        ax.set_xticks([0, 1], ["No dañado (0)", "Dañado (1)"])
        ax.set_yticks([0, 1], ["No dañado (0)", "Dañado (1)"])
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Etiqueta real xBD")
        ax.set_title(f"Matriz de confusión · {DISPLAY.get(key, key)}")
        for i in (0, 1):
            for j in (0, 1):
                code = (("TN", "FP"), ("FN", "TP"))[i][j]
                ax.text(j, i, f"{code} = {cm[i,j]:,}\n{pc[i,j]:.2f} %",
                        ha="center", va="center", fontweight="bold",
                        color="white" if pc[i,j] >= 55 else "black")
        fig.tight_layout()
        fig.savefig(OUT / f"matriz_confusion_{key}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.6, 3.5))
        tags = ["TP", "FP", "FN", "TN"]
        bars = ax.bar(tags, [values[z.lower()] for z in tags])
        ax.bar_label(bars, padding=2)
        ax.set_ylabel("Edificios")
        ax.set_title(f"Resultados de clasificación · {DISPLAY.get(key, key)}")
        fig.tight_layout()
        fig.savefig(OUT / f"conteos_fp_fn_tp_tn_{key}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    x = np.arange(len(rows))
    width = .26
    for offset, field, name in ((-width,"tp","TP: daño detectado"),
                                (0,"fp","FP: falsa alarma"),
                                (width,"fn","FN: daño omitido")):
        bars = ax.bar(x + offset, [int(r[field]) for r in rows],
                      width=width, label=name)
        ax.bar_label(bars, padding=2)
    ax.set_xticks(x, [DISPLAY.get(r["modelo"],r["modelo"]) for r in rows])
    ax.set_ylabel("Edificios")
    ax.set_title("Aciertos y errores asociados a la clase dañada")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "comparacion_fp_fn_tp.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
