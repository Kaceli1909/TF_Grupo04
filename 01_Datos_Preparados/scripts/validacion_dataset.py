import os
import json
import glob
import csv
import sys
from collections import Counter

if len(sys.argv) < 2:
    print("Uso: python validacion_dataset.py RUTA_DATASET")
    sys.exit(1)

root = sys.argv[1]

def buscar(tipo):
    patron = os.path.join(
        root, "**", "mexico-earthquake_*_" + tipo + ".json"
    )
    return glob.glob(os.path.join(root, "**", "labels",
                                  "mexico-earthquake_*_" + tipo + ".json"),
                     recursive=True)

pre_files = buscar("pre_disaster")
post_files = buscar("post_disaster")

pre = {os.path.basename(f).replace("_pre_disaster.json", ""): f
       for f in pre_files}
post = {os.path.basename(f).replace("_post_disaster.json", ""): f
        for f in post_files}

escenas = sorted(set(pre) | set(post))
danos = Counter()
filas = []
resumen = []

for escena in escenas:
    if escena not in pre or escena not in post:
        resumen.append([escena, "FALTA ARCHIVO", 0, 0, 0, 0])
        continue

    with open(pre[escena], encoding="utf-8") as f:
        a = json.load(f)["features"]["lng_lat"]

    with open(post[escena], encoding="utf-8") as f:
        b = json.load(f)["features"]["lng_lat"]

    uid_pre = [x["properties"]["uid"] for x in a]
    uid_post = [x["properties"]["uid"] for x in b]

    A, B = set(uid_pre), set(uid_post)
    coincidencias = len(A & B)

    duplicados = (len(uid_pre) != len(A) or
                  len(uid_post) != len(B))

    estado = "OK" if A == B and not duplicados else "REVISAR"

    resumen.append([
        escena, estado, len(a), len(b),
        coincidencias, len(A - B) + len(B - A)
    ])

    post_por_uid = {
        x["properties"]["uid"]: x["properties"]
        for x in b
    }

    for x in a:
        p = x["properties"]
        uid = p["uid"]
        q = post_por_uid.get(uid, {})
        dano = q.get("subtype", "SIN_ETIQUETA")

        filas.append([escena, uid, dano])
        if q.get("feature_type") == "building":
            danos.update([dano])

os.makedirs("01_Datos_Preparados/resultados", exist_ok=True)
salida = "01_Datos_Preparados/resultados"

with open(os.path.join(salida, "validacion_escenas.csv"),
          "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["escena", "estado", "edificios_pre",
                "edificios_post", "uid_coincidentes", "diferencias"])
    w.writerows(resumen)

with open(os.path.join(salida, "tabla_edificaciones.csv"),
          "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["escena", "uid", "clasificacion_dano"])
    w.writerows(filas)

with open(os.path.join(salida, "distribucion_danos.csv"),
          "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["clasificacion", "cantidad", "porcentaje"])
    total = sum(danos.values())
    for clase, cantidad in sorted(danos.items()):
        w.writerow([clase, cantidad, round(cantidad / total * 100, 4)])

with open(os.path.join(salida, "resumen_validacion.txt"),
          "w", encoding="utf-8") as f:
    f.write("VALIDACION DATASET xBD - MEXICO EARTHQUAKE\n")
    f.write(f"Archivos PRE: {len(pre_files)}\n")
    f.write(f"Archivos POST: {len(post_files)}\n")
    f.write(f"Escenas: {len(escenas)}\n")
    f.write(f"Anotaciones PRE/POST emparejadas: {len(filas)}\n")
    f.write(f"Distribucion de danos: {dict(danos)}\n")
    f.write(f"Escenas con diferencias: "
            f"{sum(x[1] != 'OK' for x in resumen)}\n")

print("VALIDACION FINALIZADA")
print("Escenas:", len(escenas))
print("Edificios emparejados:", len(filas))
print("Distribucion de daños:", dict(danos))
print("Resultados guardados en:", salida)
