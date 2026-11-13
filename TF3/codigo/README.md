# 🐍 Código fuente Python — TF3

Este directorio contiene:

- **`cimlab_ai/`**: snapshot del motor y la interfaz de CIMLAB AI, con Python de desarrollo y correcciones v0.5.1–v0.5.4. Incluye módulos de preparación, entrenamiento, comparación, auditoría y predicción, así como pruebas sintéticas. Ver [README de la aplicación](cimlab_ai/README.md).
- **`auditar_manifest.py`**: verificador independiente del manifiesto local, que usa solo bibliotecas estándar de Python. Comprueba etiquetas, identificadores, splits y origen PRE/POST; no reentrena ni modifica datos.

## Verificar la colección local

```bash
python TF3/codigo/auditar_manifest.py "J:/TF 3/Trabajo_Mexico_y_Palu/manifest.csv" --salida "resumen_manifest.json"
```

Agregar `--verificar-recortes` si las rutas locales a los JPG son accesibles. **El manifiesto debe ser el corregido con v0.5.4**, no el que se exportó antes de esa actualización.

El código Python distribuido **no sustituye la publicación posterior de modelos entrenados, hiperparámetros, versiones exactas de dependencias y métricas de evaluación verificadas**. No deben añadirse al repositorio credenciales, `.venv`, recortes masivos ni los manifiestos originales con rutas privadas de Windows.
