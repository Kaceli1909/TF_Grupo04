"""Simple local machine diagnostic; no network calls."""
from __future__ import annotations
import platform
import shutil
import subprocess
print('Python:',platform.python_version())
if shutil.which('nvidia-smi'):
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
            capture_output=True,text=True,timeout=7)
        print('NVIDIA:',p.stdout.strip() if p.returncode==0 else 'no disponible')
    except Exception as e:print('NVIDIA: error al consultar:',e)
else:print('NVIDIA: no detectada, disponible CPU')
try:
    import torch
    print('PyTorch:',torch.__version__)
    print('CUDA disponible:',torch.cuda.is_available())
    if torch.cuda.is_available():print('GPU usada:',torch.cuda.get_device_name(0))
    else:print('El entrenamiento de redes profundas se hara en CPU.')
except ImportError:print('PyTorch: no instalado, puede ejecutarse Random Forest con INSTALAR_BASE')
except Exception as e:print('Problema al importar PyTorch:',e)
