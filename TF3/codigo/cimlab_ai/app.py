"""CIMLAB AI Desktop - local tkinter UI on Windows 10, no browser required.
Long operations run in background threads, processing is local. No mock metrics.
"""
from __future__ import annotations
import json
import threading
import queue
import sys
import os
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from data_core import clone_lfs,scan,prepare,prepare_v03
from event_sources import github_events,local_events,REPO_URL,selected_roots
from labels_core import TASK_ID, TASK_NAME, save_event_mapping, require_task

HERE=Path(__file__).resolve().parent
CONFIG=HERE/'config_local.json'
BG='#101724';PANEL='#172236';CARD='#1d2c40';ACCENT='#34b9ab';TXT='#edf4fa';MUTED='#9fb4c9'

class CIMLABApp:
    def __init__(self,root):
        self.root=root
        root.title('CIMLAB AI v0.5 | Dañado / no dañado · Equipo')
        root.geometry('1170x760');root.minsize(990,660)
        root.configure(bg=BG)
        self.q=queue.Queue();self.working=False
        data={}
        try:data=json.loads(CONFIG.read_text(encoding='utf-8'))
        except Exception:pass
        self.path=tk.StringVar(value=data.get('repository',str(HERE/'repositorio')))
        self.work=tk.StringVar(value=data.get('workspace',str(HERE/'trabajo')))
        self.source_type=tk.StringVar(value=data.get('source_type','GitHub'))
        self.event_status=tk.StringVar(value='Pulsa Actualizar eventos para consultar GitHub o explorar la carpeta local.')
        self.event_names=[];self.event_catalog_source=None
        self.latest_source_audit=None
        self.device_status=tk.StringVar(value='Comprobando si hay GPU disponible...')
        self.epochs=tk.IntVar(value=20)
        self.batch=tk.IntVar(value=8)
        self.neg=tk.IntVar(value=0)
        self.pre=tk.StringVar();self.post=tk.StringVar();self.json=tk.StringVar()
        self.model_vars={k:tk.BooleanVar(value=False) for k in
            ('random_forest','dinov2_xgboost','resnet18_early','siamese_resnet18',
             'siamese_unet_lite','changeos_lite','dahitra_lite','protonet_dino')}
        self.audit_model=tk.StringVar(value='resnet18_early')
        self.audit_group=tk.StringVar(value='FN')
        self.audit_view=tk.StringVar(value='Original')
        self.quality_model=tk.StringVar(value='dinov2_xgboost')
        self.quality_status=tk.StringVar(value='Ejecuta la auditoría para identificar recortes en bordes y escenas a revisar.')
        self.audit_thr=tk.DoubleVar(value=.5)
        self.audit_data=None;self.audit_display=[];self.audit_photo=None
        self.pages={}
        self.setup_styles();self.build()
        root.after(120,self.poll)
        root.after(350,self.detect_device)
        root.protocol('WM_DELETE_WINDOW',self.on_close)

    def setup_styles(self):
        style=ttk.Style(self.root);style.theme_use('clam')
        style.configure('Dark.TCombobox',fieldbackground=CARD,background=CARD,foreground=TXT)
        style.configure('Dark.TEntry',fieldbackground=CARD,foreground=TXT,insertcolor=TXT)

    def make_label(self,parent,text,size=11,color=TXT,bold=False,wrap=900):
        return tk.Label(parent,text=text,fg=color,bg=parent.cget('bg'),font=('Segoe UI',size,'bold' if bold else 'normal'),wraplength=wrap,justify='left')

    def btn(self,parent,text,callback,primary=False,width=None):
        bg=ACCENT if primary else CARD; fg='#071b23' if primary else TXT
        return tk.Button(parent,text=text,command=callback,bg=bg,fg=fg,
                         activebackground=('#56d3c5' if primary else '#2a435e'),activeforeground=fg,
                         relief='flat',cursor='hand2',font=('Segoe UI',10,'bold'),height=2,
                         padx=14,pady=3,width=width)

    def line(self,parent,title,var,browse=False,folder=True):
        self.make_label(parent,title,10,MUTED).pack(anchor='w',pady=(10,4))
        row=tk.Frame(parent,bg=parent.cget('bg'));row.pack(fill='x')
        e=tk.Entry(row,textvariable=var,bg=CARD,fg=TXT,insertbackground=TXT,relief='flat',font=('Consolas',10))
        e.pack(side='left',fill='x',expand=True,ipady=10)
        if browse:
            def choose():
                v=filedialog.askdirectory() if folder else filedialog.askopenfilename()
                if v:var.set(v)
            self.btn(row,'Examinar',choose).pack(side='left',padx=(7,0))

    def build(self):
        sidebar=tk.Frame(self.root,bg=PANEL,width=240);sidebar.pack(side='left',fill='y');sidebar.pack_propagate(False)
        tk.Label(sidebar,text='CIMLAB',bg=PANEL,fg=ACCENT,font=('Segoe UI',25,'bold')).pack(anchor='w',padx=24,pady=(28,0))
        tk.Label(sidebar,text='AI  /  DESASTRES NATURALES',bg=PANEL,fg=MUTED,font=('Segoe UI',9,'bold')).pack(anchor='w',padx=24,pady=(0,28))
        self.outer=tk.Frame(self.root,bg=BG);self.outer.pack(side='right',fill='both',expand=True)
        for name in ('Inicio','1  Datos / GitHub','2  Preparar','3  Entrenar','4  Comparar','5  Auditoría','6  Calidad','7  Predecir'):
            b=tk.Button(sidebar,text=name,anchor='w',bg=PANEL,fg=TXT,relief='flat',bd=0,
                        activebackground=CARD,activeforeground=ACCENT,font=('Segoe UI',11),
                        padx=22,pady=12,command=lambda n=name:self.show(n))
            b.pack(fill='x',pady=1)
        tk.Label(sidebar,text='CPU o GPU NVIDIA\n8 modelos + híbrido\nVarios eventos xBD',
                 bg=PANEL,fg=MUTED,font=('Segoe UI',9),justify='left').pack(side='bottom',anchor='w',padx=23,pady=24)
        for name in ('Inicio','1  Datos / GitHub','2  Preparar','3  Entrenar','4  Comparar','5  Auditoría','6  Calidad','7  Predecir'):
            frm=tk.Frame(self.outer,bg=BG);self.pages[name]=frm
        self.page_home(self.pages['Inicio'])
        self.page_data(self.pages['1  Datos / GitHub'])
        self.page_prepare(self.pages['2  Preparar'])
        self.page_train(self.pages['3  Entrenar'])
        self.page_compare(self.pages['4  Comparar'])
        self.page_audit(self.pages['5  Auditoría'])
        self.page_quality(self.pages['6  Calidad'])
        self.page_predict(self.pages['7  Predecir'])
        self.status=tk.StringVar(value='Listo para comenzar')
        bottom=tk.Frame(self.outer,bg=BG);bottom.pack(side='bottom',fill='x')
        self.make_label(bottom,'REGISTRO DE OPERACIONES',10,MUTED,True).pack(anchor='w',padx=18,pady=(8,3))
        self.logbox=tk.Text(bottom,height=8,bg='#09111f',fg='#c9dbe5',insertbackground=TXT,
                            relief='flat',wrap='word',font=('Consolas',9),state='disabled')
        self.logbox.pack(fill='x',padx=18)
        self.make_label(bottom,'',10,MUTED).pack()
        tk.Label(bottom,textvariable=self.status,bg=PANEL,fg=ACCENT,font=('Segoe UI',9),anchor='w',padx=18).pack(fill='x')
        self.show('Inicio')

    def show(self,name):
        for f in self.pages.values():f.pack_forget()
        self.pages[name].pack(fill='both',expand=True,padx=26,pady=18)
        if name=='4  Comparar':self.refresh_comparison()

    def header(self,frame,title,subtitle):
        self.make_label(frame,title,20,TXT,True).pack(anchor='w',pady=(4,5))
        self.make_label(frame,subtitle,10,MUTED,wrap=800).pack(anchor='w',pady=(0,17))

    def card(self,frame):
        c=tk.Frame(frame,bg=PANEL,padx=18,pady=13);c.pack(fill='x',pady=6)
        return c

    def page_home(self,f):
        self.header(f,'Evaluación de daños por desastres naturales',
                    'Eventos como sismos, tsunamis e inundaciones · imágenes PRE/POST + polígonos xBD')
        c=self.card(f)
        self.make_label(c,'FLUJO DE TRABAJO · OBJETIVO ÚNICO: DAÑADO / NO DAÑADO',11,ACCENT,True).pack(anchor='w',pady=(0,8))
        for s in ('01  Elegir evento(s) locales o actualizar catálogo GitHub y descargar TIFF/JSON.',
                  '02  Verificar que las etiquetas de TODOS los eventos representan daño visible en edificaciones.',
                  '03  Preparar UN dataset conjunto de eventos; entrenar y comparar los modelos.',
                  '04  Auditar y comparar en validación; elegir modelo y evaluar test una vez.',
                  '05  Predecir daño con el ganador (sin dictaminar seguridad estructural).' ):
            self.make_label(c,s,11).pack(anchor='w',pady=6)
        c=self.card(f)
        self.make_label(c,'IMPORTANTE',11,ACCENT,True).pack(anchor='w')
        self.make_label(c,'Objetivo: dañado (1) vs no dañado (0) en todos los eventos elegidos. Daño menor, mayor y destrucción se consideran dañados. Si faltan suficientes edificios dañados, la preparación se detiene para evitar un entrenamiento inválido. No sustituye una inspección estructural.',10,TXT,wrap=740).pack(anchor='w',pady=8)

    def page_data(self,f):
        self.header(f,'1 · Fuentes y eventos de desastre',
            'No hay eventos codificados: actualiza GitHub cuando el equipo publique Palu u otros datos, o utiliza tu propia carpeta.')
        c=self.card(f)
        row=tk.Frame(c,bg=PANEL);row.pack(anchor='w',pady=(0,4))
        self.make_label(row,'ORIGEN:',10,ACCENT,True).pack(side='left',padx=(0,10))
        for v in ('GitHub','Carpeta local'):
            tk.Radiobutton(row,text=v,value=v,variable=self.source_type,bg=PANEL,fg=TXT,
                selectcolor=CARD,activebackground=PANEL,activeforeground=ACCENT,
                command=self.source_changed).pack(side='left',padx=8)
        self.line(c,'Carpeta del repositorio (GitHub) o carpeta que ya contiene TIFF + JSON',self.path,True)
        row=tk.Frame(c,bg=PANEL);row.pack(fill='x',pady=(9,3))
        self.btn(row,'Actualizar eventos disponibles',self.refresh_events,True).pack(side='left',padx=(0,8))
        self.btn(row,'Descargar eventos seleccionados',self.download).pack(side='left')
        self.make_label(c,'EVENTOS DISPONIBLES · selecciona uno o varios (Ctrl+clic)',9,ACCENT,True).pack(anchor='w',pady=(7,5))
        self.event_list=tk.Listbox(c,selectmode=tk.EXTENDED,height=5,exportselection=False,
            bg=CARD,fg=TXT,selectbackground='#287968',selectforeground='white',
            font=('Consolas',10),relief='flat',highlightthickness=0)
        self.event_list.pack(fill='x')
        tk.Label(c,textvariable=self.event_status,bg=PANEL,fg=MUTED,
            font=('Segoe UI',9),justify='left',anchor='w',wraplength=760).pack(fill='x',pady=(6,5))
        checks=tk.Frame(c,bg=PANEL);checks.pack(anchor='w',pady=(4,0))
        self.btn(checks,'Auditar TIFF + JSON + etiquetas',self.audit).pack(side='left',padx=(0,8))
        self.btn(checks,'Revisar equivalencias de etiquetas',self.review_labels).pack(side='left')
        self.make_label(c,'CLASIFICACIÓN COMÚN: dañado=1; no dañado=0. México + Palu + otros se combinan en un solo entrenamiento si sus etiquetas son compatibles.',9,ACCENT,True,wrap=745).pack(anchor='w',pady=(8,2))
        self.make_label(c,'GitHub: primero Actualizar, después selecciona eventos y Descargar. Se usa Git LFS para obtener los TIFF verdaderos, no los punteros. Local: elige una carpeta de evento, un conjunto de eventos o el repositorio que ya descargaste.',9,MUTED,wrap=745).pack(anchor='w',pady=(8,1))
        self.make_label(c,'Repositorios compatibles: carpetas data/<evento>/..., pares *_pre_disaster y *_post_disaster con TIFF/PNG/JPG y etiquetas *_post_disaster.json de xBD. Una estructura o formato distinto requerirá adaptación.',9,MUTED,wrap=745).pack(anchor='w',pady=(2,0))
        self.help_btn(f,'Datos')

    def source_changed(self):
        self.event_catalog_source=None;self.event_names=[];self.latest_source_audit=None
        self.event_list.delete(0,tk.END)
        if self.source_type.get()=='GitHub':
            self.event_status.set('GitHub: consulta dinámica. Los nuevos eventos aparecerán cuando estén publicados en la rama principal.')
        else:
            self.event_status.set('Local: selecciona tu carpeta; los TIFF y JSON deben estar completos y utilizar un formato xBD compatible.')

    def refresh_events(self):
        self.latest_source_audit=None
        mode=self.source_type.get();path=self.path.get()
        def do():
            if mode=='GitHub':
                names=github_events()
            else:
                names=list(local_events(path))
            self.q.put(('events',(names,mode,path)))
        self.submit('Consultar eventos de '+mode,do)

    def selected_events(self):
        if self.event_catalog_source is None:
            raise ValueError('Primero pulsa Actualizar eventos disponibles para detectar tus carpetas o consultar GitHub.')
        mode,path=self.event_catalog_source
        if mode!=self.source_type.get() or (mode=='Carpeta local' and str(Path(path).resolve())!=str(Path(self.path.get()).resolve())):
            raise ValueError('La fuente cambió. Vuelve a pulsar Actualizar eventos disponibles.')
        names=[self.event_names[i] for i in self.event_list.curselection()]
        if not names:
            raise ValueError('Selecciona uno o más eventos en la lista (Ctrl+clic para varios).')
        return names

    def detect_device(self):
        def worker():
            import shutil
            pieces=[]
            if shutil.which('nvidia-smi'):
                try:
                    p=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                        capture_output=True,text=True,timeout=5)
                    if p.returncode==0 and p.stdout.strip(): pieces.append('GPU NVIDIA: '+p.stdout.strip().splitlines()[0])
                except Exception:pass
            try:
                import torch
                if torch.cuda.is_available():pieces.append('PyTorch CUDA disponible: entrenamiento acelerado')
                else:pieces.append('PyTorch usará CPU (CNN y DINOv2 pueden ser lentos)')
            except ImportError:
                pieces.append('PyTorch no instalado: ejecuta el instalador CPU o GPU')
            self.q.put(('device',' | '.join(pieces) or 'CPU: sin GPU NVIDIA detectada'))
        threading.Thread(target=worker,daemon=True).start()

    def help_btn(self,f,section):
        self.btn(f,'?  Guía de esta etapa',lambda:self.show_help(section)).pack(anchor='w',pady=(9,2))

    def show_help(self,section):
        explanations={
        'Datos': 'FUENTES Y EVENTOS\n\nOBJETIVO: dañado / no dañado con TODOS los eventos seleccionados, sin distinguir el tipo de desastre en la variable a predecir. Las cuatro etiquetas xBD se asignan así: no-damage=0; minor-damage, major-damage y destroyed=1. Etiquetas desconocidas requieren revisión antes de preparar. No equiparar alto/bajo o severo con daño sin comprobar el significado de la etiqueta.\n\nGitHub consulta EN VIVO las carpetas bajo data/ del repositorio TF_Grupo04. Si el equipo publica Palu allí, pulsa Actualizar eventos; NO se requiere nueva versión de CIMLAB. Elige los eventos y descarga solo sus archivos con Git LFS.\n\nLocal usa TIFF y JSON que ya tengas. Puedes elegir varios eventos, pero todos deben seguir la convención de nombres xBD y tener imágenes PRE y POST junto con polígonos de edificios etiquetados. Se admiten imágenes TIFF, PNG y JPEG. Comprueba la auditoría antes de preparar.',
        'Preparar': 'PREPARAR\n\nUtiliza TODOS los eventos seleccionados y prepara una sola colección binaria de edificios dañados / no dañados. Ningún evento se excluye por ser sismo o tsunami. Valida las etiquetas antes de trabajar. La partición es por escena, no por edificio.\n\nConvierte los pares PRE/POST y los polígonos JSON en recortes por edificio. Separa escenas en entrenamiento (train), validación (val) y prueba (test) para evitar compartir la misma escena.\n\nMáximo NO DAÑADOS (0 = todos): se usa TODA la colección seleccionada por defecto. Si el equipo decide limitar negativos en train por desbalance, indique un número positivo. Val y test conservan todos los edificios. Al cambiar la selección de eventos, prepara en una carpeta de trabajo NUEVA para no mezclar modelos de datasets distintos.',
        'Entrenar': 'ENTRENAR\n\nSelecciona los modelos a ejecutar. Épocas = pasadas por los datos; batch = cantidad de ejemplos por actualización. Cuanto menor la VRAM disponible, menor batch. La aplicación detecta PyTorch/CUDA: si no hay GPU compatible, usa CPU, que puede tardar mucho para redes profundas.\n\nRandom Forest es una primera prueba rápida sin GPU. Entrena en validación y NO cambies el test al comparar modelos.',
        'Comparar': 'COMPARAR\n\nPOSITIVO = DAÑADO (1), NEGATIVO = NO DAÑADO (0). Comparar únicamente modelos evaluados sobre el mismo conjunto y definición de etiquetas.\n\nAP (Average Precision) mide el ordenamiento de las predicciones a diferentes umbrales; F1 equilibra precisión y recall en un umbral. Recall = proporción de dañados identificados. Precisión = proporción de alertas que corresponden a daño.\n\nPrimero compara en validación; haz UNA sola evaluación final en test cuando el equipo haya elegido su modelo.',
        'Auditoría': 'AUDITORÍA\n\nFP = falso positivo (alerta de DAÑADO en un edificio sin daño); FN = falso negativo (un dañado omitido). TP/TN son aciertos. El umbral clasifica puntuaciones sin calibrar, por eso puede producir muchas falsas alarmas.\n\nLas vistas de luz equilibrada y mapa de cambios son solo ayudas visuales: NO alteran predicciones ni demuestran daño por sí mismas.',
        'Calidad': 'CALIDAD\n\nRevisa recortes con franjas, posibles desenfoques y diferencias radiométricas extremas en train/val. No inspecciona test ni cambia las etiquetas. El informe sirve para decidir si conviene limpiar los datos antes de volver a entrenar.',
        'Predecir': 'PREDECIR\n\nPredice dañado / no dañado según el objetivo del entrenamiento actual. Usa pesos generados con el dataset binario actual: verifica el manifest y la compatibilidad del modelo antes de utilizarlo.\n\nSelecciona PRE, POST y un JSON de polígonos de la misma escena. La app estima cuáles edificios anotados presentan daño visible y guarda una tabla y un mapa. No detecta edificios nuevos ni evalúa seguridad estructural.'}
        d=tk.Toplevel(self.root);d.title('CIMLAB AI · Ayuda · '+section)
        d.geometry('650x420');d.configure(bg=BG);d.transient(self.root)
        t=tk.Text(d,wrap='word',bg=PANEL,fg=TXT,relief='flat',font=('Segoe UI',11),padx=20,pady=20)
        t.pack(fill='both',expand=True,padx=15,pady=15)
        t.insert('1.0',explanations.get(section,''));t.configure(state='disabled')
        self.btn(d,'Cerrar',d.destroy).pack(pady=(0,12))

    def page_prepare(self,f):
        self.header(f,'Preparación · daño binario','Combina todos los eventos elegidos. Recortes sin bordes negros artificiales; dataset nuevo sin sobrescribir otros.')
        c=self.card(f)
        self.line(c,'Carpeta de trabajo (recortes, métricas, modelos)',self.work,True)
        row=tk.Frame(c,bg=PANEL);row.pack(anchor='w',pady=(16,10))
        self.make_label(row,'Máximo NO DAÑADOS en train (0 = todos):',10,MUTED).pack(side='left')
        tk.Spinbox(row,from_=0,to=1000000,increment=100,textvariable=self.neg,width=7,
                   bg=CARD,fg=TXT,buttonbackground=CARD,relief='flat').pack(side='left',padx=12)
        self.btn(c,'Preparar colección conjunta · Dañado / No dañado',self.prepare,True).pack(anchor='w',pady=10)
        self.make_label(c,'Selecciona TODOS los eventos en Datos y audita sus etiquetas. La app detiene la preparación ante categorías desconocidas o si faltan suficientes edificios dañados. Para no sobrescribir resultados previos, se crea un trabajo nuevo si ya existe un manifest.',10,MUTED,wrap=700).pack(anchor='w',pady=8)
        self.help_btn(f,'Preparar')

    def page_train(self,f):
        self.header(f,'Entrenamiento de modelos',
                    '8 alternativas · una sola etiqueta positiva: DAÑADO, en todos los eventos seleccionados. CPU o GPU.')
        c=self.card(f)
        groups=[('MODELOS ORIGINALES',[
            ('random_forest','Random Forest'),
            ('dinov2_xgboost','DINOv2 + XGBoost'),
            ('resnet18_early','ResNet18 Early Fusion'),
            ('siamese_resnet18','Siamese ResNet18')]),
            ('NUEVOS · VERSIONES LIGERAS ADAPTADAS',[
            ('siamese_unet_lite','Siamese U-Net (máscaras de edificios)'),
            ('changeos_lite','ChangeOS-like (objeto y cambio)'),
            ('dahitra_lite','DAHiTrA-like (Transformer)'),
            ('protonet_dino','ProtoNet (pocos ejemplos / DINOv2)')])]
        for heading,items in groups:
            self.make_label(c,heading,9,ACCENT,True).pack(anchor='w',pady=(4,3))
            row=tk.Frame(c,bg=PANEL);row.pack(fill='x')
            for i,(key,title) in enumerate(items):
                cb=tk.Checkbutton(row,text=title,variable=self.model_vars[key],bg=PANEL,fg=TXT,
                    activebackground=PANEL,activeforeground=ACCENT,selectcolor=CARD,
                    font=('Segoe UI',10),anchor='w',wraplength=335,justify='left')
                cb.grid(row=i//2,column=i%2,sticky='w',padx=(0,25),pady=3)
        line=tk.Frame(c,bg=PANEL);line.pack(anchor='w',pady=(12,6))
        self.make_label(line,'Épocas máximas',10,MUTED).pack(side='left')
        tk.Spinbox(line,from_=2,to=100,textvariable=self.epochs,width=5,bg=CARD,fg=TXT,
                   buttonbackground=CARD,relief='flat').pack(side='left',padx=10)
        self.make_label(line,'Batch size',10,MUTED).pack(side='left',padx=(16,0))
        tk.Spinbox(line,from_=1,to=32,textvariable=self.batch,width=5,bg=CARD,fg=TXT,
                   buttonbackground=CARD,relief='flat').pack(side='left',padx=10)
        tk.Label(c,textvariable=self.device_status,bg=PANEL,fg=ACCENT,
            font=('Segoe UI',9),justify='left',anchor='w',wraplength=750).pack(fill='x',pady=(4,2))
        self.btn(c,'Entrenar modelos seleccionados',self.train,True).pack(anchor='w',pady=(2,6))
        self.help_btn(f,'Entrenar')
        self.make_label(c,'Los nuevos ChangeOS-like y DAHiTrA-like son adaptaciones ligeras de investigación; NO son reproducciones del código oficial. ProtoNet reutiliza DINOv2. Entrenar en secuencia para ahorrar VRAM.',9,MUTED,wrap=735).pack(anchor='w',pady=3)

    def page_compare(self,f):
        self.header(f,'Comparación y evaluación · Dañado / No dañado','Modelos entrenados para DAÑO VISIBLE (cualquier nivel de daño); test no participa en la elección.')
        self.table=tk.Text(f,bg=PANEL,fg=TXT,height=15,relief='flat',font=('Consolas',10),wrap='none')
        self.table.pack(fill='x',pady=10);self.table.configure(state='disabled')
        row=tk.Frame(f,bg=BG);row.pack(anchor='w',pady=8)
        self.btn(row,'Actualizar resultados',self.refresh_comparison).pack(side='left',padx=(0,12))
        self.btn(row,'Generar informe y gráficos',self.report).pack(side='left',padx=(0,12))
        self.btn(row,'Seleccionar y evaluar en test una sola vez',self.evaluate,True).pack(side='left')
        self.make_label(f,'AP: área resumida bajo precisión-recall; F1: balance entre precisión y recall. Selecciona el ganador con VALIDACIÓN y reserva test para el final. «Pendiente» significa que ese modelo no se entrenó.',10,MUTED,wrap=760).pack(anchor='w',pady=8)
        self.help_btn(f,'Comparar')

    def page_audit(self,f):
        from audit_core import MODELS,NAMES
        self.header(f,'Auditoría de daños · solo validación',
            'Revisa errores reales por edificio, cambia el umbral y mide Recall@K sin tocar test.')
        controls=tk.Frame(f,bg=PANEL,padx=14,pady=10);controls.pack(fill='x',pady=(0,8))
        row=tk.Frame(controls,bg=PANEL);row.pack(fill='x')
        self.make_label(row,'Modelo:',10,MUTED).pack(side='left')
        self.audit_box=ttk.Combobox(row,textvariable=self.audit_model,values=list(MODELS),
                                   state='readonly',width=23)
        self.audit_box.pack(side='left',padx=(7,12))
        self.btn(row,'Cargar validación',self.audit_load).pack(side='left',padx=(0,13))
        self.make_label(row,'Filtro:',10,MUTED).pack(side='left')
        self.audit_filter=ttk.Combobox(row,textvariable=self.audit_group,
                        values=['FN','FP','TP','TN','Mayores scores','Cerca del umbral','Aleatorios'],
                        state='readonly',width=18)
        self.audit_filter.pack(side='left',padx=7)
        self.audit_filter.bind('<<ComboboxSelected>>',lambda _:self.audit_refresh_cases())
        sec=tk.Frame(controls,bg=PANEL);sec.pack(fill='x',pady=(10,0))
        self.make_label(sec,'Umbral de alerta:',10,MUTED).pack(side='left')
        tk.Scale(sec,from_=0.01,to=.99,resolution=.01,orient='horizontal',length=325,
             variable=self.audit_thr,command=lambda _:self.audit_threshold_changed(),
             bg=PANEL,fg=TXT,highlightthickness=0,troughcolor=CARD,
             activebackground=ACCENT).pack(side='left',padx=10)
        self.audit_scores=tk.StringVar(value='Carga un modelo para ver sus resultados.')
        self.make_label(controls,'MÉTRICAS PARA EL UMBRAL ACTUAL',9,ACCENT,True).pack(anchor='w',pady=(8,2))
        tk.Label(controls,textvariable=self.audit_scores,bg=PANEL,fg=TXT,
                 font=('Consolas',10),anchor='w',justify='left').pack(anchor='w')
        content=tk.Frame(f,bg=BG);content.pack(fill='both',expand=True)
        left=tk.Frame(content,bg=PANEL,padx=8,pady=8);left.pack(side='left',fill='both',expand=True,padx=(0,8))
        self.make_label(left,'EDIFICIOS DE VALIDACIÓN',10,ACCENT,True).pack(anchor='w',pady=(0,5))
        columns=('outcome','score','scene','uid')
        self.audit_tree=ttk.Treeview(left,columns=columns,show='headings',height=8)
        for name,width in [('outcome',55),('score',70),('scene',180),('uid',130)]:
            self.audit_tree.heading(name,text=name.upper());self.audit_tree.column(name,width=width,anchor='w')
        self.audit_tree.pack(fill='both',expand=True)
        self.audit_tree.bind('<<TreeviewSelect>>',self.audit_selected)
        right=tk.Frame(content,bg=PANEL,padx=8,pady=8);right.pack(side='left',fill='both',expand=True)
        self.make_label(right,'PRE / POST + POLÍGONO',10,ACCENT,True).pack(anchor='w',pady=(0,5))
        opt=tk.Frame(right,bg=PANEL);opt.pack(fill='x',pady=(0,7))
        self.make_label(opt,'Visualización:',9,MUTED).pack(side='left')
        ttk.Combobox(opt,textvariable=self.audit_view,values=('Original','Equilibrar luz','Mapa de cambios'),
                     state='readonly',width=21).pack(side='left',padx=(8,0))
        self.audit_view.trace_add('write',lambda *_:self.audit_selected())

        self.audit_imglabel=tk.Label(right,bg=PANEL,fg=TXT,text='Selecciona un edificio',font=('Segoe UI',10))
        self.audit_imglabel.pack(fill='both',expand=True)
        self.btn(right,'Exportar informe de auditoría',self.audit_export).pack(anchor='w',pady=(6,0))
        self.make_label(right,'El ajuste de luz y el mapa de cambios solo afectan la vista; no cambian predicciones ni recuperan detalle satelital.',9,MUTED,wrap=350).pack(anchor='w',pady=3)
        self.make_label(right,'Scores no calibrados; el umbral es exploratorio y no puede escogerse con test.',9,MUTED,wrap=350).pack(anchor='w',pady=3)
        self.help_btn(f,'Auditoría')

    def audit_load(self):
        from audit_core import load_validation
        try:
            self.audit_data=load_validation(self.work.get(),self.audit_model.get())
            self.audit_thr.set(self.audit_data['saved_threshold'])
            self.audit_threshold_changed()
            self.log('✓ Validación auditada: '+self.audit_data['model']+' · '+str(len(self.audit_data['rows']))+' edificios.')
        except Exception as e:
            messagebox.showerror('CIMLAB AI - Auditoría',str(e))

    def audit_threshold_changed(self):
        if self.audit_data is None:return
        from audit_core import evaluate,topk
        m=evaluate(self.audit_data['rows'],self.audit_thr.get())
        ranked=topk(self.audit_data['rows'],(50,100))
        self.audit_scores.set('F1 {:0.4f}   Recall {:0.2%}   Precisión {:0.2%}   AP {:0.4f}   TP={} FP={} FN={}\n'.format(
            m['f1'],m['recall'],m['precision'],m['ap'],m['tp'],m['fp'],m['fn'])+
            'Recall@50: {:0.2%}   Recall@100: {:0.2%}   (solo validación)'.format(ranked[0]['recall_at_k'],ranked[1]['recall_at_k']))
        self.audit_refresh_cases()

    def audit_refresh_cases(self):
        if self.audit_data is None:return
        from audit_core import categorized
        self.audit_display,_=categorized(self.audit_data['rows'],self.audit_thr.get(),
                                         self.audit_group.get(),limit=120)
        for row in self.audit_tree.get_children():self.audit_tree.delete(row)
        for i,r in enumerate(self.audit_display):
            self.audit_tree.insert('',tk.END,iid=str(i),values=(
                r['outcome'],f'{r["score"]:.4f}',r['scene'],r['uid'][:12]))
        self.audit_imglabel.configure(image='',text='Selecciona un edificio')
        self.audit_photo=None

    def audit_selected(self,event=None):
        sel=self.audit_tree.selection()
        if not sel or self.audit_data is None:return
        try:
            from audit_core import make_visual
            from PIL import ImageTk
            data=self.audit_display[int(sel[0])]
            mode={'Original':'original','Equilibrar luz':'balanced','Mapa de cambios':'change'}.get(self.audit_view.get(),'original')
            img=make_visual(data,True,124 if mode=='change' else 195,view_mode=mode)
            self.audit_photo=ImageTk.PhotoImage(img)
            self.audit_imglabel.configure(image=self.audit_photo,text='')
        except Exception as e:
            self.audit_imglabel.configure(text='No se pudo abrir el recorte: '+str(e)[:160],image='')

    def audit_export(self):
        if self.audit_data is None:
            messagebox.showinfo('CIMLAB AI','Primero carga la validación de un modelo.');return
        dataset=self.audit_data;threshold=self.audit_thr.get()
        def run():
            from audit_core import export_audit
            out=export_audit(dataset,threshold,self.log)
            self.log('Abre el informe en tu navegador: '+str(out))
        self.submit('Informe de auditoría (solo validación)',run)

    def page_quality(self,f):
        self.header(f,'Control de calidad de imágenes',
            'Analiza recortes y posibles inconsistencias por escena, exclusivamente en train/val.')
        c=self.card(f)
        self.make_label(c,'AUDITORÍA SIN MODIFICAR LOS DATOS',11,ACCENT,True).pack(anchor='w',pady=(0,8))
        self.make_label(c,'Trabaja sobre el manifest.csv de la carpeta indicada en Preparar. Detecta bordes fuera de imagen, zonas negras, cambios visuales extremos, recortes faltantes y caídas de nitidez. Las banderas requieren revisión humana.',10,TXT,wrap=720).pack(anchor='w',pady=5)
        row=tk.Frame(c,bg=PANEL);row.pack(fill='x',pady=(10,8))
        self.make_label(row,'Modelo de validación (si existe):',10,MUTED).pack(side='left')
        from audit_core import MODELS
        ttk.Combobox(row,textvariable=self.quality_model,values=list(MODELS),state='readonly',width=22).pack(side='left',padx=10)
        self.btn(c,'Analizar train/val y exportar informe de calidad',self.quality_run,True).pack(anchor='w',pady=8)
        self.btn(c,'Abrir último informe de calidad',self.quality_open).pack(anchor='w',pady=4)
        tk.Label(c,textvariable=self.quality_status,fg=MUTED,bg=PANEL,
                 font=('Segoe UI',10),justify='left',wraplength=680,anchor='w').pack(fill='x',pady=8)
        c=self.card(f)
        self.make_label(c,'RECORTES · PROCESO SEPARADO',11,ACCENT,True).pack(anchor='w',pady=(0,7))
        self.make_label(c,'Después de revisar el informe, utiliza Preparar para crear un nuevo dataset de daños binarios. NO se eliminan tus recortes, modelos, validaciones ni archivos de GitHub. Los modelos anteriores deben seguir evaluándose con su dataset original.',10,TXT,wrap=710).pack(anchor='w',pady=6)
        self.make_label(c,'Compara las imágenes en 5 Auditoría con tres vistas: Original, Equilibrar luz y Mapa de cambios. Son ayudas visuales, sin alterar el entrenamiento.',10,MUTED,wrap=700).pack(anchor='w',pady=4)
        self.make_label(c,'La auditoría no inspecciona TEST ni corrige etiquetas automáticamente.',10,MUTED).pack(anchor='w',pady=4)
        self.help_btn(f,'Calidad')

    def quality_run(self):
        ws=self.work.get();model=self.quality_model.get()
        if not (Path(ws)/'manifest.csv').is_file():
            messagebox.showerror('CIMLAB AI','Selecciona la carpeta de trabajo con manifest.csv en Preparar.');return
        def do():
            from quality_core import audit_quality
            report,summary=audit_quality(ws,model=model,log=self.log)
            self.q.put(('quality_summary',(str(report),summary['inspected'],summary['flagged'])))
        self.submit('Control de calidad train/val',do)

    def quality_open(self):
        report=Path(self.work.get())/'calidad_v03'/'informe_calidad.html'
        if not report.is_file():
            messagebox.showinfo('CIMLAB AI','Todavía no hay informe. Pulsa Analizar train/val.');return
        import webbrowser
        webbrowser.open(report.resolve().as_uri())

    def page_predict(self,f):
        self.header(f,'Predicción · daños visibles','Requiere PRE/POST de la misma escena y un JSON de polígonos.')
        c=self.card(f)
        self.line(c,'Imagen PRE (.tif, .png)',self.pre,True,False)
        self.line(c,'Imagen POST (.tif, .png)',self.post,True,False)
        self.line(c,'Polígonos JSON (.json) de la escena',self.json,True,False)
        self.btn(c,'Predecir y exportar mapa coloreado + CSV',self.predict,True).pack(anchor='w',pady=15)
        self.make_label(c,'Las etiquetas del JSON no se utilizan al predecir. El modelo estima DAÑO visible dentro de polígonos existentes; no detecta edificios nuevos y no reemplaza inspecciones.',10,MUTED,wrap=710).pack(anchor='w')
        self.help_btn(f,'Predecir')

    def log(self,msg):
        self.q.put(('log',str(msg)))

    def poll(self):
        try:
            while True:
                kind,data=self.q.get_nowait()
                if kind=='log':
                    self.logbox.configure(state='normal');self.logbox.insert('end',data+'\n');self.logbox.see('end');self.logbox.configure(state='disabled')
                elif kind=='done':
                    self.working=False;self.status.set('✓ Terminado: '+data)
                    self.refresh_comparison()
                elif kind=='events':
                    names,mode,path=data
                    self.event_names=names
                    self.event_catalog_source=(mode,path)
                    self.event_list.delete(0,tk.END)
                    for name in names:self.event_list.insert(tk.END,name)
                    self.event_status.set(f'{len(names)} evento(s) en {mode}. Selecciona uno o varios. El catálogo puede cambiar: vuelve a actualizar cuando publiquen más datos.')
                elif kind=='source_audit':
                    self.latest_source_audit=data
                    unknown=data.get('incompatible_labels',{})
                    if unknown:
                        desc='; '.join(f'{k}: {v} edificios' for k,v in unknown.items())
                        messagebox.showwarning('ETIQUETAS INCOMPATIBLES · revisar antes de preparar',
                            'Se detectaron categorías ajenas a las etiquetas reconocidas para DAÑADO / NO DAÑADO:\n\n'+desc+
                            '\n\nNo serán incorporadas silenciosamente. Pulsa Revisar equivalencias de etiquetas y verifica su significado con la fuente.')
                    else:
                        self.log('✓ Etiquetas reconocidas. Edificios dañados: '+str(data.get('damaged',0)))
                elif kind=='device':
                    self.device_status.set(data)
                elif kind=='quality_summary':
                    path,n,k=data
                    self.quality_status.set(f'✓ Analizados {n:,} edificios de train/val. {k:,} con una o más alertas.\nInforme: {path}')
                elif kind=='error':
                    self.working=False;self.status.set('Error: '+data)
                    messagebox.showerror('CIMLAB AI',data)
        except queue.Empty:pass
        self.root.after(120,self.poll)

    def submit(self,label,fn):
        if self.working:
            messagebox.showinfo('Proceso en curso','Hay una operación en ejecución. Revisa el registro inferior.');return
        self.working=True;self.status.set('En proceso: '+label)
        def worker():
            try:
                fn();self.q.put(('done',label))
            except Exception as e:
                self.log('ERROR: '+str(e));self.q.put(('error',str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def download(self):
        if self.source_type.get()!='GitHub':
            messagebox.showinfo('Fuente local','En modo local no es necesario descargar. Selecciona eventos y pulsa Auditar.');return
        try:names=self.selected_events()
        except Exception as e:messagebox.showwarning('Eventos',str(e));return
        p=Path(self.path.get())
        self.submit('Descarga GitHub / Git LFS',lambda:clone_lfs(p,names,log=self.log))

    def audit(self):
        try:names=self.selected_events()
        except Exception as e:messagebox.showwarning('Eventos',str(e));return
        def do():
            rows,info=scan(self.path.get(),self.log,events=names)
            self.q.put(('source_audit',info))
        self.submit('Auditoría de fuentes y compatibilidad',do)

    def review_labels(self):
        """Manual, per-event mapping for otherwise incompatible label taxonomies."""
        try:
            events=self.selected_events()
            roots=selected_roots(self.path.get(),events)
            info=self.latest_source_audit
            if info is None or sorted(info.get('events',[]))!=sorted(events):
                messagebox.showwarning('Revisar etiquetas',
                    'Primero pulsa «Auditar TIFF + JSON + etiquetas» con los eventos seleccionados.')
                return
        except Exception as e:
            messagebox.showerror('Revisar etiquetas',str(e));return
        pending=[]
        from labels_core import event_mapping, normalize
        for rep in info['event_reports']:
            root=next(r for r in roots if r.name==rep['event'])
            for raw,count in rep.get('incompatible_labels',{}).items():
                pending.append((root,raw,count))
        if not pending:
            messagebox.showinfo('Etiquetas','No hay etiquetas pendientes. Se utilizan las cuatro clases xBD conocidas y equivalencias inequívocas.');return
        if len(pending)>25:
            messagebox.showerror('Etiquetas','Hay más de 25 etiquetas distintas. Revisa primero las anotaciones o el esquema de datos.');return
        dlg=tk.Toplevel(self.root);dlg.title('CIMLAB AI · Compatibilidad de etiquetas por evento')
        dlg.geometry('800x590');dlg.minsize(700,480);dlg.configure(bg=BG)
        self.make_label(dlg,'REVISIÓN MANUAL OBLIGATORIA',15,ACCENT,True).pack(anchor='w',padx=18,pady=(15,5))
        self.make_label(dlg,'Indica qué significa CADA etiqueta en la documentación ORIGINAL. '
          '1 = edificio con daño menor, mayor o destruido; 0 = edificio sin daño; excluir = no usar esa anotación. '
          'Si alto/bajo representa altura, riesgo u otra variable, NO lo conviertas en daño.',10,TXT,wrap=740).pack(anchor='w',padx=18,pady=(0,12))
        container=tk.Frame(dlg,bg=PANEL);container.pack(fill='both',expand=True,padx=18)
        canvas=tk.Canvas(container,bg=PANEL,highlightthickness=0)
        bar=ttk.Scrollbar(container,orient='vertical',command=canvas.yview)
        inside=tk.Frame(canvas,bg=PANEL)
        inside.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0),window=inside,anchor='nw')
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        selections=[]
        values=['Sin definir','0 · No dañado','1 · Dañado','Excluir anotación']
        for root,raw,count in pending:
            row=tk.Frame(inside,bg=PANEL);row.pack(fill='x',padx=12,pady=7)
            self.make_label(row,f'{root.name} / {raw!r} ({count})',10,TXT,wrap=400).pack(side='left',fill='x',expand=True)
            v=tk.StringVar(value='Sin definir')
            ttk.Combobox(row,textvariable=v,values=values,state='readonly',width=19).pack(side='right',padx=9)
            selections.append((root,raw,v))
        confirmed=tk.BooleanVar(value=False)
        tk.Checkbutton(dlg,text='Confirmo que las asignaciones 0/1 describen daños visibles en edificios y que las excluidas quedan fuera del análisis.',
            variable=confirmed,bg=BG,fg=TXT,selectcolor=CARD,activebackground=BG,activeforeground=TXT,
            font=('Segoe UI',9),wraplength=730,justify='left').pack(anchor='w',padx=18,pady=10)
        def save():
            if not confirmed.get():
                messagebox.showwarning('Confirmación','Antes de guardar, confirma que la variable original realmente mide daño visible en edificios.',parent=dlg);return
            grouped={}
            for root,raw,v in selections:
                value=v.get()
                if value=='Sin definir':
                    messagebox.showwarning('Revisión incompleta',f'Falta definir {root.name} / {raw}.',parent=dlg);return
                grouped.setdefault(root,{})[raw]= 0 if value.startswith('0 ') else 1 if value.startswith('1 ') else -1
            if not messagebox.askyesno('Confirmar equivalencias','Las correspondencias son una decisión metodológica del equipo. ¿Guardar las asignaciones revisadas?',parent=dlg):return
            try:
                for root,mapping in grouped.items():
                    saved=save_event_mapping(root,mapping)
                    self.log('✓ Equivalencias guardadas: '+str(saved))
                dlg.destroy()
                self.log('Vuelve a pulsar Auditar para comprobar compatibilidad.')
            except Exception as e:messagebox.showerror('Error',str(e),parent=dlg)
        row=tk.Frame(dlg,bg=BG);row.pack(anchor='e',padx=18,pady=(0,14))
        self.btn(row,'Guardar correspondencias',save,True).pack(side='left',padx=8)
        self.btn(row,'Cancelar',dlg.destroy).pack(side='left')

    def prepare(self):
        try:events=self.selected_events()
        except Exception as e:messagebox.showwarning('Eventos',str(e));return
        p=self.path.get();w=self.work.get()
        try:neg=self.neg.get()
        except tk.TclError:
            messagebox.showerror('CIMLAB AI','Indica un número válido de no dañados.');return
        if neg<0:
            messagebox.showerror('CIMLAB AI','Utiliza 0 (todos) o un máximo positivo.');return
        old=Path(w)
        if (old/'manifest.csv').exists():
            dest=old.with_name(old.name+'_dano_binario_v1')
            if dest.exists():
                messagebox.showwarning('Protección de datos','Ya existe '+str(dest)+'. Selecciona otra carpeta NUEVA y vacía.');return
            if not messagebox.askyesno('Nuevo objetivo binario',
                'La carpeta seleccionada ya contiene un dataset. Para evitar sobrescribirlo, '
                'se creará otro trabajo en:\n'+str(dest)+'\n\n¿Continuar?'):
                return
            self.work.set(str(dest));w=str(dest)
        if Path(w).exists() and any(Path(w).iterdir()):
            messagebox.showwarning('Carpeta en uso','Selecciona una carpeta de trabajo nueva y VACÍA. No se sobrescribirán modelos.');return
        if not messagebox.askyesno('Preparar dataset conjunto',
            'Eventos seleccionados: '+', '.join(events)+'\nOBJETIVO: DAÑADO / NO DAÑADO\n'
            'Destino: '+str(w)+'\n\nAntes de extraer recortes se verificará la compatibilidad de etiquetas. ¿Continuar?'):
            return
        self.submit('Preparación de daño binario',
                    lambda:prepare(p,w,max_train_neg=neg,log=self.log,events=events))

    def train(self):
        models=[k for k,v in self.model_vars.items() if v.get()]
        if not models:
            messagebox.showwarning('CIMLAB AI','Selecciona al menos un modelo.');return
        ep=self.epochs.get();batch=self.batch.get();work=self.work.get()
        def do():
            from models_core import train_selected
            train_selected(work,models,ep,batch,log=self.log)
        self.submit('Entrenamiento de '+str(len(models))+' modelos',do)

    def report(self):
        def do():
            from report_core import make_report
            import webbrowser
            report=make_report(self.work.get(),self.log)
            webbrowser.open(report.resolve().as_uri())
        self.submit('Informe y gráficos',do)

    def evaluate(self):
        if messagebox.askyesno('Evaluación final','El conjunto de prueba se reserva para evaluar el ganador. ¿Confirmas que terminaste de comparar modelos en validación?'):
            def do():
                from models_core import test_winner
                path=Path(self.work.get())/'resultados/evaluacion_final.json'
                if path.exists():
                    raise RuntimeError('La prueba final ya se ejecutó. Para preservar su independencia, no la repetimos.')
                test_winner(self.work.get(),self.log)
            self.submit('Evaluación final',do)

    def refresh_comparison(self):
        if not hasattr(self,'table'):return
        path=Path(self.work.get())/'resultados'
        lines=['OBJETIVO: DAÑADO (1) vs NO DAÑADO (0)', 'MODELO                   AP-val     F1-val     Recall    Precision',
               '─'*72]
        try:
            require_task(self.work.get())
        except Exception:
            lines=['DATOS ANTIGUOS / OBJETIVO DIFERENTE',
                   'Esta carpeta no tiene metadatos compatibles con el experimento binario de DAÑO.',
                   'Utiliza Preparar para crear una carpeta NUEVA; no se comparan métricas anteriores.']
            self.table.configure(state='normal');self.table.delete('1.0','end');self.table.insert('1.0','\n'.join(lines));self.table.configure(state='disabled')
            return
        for model in ('random_forest','dinov2_xgboost','resnet18_early','siamese_resnet18',
                      'siamese_unet_lite','changeos_lite','dahitra_lite','protonet_dino','hybrid'):
            fp=path/(model+'_val.json')
            if fp.exists():
                try:
                    m=json.loads(fp.read_text(encoding='utf-8'))['validation']
                    vals=[m.get('average_precision'),m.get('f1_damage'),m.get('recall_damage'),m.get('precision_damage')]
                    line=f'{model:24} '+''.join(f'{v:>10.4f}' if isinstance(v,(int,float)) else '  n/d     ' for v in vals)
                except Exception:line=f'{model:24} ARCHIVO NO VÁLIDO'
            else:line=f'{model:24} Pendiente'
            lines.append(line)
        result=path/'evaluacion_final.json'
        if result.exists():
            r=json.loads(result.read_text(encoding='utf-8'))
            lines+=['','EVALUACIÓN FINAL: '+str(r['selected_model']),json.dumps(r['test'],ensure_ascii=False,indent=2)]
        else:lines+=['','PRUEBA FINAL: aún no realizada']
        self.table.configure(state='normal');self.table.delete('1.0','end');self.table.insert('1.0','\n'.join(lines));self.table.configure(state='disabled')

    def predict(self):
        p,q,j=self.pre.get(),self.post.get(),self.json.get();work=self.work.get()
        if not all([p,q,j]):messagebox.showwarning('CIMLAB AI','Selecciona PRE, POST y JSON.');return
        def do():
            from predict_core import predict_scene
            predict_scene(work,p,q,j,log=self.log)
        self.submit('Predicción',do)

    def on_close(self):
        if self.working and not messagebox.askyesno('CIMLAB AI','Hay una tarea en curso. ¿Cerrar igualmente?'):return
        CONFIG.write_text(json.dumps({'repository':self.path.get(),'workspace':self.work.get(),
            'source_type':self.source_type.get()},ensure_ascii=False,indent=2),encoding='utf-8')
        self.root.destroy()


def main():
    r=tk.Tk(); CIMLABApp(r);r.mainloop()

if __name__=='__main__':main()
