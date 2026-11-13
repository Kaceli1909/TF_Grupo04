"""CIMLAB AI - dataset indexing, crop extraction, scene-disjoint split, feature extraction.
v0.5.1 DIAGNOSTICO RECORTES
"""
from __future__ import annotations
import csv
import json
import math
import random
import subprocess
import os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from labels_core import TASK_ID, TASK_NAME, classify, event_mapping, IncompatibleLabelsError
from tiff_support import open_rgb, describe  # CIMLAB TIFF v0.5.2

REPO = 'https://github.com/Kaceli1909/TF_Grupo04.git'
CLASSES = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 1, 'destroyed': 1}
CLASS3 = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}
FIELDS = ['scene','uid','label','label3','original_label','split','pre','post','wkt','x0','y0','x1','y1','crop_pre','crop_post']


def logfn(log, msg):
    if log:
        log(str(msg))


def run_cmd(args, cwd=None, env=None, log=None):
    logfn(log, '▶ ' + ' '.join(map(str, args)))
    p = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in p.stdout:
        logfn(log, line.rstrip())
    rc = p.wait()
    if rc:
        raise RuntimeError(f'Comando falló (código {rc}): {" ".join(args)}')


def clone_lfs(dst: Path, events=None, log=None):
    """Clona/actualiza TF_Grupo04 y descarga via Git LFS solo los eventos elegidos.

    `events` must be selected folder names from GitHub data/. If omitted, only
    metadata is updated (no large TIFF are downloaded by surprise).
    """
    from shutil import which
    from event_sources import valid_event
    events=list(dict.fromkeys(events or []))
    if any(not valid_event(e) for e in events):
        raise ValueError('Nombre de evento de GitHub inválido.')
    if not which('git'):
        raise RuntimeError('No se encontró Git. Instálalo: https://git-scm.com/download/win')
    p = subprocess.run(['git','lfs','version'],capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError('No se encontró Git LFS. Instálalo: https://git-lfs.com/')
    dst = Path(dst).expanduser().resolve()
    if not (dst / '.git').exists():
        if dst.exists() and any(dst.iterdir()):
            raise RuntimeError(f'La carpeta {dst} ya contiene archivos y no es un repositorio Git. Escoge una carpeta nueva o una clonación existente.')
        dst.parent.mkdir(parents=True,exist_ok=True)
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
        run_cmd(['git','clone','--progress',REPO,str(dst)],env=env,log=log)
    else:
        env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        run_cmd(['git','pull','--ff-only'],cwd=dst,env=env,log=log)
    if not events:
        logfn(log,'✓ Se actualizaron archivos del repositorio. Selecciona uno o más eventos para descargar imágenes LFS.')
        return dst
    for event in events:
        event_path=dst/'data'/event
        if not event_path.is_dir():
            raise FileNotFoundError(f'El evento {event} no existe en la rama actual de GitHub (data/{event}).')
        logfn(log,f'Descargando datos Git LFS de {event}...')
        run_cmd(['git','lfs','pull','--include='+f'data/{event}/**','--exclude='],cwd=dst,log=log)
        tifs=[p for p in event_path.rglob('*') if p.suffix.lower() in ('.tif','.tiff')]
        if not tifs:
            logfn(log,f'AVISO {event}: no se localizaron TIFF. Revisa la carpeta del evento.')
            continue
        pointers=[p for p in tifs if is_lfs_pointer(p)]
        if pointers:
            raise RuntimeError(f'{event}: {len(pointers)} TIFF son punteros LFS y no imágenes. Verifica cuota/permisos del repositorio.')
        # Verificación real del formato. El conteo y la ausencia de puntero LFS no bastan.
        sample=tifs[:min(3,len(tifs))]
        for sample_path in sample:
            try:
                check=open_rgb(sample_path)
                check.close()
            except Exception as exc:
                raise RuntimeError(f'{event}: archivo no decodificable {sample_path}. {exc}') from exc
        logfn(log,f'✓ {event}: {len(tifs)} archivos TIFF presentes; {len(sample)} muestras abiertas y decodificadas.')
    return dst


def is_lfs_pointer(path: Path) -> bool:
    path=Path(path)
    if not path.is_file():return True
    if path.stat().st_size < 256:
        return path.read_bytes()[:60].startswith(b'version https://git-lfs.github.com/spec/v1')
    return False


def find_root(path: Path) -> Path:
    """Compatibility helper for callers using one event folder."""
    from event_sources import selected_roots
    roots=selected_roots(path)
    if len(roots)!=1:
        raise ValueError('Se detectaron varios eventos. Selecciona uno o utiliza scan/prepare con la lista events.')
    return roots[0]


def _scene_token(root,name):
    """Preserve xBD's original scene identifier; namespace generic local filenames."""
    if name.startswith(root.name+'_') or name.startswith(root.name+'-'):
        return name
    return root.name+'__'+name


def image_index(root):
    """Look for scene pair filenames anywhere inside one event, not just images/."""
    root=Path(root)
    paths={}; pref={'.tif':4,'.tiff':4,'.png':3,'.jpg':2,'.jpeg':2}
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in pref: continue
        for typ in ('pre','post'):
            suffix=f'_{typ}_disaster{p.suffix}'
            if p.name.lower().endswith(suffix.lower()):
                stem=p.name[:-len(suffix)]
                scene=_scene_token(root,stem)
                key=(scene,typ)
                if key not in paths or pref[p.suffix.lower()] > pref[paths[key].suffix.lower()]:
                    paths[key]=p
                break
    out={}
    for (scene,typ),path in paths.items():
        out.setdefault(scene,{})[typ]=path
    return out


def json_index(root):
    """Labels in labels/, tier1/labels/, or any nested structure are accepted."""
    root=Path(root);out={}
    for p in root.rglob('*_post_disaster.json'):
        scene=_scene_token(root,p.name[:-len('_post_disaster.json')])
        if scene not in out or ('/tier1/' in p.as_posix() and '/tier1/' not in out[scene].as_posix()):
            out[scene]=p
    return out


def polygon_from_wkt(wkt):
    """Read xy polygon (pixel-coordinate) of xBD, no shapely dependency."""
    from shapely import wkt as shp_wkt
    poly=shp_wkt.loads(wkt)
    if poly.is_empty or not poly.is_valid: raise ValueError('Geometría inválida')
    if poly.geom_type not in ('Polygon','MultiPolygon'): raise ValueError('Geometría no poligonal')
    return poly


def _scan_one(root,log=None):
    root=Path(root)
    imgs=image_index(root)
    labels=json_index(root)
    rows=[]; invalid=0; excluded=Counter(); incompatible=Counter(); originals=Counter(); issues=[]
    user_map=event_mapping(root)
    for scene,json_path in sorted(labels.items()):
        pair=imgs.get(scene,{})
        if 'pre' not in pair or 'post' not in pair:
            issues.append(f'{scene}: faltan imágenes PRE/POST')
            continue
        if is_lfs_pointer(pair['pre']) or is_lfs_pointer(pair['post']):
            issues.append(f'{scene}: imagen es un puntero Git LFS, faltan píxeles')
            continue
        try:
            data=json.loads(json_path.read_text(encoding='utf-8'))
            feats=data.get('features',{}).get('xy',[])
        except Exception as e:
            issues.append(f'{scene}: error al abrir JSON ({e})');continue
        if not isinstance(feats,list):
            issues.append(f'{scene}: JSON sin lista features.xy compatible con xBD');continue
        for feat in feats:
            props=feat.get('properties') or {}
            label=str(props.get('subtype','un-classified'))
            originals[label]+=1
            target=classify(label,user_map)
            if target=='INCOMPATIBLE':
                incompatible[label]+=1;continue
            if target is None or target==-1:
                excluded[label]+=1;continue
            try:
                p=polygon_from_wkt(feat['wkt'])
                x0,y0,x1,y1=p.bounds
            except Exception:
                invalid+=1;continue
            uid=str(props.get('uid') or '')
            if not uid:
                invalid+=1;continue
            rows.append(dict(scene=scene,uid=uid,label=target,label3=CLASS3.get(label,-1),
                             original_label=label,split='',pre=str(pair['pre']),post=str(pair['post']),
                             wkt=feat['wkt'],x0=x0,y0=y0,x1=x1,y1=y1,crop_pre='',crop_post=''))
    return rows,dict(event=root.name,scenes_json=len(labels),scenes_tiff=len(imgs),
        scenes_with_labels=len({r['scene'] for r in rows}),
        pairs_tiff=sum('pre' in x and 'post' in x for x in imgs.values()),
        buildings=len(rows),damaged=sum(r['label'] for r in rows),
        invalid_polygons=invalid,excluded=dict(excluded),
        incompatible_labels=dict(incompatible),original_labels=dict(originals),issues=issues)


def scan(root,log=None,events=None,strict=False):
    """Audit single/multiple xBD event folders; no assumed disaster or event name."""
    from event_sources import selected_roots
    roots=selected_roots(root,events)
    rows=[];reports=[]
    for r in roots:
        group,rep=_scan_one(r,log)
        rows.extend(group);reports.append(rep)
        logfn(log,f'Evento {r.name}: {rep["pairs_tiff"]} pares; {rep["buildings"]} edificios; {rep["damaged"]} DAÑADOS.')
        if rep['incompatible_labels']:
            logfn(log,'⚠ ETIQUETAS INCOMPATIBLES en '+r.name+': '+json.dumps(rep['incompatible_labels'],ensure_ascii=False)+'. Revisar antes de preparar.')
        for issue in rep['issues'][:12]:logfn(log,'AVISO: '+issue)
    info=dict(events=[r.name for r in roots],event_reports=reports,
        scenes_json=sum(r['scenes_json'] for r in reports),
        scenes_tiff=sum(r['scenes_tiff'] for r in reports),
        scenes_with_labels=sum(r['scenes_with_labels'] for r in reports),
        pairs_tiff=sum(r['pairs_tiff'] for r in reports),
        buildings=len(rows),damaged=sum(r['label'] for r in rows),
        invalid_polygons=sum(r['invalid_polygons'] for r in reports),
        excluded=dict(sum((Counter(r['excluded']) for r in reports),Counter())),
        incompatible_labels=dict(sum((Counter(r['incompatible_labels']) for r in reports),Counter())),
        original_labels=dict(sum((Counter(r['original_labels']) for r in reports),Counter())),
        task=TASK_ID, positive='dañado', negative='no dañado',
        issues=[issue for report in reports for issue in report['issues']])
    logfn(log,'AUDITORÍA: '+json.dumps({k:v for k,v in info.items() if k not in ('issues','event_reports')},ensure_ascii=False))
    if strict and info['incompatible_labels']:
        details='; '.join(x['event']+': '+str(x['incompatible_labels']) for x in reports if x['incompatible_labels'])
        raise IncompatibleLabelsError('No se preparó el dataset. Se encontraron etiquetas de otra escala o sin equivalencia: '+details+
          '. Ve a Datos > Revisar etiquetas y confirma manualmente la correspondencia. No se presuponen significados como alto/bajo o severo.')
    return rows,info


def allocate_splits(rows, seed=42):
    """CIMLAB AI v0.5.4 - respetar la particion del origen.

    xBD: tier1 -> train, hold -> val, test -> test. No redistribuir escenas.
    Datos locales SIN esas carpetas: particion alternativa por escena con semilla.
    Mezclas ambiguas: detener y solicitar organizacion explicita.
    """
    from collections import defaultdict
    original={"tier1":"train", "hold":"val", "test":"test"}
    def lookup(path):
        names=set(str(path).replace("\\", "/").lower().split("/"))
        found={v for k,v in original.items() if k in names}
        if len(found)>1:raise RuntimeError("Ruta con particiones ambiguas: " + str(path))
        return next(iter(found)) if found else None
    found=[]
    for row in rows:
        a=lookup(row['pre']);b=lookup(row['post'])
        if a!=b:raise RuntimeError('PRE y POST tienen particiones diferentes: ' + row['scene'])
        found.append(a)
    if any(found):
        if not all(found):
            raise RuntimeError('Hay escenas con particion xBD y otras sin ella. Organiza el origen antes de preparar.')
        selected={'train':set(),'val':set(),'test':set()}
        scene_splits={}
        for row, split in zip(rows,found):
            scene=row['scene']
            if scene in scene_splits and scene_splits[scene]!=split:
                raise RuntimeError('Escena presente en varios splits originales: ' + scene)
            scene_splits[scene]=split
            row['split']=split
            selected[split].add(scene)
        return selected
    # Alternativa para datasets locales sin carpetas oficiales xBD.
    grouped=defaultdict(list)
    for row in rows:grouped[row['scene']].append(row)
    classes=defaultdict(list)
    for scene,group in grouped.items():
        classes[int(any(r['label'] for r in group))].append(scene)
    rng=random.Random(seed)
    selected={'train':set(),'val':set(),'test':set()}
    for has_damage,scenes in sorted(classes.items()):
        rng.shuffle(scenes)
        n=len(scenes)
        nt=int(round(n*.15));nv=int(round(n*.15))
        if n>=10:nt=max(1,nt);nv=max(1,nv)
        selected['test'].update(scenes[:nt]);selected['val'].update(scenes[nt:nt+nv])
        selected['train'].update(scenes[nt+nv:])
    for s,ss in selected.items():
        for row in rows:
            if row['scene'] in ss:row['split']=s
    return selected


def save_rows(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows([{k:r.get(k,'') for k in FIELDS} for r in rows])


def load_rows(path):
    with Path(path).open('r',newline='',encoding='utf-8-sig') as f:
        return [{**r,'label':int(r['label']),'label3':int(r['label3']),
                 'x0':float(r['x0']),'y0':float(r['y0']),'x1':float(r['x1']),'y1':float(r['y1'])} for r in csv.DictReader(f)]


def crop_window(bounds,pad=12):
    """Ventana cuadrada compartida, expresada en píxeles de la imagen original."""
    x0,y0,x1,y1=bounds
    if not all(math.isfinite(float(v)) for v in bounds):
        raise ValueError('Coordenadas de polígono no finitas')
    if x1<=x0 or y1<=y0:
        raise ValueError('Caja de polígono vacía o invertida')
    cx=(x0+x1)/2;cy=(y0+y1)/2
    side=max((x1-x0)+2*pad,(y1-y0)+2*pad,32)
    left=int(math.floor(cx-side/2));top=int(math.floor(cy-side/2))
    right=int(math.ceil(cx+side/2));bottom=int(math.ceil(cy+side/2))
    return left,top,right,bottom


def _reflect_crop(img,box,size):
    """Extrae la ventana sin introducir bordes NEGROS sintéticos.

    El relleno reflejado solo afecta a píxeles FUERA de la imagen; no agrega
    información real. La geometría de los polígonos continúa sin desplazarse.
    """
    rgb=img.convert('RGB');left,top,right,bottom=box
    w,h=rgb.size
    if w<=0 or h<=0:raise ValueError('Imagen vacía')
    if not (0 <= (left+right)/2 < w and 0 <= (top+bottom)/2 < h):
        raise ValueError('Centro del recorte fuera de la imagen: revisar polígono')
    outw=right-left;outh=bottom-top
    if outw<=0 or outh<=0:raise ValueError('Ventana inválida')
    # Recortes fuera casi por completo son datos que deben revisarse, no rellenarse.
    overlap=max(0,min(right,w)-max(left,0))*max(0,min(bottom,h)-max(top,0))
    if overlap / (outw*outh)<.20:
        raise ValueError('Recorte mayoritariamente fuera de la imagen: revisar polígono')
    if left>=0 and top>=0 and right<=w and bottom<=h:
        return rgb.crop(box).resize((size,size),Image.Resampling.BILINEAR)
    # Reflejar bordes sin materializar toda la imagen extendida: indexar solo ventana.
    def reflect_index(x,length):
        if length==1:return np.zeros_like(x,dtype=np.int64)
        period=2*length-2
        z=np.mod(x,period)
        return np.where(z<length,z,period-z).astype(np.int64)
    xi=reflect_index(np.arange(left,right,dtype=np.int64),w)
    yi=reflect_index(np.arange(top,bottom,dtype=np.int64),h)
    a=np.asarray(rgb)
    result=Image.fromarray(a[np.ix_(yi,xi)],mode='RGB')
    return result.resize((size,size),Image.Resampling.BILINEAR)


def crop_pair(pre_img,post_img,poly,pad=12,size=224):
    """Misma caja PRE/POST; relleno reflejado en bordes (v0.3)."""
    if pre_img.size!=post_img.size:raise ValueError('Dimensiones PRE/POST distintas')
    box=crop_window(poly.bounds,pad)
    return _reflect_crop(pre_img,box,size),_reflect_crop(post_img,box,size)


def load_pair_rgb(pre_path,post_path):
    """Lectura segura de ambas imágenes, usada también en predicción."""
    return open_rgb(pre_path),open_rgb(post_path)


def prepare(root,workspace,max_train_neg=0,seed=42,limit_scenes=None,log=None,events=None):
    workspace=Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    rows,info=scan(root,log,events=events,strict=True)
    if not rows: raise RuntimeError('No hay pares de imágenes reales + JSON utilizables. Descarga Git LFS primero.')
    if sum(int(r['label']) for r in rows)<4 or len(set(r['scene'] for r in rows if r['label']==1))<4:
        raise RuntimeError('Insuficientes edificios/escenas con etiqueta DAÑADO para separar train/val/test con positivos. '
                           'Agrega más eventos con daños confirmados y etiquetas xBD compatibles.')
    if limit_scenes:
        keep=set(sorted({r['scene'] for r in rows})[:limit_scenes]);rows=[r for r in rows if r['scene'] in keep]
    selected=allocate_splits(rows,seed)
    for part in ('train','val','test'):
        if not any(r['split']==part and r['label']==1 for r in rows):
            raise RuntimeError(f'La partición {part} quedó sin edificios dañados. Agrega más escenas con daño.')
    if not selected['train'] or not selected['val'] or not selected['test']:
        raise RuntimeError('Hay muy pocas escenas para crear divisiones train/val/test.')
    # Preserve all damaged buildings; all events included. 0 = all no-damage buildings.
    if max_train_neg is None or max_train_neg==0:
        max_train_neg=len([r for r in rows if r['split']=='train' and r['label']==0])
    if not isinstance(max_train_neg,int) or max_train_neg<0:
        raise ValueError('max_train_neg debe ser 0 (todos) o entero positivo.')
    train_negative=[r for r in rows if r['split']=='train' and r['label']==0]
    rng=random.Random(seed);rng.shuffle(train_negative)
    n_drop=max(0,len(train_negative)-max_train_neg)
    drop={(r['scene'],r['uid']) for r in train_negative[max_train_neg:]}
    kept=[r for r in rows if (r['scene'],r['uid']) not in drop]
    logfn(log,f'Escenas {len(selected["train"])} entrenamiento / {len(selected["val"])} validación / {len(selected["test"])} prueba.')
    logfn(log,f'Entrenamiento: se seleccionan {min(len(train_negative),max_train_neg)} negativos; val/test completos.')
    groups=defaultdict(list)
    for r in kept:groups[r['scene']].append(r)
    failed=[]
    for i,(scene,group) in enumerate(sorted(groups.items()),1):
        try:
            pre=open_rgb(group[0]['pre']);post=open_rgb(group[0]['post'])
            if pre.size != post.size:
                failed.append(f'{scene}: dimensiones PRE/POST distintas');continue
            for row in group:
                try:
                    p=polygon_from_wkt(row['wkt'])
                    a,b=crop_pair(pre,post,p)
                    stem=f'{row["uid"].replace("/","_")}'
                    folder=workspace/'crops'/row['split']/scene
                    folder.mkdir(parents=True,exist_ok=True)
                    af=folder/(stem+'_pre.jpg'); bf=folder/(stem+'_post.jpg')
                    a.save(af,quality=89);b.save(bf,quality=89)
                    row['crop_pre']=str(af);row['crop_post']=str(bf)
                except Exception as e:
                    error=f'{scene}/{row["uid"]}: {type(e).__name__}: {e}'
                    failed.append(error)
                    if len(failed)<=6:
                        logfn(log,'ERROR RECORTE: '+error)
        except Exception as e:
            error=f'{scene}: {type(e).__name__}: {e}'
            failed.append(error)
            if len(failed)<=6:
                logfn(log,'ERROR LECTURA TIFF: '+error)
        if i%10==0 or i==len(groups):logfn(log,f'Recortes: {i}/{len(groups)} escenas.')
        if i==min(12,len(groups)) and not any(r['crop_pre'] and r['crop_post'] for r in kept):
            failure_path=workspace/'ERRORES_PREPARACION_v051.txt'
            failure_path.write_text('\n'.join(failed[:200]),encoding='utf-8')
            example=failed[0] if failed else 'No se pudo determinar la excepción'
            raise RuntimeError('Se detuvo la preparación: ninguna de las primeras '+str(i)+
                ' escenas produjo recortes. Primera causa: '+example+
                '. Informe: '+str(failure_path))
    kept=[r for r in kept if r['crop_pre'] and r['crop_post']]
    if not kept:
        failure_path=workspace/'ERRORES_PREPARACION_v051.txt'
        failure_path.write_text('\n'.join(failed[:200]),encoding='utf-8')
        raise RuntimeError('No se pudo crear ningún recorte. Primer error: '+
            (failed[0] if failed else 'No se registró detalle')+'; informe: '+str(failure_path))
    out=workspace/'manifest.csv';save_rows(out,kept)
    counts={s:dict(total=sum(r['split']==s for r in kept),damaged=sum(r['split']==s and r['label'] for r in kept)) for s in ('train','val','test')}
    report={**info,'task':TASK_ID,'task_name':TASK_NAME,'scenes_split':{s:len(ss) for s,ss in selected.items()},'building_split':counts,
            'train_negatives_dropped':n_drop,'prepared':len(kept),'crop_errors':failed[:100],
            'manifest':str(out)}
    (workspace/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (workspace/'dataset_meta.json').write_text(json.dumps({
      'task':TASK_ID,'task_name':TASK_NAME,'events':info['events'],
      'positive_class':1,'positive_name':'dañado',
      'negative_class':0,'negative_name':'no dañado',
      'original_labels':info['original_labels'],'building_split':counts,
      'note':'Mismo objetivo semántico que modelos de daño previos; no reutilizar resultados sin verificar splits, recortes y metadatos.'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    logfn(log,'✓ Manifiesto y recortes listos: '+str(out))
    return out,report


def handcrafted_features(pre_path,post_path):
    """Visual change features + appearance summaries; never use label or uid."""
    with Image.open(pre_path) as a,Image.open(post_path) as b:
        aa=np.asarray(a.convert('RGB').resize((64,64)),dtype=np.float32)/255.
        bb=np.asarray(b.convert('RGB').resize((64,64)),dtype=np.float32)/255.
    diff=bb-aa;absd=np.abs(diff)
    def gray(x):return .299*x[:,:,0]+.587*x[:,:,1]+.114*x[:,:,2]
    ga=gray(aa);gb=gray(bb)
    gy0,gx0=np.gradient(ga);gy1,gx1=np.gradient(gb)
    e0=np.hypot(gx0,gy0);e1=np.hypot(gx1,gy1)
    scores=[]
    for im in (aa,bb,diff,absd):
        for c in range(3):
            a=im[:,:,c];scores.extend([float(a.mean()),float(a.std()),float(np.quantile(a,.1)),float(np.quantile(a,.9))])
    scores += [float(absd.mean()),float((absd>.2).mean()),float((absd>.4).mean()),
               float(np.mean(np.abs(ga-gb))),float(e0.mean()),float(e1.mean()),
               float(np.abs(e1-e0).mean()),float(np.corrcoef(ga.ravel(),gb.ravel())[0,1]) if ga.std()>1e-6 and gb.std()>1e-6 else 0.]
    return np.nan_to_num(np.asarray(scores,dtype=np.float32),nan=0.,posinf=0.,neginf=0.)


def extract_feature_table(rows,log=None):
    x=[];y=[];ids=[];skipped=0
    for i,r in enumerate(rows,1):
        try:
            feat=handcrafted_features(r['crop_pre'],r['crop_post'])
            x.append(feat);y.append(int(r['label']));ids.append(r)
        except Exception as e:
            skipped+=1
            if skipped<6:logfn(log,f'No se pudo leer recorte {r["uid"]}: {e}')
        if i%1000==0:logfn(log,f'Características: {i}/{len(rows)}')
    if not x:raise RuntimeError('No hay recortes válidos para características.')
    return np.stack(x),np.asarray(y,dtype=int),ids


def prepare_v03(root, previous_workspace, max_train_neg=5000, seed=42, log=None, events=None):
    """Prepara recortes con bordes corregidos en una carpeta NUEVA.

    Nunca sobrescribe un manifest anterior ni elimina resultados. Comprueba que los
    splits asignados a escenas ya presentes se mantienen respecto al manifest viejo.
    """
    previous=Path(previous_workspace).expanduser().resolve()
    old_manifest=previous/'manifest.csv'
    if not old_manifest.is_file():
        raise FileNotFoundError('Primero selecciona la carpeta de trabajo ORIGINAL con manifest.csv.')
    # Mantiene un nombre distinto; los modelos entrenados con recortes v0.2 NO son
    # automáticamente compatibles con los nuevos recortes.
    target=previous.with_name(previous.name+'_v03')
    if target.exists():
        raise FileExistsError('La carpeta de v0.3 ya existe, no se modificó nada: '+str(target))
    if not isinstance(max_train_neg,int) or max_train_neg<=0:
        raise ValueError('El máximo de negativos debe ser positivo.')
    old=load_rows(old_manifest)
    old_scene_split={}
    for r in old:
        scene=r['scene']
        if scene in old_scene_split and old_scene_split[scene]!=r['split']:
            raise RuntimeError('El manifest anterior mezcla splits para '+scene)
        old_scene_split[scene]=r['split']
    fresh,info=scan(root,log=None,events=events)
    chosen=allocate_splits(fresh,seed)
    new_scene_split={r['scene']:r['split'] for r in fresh}
    mismatch=[(scene,split,new_scene_split.get(scene)) for scene,split in old_scene_split.items()
              if split!=new_scene_split.get(scene)]
    if mismatch:
        raise RuntimeError('Los splits del manifest actual no coinciden con la semilla '+str(seed)+
                           '. No se creó ningún dataset. Primera diferencia: '+str(mismatch[0]))
    logfn(log,'✓ Misma asignación de escenas train/val/test respecto a v0.2.')
    logfn(log,'✓ Preparación no destructiva en '+str(target))
    out,report=prepare(root,target,max_train_neg=max_train_neg,seed=seed,log=log,events=events)
    report['previous_workspace']=str(previous)
    report['crop_policy']='reflect-edge-v03; fuera de imagen >50% excluido'
    (target/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    logfn(log,'✓ v0.3 creada. Para entrenar nuevos modelos, seleccionar esta NUEVA carpeta de trabajo.')
    return out,report
