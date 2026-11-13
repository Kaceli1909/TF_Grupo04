"""CIMLAB AI v0.2 — auditoría visual y de decisiones SOBRE VALIDACIÓN.

Abre SOLO manifest split=val y *_val.csv. No entrena, no altera modelos
ni consulta split=test. No escribe dentro del repositorio Git ni crops.
"""
from __future__ import annotations
import csv
import html
import json
import math
import random
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
from labels_core import require_task

MODELS = ('random_forest', 'dinov2_xgboost', 'resnet18_early',
          'siamese_resnet18', 'siamese_unet_lite', 'changeos_lite',
          'dahitra_lite', 'protonet_dino', 'hybrid')
NAMES = {'random_forest': 'Random Forest', 'dinov2_xgboost': 'DINOv2 + XGBoost',
         'resnet18_early': 'ResNet18 Early Fusion',
         'siamese_resnet18': 'Siamese ResNet18', 'hybrid': 'Híbrido',
         'siamese_unet_lite': 'Siamese U-Net (adaptación)',
         'changeos_lite':'ChangeOS-like (adaptación)',
         'dahitra_lite':'DAHiTrA-like (adaptación)',
         'protonet_dino':'ProtoNet con DINOv2'}


def available_models(workspace):
    res = Path(workspace) / 'resultados'
    return [m for m in MODELS if (res/(m+'_val.json')).exists() and
            (res/(m+'_val.csv')).exists()]


def load_validation(workspace, model):
    """Relaciona pronósticos y recortes por (scene, uid), nunca por orden de filas."""
    if model not in MODELS:
        raise ValueError('Modelo inválido: '+str(model))
    ws = Path(workspace)
    require_task(ws)
    if not (ws/'manifest.csv').is_file():
        raise FileNotFoundError('Falta manifest.csv. Selecciona el trabajo del experimento actual.')
    res = ws/'resultados'
    jsonfile, predfile = res/(model+'_val.json'), res/(model+'_val.csv')
    if not jsonfile.exists() or not predfile.exists():
        raise FileNotFoundError(f'No hay resultados de validación para {model}.')
    with (ws/'manifest.csv').open(newline='', encoding='utf-8-sig') as f:
        man = list(csv.DictReader(f))
    valid = {}
    split_scenes = {}
    for r in man:
        key = (r.get('scene',''),r.get('uid',''))
        if not key[0] or not key[1]:
            raise ValueError('El manifest contiene escena/uid vacío.')
        sp = r.get('split')
        if sp not in ('train','val','test'):
            raise ValueError('Split inválido del manifest: '+str(sp))
        if key in valid or key in split_scenes:
            raise ValueError(f'Duplicado en manifest: {key}')
        split_scenes[key] = sp
        if sp == 'val':
            valid[key] = r
    if not valid:
        raise ValueError('No hay registros de validación en manifest.csv.')
    with predfile.open(newline='',encoding='utf-8-sig') as f:
        predicted = list(csv.DictReader(f))
    rows,seen = [],set()
    for pred in predicted:
        key = (pred.get('scene',''),pred.get('uid',''))
        if key in seen:
            raise ValueError(f'Duplicado en predicciones {model}: {key}')
        seen.add(key)
        if key not in valid:
            sp = split_scenes.get(key,'no encontrado')
            raise ValueError(f'PREDICCIÓN FUERA DE VALIDACIÓN: {key} ({sp}). Se detuvo la auditoría.')
        r = dict(valid[key])
        label = int(r['label'])
        if int(pred['label']) != label:
            raise ValueError(f'Inconsistencia de etiqueta para {key}')
        p = float(pred['p_damage'])
        if not math.isfinite(p) or not (0 <= p <= 1):
            raise ValueError(f'Score inválido para {key}: {p}')
        r['label'] = label
        r['score'] = p
        rows.append(r)
    if len(rows)!=len(valid):
        missing = sorted(set(valid)-seen)
        raise ValueError(f'Faltan {len(missing)} predicciones de validación. Ejemplo: {missing[:1]}')
    report = json.loads(jsonfile.read_text(encoding='utf-8'))
    if report.get('model')!=model:
        raise ValueError('El JSON de validación no corresponde al modelo seleccionado.')
    threshold=float(report.get('validation',{}).get('threshold',.5))
    return {'workspace': str(ws), 'model': model, 'rows': rows,
            'saved_threshold': threshold, 'report': report,
            'scenes': len({r['scene'] for r in rows})}


def evaluate(rows, threshold):
    from sklearn.metrics import average_precision_score,precision_recall_fscore_support
    if not rows:
        raise ValueError('No hay predicciones.')
    y = np.array([int(r['label']) for r in rows], dtype=int)
    p = np.array([r['score'] for r in rows], dtype=float)
    pred = p >= float(threshold)
    tp = int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred))
    tn = int(np.sum((y==0)&~pred));fn=int(np.sum((y==1)&~pred))
    pr = tp/max(tp+fp,1);re=tp/max(tp+fn,1)
    f1 = 2*pr*re/max(pr+re,1e-15)
    specificity = tn/max(tn+fp,1)
    ap = float(average_precision_score(y,p)) if len(set(y))==2 else float('nan')
    return {'threshold': float(threshold), 'precision':pr,'recall':re,'f1':f1,
            'ap':ap, 'prevalence':float(y.mean()), 'balanced_accuracy':(re+specificity)/2,
            'tp':tp,'fp':fp,'tn':tn,'fn':fn,'positives':int(y.sum()),'n':len(rows),
            'alerts':tp+fp}


def categorized(rows, threshold, group='FN', limit=200):
    """Filtra y ordena por casos de mayor interés diagnóstico."""
    threshold=float(threshold)
    annotated=[]
    for r in rows:
        hit = r['score']>=threshold
        outcome=('TP' if hit else 'FN') if r['label']==1 else ('FP' if hit else 'TN')
        x=dict(r);x['outcome']=outcome;x['pred']=int(hit)
        annotated.append(x)
    if group in ('FN','FP','TP','TN'):
        data=[r for r in annotated if r['outcome']==group]
        if group in ('FP','TP'): data.sort(key=lambda r:r['score'],reverse=True)
        elif group=='FN':data.sort(key=lambda r:r['score'])
        else: data.sort(key=lambda r:r['score'],reverse=True)
    elif group=='Mayores scores':
        data=sorted(annotated,key=lambda r:r['score'],reverse=True)
    elif group=='Cerca del umbral':
        data=sorted(annotated,key=lambda r:abs(r['score']-threshold))
    elif group=='Aleatorios':
        data=annotated[:]
        random.Random(42).shuffle(data)
    else:
        raise ValueError('Filtro no reconocido')
    return data[:int(limit)],len(data)


def topk(rows, k_values=(20,50,100,200,500,1000)):
    sorted_rows=sorted(rows,key=lambda r:r['score'],reverse=True)
    positives=sum(int(r['label']) for r in rows)
    out=[]
    for k in k_values:
        k=min(k,len(rows))
        if not k:continue
        hits=sum(int(r['label']) for r in sorted_rows[:k])
        expected = k*positives/len(rows)
        out.append({'k':k,'damaged_found':hits,'precision_at_k':hits/k,
                    'recall_at_k':hits/max(positives,1),
                    'random_expected':expected,'lift_vs_random': hits/expected if expected>0 else 0})
    return out


def polygon_on_crop(row, size=(224,224), outline=(37,210,201)):
    """Calcula la geometría en el recorte con la MISMA transformación crop_pair()."""
    from PIL import ImageDraw, Image
    from shapely import wkt
    x0,y0,x1,y1 = (float(row[k]) for k in ('x0','y0','x1','y1'))
    pad=12
    cx=(x0+x1)/2; cy=(y0+y1)/2
    side=max(x1-x0+2*pad,y1-y0+2*pad,32)
    left=math.floor(cx-side/2);top=math.floor(cy-side/2)
    right=math.ceil(cx+side/2);bottom=math.ceil(cy+side/2)
    sx,sy=size[0]/(right-left),size[1]/(bottom-top)
    geometry=wkt.loads(row['wkt'])
    parts=list(geometry.geoms) if geometry.geom_type=='MultiPolygon' else [geometry]
    def draw_poly(img):
        dr=ImageDraw.Draw(img)
        for poly in parts:
            outer=[((x-left)*sx,(y-top)*sy) for x,y in poly.exterior.coords]
            dr.line(outer,fill=outline,width=3,joint='curve')
            for ring in poly.interiors:
                holes=[((x-left)*sx,(y-top)*sy) for x,y in ring.coords]
                dr.line(holes,fill=(240,196,79),width=2)
        return img
    return draw_poly


def make_visual(row, with_polygon=True, image_size=250, view_mode='original'):
    """Compara PRE/POST originales, luz ajustada o magnitud de diferencias.

    El ajuste fotométrico es solo de VISUALIZACIÓN; las predicciones y los
    recortes de entrenamiento nunca se modifican.
    """
    from PIL import Image,ImageDraw
    from comparison_core import view_images
    paths=(Path(row['crop_pre']),Path(row['crop_post']))
    if not all(p.exists() for p in paths):
        missing=next(p for p in paths if not p.exists())
        raise FileNotFoundError('No se encontró el recorte '+str(missing))
    with Image.open(paths[0]) as ia,Image.open(paths[1]) as ib:
        a=ia.convert('RGB');b=ib.convert('RGB')
    if a.size!=b.size:
        raise ValueError('Recortes PRE/POST con dimensiones diferentes')
    images=view_images(a,b,view_mode)
    for j,(title,im) in enumerate(images):
        if with_polygon and j<2:
            try:polygon_on_crop(row,size=im.size)(im)
            except Exception:pass
        im.thumbnail((image_size,image_size))
    n=len(images);wid=max(im.width for _,im in images);hei=max(im.height for _,im in images)
    canvas=Image.new('RGB',(n*wid+(n+1)*8,hei+72),(15,24,38));draw=ImageDraw.Draw(canvas)
    for i,(title,im) in enumerate(images):
        left=8+i*(wid+8)
        canvas.paste(im,(left,32))
        draw.text((left,10),title,fill=(204,229,234))
    rawscore=row.get('score',0)
    try:score=float(rawscore)
    except (TypeError,ValueError):score=0.
    label='DAÑADO' if int(row['label']) else 'NO DAÑADO'
    draw.text((8,hei+43),f'Etiqueta: {label} | Score: {score:.4f}',fill=(204,229,234))
    return canvas


def export_audit(dataset, threshold, log=None):
    """Informe nuevo, separado del reporte original. Usa exclusivamente VALIDACIÓN."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    ws=Path(dataset['workspace']);model=dataset['model'];rows=dataset['rows']
    dest=ws/'auditoria_v02'/model
    dest.mkdir(parents=True,exist_ok=True)
    m=evaluate(rows,threshold)
    thresholds = sorted(set([round(t,4) for t in np.linspace(.025,.975,39)]+[
        round(float(threshold),4),round(dataset['saved_threshold'],4),.5]))
    threshold_rows=[evaluate(rows,t) for t in thresholds]
    with (dest/'tabla_umbrales_validacion.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(threshold_rows[0]));w.writeheader();w.writerows(threshold_rows)
    tk=topk(rows)
    with (dest/'priorizacion_topk_validacion.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(tk[0]));w.writeheader();w.writerows(tk)
    with (dest/'edificios_validacion_diagnostico.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['scene','uid','original_label','label','score','pred','outcome','crop_pre','crop_post']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for row in rows:
            ispos=int(row['score']>=threshold);tr=int(row['label'])
            d='TP' if ispos and tr else 'FP' if ispos else 'FN' if tr else 'TN'
            w.writerow({**{k:row.get(k,'') for k in cols},'pred':ispos,'outcome':d})
    yy=np.array([int(r['label']) for r in rows]);p=np.array([r['score'] for r in rows])
    precision,recall,_=precision_recall_curve(yy,p)
    fig,ax=plt.subplots(figsize=(7.5,4.5));ax.plot(recall,precision,label=model)
    ax.axhline(m['prevalence'],color='grey',linestyle='--',label=f'Base aleatoria {m["prevalence"]:.2%}')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title='Precision-Recall · solo validación')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(dest/'curva_pr.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.plot([t['threshold'] for t in threshold_rows],[t['f1'] for t in threshold_rows],label='F1 daño')
    ax.plot([t['threshold'] for t in threshold_rows],[t['recall'] for t in threshold_rows],label='Recall daño')
    ax.plot([t['threshold'] for t in threshold_rows],[t['precision'] for t in threshold_rows],label='Precision daño')
    ax.axvline(threshold,color='black',linestyle='--',label=f'Umbral actual {threshold:.3f}')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Umbral de decisión',ylabel='Métrica',title='Análisis exploratorio de umbrales (validación)')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(dest/'curvas_umbrales.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.hist(p[yy==0],bins=np.linspace(0,1,30),alpha=.65,label=f'No dañados n={(yy==0).sum()}')
    ax.hist(p[yy==1],bins=np.linspace(0,1,30),alpha=.65,label=f'Dañados n={(yy==1).sum()}')
    ax.axvline(threshold,color='black',linestyle='--')
    ax.set(xlabel='Score del modelo (NO probabilidad calibrada)',ylabel='Edificios',title='Distribución de scores en validación')
    ax.legend();fig.tight_layout();fig.savefig(dest/'distribucion_scores.png',dpi=140);plt.close(fig)
    # 6 casos por tipo max: creates readable snapshots; never draws from test.
    cases=[]
    for g in ('FN','FP','TP','TN'):
        subset,total=categorized(rows,threshold,g,limit=6)
        for j,r in enumerate(subset,1):
            try:
                name=f'{g}_{j:02d}.jpg'
                make_visual(r,True,270).save(dest/name,quality=87)
                cases.append((g,r,name))
            except Exception as e:
                if log:log(f'AVISO: vista {g}/{j} no exportada: {e}')
    def fmt(x):return f'{x:.4f}'
    case_html=''.join('<div class="sample"><div><b>'+html.escape(g)+'</b> · '+html.escape(r['scene'])+' · '
                      +html.escape(r['uid'][:12])+' · score '+fmt(r['score'])+'</div><img src="'+name+'"></div>'
                      for g,r,name in cases)
    top_html=''.join('<tr>'+''.join('<td>'+str(round(v,4) if isinstance(v,float) else v)+'</td>' for v in t.values())+'</tr>' for t in tk)
    html_file=dest/'informe_auditoria.html'
    html_file.write_text(f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>CIMLAB AI · Daños | {html.escape(model)}</title>
<style>body{{background:#101724;color:#eaf0f4;font:15px Segoe UI,Arial,sans-serif;margin:32px auto;padding:0 25px;max-width:1120px}}h1{{color:#46c6b7}}p{{line-height:1.6}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #36475b;text-align:left}}th{{background:#1c2c40}}.warning{{padding:15px;background:#233044;border-left:4px solid #e5af51;margin:20px 0}}.plots{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}.plots img{{max-width:100%;background:white;border-radius:9px}}.samples{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.sample{{background:#18263a;padding:12px;border-radius:8px;font-size:12px;overflow-wrap:anywhere}}.sample img{{width:100%;margin-top:8px}}</style></head>
<body><h1>CIMLAB AI · Auditoría de daños · validación</h1><p>Modelo: <b>{html.escape(NAMES.get(model,model))}</b> · Escenas de validación: {dataset['scenes']} · edificios: {m['n']} · dañados: {m['positives']}.</p>
<div class="warning"><b>Solo VALIDACIÓN.</b> No se utiliza la prueba final. Las etiquetas son daños visuales, no un dictamen de seguridad estructural. Un score de 0.9 no representa automáticamente 90 % de certeza. La exploración de umbrales sobre validación puede sobreajustar decisiones: no elegir el umbral usando test.</div>
<h2>Métricas para umbral {threshold:.3f}</h2><table><tr><th>Average Precision (no depende del umbral)</th><th>F1</th><th>Recall</th><th>Precision</th><th>TP / FP / FN / TN</th></tr><tr><td>{fmt(m['ap'])}</td><td>{fmt(m['f1'])}</td><td>{fmt(m['recall'])}</td><td>{fmt(m['precision'])}</td><td>{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}</td></tr></table>
<h2>Gráficos</h2><div class="plots"><img src="curva_pr.png"><img src="curvas_umbrales.png"><img src="distribucion_scores.png"></div>
<h2>Priorización de inspección (exploratoria)</h2><p>Recall@K indica qué proporción de daños conocidos aparece entre los K edificios con mayor score.</p><table><tr><th>K</th><th>Daños encontrados</th><th>Precision@K</th><th>Recall@K</th><th>Azar esperado</th><th>Lift vs azar</th></tr>{top_html}</table>
<h2>Ejemplos visuales (revisar individualmente)</h2><p>FN: edificios dañados omitidos; FP: falsas alertas de daño; TP: dañados detectados; TN: aciertos sin daño. Contornos cian indican la geometría original proyectada sobre el recorte.</p><div class="samples">{case_html}</div><p>Archivos CSV adicionales disponibles en esta carpeta. No se han modificado los TIFF, recortes, modelos ni resultados anteriores.</p></body></html>''',encoding='utf-8')
    if log:log('✓ Auditoría de daños exportada: '+str(html_file))
    return html_file
