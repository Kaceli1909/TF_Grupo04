"""CIMLAB AI v0.3: auditoría NO DESTRUCTIVA de manifest y recortes de train/val.

No lee imágenes de test ni altera recortes, pesos ni predicciones. Las métricas de
calidad son INDICIOS visuales, no errores de etiqueta confirmados.
"""
from __future__ import annotations
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

from data_core import logfn, crop_window, load_rows


def _rgb_small(path, size=64):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.


def _corr(a,b):
    x=(a.mean(axis=2)).ravel();y=(b.mean(axis=2)).ravel()
    if x.std()<1e-5 or y.std()<1e-5:return 0.0
    return float(np.clip(np.corrcoef(x,y)[0,1],-1,1))


def _sharpness(img):
    g=img.mean(axis=2)
    dx=np.diff(g,axis=1);dy=np.diff(g,axis=0)
    return float(np.mean(dx*dx)+np.mean(dy*dy))


def _crop_outside_fraction(box,width,height):
    x0,y0,x1,y1=box
    area=max(1,(x1-x0)*(y1-y0))
    overlap=max(0,min(x1,width)-max(x0,0))*max(0,min(y1,height)-max(y0,0))
    return max(0.0,min(1.0,1.-overlap/area))


def _read_predictions(workspace,model):
    p=workspace/'resultados'/(model+'_val.csv')
    if not p.is_file():return {}
    predictions={}
    with p.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key=(r['scene'],r['uid'])
            if key in predictions:raise ValueError('Duplicados en resultados de validación: '+str(key))
            predictions[key]=(int(r['label']),float(r['p_damage']))
    return predictions


def _read_threshold(workspace,model):
    p=workspace/'resultados'/(model+'_val.json')
    if p.is_file():
        return float(json.loads(p.read_text(encoding='utf-8')).get('validation',{}).get('threshold',.5))
    return .5


def _validate_manifest_scenes(rows):
    scenes={}
    for r in rows:
        prev=scenes.setdefault(r['scene'],r['split'])
        if r['split']!=prev:
            raise RuntimeError('FUGA ENTRE SPLITS: escena '+r['scene']+' aparece en '+prev+' y '+r['split'])
    return scenes


def audit_quality(workspace,model='dinov2_xgboost',log=None,max_previews=18):
    """Lee el manifiesto existente, solo procesa recortes train/val y escribe un informe nuevo.

    No se emplea test para decisiones de calidad; solo se contrastan nombres de escena
    para verificar que las particiones son disjuntas.
    """
    ws=Path(workspace).expanduser().resolve()
    manifest=ws/'manifest.csv'
    if not manifest.exists():raise FileNotFoundError('Falta manifest.csv en la carpeta de trabajo seleccionada.')
    rows=load_rows(manifest)
    if not rows:raise ValueError('El manifest está vacío.')
    _validate_manifest_scenes(rows)
    seen=set(); filtered=[]
    for r in rows:
        key=(r['scene'],r['uid'])
        if key in seen:raise RuntimeError('Edificio duplicado: '+str(key))
        seen.add(key)
        if r['split'] in ('train','val'):filtered.append(r)
    if not filtered:raise RuntimeError('No hay edificios de train/val para auditar.')
    predictions=_read_predictions(ws,model)
    threshold=_read_threshold(ws,model)
    # Las escenas se abren una sola vez para dimensiones, nunca se lee el test.
    scene_info={};grouped=defaultdict(list)
    for r in filtered:grouped[r['scene']].append(r)
    diagnostics=[]; failures=[]; examples=[]
    totals=Counter();totals_by_split=defaultdict(Counter)
    logfn(log,f'Calidad v0.3: {len(filtered)} edificios de train/val · {len(grouped)} escenas; test sin analizar.')
    for i,(scene,group) in enumerate(sorted(grouped.items()),1):
        a,b=Path(group[0]['pre']),Path(group[0]['post'])
        try:
            with Image.open(a) as pre,Image.open(b) as post:
                wh1=pre.size;wh2=post.size
        except Exception as exc:
            failures.append(scene+': no se pudieron abrir originales: '+str(exc))
            continue
        dimensions_ok=wh1==wh2
        scene_info[scene]={'width':wh1[0],'height':wh1[1],'pair_dims_ok':dimensions_ok}
        for r in group:
            spl=r['split'];totals[spl]+=1;totals_by_split[spl][int(r['label'])]+=1
            box=crop_window((r['x0'],r['y0'],r['x1'],r['y1']))
            outside=_crop_outside_fraction(box,*wh1)
            crop_side=max(1,(box[2]-box[0])*(box[3]-box[1]))
            bbox_fraction=max(0.,(r['x1']-r['x0'])*(r['y1']-r['y0']))/crop_side
            flags=[]
            if not dimensions_ok:flags.append('dimensiones_distintas')
            if outside>.01:flags.append('recorte_fuera_imagen')
            if bbox_fraction<.035:flags.append('edificio_muy_pequeno')
            similarity=None;sharp_ratio=None;black_border=None
            path_a,path_b=Path(r['crop_pre']),Path(r['crop_post'])
            if not path_a.is_file() or not path_b.is_file():
                flags.append('recortes_faltantes')
            else:
                try:
                    pre=_rgb_small(path_a)
                    post=_rgb_small(path_b)
                    similarity=_corr(pre,post)
                    sharp_ratio=_sharpness(post)/max(_sharpness(pre),1e-7)
                    rim=np.concatenate((pre[:5],pre[-5:],post[:5],post[-5:]),axis=0)
                    black_border=float(np.mean(np.max(rim,axis=2)<.035))
                    if similarity<.10:flags.append('cambio_visual_extremo')
                    if sharp_ratio<.30:flags.append('post_poco_nitida')
                    if black_border>.18:flags.append('borde_negro')
                except Exception as exc:
                    flags.append('recorte_ilegible')
                    if len(failures)<200:failures.append(scene+'/'+r['uid']+': '+str(exc))
            truth,score=predictions.get((scene,r['uid']),(None,None)) if spl=='val' else (None,None)
            # No confiar en CSV de predicción si etiquetas cambiaron.
            if truth is not None and truth!=int(r['label']):
                raise RuntimeError('Inconsistencia entre etiqueta y predicción: '+scene+'/'+r['uid'])
            outcome=''
            if score is not None:
                outcome=('TP' if score>=threshold else 'FN') if r['label']==1 else ('FP' if score>=threshold else 'TN')
            record={'split':spl,'scene':scene,'uid':r['uid'],'original_label':r['original_label'],
                    'label':r['label'],'outside_pct':round(outside*100,2),
                    'bbox_frac_pct':round(bbox_fraction*100,2),
                    'pre_post_corr':round(similarity,3) if similarity is not None else '',
                    'sharp_ratio_post_pre':round(sharp_ratio,3) if sharp_ratio is not None else '',
                    'black_border_pct':round(black_border*100,2) if black_border is not None else '',
                    'flags':' | '.join(flags),'score':round(score,6) if score is not None else '',
                    'outcome':outcome,'crop_pre':r['crop_pre'],'crop_post':r['crop_post'],
                    'wkt':r['wkt'],'x0':r['x0'],'y0':r['y0'],'x1':r['x1'],'y1':r['y1']}
            diagnostics.append(record)
        if i%15==0 or i==len(grouped):logfn(log,f'Calidad de datos: {i}/{len(grouped)} escenas.')
    if not diagnostics:raise RuntimeError('No se pudieron examinar edificios. Revisa la ubicación del repositorio.')
    dest=ws/'calidad_v03';dest.mkdir(parents=True,exist_ok=True)
    csv_file=dest/'edificios_train_val.csv'
    # Los paths en CSV son solo referencias; las carpetas originales nunca se alteran.
    fields=['split','scene','uid','original_label','label','outside_pct','bbox_frac_pct',
            'pre_post_corr','sharp_ratio_post_pre','black_border_pct','flags','score','outcome',
            'crop_pre','crop_post']
    with csv_file.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k,'') for k in fields} for r in diagnostics)
    grouped_out=defaultdict(list)
    for r in diagnostics:grouped_out[r['scene']].append(r)
    scenes=[]
    for scene,items in grouped_out.items():
        val=[r for r in items if r['split']=='val']
        fps=sum(r['outcome']=='FP' for r in val)
        fns=sum(r['outcome']=='FN' for r in val)
        alerts=sum(bool(r['flags']) for r in items)
        scenes.append({'scene':scene,'split':items[0]['split'],'edificios':len(items),
                       'danados':sum(int(r['label']) for r in items),'alertas_calidad':alerts,
                       'alerta_pct':round(alerts*100/len(items),2),
                       'fp':fps,'fn':fns,'fp_100':round(fps*100/max(1,sum(r['split']=='val' and r['label']==0 for r in items)),2)})
    scenes.sort(key=lambda r:(r['fp'],r['alertas_calidad'],r['edificios']),reverse=True)
    with (dest/'escenas_train_val.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=list(scenes[0]);w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(scenes)
    flag_counts=Counter(f for r in diagnostics for f in r['flags'].split(' | ') if f)
    # Priorizar muestras: FP/FN que además tienen banderas, luego errores relevantes.
    candidates=sorted(diagnostics,key=lambda r:(
        bool(r['flags']) and r['outcome'] in ('FP','FN'),r['outcome'] in ('FP','FN'),
        bool(r['flags']),r['outside_pct'],float(r['score']) if r['score']!='' else -1),reverse=True)
    used_scenes=Counter()
    try:
        from audit_core import make_visual
        for r in candidates:
            if len(examples)>=max_previews:break
            if used_scenes[r['scene']]>=4:continue
            if not Path(r['crop_pre']).is_file() or not Path(r['crop_post']).is_file():continue
            try:
                visual=dict(r)
                # Training rows do not have scores. They still deserve previews.
                visual['score']=float(r['score']) if r['score']!='' else 0.0
                number=len(examples)+1
                im=make_visual(visual,True,240)
                name=f'caso_{number:02d}.jpg';im.save(dest/name,quality=85)
                diff=make_visual(visual,True,170,view_mode='change')
                diffname=f'cambio_{number:02d}.jpg';diff.save(dest/diffname,quality=85)
                examples.append((r,name,diffname));used_scenes[r['scene']]+=1
            except Exception:pass
    except Exception as e:logfn(log,'Aviso: no se pudieron crear las vistas de polígonos: '+str(e))
    summary={'version':'0.3','workspace':str(ws),'generated':datetime.now().isoformat(timespec='seconds'),
             'audited_splits':['train','val'],'test_used':False,'inspected':len(diagnostics),
             'totals':dict(totals),'labels':{k:dict(v) for k,v in totals_by_split.items()},
             'flags':dict(flag_counts),'flagged':sum(bool(r['flags']) for r in diagnostics),
             'scenes':len(scenes),'model_validation':model if predictions else None,
             'saved_threshold':threshold if predictions else None,'failures':failures[:200],
             'manifest':str(manifest),'report':str(dest/'informe_calidad.html')}
    (dest/'resumen_calidad.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    tr=''.join('<tr>'+''.join('<td>'+html.escape(str(x.get(c,'')))+'</td>' for c in
        ('scene','split','edificios','danados','alertas_calidad','fp','fn'))+'</tr>' for x in scenes[:40])
    flags_html=''.join('<tr><td>'+html.escape(k)+'</td><td>'+str(v)+'</td></tr>' for k,v in flag_counts.most_common())
    previews=''.join('<div class="sample"><b>'+html.escape(r['outcome'] or r['split'])+'</b> · '+
                     html.escape(r['scene'])+' · '+html.escape(r['uid'])+'<p>'+html.escape(r['flags'] or 'Sin alerta automática')+
                     '</p><img src="'+name+'"><details><summary>Ver mapa de diferencias (orientativo)</summary><img src="'+diffname+'"></details></div>' for r,name,diffname in examples)
    label_str='; '.join(f'{k}: {dict(v)}' for k,v in totals_by_split.items())
    html_doc='''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>CIMLAB AI · Control de calidad v0.3</title>
<style>body{font:14px Segoe UI,Arial,sans-serif;background:#101724;color:#e8eef5;max-width:1220px;padding:22px;margin:auto}h1,h2{color:#44c9bb}p{line-height:1.6}.note{border-left:4px solid #deb066;background:#1a293f;padding:14px;margin:20px 0}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #344357;padding:8px;text-align:left}th{background:#1c2b40}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.sample{background:#1d2b40;padding:12px;border-radius:7px}.sample img{width:100%}small{color:#acbed3}</style></head><body>
<h1>CIMLAB AI · Control de calidad de datos v0.3</h1>
<div class="note"><b>Auditoría no destructiva.</b> Se analizan únicamente TRAIN y VALIDACIÓN; TEST permanece sin inspeccionar. Las banderas son <b>indicios</b>, no confirmaciones de errores de etiqueta, daños o desalineamiento. Los scores originales no se alteran.</div>
<p>Edificios analizados: <b>__N__</b> · escenas: <b>__SCENES__</b> · con alguna alerta: <b>__FLAGGED__</b>.</p>
<p>Distribución: __LABELS__. Modelo de validación: __MODEL__; umbral original: __THRESHOLD__.</p>
<h2>Alertas de calidad</h2><table><thead><tr><th>Indicador</th><th>Edificios</th></tr></thead><tbody>__FLAGS__</tbody></table>
<small>cambio_visual_extremo = correlación de apariencia baja, no prueba de desregistro; post_poco_nitida = menor energía de bordes en POST; borde_negro = píxeles oscuros en el borde del recorte, que también pueden ser reales. No son reglas para excluir imágenes automáticamente. Los mapas de diferencias son ayudas visuales: las diferencias pueden ser de sombras, luz, vegetación o desplazamiento, no necesariamente daños.</small>
<h2>Escenas prioritarias (hasta 40)</h2><table><thead><tr><th>Escena</th><th>Split</th><th>Edificios</th><th>Dañados</th><th>Alertas</th><th>FP</th><th>FN</th></tr></thead><tbody>__SCENE_TABLE__</tbody></table>
<h2>Ejemplos a revisar</h2><div class="gallery">__PREVIEWS__</div>
<p><small>CSVs completos: edificios_train_val.csv y escenas_train_val.csv. No se han recreado recortes ni modificado pesos.</small></p></body></html>'''
    for k,v in {'__N__':len(diagnostics),'__SCENES__':len(scenes),'__FLAGGED__':summary['flagged'],
                '__LABELS__':html.escape(label_str),'__MODEL__':html.escape(model if predictions else '(no disponible)'),
                '__THRESHOLD__':html.escape(f'{threshold:.3f}' if predictions else '-'),
                '__FLAGS__':flags_html,'__SCENE_TABLE__':tr,'__PREVIEWS__':previews}.items():
        html_doc=html_doc.replace(k,str(v))
    (dest/'informe_calidad.html').write_text(html_doc,encoding='utf-8')
    logfn(log,f'✓ Control de calidad exportado sin tocar datos existentes: {dest / "informe_calidad.html"}')
    return dest/'informe_calidad.html',summary
