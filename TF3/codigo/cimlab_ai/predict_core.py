"""Predict visible damage from given PRE/POST files and xBD polygon JSON."""
from pathlib import Path
import json
import csv
from collections import Counter
from PIL import Image,ImageDraw
from data_core import polygon_from_wkt,crop_pair,is_lfs_pointer,logfn,load_pair_rgb
from models_core import predict_prob
from labels_core import require_task


def predict_scene(workspace,pre_path,post_path,json_path,log=None):
    work=Path(workspace).expanduser().resolve()
    require_task(work)
    result_path=work/'resultados'/'evaluacion_final.json'
    if not result_path.exists():
        raise RuntimeError('Primero ejecuta Comparar y prueba final para elegir un modelo validado.')
    results=json.loads(result_path.read_text(encoding='utf-8'))
    name=results['selected_model']
    threshold=results['validation']['threshold']
    pre_path=Path(pre_path);post_path=Path(post_path);json_path=Path(json_path)
    for p in [pre_path,post_path,json_path]:
        if not p.exists():raise FileNotFoundError(str(p))
    for p in [pre_path,post_path]:
        if is_lfs_pointer(p):raise RuntimeError(f'{p.name}: es un puntero LFS; descarga TIFF real primero.')
    data=json.loads(json_path.read_text(encoding='utf-8'))
    features=data.get('features',{}).get('xy',[])
    if not features:raise RuntimeError('El JSON no contiene polígonos en features.xy.')
    dest=work/'predicciones'/post_path.stem
    dest.mkdir(parents=True,exist_ok=True)
    rows=[];polys=[]
    a,b=load_pair_rgb(pre_path,post_path)
    if a.size!=b.size:raise RuntimeError('Las imágenes PRE y POST deben tener las mismas dimensiones.')
    for n,feat in enumerate(features):
        uid=str(feat.get('properties',{}).get('uid') or n)
        try:
            poly=polygon_from_wkt(feat['wkt'])
            c,d=crop_pair(a,b,poly)
            outa=dest/f'{n:05d}_pre.jpg';outb=dest/f'{n:05d}_post.jpg'
            c.save(outa,quality=87);d.save(outb,quality=87)
            rows.append({'scene':post_path.stem,'uid':uid,'label':-1,'split':'new',
                         'crop_pre':str(outa),'crop_post':str(outb),
                         'wkt':feat['wkt'],
                         'x0':poly.bounds[0],'y0':poly.bounds[1],
                         'x1':poly.bounds[2],'y1':poly.bounds[3]})
            polys.append(poly)
        except Exception as e:
            logfn(log,f'Se omitió geometría {uid}: {e}')
    if not rows:raise RuntimeError('No hay polígonos válidos para predecir.')
    if name=='hybrid':
        cfg=json.loads((work/'resultados'/'hybrid_config.json').read_text(encoding='utf-8'))
        vals=[]
        for member in cfg['members']:
            j=json.loads((work/'resultados'/(member+'_val.json')).read_text(encoding='utf-8'))
            vals.append(predict_prob(member,j['model_path'],rows,work,log))
        import numpy as np
        prob=np.mean(vals,axis=0)
    else:
        j=json.loads((work/'resultados'/(name+'_val.json')).read_text(encoding='utf-8'))
        prob=predict_prob(name,j['model_path'],rows,work,log)
    counts=Counter()
    layer=Image.new('RGBA',b.size,(0,0,0,0))
    draw=ImageDraw.Draw(layer,'RGBA')
    with (dest/'predicciones.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scene','uid','score_dano_no_calibrado','clasificacion','umbral'])
        for r,p,poly in zip(rows,prob,polys):
            prediction='Posible daño' if p>=threshold else 'No dañado' 
            counts[prediction]+=1
            writer.writerow([r['scene'],r['uid'],round(float(p),6),prediction,threshold])
            color=(235,79,72,100) if p>=threshold else (38,183,130,70)
            parts=[poly] if poly.geom_type=='Polygon' else list(poly.geoms)
            for shape in parts:
                points=list(shape.exterior.coords)
                draw.polygon(points,fill=color,outline=(*color[:3],240),width=2)
    overlay=Image.alpha_composite(b.convert('RGBA'),layer).convert('RGB')
    overlay.save(dest/'mapa_prediccion.png')
    summary={'model':name,'threshold':threshold,'counts':dict(counts),
             'note':'Daño visible estimado por imágenes; NO es dictamen de seguridad estructural.',
             'results_folder':str(dest)}
    (dest/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    logfn(log,'✓ Predicción guardada en '+str(dest))
    return summary
