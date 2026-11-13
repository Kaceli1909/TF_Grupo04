"""Create transparent HTML / PNG experiment comparison; never manufacture metrics."""
from pathlib import Path
import html
import json
import csv
import numpy as np
from labels_core import require_task

MODELS=('random_forest','dinov2_xgboost','resnet18_early','siamese_resnet18','siamese_unet_lite','changeos_lite','dahitra_lite','protonet_dino','hybrid')
LABELS={'random_forest':'Random Forest','dinov2_xgboost':'DINOv2 + XGBoost',
        'resnet18_early':'ResNet18 Early Fusion','siamese_resnet18':'Siamese ResNet18',
        'hybrid':'Ensemble (promedio)',
        'siamese_unet_lite':'Siamese U-Net (adaptada)',
        'changeos_lite':'ChangeOS-like (adaptada)',
        'dahitra_lite':'DAHiTrA-like (adaptada)',
        'protonet_dino':'ProtoNet/DINOv2'}


def make_report(workspace,log=None):
    from matplotlib import pyplot as plt
    from sklearn.metrics import precision_recall_curve,ConfusionMatrixDisplay,confusion_matrix
    work=Path(workspace);require_task(work);results=work/'resultados'
    results.mkdir(exist_ok=True,parents=True)
    entries=[]
    predictions=[]
    for model in MODELS:
        p=results/f'{model}_val.json';pred_file=results/f'{model}_val.csv'
        if not p.is_file() or not pred_file.is_file():continue
        d=json.loads(p.read_text(encoding='utf-8'))
        with pred_file.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        if not rows:continue
        y=np.asarray([int(r['label']) for r in rows]);pr=np.asarray([float(r['p_damage']) for r in rows])
        entries.append(d)
        predictions.append((model,y,pr))
    if not entries:raise RuntimeError('Primero entrena al menos un modelo; no hay resultados para graficar.')
    # Metric comparison
    fig,ax=plt.subplots(figsize=(max(12,1.9*len(entries)),6.0))
    indices=np.arange(len(entries));width=.20
    keys=['average_precision','f1_damage','recall_damage','precision_damage']
    for i,key in enumerate(keys):
        vals=[float(e['validation'].get(key) or 0) for e in entries]
        ax.bar(indices+(i-1.5)*width,vals,width,label=key.replace('_',' ').title())
    ax.set_xticks(indices,[LABELS.get(e['model'],e['model']) for e in entries],rotation=25,ha='right')
    ax.set_ylim(0,1.04);ax.set_ylabel('Métrica de validación [0,1]')
    ax.set_title('CIMLAB AI · Comparativa de modelos (VALIDACIÓN, sin test)')
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15))
    fig.tight_layout();fig.savefig(results/'comparativa_metricas.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for model,y,pr in predictions:
        if len(np.unique(y))!=2:continue
        precision,recall,_=precision_recall_curve(y,pr)
        ax.plot(recall,precision,label=LABELS.get(model,model),linewidth=2)
    # natural prevalence of validation set
    base=float(predictions[0][1].mean())
    ax.axhline(base,linestyle='--',linewidth=1,color='gray',label=f'Prevalencia de daño ({base:.2%})')
    ax.set(xlabel='Recall de dañados',ylabel='Precisión de dañados',title='Curvas Precision-Recall (validación)',xlim=(-.02,1.02),ylim=(-.02,1.02))
    ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(results/'precision_recall.png',dpi=150);plt.close(fig)
    # confusion matrix on val-selected thresholds
    count=len(predictions); ncols=min(3,count);nrows=(count+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(4.6*ncols,4.2*nrows))
    axes=np.asarray(axes).reshape(-1)
    for ax,(model,y,pred),entry in zip(axes,predictions,entries):
        t=float(entry['validation']['threshold'])
        cm=confusion_matrix(y,(pred>=t).astype(int),labels=[0,1])
        ConfusionMatrixDisplay(cm,display_labels=['No dañado','Dañado']).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
        ax.set_title(LABELS.get(model,model))
    for ax in axes[count:]:ax.axis('off')
    fig.suptitle('Matrices de confusión · validación')
    fig.tight_layout();fig.savefig(results/'matrices_confusion.png',dpi=150);plt.close(fig)
    # HTML report with exact source numbers
    headers=['Modelo','AP','F1 daño','Recall daño','Precisión daño','Umbral','TN','FP','FN','TP']
    rows_html=[]
    for d in entries:
        m=d['validation']
        cells=[LABELS.get(d['model'],d['model']),m.get('average_precision'),m['f1_damage'],
               m['recall_damage'],m['precision_damage'],m['threshold'],m['tn'],m['fp'],m['fn'],m['tp']]
        rows_html.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>')
    final=results/'evaluacion_final.json'
    add=''
    if final.exists():
        d=json.loads(final.read_text(encoding='utf-8'))
        add=f'<h2>Prueba final reservada</h2><p>Modelo elegido en validación: <strong>{html.escape(str(d["selected_model"]))}</strong></p><pre>{html.escape(json.dumps(d["test"],indent=2,ensure_ascii=False))}</pre>'
    output=f'''<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>CIMLAB AI · Comparación multievento</title>
<style>body{{background:#101724;color:#eaf3f8;font:15px Segoe UI,Arial,sans-serif;margin:0;padding:35px;max-width:1150px;margin:auto}} h1{{font-size:30px;color:#5ad3c5}}h2{{margin-top:38px}}p{{line-height:1.6}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #34475c;padding:12px;text-align:left}}th{{background:#1b2c41}}img{{width:100%;background:white;border-radius:12px;margin-bottom:10px}}.note{{padding:17px 20px;background:#172236;border-left:4px solid #5ad3c5;margin:22px 0}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>CIMLAB AI · Comparación de modelos</h1><p>Colección conjunta multievento · clasificación binaria <b>dañado / no dañado</b> por edificio</p>
<div class="note">Los valores son resultados calculados, no ejemplos. Selección por Average Precision en validación. No interpretar un score de salida sin calibrar como certeza o seguridad estructural.</div>
<h2>Resultados de validación</h2><table><thead><tr>{''.join('<th>'+h+'</th>' for h in headers)}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>
<h2>Comparación de métricas</h2><img src="comparativa_metricas.png" alt="Comparativa de métricas">
<h2>Curvas Precision-Recall</h2><img src="precision_recall.png" alt="Curvas Precision-Recall">
<h2>Matrices de confusión</h2><img src="matrices_confusion.png" alt="Matrices de confusión">
{add}<p><small>Las etiquetas expresan daño superficial visible en imágenes satelitales. Los resultados no constituyen inspección estructural ni habilitabilidad.</small></p></body></html>'''
    out=results/'reporte_modelos.html';out.write_text(output,encoding='utf-8')
    if log:log(f'✓ Reporte HTML y gráficos: {out}')
    return out
