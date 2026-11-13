"""Safety: unknown annotations never become damage labels by accident."""
import sys,json,tempfile
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labels_core import classify,save_event_mapping,require_task,TASK_ID
from data_core import scan,prepare,load_rows

assert classify('destroyed')==1
assert classify('destruido')==1
assert classify('collapsed')==1
assert classify('no-damage')==0
for label in ['minor-damage','major-damage','destroyed','dano-menor','daño mayor','collapsed']:
    assert classify(label)==1,label
for label in ['sin daño','undamaged']:
    assert classify(label)==0,label
assert classify('not destroyed')=='INCOMPATIBLE'
for label in ['severe','severo','alto','bajo','height-high']:
    assert classify(label)=='INCOMPATIBLE',label
assert classify('un-classified') is None

with tempfile.TemporaryDirectory() as td:
    base=Path(td)/'repo'/'data';event=base/'palu-tsunami'
    img=event/'images';ann=event/'labels';img.mkdir(parents=True);ann.mkdir(parents=True)
    for k in range(16):
        scene=f'palu-tsunami_{k:08d}'
        pre=Image.new('RGB',(80,80),(70,80,90));post=Image.new('RGB',(80,80),(80,80,90))
        pre.save(img/(scene+'_pre_disaster.tif'));post.save(img/(scene+'_post_disaster.tif'))
        labels=[('b1','destroyed' if k%2 else 'no-damage'),('b2','alto')]
        features={'features':{'xy':[{'properties':{'uid':uid+str(k),'subtype':typ},
            'wkt':'POLYGON ((10 10, 40 10, 40 40, 10 40, 10 10))'} for uid,typ in labels]}}
        (ann/(scene+'_post_disaster.json')).write_text(json.dumps(features),encoding='utf-8')
    rows,info=scan(base)
    assert info['damaged']==8,info
    assert info['incompatible_labels']=={'alto':16},info
    # NO dataset should be prepared with an unexplained label.
    try:prepare(base,Path(td)/'work',max_train_neg=300)
    except ValueError as exc:assert 'incompatibles' in str(exc).lower() or 'otra escala' in str(exc).lower(),str(exc)
    else:raise AssertionError('Unknown data were silently incorporated')
    # Following documented review, high/low can be explicitly excluded.
    saved=save_event_mapping(event,{'alto':-1})
    assert saved.is_file()
    rows,info=scan(base,strict=True)
    assert not info['incompatible_labels']
    assert len(rows)==16 and sum(r['label'] for r in rows)==8
    out,report=prepare(base,Path(td)/'new_work',max_train_neg=300)
    loaded=load_rows(out)
    assert len(loaded)==16
    assert {int(x['label']) for x in loaded}=={0,1}
    assert require_task(Path(td)/'new_work')['task']==TASK_ID
    # Legacy workspaces without task metadata cannot train a new binary target.
    try: require_task(Path(td)/'old_work')
    except RuntimeError:pass
    else:raise AssertionError('Legacy dataset was accepted for training')
print('✓ ETIQUETAS OK: xBD no-damage=0, minor/major/destroyed=1; unknown blocked; explicit mappings & metadata.')
