"""Self-contained synthetic data smoke test; DOES NOT measure real-world model performance."""
import sys
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from data_core import prepare,load_rows,scan,is_lfs_pointer
from models_core import train_selected,test_winner
from predict_core import predict_scene
from report_core import make_report

with tempfile.TemporaryDirectory() as t:
    root=Path(t)/'repo/data/mexico-earthquake'
    for k in range(18):
        part='tier1' if k<10 else ('hold' if k<14 else 'test')
        img=root/part/'images';lab=root/part/'labels'
        img.mkdir(parents=True,exist_ok=True);lab.mkdir(parents=True,exist_ok=True)
        scene=f'mexico-earthquake_{k:08d}'
        pre=Image.new('RGB',(256,256),(130,150,135));post=pre.copy()
        before=ImageDraw.Draw(pre);after=ImageDraw.Draw(post)
        feats=[]
        # 2 buildings every scene; half have 1 damaged
        for j in range(2):
            x=25+j*110;y=38
            before.rectangle((x,y,x+66,y+77),fill=(160,160,155))
            damaged=j==0 and k%2==0
            after.rectangle((x,y,x+66,y+77),fill=(95,95,95) if damaged else (160,160,155))
            wkt=f'POLYGON (({x} {y}, {x+66} {y}, {x+66} {y+77}, {x} {y+77}, {x} {y}))'
            feats.append({'properties':{'uid':f'uid-{k}-{j}','subtype':'destroyed' if damaged else 'no-damage'},'wkt':wkt})
        pre.save(img/f'{scene}_pre_disaster.tif')
        post.save(img/f'{scene}_post_disaster.tif')
        (lab/f'{scene}_post_disaster.json').write_text(json.dumps({'features':{'xy':feats}}))
    a,b=scan(root)
    assert b['scenes_with_labels']==18 and b['damaged']==9
    man,report=prepare(root,Path(t)/'work',max_train_neg=100,log=lambda s:None)
    rows=load_rows(man)
    assert len(rows)==36
    results=train_selected(Path(t)/'work',['random_forest'],log=lambda s:None)
    assert results[0]['validation']['average_precision'] is not None
    report_file=make_report(Path(t)/'work',log=lambda s:None)
    assert report_file.is_file() and (report_file.parent/'precision_recall.png').is_file()
    final=test_winner(Path(t)/'work',log=lambda s:None)
    assert final['selected_model']=='random_forest'
    test_scene='mexico-earthquake_00000000'
    img=root/'tier1'/'images';lab=root/'tier1'/'labels'
    r=predict_scene(Path(t)/'work',img/f'{test_scene}_pre_disaster.tif',img/f'{test_scene}_post_disaster.tif',lab/f'{test_scene}_post_disaster.json',log=lambda s:None)
    assert sum(r['counts'].values())==2
    print('✓ SMOKE OK: 18 scenes; 36 polygons; RF trained; val/test generated; new-scene predictions saved.')
