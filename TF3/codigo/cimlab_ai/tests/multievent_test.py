"""Tests multi-event discovery and preparation using SYNTHETIC xBD-like data."""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from event_sources import local_events,selected_roots,github_events
from data_core import scan,prepare,load_rows

with tempfile.TemporaryDirectory() as tmp:
    data=Path(tmp)/'repo'/'data'
    for event in ('mexico-earthquake','palu-tsunami'):
        base=data/event
        for i in range(16):
            part='tier1' if i<8 else ('hold' if i<12 else 'test')
            img=base/part/'images';lab=base/part/'labels'
            img.mkdir(parents=True,exist_ok=True);lab.mkdir(parents=True,exist_ok=True)
            name=f'{event}_{i:08d}'
            pre=Image.new('RGB',(120,120),'#aaaaaa');post=pre.copy()
            im=ImageDraw.Draw(pre);pm=ImageDraw.Draw(post)
            im.rectangle((20,20,80,80),fill='gray')
            damaged=i%2==0
            pm.rectangle((20,20,80,80),fill='red' if damaged else 'gray')
            pre.save(img/(name+'_pre_disaster.tif'))
            post.save(img/(name+'_post_disaster.tif'))
            labels={'features':{'xy':[{'wkt':'POLYGON ((20 20, 80 20, 80 80, 20 80, 20 20))',
                'properties':{'uid':f'{event}-{i}','subtype':(['minor-damage','major-damage','destroyed'][i%3] if damaged else 'no-damage')}}]}}
            (lab/(name+'_post_disaster.json')).write_text(json.dumps(labels))
    catalog=local_events(data.parent)
    assert set(catalog)=={'mexico-earthquake','palu-tsunami'},catalog
    assert len(selected_roots(data.parent,['palu-tsunami']))==1
    rows,info=scan(data.parent,events=['palu-tsunami'])
    assert len(rows)==16 and info['buildings']==16 and info['damaged']==8
    combined,info=scan(data.parent,events=['mexico-earthquake','palu-tsunami'])
    assert len(combined)==32 and len(set(r['scene'] for r in combined))==32
    assert set(r['original_label'] for r in combined)=={'no-damage','minor-damage','major-damage','destroyed'}
    assert all(int(r['label'])==int(r['original_label']!='no-damage') for r in combined)
    manifest,_=prepare(data.parent,Path(tmp)/'work',events=['mexico-earthquake','palu-tsunami'],max_train_neg=100)
    saved=load_rows(manifest)
    assert len(saved)==32
    assert {r['label'] for r in saved}=={0,1}
    assert (Path(tmp)/'work'/'dataset_meta.json').is_file()
    assert len({r['scene'] for r in saved})==32
    assert set(r['split'] for r in saved)=={'train','val','test'}
    assert all(('tier1' in r['pre'] and r['split']=='train') or
               ('hold' in r['pre'] and r['split']=='val') or
               ('test' in r['pre'] and r['split']=='test') for r in saved)
    assert all(Path(r['crop_pre']).is_file() and Path(r['crop_post']).is_file() for r in saved)
    # The event list is LIVE; test the API decoder with a mocked response.
    import io
    class FakeResponse:
        def __enter__(self):return io.BytesIO(json.dumps([
            {'name':'mexico-earthquake','type':'dir'},
            {'name':'palu-tsunami','type':'dir'},
            {'name':'random.txt','type':'file'}]).encode())
        def __exit__(self,*args):pass
    with patch('urllib.request.urlopen',return_value=FakeResponse()):
        assert github_events()==['mexico-earthquake','palu-tsunami']
    print('✓ MULTIEVENT OK: 2 events, 32 pairs, CSV + crops, dynamic GitHub catalog mocked.')
