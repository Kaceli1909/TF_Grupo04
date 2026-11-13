"""Binary damage detection experiments (baseline, DINOv2, ResNet18 pair models).

Methodological safeguards: scene-disjoint held-out set, validation-only threshold
selection, no fake metrics, test evaluated only for validation-selected model.
"""
from __future__ import annotations
import csv
import json
import copy
import hashlib
import math
import random
from pathlib import Path
import numpy as np
from data_core import load_rows, extract_feature_table, logfn
from labels_core import require_task, TASK_ID

SEED=42

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    except ImportError:pass


def metrics(y,prob,threshold=.5):
    from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_fscore_support,balanced_accuracy_score
    y=np.asarray(y,dtype=int);p=np.asarray(prob,dtype=float)
    pred=(p>=threshold).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=[1],zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'threshold':round(float(threshold),5),'precision_damage':round(float(pr[0]),5),
            'recall_damage':round(float(re[0]),5),'f1_damage':round(float(f1[0]),5),
            'average_precision':round(float(average_precision_score(y,p)),5) if len(np.unique(y))>1 else None,
            'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),5) if len(np.unique(y))>1 else None,
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'total':int(len(y)),'positives':int(y.sum())}


def choose_threshold(y,prob):
    """Optimizes F1 on validation set ONLY, with fixed deterministic tie-breaker."""
    y=np.asarray(y);prob=np.asarray(prob)
    best=(-1.,.5)
    if len(np.unique(y))<2:return .5
    for t in np.arange(.05,.951,.025):
        m=metrics(y,prob,t)
        if m['f1_damage']>best[0]:best=(m['f1_damage'],float(t))
    return best[1]


def save_pred(path,rows,y,prob):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['scene','uid','label','p_damage'])
        for r,yy,pp in zip(rows,y,prob):w.writerow([r['scene'],r['uid'],int(yy),round(float(pp),8)])


def check_splits(rows):
    splits={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    for s,x in splits.items():
        if not x:raise RuntimeError(f'El conjunto {s} está vacío.')
        if len(set(r['label'] for r in x))<2:raise RuntimeError(f'El conjunto {s} solo contiene una clase. Cambia la semilla y revisa las escenas.')
    if any(set(x['scene'] for x in splits[a]) & set(x['scene'] for x in splits[b])
           for a,b in [('train','val'),('train','test'),('val','test')]):
        raise RuntimeError('FUGA DE DATOS: escena repetida entre splits.')
    return splits


def save_result(workspace,name,model_filename,ys,ps,rows,extra=None):
    workspace=Path(workspace);out=workspace/'resultados';out.mkdir(exist_ok=True,parents=True)
    t=choose_threshold(ys,ps)
    m=metrics(ys,ps,t)
    data={'model':name,'model_path':str(model_filename),'task':TASK_ID,'validation':m, 'extra':extra or {}}
    (out/(name+'_val.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    save_pred(out/(name+'_val.csv'),rows,ys,ps)
    return data


def train_rf(splits,workspace,log=None):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    workspace=Path(workspace);modeldir=workspace/'modelos';modeldir.mkdir(parents=True,exist_ok=True)
    logfn(log,'Random Forest: extrayendo características de píxeles PRE/POST')
    a,b,ra=extract_feature_table(splits['train'],log)
    c,d,rb=extract_feature_table(splits['val'],log)
    rf=RandomForestClassifier(n_estimators=250,max_depth=18,min_samples_leaf=2,
                              class_weight='balanced_subsample',n_jobs=-1,random_state=SEED)
    rf.fit(a,b);p=rf.predict_proba(c)[:,list(rf.classes_).index(1)]
    path=modeldir/'random_forest.joblib';joblib.dump(rf,path)
    return save_result(workspace,'random_forest',path,d,p,rb,{'features':'visuales PRE/POST; sin área/uid'})


def torch_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_pair_tensors(pre,post):
    from PIL import Image
    import torch
    with Image.open(pre) as a,Image.open(post) as b:
        x=np.asarray(a.convert('RGB').resize((224,224)),dtype=np.float32).copy()/255.
        y=np.asarray(b.convert('RGB').resize((224,224)),dtype=np.float32).copy()/255.
    mean=np.array([.485,.456,.406],dtype=np.float32)
    std=np.array([.229,.224,.225],dtype=np.float32)
    a=torch.from_numpy(((x-mean)/std).transpose(2,0,1).copy())
    b=torch.from_numpy(((y-mean)/std).transpose(2,0,1).copy())
    return a,b


def build_cnn(name,pretrained=True):
    import torch
    from torch import nn
    from torchvision.models import resnet18,ResNet18_Weights
    weights=ResNet18_Weights.DEFAULT if pretrained else None
    if name=='resnet18_early':
        backbone=resnet18(weights=weights)
        old=backbone.conv1
        new=nn.Conv2d(6,64,kernel_size=7,stride=2,padding=3,bias=False)
        with torch.no_grad():
            if pretrained:
                new.weight[:,:3]=old.weight*.5
                new.weight[:,3:]=old.weight*.5
        backbone.conv1=new
        backbone.fc=nn.Sequential(nn.Dropout(.35),nn.Linear(512,1))
        return backbone
    if name=='siamese_resnet18':
        class Siamese(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=resnet18(weights=weights)
                self.encoder.fc=nn.Identity()
                self.head=nn.Sequential(nn.Linear(512*3,128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,1))
            def forward(self,x):
                # x = Nx6xHxW with two 3-channel images
                pre=self.encoder(x[:,:3]);post=self.encoder(x[:,3:])
                return self.head(torch.cat([pre,post,torch.abs(post-pre)],dim=1))
        return Siamese()
    raise ValueError('Modelo CNN no reconocido: '+name)


def train_cnn(name,splits,workspace,epochs=20,batch_size=8,log=None):
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
    workspace=Path(workspace);modeldir=workspace/'modelos';modeldir.mkdir(parents=True,exist_ok=True)
    device=torch.device(torch_device());logfn(log,f'{name}: usando {device}')
    if device.type=='cpu':logfn(log,'AVISO: no se detecta CUDA; el entrenamiento será mucho más lento.')
    class PairDataset(Dataset):
        def __init__(self,rs,augment=False):self.rs=rs;self.aug=augment
        def __len__(self):return len(self.rs)
        def __getitem__(self,i):
            r=self.rs[i];a,b=load_pair_tensors(r['crop_pre'],r['crop_post'])
            if self.aug:
                # SAME geometric transform for PRE and POST: avoid inventing change.
                if random.random()<.5:a=a.flip(2);b=b.flip(2)
                if random.random()<.5:a=a.flip(1);b=b.flip(1)
                k=random.randrange(4);a=a.rot90(k,(1,2));b=b.rot90(k,(1,2))
            return torch.cat([a,b],dim=0),torch.tensor(float(r['label']))
    # Weighted sampler takes positives often enough to learn without duplicating val/test.
    tr=PairDataset(splits['train'],True);va=PairDataset(splits['val'])
    labs=np.array([r['label'] for r in splits['train']]);ns=np.bincount(labs,minlength=2)
    ws=np.array([1/max(1,ns[v]) for v in labs]);sampler=WeightedRandomSampler(ws,len(tr),replacement=True)
    dl=DataLoader(tr,batch_size=batch_size,sampler=sampler,num_workers=0)
    vl=DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=0)
    model=build_cnn(name,True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=.01)
    lossfn=nn.BCEWithLogitsLoss()
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda'))
    best=-1.;best_model=None;no_improve=0;history=[]
    for ep in range(1,int(epochs)+1):
        model.train();loss_sum=0.
        for xx,yy in dl:
            xx=xx.to(device);yy=yy.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                output=model(xx).flatten();loss=lossfn(output,yy)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            loss_sum+=float(loss.detach().cpu())*len(yy)
        model.eval();prob=[];true=[]
        with torch.inference_mode():
            for xx,yy in vl:
                xx=xx.to(device)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    out=model(xx).flatten().float().sigmoid().cpu().numpy()
                prob.extend(out);true.extend(yy.numpy())
        m=metrics(true,prob,choose_threshold(true,prob))
        score=m['average_precision'] or 0.
        history.append({'epoch':ep,'train_loss':round(loss_sum/len(tr),5),'val_ap':score,'val_f1':m['f1_damage']})
        logfn(log,f'{name}: época {ep}/{epochs}, pérdida {loss_sum/len(tr):.4f}, AP val {score:.4f}, F1 val {m["f1_damage"]:.4f}')
        if score>best+1e-5:
            best=score;no_improve=0;best_model=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            no_improve+=1
            if no_improve>=5:logfn(log,'Early stopping.');break
    if best_model is None:raise RuntimeError('No fue posible entrenar la CNN.')
    model.load_state_dict(best_model);model.to(device).eval()
    # validation predictions only for best checkpoint
    prob=[];true=[]
    with torch.inference_mode():
        for xx,yy in vl:
            xx=xx.to(device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                out=model(xx).flatten().float().sigmoid().cpu().numpy()
            prob.extend(out);true.extend(yy.numpy())
    path=modeldir/(name+'.pt')
    torch.save({'state':best_model,'model':name,'image_size':224,'pretrained_source':'torchvision imagenet'},path)
    return save_result(workspace,name,path,true,prob,splits['val'],{'epochs_run':len(history),'history':history,'device':str(device)})


def dino_extract(rows,workspace,split,log=None,batch_size=12):
    """Features from frozen DINOv2-S/14, computed once and cached as NPZ."""
    import torch
    from transformers import AutoImageProcessor,AutoModel
    from PIL import Image
    out=Path(workspace)/'features';out.mkdir(parents=True,exist_ok=True)
    cache=out/f'dinov2_{split}.npz'
    # Do not reuse embeddings for different pixels just because building UIDs coincide.
    cache_keys=[]
    for r in rows:
        cache_keys.append('|'.join(f'{Path(r[f]).resolve()}:{Path(r[f]).stat().st_size}:{Path(r[f]).stat().st_mtime_ns}' for f in ('crop_pre','crop_post')))
    fingerprint=hashlib.sha256(('\n'.join(cache_keys)).encode('utf-8')).hexdigest()
    if cache.exists():
        c=np.load(cache)
        if len(c['y'])==len(rows) and str(c.get('fingerprint',''))==fingerprint:
            logfn(log,'DINOv2: reutilizando características '+str(cache))
            return c['x'],c['y']
    device=torch.device(torch_device());logfn(log,'Descargando/cargando DINOv2-small (requiere Internet la primera vez)')
    processor=AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model=AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
    xx=[];yy=[]
    for i in range(0,len(rows),batch_size):
        batch=rows[i:i+batch_size];imgs=[]
        for r in batch:
            with Image.open(r['crop_pre']) as a,Image.open(r['crop_post']) as b:
                imgs.extend([a.convert('RGB').copy(),b.convert('RGB').copy()])
        inp=processor(images=imgs,return_tensors='pt')
        inp={k:v.to(device) for k,v in inp.items()}
        with torch.inference_mode():
            emb=model(**inp).last_hidden_state[:,0].float().cpu().numpy()
        a=emb[0::2];b=emb[1::2]
        xx.extend(np.concatenate([a,b,np.abs(a-b)],axis=1))
        yy.extend(r['label'] for r in batch)
        if (i//batch_size)%40==0:logfn(log,f'DINOv2 {split}: {min(len(rows),i+batch_size)}/{len(rows)}')
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    x=np.asarray(xx,dtype=np.float32);y=np.asarray(yy,dtype=int)
    np.savez_compressed(cache,x=x,y=y,ids=np.asarray([r['uid'] for r in rows],dtype=str),fingerprint=fingerprint)
    return x,y


def train_dino(splits,workspace,log=None):
    from xgboost import XGBClassifier
    import joblib
    workspace=Path(workspace);modeldir=workspace/'modelos';modeldir.mkdir(parents=True,exist_ok=True)
    a,b=dino_extract(splits['train'],workspace,'train',log)
    c,d=dino_extract(splits['val'],workspace,'val',log)
    scale=min(30.,max(1.,float((b==0).sum())/max(1,(b==1).sum())))
    model=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.05,subsample=.85,
                        colsample_bytree=.85,reg_lambda=3.,scale_pos_weight=scale,
                        eval_metric='logloss',n_jobs=4,tree_method='hist',random_state=SEED)
    model.fit(a,b);p=model.predict_proba(c)[:,1]
    path=modeldir/'dinov2_xgboost.joblib';joblib.dump(model,path)
    return save_result(workspace,'dinov2_xgboost',path,d,p,splits['val'],{'feature_model':'facebook/dinov2-small','frozen':True})


def make_ensemble(workspace,log=None):
    """Simple soft-vote; only if >=2 trained models with same validation IDs."""
    work=Path(workspace);preds=[];names=[]
    for p in sorted((work/'resultados').glob('*_val.csv')):
        if p.stem=='hybrid_val':continue
        with p.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        key=[(r['scene'],r['uid']) for r in rows]
        # Other algorithms may have skipped failed files: require identical manifests.
        if preds and key!=preds[0][0]:
            logfn(log,f'Híbrido: omitido {p.stem}, sus edificios no coinciden.');continue
        vals=np.array([float(r['p_damage']) for r in rows]);y=np.array([int(r['label']) for r in rows])
        preds.append((key,vals,y,rows));names.append(p.stem.replace('_val',''))
    if len(preds)<2:
        logfn(log,'Híbrido requiere al menos dos modelos con las mismas predicciones de validación.');return None
    y=preds[0][2]
    # Equal weight soft vote, no optimization on held-out test.
    prob=np.mean([p[1] for p in preds],axis=0)
    rows=[{'scene':r['scene'],'uid':r['uid']} for r in preds[0][3]]
    result=save_result(work,'hybrid',work/'resultados/hybrid_config.json',y,prob,rows,{'members':names,'method':'equal_probability_mean'})
    (work/'resultados/hybrid_config.json').write_text(json.dumps({'members':names,'method':'mean'},indent=2),encoding='utf-8')
    return result


def infer_cnn(name,path,rows,batch_size=8,log=None):
    import torch
    from torch.utils.data import DataLoader,Dataset
    device=torch.device(torch_device())
    model=build_cnn(name,False)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state']);model.to(device).eval()
    class Pairs(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            a,b=load_pair_tensors(rows[i]['crop_pre'],rows[i]['crop_post'])
            return torch.cat([a,b],0)
    ps=[]
    with torch.inference_mode():
        for xx in DataLoader(Pairs(),batch_size=batch_size,num_workers=0):
            v=model(xx.to(device)).flatten().float().sigmoid().cpu().numpy()
            ps.extend(v)
    return np.array(ps)


def predict_prob(name,model_path,rows,workspace,log=None):
    import joblib
    if name=='random_forest':
        x,y,ids=extract_feature_table(rows,log)
        if len(ids)!=len(rows):raise RuntimeError('Faltan recortes para predecir.')
        m=joblib.load(model_path)
        return m.predict_proba(x)[:,list(m.classes_).index(1)]
    if name=='dinov2_xgboost':
        # cache key from row ids for training splits; arbitrary new rows need unique key
        key='test' if all(r['split']=='test' for r in rows) else 'new'
        x,y=dino_extract(rows,workspace,key,log)
        return joblib.load(model_path).predict_proba(x)[:,1]
    if name in ('siamese_resnet18','resnet18_early'):
        return infer_cnn(name,model_path,rows,log=log)
    if name in ('siamese_unet_lite','changeos_lite','dahitra_lite'):
        from advanced_core import infer_advanced
        return infer_advanced(name,model_path,rows,log=log)
    if name=='protonet_dino':
        from advanced_core import infer_protonet
        return infer_protonet(model_path,rows,workspace,log)
    raise ValueError('Modelo desconocido: '+name)


def test_winner(workspace,log=None):
    """Select highest validation AP (tie F1). Test only selected candidate."""
    work=Path(workspace);require_task(work);splits=check_splits(load_rows(work/'manifest.csv'))
    if (work/'resultados'/'evaluacion_final.json').exists():
        raise RuntimeError('Prueba final ya ejecutada: no repetir selección usando el conjunto de test.')
    reports=[]
    for path in sorted((work/'resultados').glob('*_val.json')):
        j=json.loads(path.read_text(encoding='utf-8'))
        if j.get('validation',{}).get('average_precision') is not None:reports.append(j)
    if not reports:raise RuntimeError('No hay resultados de validación. Entrena uno o más modelos.')
    best=max(reports,key=lambda j:(j['validation']['average_precision'],j['validation']['f1_damage']))
    chosen=best['model'];rows=splits['test'];y=[r['label'] for r in rows]
    logfn(log,'GANADOR según validación: '+chosen+' (sin consultar antes test)')
    if chosen=='hybrid':
        members=best['extra']['members'];vals=[]
        for m in members:
            j=json.loads((work/'resultados'/(m+'_val.json')).read_text(encoding='utf-8'))
            vals.append(predict_prob(m,j['model_path'],rows,work,log))
        prob=np.mean(vals,axis=0)
    else:
        prob=predict_prob(chosen,best['model_path'],rows,work,log)
    m=metrics(y,prob,best['validation']['threshold'])
    result={'selected_model':chosen,'task':TASK_ID,'selection_metric':'validation average_precision',
            'validation':best['validation'],'test':m,'split_policy':'scene-disjoint; test once after selecting'}
    (work/'resultados'/'evaluacion_final.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    save_pred(work/'resultados'/'evaluacion_final_test.csv',rows,y,prob)
    logfn(log,'PRUEBA FINAL: '+json.dumps(m,ensure_ascii=False))
    return result


def train_selected(workspace,names,epochs=20,batch_size=8,log=None):
    seed_all()
    work=Path(workspace);require_task(work);splits=check_splits(load_rows(work/'manifest.csv'))
    res=[]
    for name in names:
        logfn(log,'\n'+'='*22+' '+name+' '+'='*22)
        if name=='random_forest':r=train_rf(splits,work,log)
        elif name=='dinov2_xgboost':r=train_dino(splits,work,log)
        elif name in ('resnet18_early','siamese_resnet18'):r=train_cnn(name,splits,work,epochs,batch_size,log)
        elif name in ('siamese_unet_lite','changeos_lite','dahitra_lite'):
            from advanced_core import train_segment_or_class
            r=train_segment_or_class(name,splits,work,epochs,batch_size,log)
        elif name=='protonet_dino':
            from advanced_core import train_protonet
            r=train_protonet(splits,work,epochs,log)
        else:raise ValueError('Modelo inválido: '+name)
        logfn(log,'✓ Validación: '+json.dumps(r['validation'],ensure_ascii=False))
        res.append(r)
    if len(list((work/'resultados').glob('*_val.csv')))>=2:
        e=make_ensemble(work,log)
        if e:res.append(e)
    # Test intentionally not executed here: user chooses when finalizing model comparison.
    return res
