"""CIMLAB v0.2 research extensions: small, documented *adaptations*, not
reproductions of ChangeOS/DAHiTrA research code/weights.

All experiments use existing scene-disjoint manifest; no test during training.
Siamese U-Net-lite: instance-mask conditioned binary damage segmentation.
ChangeOS-lite: object-aware bi-temporal classifier + building-mask auxiliary task.
DAHiTrA-lite: frozen multi-scale paired ResNet features + transformer fusion.
ProtoNet: episodically optimized projection over *frozen DINOv2* embeddings.

Scores are NOT calibrated probabilities, and an instance mask is assumed known.
"""
from __future__ import annotations
import copy, json, math, random
from pathlib import Path
import numpy as np
from models_core import SEED, metrics, choose_threshold, save_result, dino_extract, torch_device
from data_core import logfn

ADVANCED_MODELS = ('siamese_unet_lite','changeos_lite','dahitra_lite','protonet_dino')
ADVANCED_LABELS = {
    'siamese_unet_lite':'Siamese U-Net ligera (adaptación)',
    'changeos_lite':'ChangeOS-like orientado a objetos (adaptación)',
    'dahitra_lite':'DAHiTrA-like jerárquico (adaptación)',
    'protonet_dino':'ProtoNet episodica + DINOv2 congelado',
}


def polygon_mask(row, side=224):
    """Reproduce EXACTAMENTE crop_pair's bounding box transform (pad=12).
    The crop JPEG is 224px; mask itself is freshly rasterized, never guessed.
    """
    from shapely import wkt
    from PIL import Image, ImageDraw
    g=wkt.loads(row['wkt'])
    xmin,ymin,xmax,ymax=[float(row[k]) for k in ('x0','y0','x1','y1')]
    cx=(xmin+xmax)/2;cy=(ymin+ymax)/2
    span=max(xmax-xmin+24,ymax-ymin+24,32)
    left=math.floor(cx-span/2);top=math.floor(cy-span/2)
    right=math.ceil(cx+span/2);bottom=math.ceil(cy+span/2)
    sx=side/(right-left);sy=side/(bottom-top)
    mask=Image.new('L',(side,side),0);d=ImageDraw.Draw(mask)
    parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
    for poly in parts:
        xy=[((x-left)*sx,(y-top)*sy) for x,y in poly.exterior.coords]
        d.polygon(xy,fill=255)
        for hole in poly.interiors:
            xy=[((x-left)*sx,(y-top)*sy) for x,y in hole.coords]
            d.polygon(xy,fill=0)
    out=np.asarray(mask,dtype=np.float32).copy()/255.
    if float(out.sum())<3:
        raise ValueError('Polígono no visible en el recorte (revisar transformación)')
    return out


def pair_encoder(pretrained=True):
    from torchvision.models import resnet18,ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)


def features(model,x):
    """ResNet18 feature maps at 56, 28, 14 and 7 pixel resolution."""
    x=model.relu(model.bn1(model.conv1(x)))
    x=model.maxpool(x)
    a=model.layer1(x);b=model.layer2(a);c=model.layer3(b);d=model.layer4(c)
    return a,b,c,d


def fuse(a,b):
    import torch
    return torch.cat((a,b,torch.abs(b-a)),dim=1)


def conv_block(cin,cout):
    from torch import nn
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),
                         nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),
                         nn.BatchNorm2d(cout),nn.ReLU(inplace=True))


def build_advanced(name,pretrained=True):
    import torch
    from torch import nn
    from torch.nn import functional as F
    if name=='siamese_unet_lite':
        class SiamUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=pair_encoder(pretrained)
                # Freeze first encoder stage to reduce memory on RTX 3060.
                for param in list(self.encoder.conv1.parameters())+list(self.encoder.bn1.parameters())+list(self.encoder.layer1.parameters()):
                    param.requires_grad=False
                self.reduce4=nn.Sequential(nn.Conv2d(1536,160,1),nn.ReLU(inplace=True))
                self.dec3=conv_block(160+768,128)
                self.dec2=conv_block(128+384,64)
                self.dec1=conv_block(64+192,48)
                self.out=nn.Conv2d(48,1,1)
            def forward(self,x,mask):
                p=features(self.encoder,x[:,:3]);q=features(self.encoder,x[:,3:])
                s1,s2,s3,s4=[fuse(a,b) for a,b in zip(p,q)]
                v=self.reduce4(s4)
                v=self.dec3(torch.cat([F.interpolate(v,size=s3.shape[-2:],mode='bilinear',align_corners=False),s3],dim=1))
                v=self.dec2(torch.cat([F.interpolate(v,size=s2.shape[-2:],mode='bilinear',align_corners=False),s2],dim=1))
                v=self.dec1(torch.cat([F.interpolate(v,size=s1.shape[-2:],mode='bilinear',align_corners=False),s1],dim=1))
                seg=self.out(v)  # [B,1,56,56]
                # Building damage score = logit pooled within user-provided footprint.
                roi=F.interpolate(mask,size=seg.shape[-2:],mode='nearest')
                scores=(seg*roi).sum((2,3))/roi.sum((2,3)).clamp_min(1.)
                return scores[:,0],seg
        return SiamUNet()
    if name=='changeos_lite':
        class ObjectAware(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=pair_encoder(pretrained)
                # use Early Fusion with 6 channels for pre/post.
                old=self.net.conv1
                new=nn.Conv2d(6,64,7,stride=2,padding=3,bias=False)
                if pretrained:
                    with torch.no_grad():new.weight.copy_(torch.cat([old.weight*.5,old.weight*.5],dim=1))
                self.net.conv1=new
                self.net.fc=nn.Identity()
                self.object_classifier=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Dropout(.35),nn.Linear(128,1))
                self.loc_head=nn.Conv2d(512,1,1)
            def forward(self,x,mask):
                _,_,_,v=features(self.net,x)
                roi=F.interpolate(mask,size=v.shape[-2:],mode='nearest')
                obj=(v*roi).sum((2,3))/roi.sum((2,3)).clamp_min(1.)
                ctx=v.mean((2,3))
                damage=self.object_classifier(torch.cat([obj,ctx],dim=1))[:,0]
                loc=self.loc_head(v)
                return damage,loc
        return ObjectAware()
    if name=='dahitra_lite':
        class MultiScaleTrans(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc=pair_encoder(pretrained)
                for p in self.enc.parameters():p.requires_grad=False
                self.proj3=nn.Conv2d(256*3,192,1)
                self.proj4=nn.Conv2d(512*3,192,1)
                self.cls=nn.Parameter(torch.zeros(1,1,192))
                self.pos=nn.Parameter(torch.zeros(1,66,192))
                layer=nn.TransformerEncoderLayer(d_model=192,nhead=4,dim_feedforward=384,
                                dropout=.15,batch_first=True,norm_first=True)
                self.trans=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
                self.head=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
            def forward(self,x,mask):
                with torch.no_grad():
                    a=features(self.enc,x[:,:3]);b=features(self.enc,x[:,3:])
                a3=F.adaptive_avg_pool2d(self.proj3(fuse(a[2],b[2])),(4,4))
                a4=F.adaptive_avg_pool2d(self.proj4(fuse(a[3],b[3])),(7,7))
                tokens=torch.cat([a3.flatten(2),a4.flatten(2)],dim=2).transpose(1,2)
                cls=self.cls.expand(tokens.shape[0],-1,-1)
                out=self.trans(torch.cat([cls,tokens],dim=1)+self.pos)
                return self.head(out[:,0])[:,0],None
        return MultiScaleTrans()
    raise ValueError('Arquitectura avanzada desconocida: '+str(name))


def train_segment_or_class(name,splits,workspace,epochs=20,batch_size=8,log=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from models_core import load_pair_tensors
    work=Path(workspace);modeldir=work/'modelos';modeldir.mkdir(parents=True,exist_ok=True)
    dev=torch.device(torch_device())
    # Validation is still COMPLETE. All original negatives preserved.
    class Pairs(Dataset):
        def __init__(self,rows,augment=False):self.rows=rows;self.augment=augment
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i];a,b=load_pair_tensors(r['crop_pre'],r['crop_post'])
            mask=torch.from_numpy(polygon_mask(r)).unsqueeze(0)
            if self.augment:
                if random.random()<.5:a=a.flip(-1);b=b.flip(-1);mask=mask.flip(-1)
                if random.random()<.5:a=a.flip(-2);b=b.flip(-2);mask=mask.flip(-2)
                k=random.randrange(4)
                a=a.rot90(k,(1,2));b=b.rot90(k,(1,2));mask=mask.rot90(k,(1,2))
            return torch.cat((a,b),dim=0),mask,torch.tensor(float(r['label']))
    train=Pairs(splits['train'],True);valid=Pairs(splits['val'])
    # Early stopping only: positives + a fixed deterministic subset of negatives.
    # Final validation ALWAYS uses the entire original validation split.
    rng=np.random.default_rng(SEED)
    negatives=[i for i,r in enumerate(splits['val']) if int(r['label'])==0]
    positive=[i for i,r in enumerate(splits['val']) if int(r['label'])==1]
    early_ids=sorted(positive+rng.choice(negatives,min(1000,len(negatives)),replace=False).tolist())
    valid_fast=Pairs([splits['val'][i] for i in early_ids])
    labs=np.asarray([r['label'] for r in splits['train']],dtype=int)
    ns=np.bincount(labs,minlength=2)
    sample=WeightedRandomSampler([1/max(ns[int(x)],1) for x in labs],len(labs),replacement=True)
    # 8GB VRAM: keep a safe per-architecture cap; no intentional full-GPU memory use.
    cap=2 if name=='siamese_unet_lite' else 4 if name=='changeos_lite' else 4
    effective_batch=min(int(batch_size),cap)
    logfn(log,f'{name} | dispositivo={dev} | lote={effective_batch} (limite de VRAM conservador)')
    tl=DataLoader(train,batch_size=effective_batch,sampler=sample,num_workers=0)
    vl=DataLoader(valid,batch_size=effective_batch,shuffle=False,num_workers=0)
    quick_val=DataLoader(valid_fast,batch_size=effective_batch,shuffle=False,num_workers=0)
    logfn(log,f'Early stopping con {len(valid_fast)} casos de validación; métricas finales con {len(valid)} completos')
    model=build_advanced(name,True).to(dev)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1.2e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    bce=nn.BCEWithLogitsLoss()
    best=-1.;best_state=None;bad=0;history=[]
    def forward_loss(xx,mask,yy):
        logits,spatial=model(xx,mask)
        clas=bce(logits,yy)
        if spatial is None:return clas,logits
        if name=='siamese_unet_lite':
            # Positive = building footprint with damage; no-damage = zero mask.
            target=F.interpolate(mask,size=spatial.shape[-2:],mode='nearest')*yy[:,None,None,None]
            segloss=F.binary_cross_entropy_with_logits(spatial,target)
            pr=spatial.sigmoid();intersection=(pr*target).sum((1,2,3))
            dice=1-(2*intersection+1)/(pr.sum((1,2,3))+target.sum((1,2,3))+1)
            return clas+.30*segloss+.10*dice.mean(),logits
        # ChangeOS-lite: auxiliary building localization from known footprints.
        target=F.interpolate(mask,size=spatial.shape[-2:],mode='nearest')
        return clas+.15*F.binary_cross_entropy_with_logits(spatial,target),logits
    for ep in range(1,int(epochs)+1):
        model.train();loss_sum=0.
        for xx,mask,yy in tl:
            xx=xx.to(dev);mask=mask.to(dev);yy=yy.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                loss,_=forward_loss(xx,mask,yy)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            loss_sum+=loss.item()*len(yy)
        model.eval();true=[];probs=[]
        with torch.inference_mode():
            for xx,mask,yy in quick_val:
                xx=xx.to(dev);mask=mask.to(dev)
                with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                    logits,_=model(xx,mask)
                probs.extend(logits.float().sigmoid().cpu().numpy().tolist());true.extend(yy.numpy().tolist())
        met=metrics(true,probs,choose_threshold(true,probs))
        ap=met['average_precision'] or 0.
        history.append(dict(epoch=ep,loss=round(loss_sum/len(train),5),val_ap=ap,val_f1=met['f1_damage']))
        logfn(log,f'{name}: epoca {ep}/{epochs}, perdida {loss_sum/len(train):.4f}, AP val {ap:.4f}, F1 {met["f1_damage"]:.4f}')
        if ap>best+1e-5:
            best=ap;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=5:
                logfn(log,'Early stopping (AP de validacion)');break
    if best_state is None:raise RuntimeError('No se obtuvo checkpoint entrenado')
    model.load_state_dict(best_state);model.eval()
    final=[];actual=[]
    with torch.inference_mode():
        for xx,mask,yy in vl:
            xx=xx.to(dev);mask=mask.to(dev)
            logits,_=model(xx,mask)
            final.extend(logits.float().sigmoid().cpu().numpy().tolist());actual.extend(yy.numpy().tolist())
    path=modeldir/(name+'.pt')
    torch.save(dict(model=name,state=best_state,architecture='CIMLAB-adapted-lite',version='0.2'),path)
    return save_result(work,name,path,actual,final,splits['val'],
        {'backbone':'ImageNet ResNet18','status':'adaptacion ligera; no pesos oficiales ni resultados de articulo',
         'history':history,'epochs_run':len(history),'effective_batch':effective_batch,'device':str(dev)})


def infer_advanced(name,path,rows,batch_size=4,log=None):
    import torch
    from torch.utils.data import DataLoader,Dataset
    from models_core import load_pair_tensors
    dev=torch.device(torch_device())
    class Pairs(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[i];a,b=load_pair_tensors(r['crop_pre'],r['crop_post'])
            return torch.cat((a,b),0),torch.from_numpy(polygon_mask(r)).unsqueeze(0)
    net=build_advanced(name,False)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('model')!=name:raise RuntimeError('Checkpoint incompatible con '+name)
    net.load_state_dict(ck['state']);net.to(dev).eval()
    vals=[]
    with torch.inference_mode():
        for x,mask in DataLoader(Pairs(),batch_size=min(4,batch_size),num_workers=0):
            x=x.to(dev);mask=mask.to(dev)
            y,_=net(x,mask);vals.extend(y.float().sigmoid().cpu().numpy().tolist())
    return np.asarray(vals)


# A legitimate episodic prototypical embedding learner. DINOv2 backbone frozen,
# supervised projection trained on real train-only 2-way support/query episodes.
def train_protonet(splits,workspace,epochs=20,log=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    work=Path(workspace);(work/'modelos').mkdir(parents=True,exist_ok=True)
    x,y=dino_extract(splits['train'],work,'train',log)
    xv,yv=dino_extract(splits['val'],work,'val',log)
    device=torch.device(torch_device());x=torch.tensor(x,dtype=torch.float32,device=device)
    xv=torch.tensor(xv,dtype=torch.float32,device=device)
    y=np.asarray(y);yv=np.asarray(yv)
    if np.min(np.bincount(y,minlength=2))<8:raise ValueError('ProtoNet requiere al menos ocho edificios reales de cada clase en train')
    class Projection(nn.Module):
        def __init__(self,indim):
            super().__init__();self.net=nn.Sequential(nn.LayerNorm(indim),nn.Linear(indim,256),nn.ReLU(),
                       nn.Dropout(.15),nn.Linear(256,64))
        def forward(self,inp):return F.normalize(self.net(inp),dim=-1)
    model=Projection(x.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.002)
    inds=[np.flatnonzero(y==c) for c in (0,1)]
    rng=np.random.default_rng(SEED)
    best=-1.;state=None;patience=0;history=[];proto_best=None
    for ep in range(1,int(epochs)+1):
        model.train();errors=[]
        for k in range(20):
            support=[];queries=[];qtrue=[]
            for cls in (0,1):
                ids=rng.choice(inds[cls],size=13,replace=False)
                support.append(torch.tensor(ids[:5],device=device))
                queries.append(torch.tensor(ids[5:],device=device))
                qtrue.extend([cls]*8)
            # Project only episodic samples, not all ~1,700 training buildings.
            prototypes=torch.stack([model(x[idx]).mean(0) for idx in support])
            q=torch.cat([model(x[idx]) for idx in queries])
            logit=-torch.cdist(q,prototypes).square()*8
            yy=torch.tensor(qtrue,device=device)
            loss=F.cross_entropy(logit,yy)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            errors.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            embed=model(x);p=torch.stack([embed[torch.tensor(ind,device=device)].mean(0) for ind in inds])
            logits=-torch.cdist(model(xv),p).square()*8
            probs=torch.softmax(logits,dim=-1)[:,1].cpu().numpy()
        m=metrics(yv,probs,choose_threshold(yv,probs));ap=m['average_precision'] or 0.
        history.append({'epoch':ep,'episodic_loss':round(float(np.mean(errors)),4),'val_ap':ap})
        logfn(log,f'protonet_dino: época {ep}/{epochs}, loss {np.mean(errors):.4f}, AP val {ap:.4f}')
        if ap>best+1e-5:
            best=ap;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            proto_best=p.detach().cpu().clone();patience=0
        else:
            patience+=1
            if patience>=5:break
    if state is None:raise RuntimeError('ProtoNet no produjo un modelo utilizable')
    model.load_state_dict(state);model.eval()
    with torch.inference_mode():
        z=torch.softmax(-torch.cdist(model(xv),proto_best.to(device)).square()*8,dim=-1)[:,1].cpu().numpy()
    path=work/'modelos'/'protonet_dino.pt'
    torch.save(dict(model='protonet_dino',state=state,prototypes=proto_best,input_dim=int(x.shape[1]),
                    backbone='facebook/dinov2-small',version='0.2'),path)
    return save_result(work,'protonet_dino',path,yv,z,splits['val'],
        {'method':'episodic prototypical projection; frozen DINOv2 embeddings', 'history':history,'epochs_run':len(history)})


def infer_protonet(path,rows,workspace,log=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from models_core import dino_extract
    class Projection(nn.Module):
        def __init__(self,dim):
            super().__init__();self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.ReLU(),
                           nn.Dropout(.15),nn.Linear(256,64))
        def forward(self,x):return F.normalize(self.net(x),dim=-1)
    inp,_=dino_extract(rows,workspace,'test' if all(r['split']=='test' for r in rows) else 'new_protonet',log)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=Projection(ck['input_dim']);model.load_state_dict(ck['state']);model.eval()
    with torch.inference_mode():
        z=model(torch.from_numpy(inp).float())
        prob=torch.softmax(-torch.cdist(z,ck['prototypes']).square()*8,dim=1)[:,1].numpy()
    return prob
