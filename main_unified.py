import os, glob, math, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("Expected boolean value.")

# --------------------
# Dataset
# --------------------
def compute_db_stats(files, min_db, max_db):
    vals = []
    for f in files:
        x = np.load(f)
        db = 20*np.log10(x+1e-8)
        db = np.clip(db, min_db, max_db)
        vals.append(db)
    vals = np.stack(vals,0)
    return float(vals.mean()), float(vals.std()+1e-12)

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, is_phase=False, min_db=-80, max_db=0,
                 conditional=False, norm_type="zscore", db_mu=None, db_sigma=None):
        self.files = glob.glob(os.path.join(data_dir,"*.npy"))
        self.is_phase = is_phase
        self.min_db, self.max_db = min_db, max_db
        self.conditional, self.norm_type = conditional, norm_type

        if self.conditional:
            self.conditions = [os.path.basename(f).split("_")[0] for f in self.files]
            unique = sorted(set(self.conditions))
            self.cond2idx = {c:i for i,c in enumerate(unique)}

        if not self.is_phase and self.norm_type=="zscore":
            if db_mu is None or db_sigma is None:
                self.db_mu, self.db_sigma = compute_db_stats(self.files,self.min_db,self.max_db)
            else:
                self.db_mu, self.db_sigma = db_mu, db_sigma
        else:
            self.db_mu,self.db_sigma=None,None

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])
        if self.is_phase:
            phase=spec
            spec_norm=np.stack([np.sin(phase),np.cos(phase)],axis=0)
        else:
            db=20*np.log10(spec+1e-8)
            db=np.clip(db,self.min_db,self.max_db)
            if self.norm_type=="zscore":
                spec_norm=(db-self.db_mu)/self.db_sigma
            else:
                spec_01=(db-self.min_db)/(self.max_db-self.min_db+1e-12)
                spec_norm=spec_01 if self.norm_type=="0to1" else (spec_01*2-1)
            spec_norm=np.expand_dims(spec_norm,0)
        spec_norm=torch.tensor(spec_norm,dtype=torch.float32)
        if self.conditional:
            cond_idx=self.cond2idx[self.conditions[idx]]
            return spec_norm,cond_idx
        return spec_norm

# --------------------
# UNet (동일)
# --------------------
class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,1,1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,1,1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

def sinusoidal_embedding(t,dim=128):
    device=t.device;half=dim//2
    emb=torch.log(torch.tensor(10000.0))/ (half-1)
    emb=torch.exp(torch.arange(half,device=device)*-emb)
    emb=t[:,None].float()*emb[None,:]
    return torch.cat([torch.sin(emb),torch.cos(emb)],1)

class UNet(nn.Module):
    def __init__(self,in_ch=1,out_ch=1,base_ch=32,conditional=False,num_classes=0,emb_dim=128):
        super().__init__()
        self.conditional=conditional
        self.enc1=ConvBlock(in_ch,base_ch);self.pool1=nn.MaxPool2d(2)
        self.enc2=ConvBlock(base_ch,base_ch*2);self.pool2=nn.MaxPool2d(2)
        self.enc3=ConvBlock(base_ch*2,base_ch*4);self.pool3=nn.MaxPool2d(2)
        self.bottleneck=ConvBlock(base_ch*4,base_ch*8)
        self.up3=nn.ConvTranspose2d(base_ch*8,base_ch*4,2,2)
        self.dec3=ConvBlock(base_ch*8,base_ch*4)
        self.up2=nn.ConvTranspose2d(base_ch*4,base_ch*2,2,2)
        self.dec2=ConvBlock(base_ch*4,base_ch*2)
        self.up1=nn.ConvTranspose2d(base_ch*2,base_ch,2,2)
        self.dec1=ConvBlock(base_ch*2,base_ch)
        self.out_conv=nn.Conv2d(base_ch,out_ch,1)
        if self.conditional:
            self.cond_emb=nn.Embedding(num_classes,emb_dim)
            self.fc=nn.Linear(emb_dim*2,base_ch*8)
    def forward(self,x,t=None,cond=None):
        e1=self.enc1(x);p1=self.pool1(e1)
        e2=self.enc2(p1);p2=self.pool2(e2)
        e3=self.enc3(p2);p3=self.pool3(e3)
        b=self.bottleneck(p3)
        if self.conditional and t is not None and cond is not None:
            t_emb=sinusoidal_embedding(t,dim=self.cond_emb.embedding_dim)
            c_emb=self.cond_emb(cond)
            joint=torch.cat([t_emb,c_emb],1)
            joint_vec=self.fc(joint)[:,:,None,None]
            b=b+joint_vec
        u3=self.up3(b);u3=F.interpolate(u3,size=e3.shape[2:],mode="bilinear",align_corners=False)
        d3=self.dec3(torch.cat([u3,e3],1))
        u2=self.up2(d3);u2=F.interpolate(u2,size=e2.shape[2:],mode="bilinear",align_corners=False)
        d2=self.dec2(torch.cat([u2,e2],1))
        u1=self.up1(d2);u1=F.interpolate(u1,size=e1.shape[2:],mode="bilinear",align_corners=False)
        d1=self.dec1(torch.cat([u1,e1],1))
        return self.out_conv(d1)

# --------------------
# Diffusion
# --------------------
def linear_beta_schedule(timesteps,start=1e-4,end=0.02): return torch.linspace(start,end,timesteps)
class Diffusion:
    def __init__(self,timesteps=300,device="cuda"):
        self.device=device;self.timesteps=timesteps
        self.betas=linear_beta_schedule(timesteps).to(device)
        self.alphas=1.-self.betas;self.alpha_hat=torch.cumprod(self.alphas,0)
    def add_noise(self,x0,t):
        sa=torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        so=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        noise=torch.randn_like(x0)
        return sa*x0+so*noise,noise

# --------------------
# Train
# --------------------
def train_diffusion(is_phase,conditional,data_dir,epochs=10,batch_size=16,lr=1e-4,
                    timesteps=300,base_ch=32,ckpt_dir="checkpoints",
                    device="cuda",save_interval=5,norm_type="zscore",min_db=-80,max_db=0):
    os.makedirs(ckpt_dir,exist_ok=True)
    dataset=SpectrogramDataset(data_dir,is_phase, min_db,max_db, conditional,norm_type)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    num_classes=len(dataset.cond2idx) if conditional else 0
    if conditional:
        with open(os.path.join(ckpt_dir,"cond2idx.json"),"w") as f: json.dump(dataset.cond2idx,f,indent=4)
    in_ch=2 if is_phase else 1; out_ch=2 if is_phase else 1
    model=UNet(in_ch,out_ch,base_ch,conditional,num_classes).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=lr); mse=nn.MSELoss()
    diffusion=Diffusion(timesteps,device)
    for epoch in range(epochs):
        for step,batch in enumerate(dataloader):
            if conditional: x,cond=batch; x,cond=x.to(device),cond.to(device)
            else: x=batch.to(device); cond=None
            t=torch.randint(0,diffusion.timesteps,(x.shape[0],),device=device).long()
            noisy,noise=diffusion.add_noise(x,t)
            pred=model(noisy,t,cond) if conditional else model(noisy,t)
            loss=mse(pred,noise)
            optim.zero_grad(); loss.backward(); optim.step()
        print(f"Epoch {epoch+1}/{epochs} Loss {loss.item():.4f}")
        if (epoch+1)%save_interval==0 or (epoch+1)==epochs:
            torch.save({
                "epoch":epoch+1,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optim.state_dict(),
                "loss":loss.item(),
                "is_phase":is_phase,
                "conditional":conditional,
                "norm_type":norm_type,
                "min_db":min_db,"max_db":max_db,
                "db_mu":dataset.db_mu,"db_sigma":dataset.db_sigma
            },os.path.join(ckpt_dir,f"ckpt_epoch_{epoch+1:04d}.pt"))
    return model,diffusion

# --------------------
# Sample (역변환 포함)
# --------------------
@torch.no_grad()
def sample(model,diffusion,shape=(1,1,129,376),device="cuda",
           min_db=-80,max_db=0,is_phase=False,conditional=False,cond=None,
           norm_type="zscore",db_mu=None,db_sigma=None):
    img=torch.randn(shape,device=device)
    for t in reversed(range(diffusion.timesteps)):
        tt=torch.tensor([t],device=device).long()
        pred=model(img,tt,cond) if conditional and cond is not None else model(img,tt)
        a=diffusion.alphas[t];ah=diffusion.alpha_hat[t];b=diffusion.betas[t]
        noise=torch.randn_like(img) if t>0 else torch.zeros_like(img)
        img=1/torch.sqrt(a)*(img-((1-a)/torch.sqrt(1-ah))*pred)+torch.sqrt(b)*noise
    img=img.squeeze().cpu().numpy()
    if is_phase:
        return np.arctan2(img[0],img[1])
    else:
        if norm_type=="zscore":
            spec_db=img*db_sigma+db_mu
        elif norm_type=="0to1":
            spec_db=img*(max_db-min_db)+min_db
        else:
            spec_db=(img+1)/2*(max_db-min_db)+min_db
        return 10**(spec_db/20.0)

# --------------------
# Main
# --------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--is_phase",type=str2bool,default=False)
    p.add_argument("--conditional",action="store_true")
    p.add_argument("--data_dir",type=str,default="data/magnitude/")
    p.add_argument("--ckpt_dir",type=str,default="checkpoints/magnitude/")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--timesteps",type=int,default=300)
    p.add_argument("--base_ch",type=int,default=32)
    p.add_argument("--save_interval",type=int,default=10)
    p.add_argument("--norm_type",type=str,choices=["zscore","0to1","-1to1"],default="zscore")
    p.add_argument("--min_db",type=float,default=-80.0)
    p.add_argument("--max_db",type=float,default=0.0)
    args=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.data_dir,exist_ok=True); os.makedirs(args.ckpt_dir,exist_ok=True)
    model,diff=train_diffusion(args.is_phase,args.conditional,args.data_dir,
        args.epochs,args.batch_size,args.lr,args.timesteps,args.base_ch,
        args.ckpt_dir,device,args.save_interval,args.norm_type,args.min_db,args.max_db)
    shape=(1,2,129,376) if args.is_phase else (1,1,129,376)
    cond=torch.tensor([0],device=device).long() if args.conditional else None
    gen=sample(model,diff,shape,device,args.min_db,args.max_db,args.is_phase,
               args.conditional,cond,args.norm_type,
               db_mu=None,db_sigma=None)
    plt.imshow(20*np.log10(gen+1e-8) if not args.is_phase else gen,aspect="auto",origin="lower",cmap="jet")
    plt.colorbar(); plt.savefig("generated_example.png",dpi=300)