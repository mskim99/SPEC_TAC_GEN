import os,argparse,json,math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from main_unified import UNet,Diffusion,str2bool

# ----------------------------
def load_model(ckpt_path,device="cuda",base_ch=32,timesteps=300,
               conditional=False,num_classes=0):
    model=UNet(in_ch=1,out_ch=1,base_ch=base_ch,
               conditional=conditional,num_classes=num_classes).to(device)
    diffusion=Diffusion(timesteps,device)
    ckpt=torch.load(ckpt_path,map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    print(f"✅ Loaded ckpt {ckpt_path}, epoch={ckpt['epoch']}")
    return model,diffusion,ckpt

@torch.no_grad()
def sample(model,diffusion,shape,device,is_phase,norm_type,min_db,max_db,
           db_mu,db_sigma,conditional=False,cond=None):
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
        if norm_type=="zscore": spec_db=img*db_sigma+db_mu
        elif norm_type=="0to1": spec_db=img*(max_db-min_db)+min_db
        else: spec_db=(img+1)/2*(max_db-min_db)+min_db
        return 10**(spec_db/20.0)

# ----------------------------
def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir,exist_ok=True)
    num_classes=0; cond_map=None
    if args.conditional:
        cond_map=json.load(open(os.path.join(os.path.dirname(args.ckpt),"cond2idx.json")))
        num_classes=len(cond_map)
    model,diff,ckpt=load_model(args.ckpt,device,args.base_ch,args.timesteps,args.conditional,num_classes)
    norm_type=ckpt.get("norm_type","zscore")
    min_db,max_db=ckpt.get("min_db",-80),ckpt.get("max_db",0)
    db_mu,db_sigma=ckpt.get("db_mu",None),ckpt.get("db_sigma",None)
    is_phase=ckpt.get("is_phase",args.is_phase)
    print(f"✅ Norm from ckpt: {norm_type}, mu={db_mu}, sigma={db_sigma}")
    if args.conditional:
        if args.all_conditions: conditions=cond_map.keys()
        else:
            if args.condition not in cond_map: raise ValueError("Unknown cond")
            conditions=[args.condition]
    else: conditions=[None]
    for cname in conditions:
        cond_idx=None;out_dir=args.out_dir
        if args.conditional:
            cond_idx=cond_map[cname];out_dir=os.path.join(args.out_dir,cname);os.makedirs(out_dir,exist_ok=True)
        for i in range(args.num_samples):
            spec=sample(model,diff,(1,2,129,376) if is_phase else (1,1,129,376),
                        device,is_phase,norm_type,min_db,max_db,db_mu,db_sigma,
                        args.conditional,cond_idx)
            np.save(f"{out_dir}/sample_{i+1:03d}.npy",spec)
            if is_phase: plt.imsave(f"{out_dir}/sample_{i+1:03d}.png",spec,cmap="jet")
            else: plt.imsave(f"{out_dir}/sample_{i+1:03d}.png",20*np.log10(spec+1e-8),cmap="jet")
        print(f"Saved {args.num_samples} samples to {out_dir}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--is_phase",type=str2bool,default=False)
    p.add_argument("--conditional",action="store_true")
    p.add_argument("--ckpt",type=str,required=True)
    p.add_argument("--out_dir",type=str,default="gen_samples")
    p.add_argument("--num_samples",type=int,default=16)
    p.add_argument("--timesteps",type=int,default=300)
    p.add_argument("--base_ch",type=int,default=32)
    p.add_argument("--condition",type=str,default=None)
    p.add_argument("--all_conditions",action="store_true")
    args=p.parse_args(); main(args)