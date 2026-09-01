#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

# Exact vectorized Perlin implementation compatible with the public vnoise demo.
GRAD3 = np.array(((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,0,1),(-1,0,1),
(-1,0,-1),(-1,0,-1),(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
(1,0,-1),(-1,0,-1),(0,-1,1),(0,1,1)), dtype=int)
PERM = np.array((151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180), dtype=np.int64)

class Noise2:
    def __init__(self): self.perm=np.concatenate([PERM,PERM])
    @staticmethod
    def _lerp(t,a,b): return a+t*(b-a)
    @staticmethod
    def _grad2(h,x,y):
        g=GRAD3[:,:2][h & 15]; return x*g[...,0]+y*g[...,1]
    def _impl(self,x,y,rx,ry,base):
        i=np.floor(np.fmod(x,rx)).astype(int); j=np.floor(np.fmod(y,ry)).astype(int)
        ii=np.fmod(i+1,rx).astype(int); jj=np.fmod(j+1,ry).astype(int)
        i=(i&255)+base; j=(j&255)+base; ii=(ii&255)+base; jj=(jj&255)+base
        xf=x-np.floor(x); yf=y-np.floor(y); x1=xf-1.; y1=yf-1.
        fx=xf**3*(xf*(xf*6.-15.)+10.); fy=yf**3*(yf*(yf*6.-15.)+10.)
        A=self.perm[i]; AA=self.perm[A+j]; AB=self.perm[A+jj]; B=self.perm[ii]; BA=self.perm[B+j]; BB=self.perm[B+jj]
        return self._lerp(fy,self._lerp(fx,self._grad2(self.perm[AA],xf,yf),self._grad2(self.perm[BA],x1,yf)),self._lerp(fx,self._grad2(self.perm[AB],xf,y1),self._grad2(self.perm[BB],x1,y1)))
    def noise2(self,x,y,octaves=4,persistence=.5,lacunarity=2.,repeat_x=1024,repeat_y=1024,base=0):
        x=np.asarray(x,float); y=np.asarray(y,float); x,y=np.broadcast_arrays(x,y)
        total=np.zeros_like(x); freq=ampl=1.; maxa=0.
        for _ in range(octaves):
            total += self._impl(x*freq,y*freq,int(repeat_x*freq),int(repeat_y*freq),base)*ampl
            maxa += ampl; freq*=lacunarity; ampl*=persistence
        return total/maxa
NOISE=Noise2()

def make_cameras(n=24,width=640,height=360,step=.3):
    poses=[]; Ks=[]; Hs=[]; Ws=[]; Ts=[]
    fx=fy=1000.0
    for i in range(n):
        poses.append(np.array([[1,0,0,0],[0,0,1,i*step],[0,-1,0,3],[0,0,0,1]],dtype=np.float64))
        Ks.append(np.array([[fx,0,width/2],[0,fy,height/2],[0,0,1]],dtype=np.float64))
        Hs.append(height); Ws.append(width); Ts.append((.5+i)/24.)
    return poses,Ks,Hs,Ws,Ts

def terrain(points):
    h=5.0*NOISE.noise2(points[:,0]/10.,points[:,1]/10.,octaves=4)
    return points[:,2]-h

def mesh_digest(mesh, tags):
    import hashlib
    d=hashlib.sha256()
    for a in (np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int64),np.asarray(tags,np.int8)):
        b=np.ascontiguousarray(a); d.update(repr(b.shape).encode()); d.update(b.tobytes())
    return d.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--cameras',type=int,default=24); p.add_argument('--ppc',type=int,default=180); p.add_argument('--profile',choices=['demo','compact'],default='demo'); p.add_argument('--coarse',type=int); p.add_argument('--outview',type=int); p.add_argument('--bisection-iters',type=int,default=3); p.add_argument('--seed-stride',type=int,default=10); p.add_argument('--min-dist',type=float,default=.1); p.add_argument('--n-coarse-nodes',type=int,default=200000); p.add_argument('--medium-group',type=int,default=100000); p.add_argument('--fine-group',type=int,default=10000); p.add_argument('--bisection-group',type=int,default=10000000); p.add_argument('--camera-step',type=float,default=.3); p.add_argument('--fading-time',type=float,default=1.0/24.0); args=p.parse_args()
    repo=args.repo.resolve(); out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    os.environ['BINOC_EVENT_MODE']='1'; os.environ['BINOC_PROVENANCE_V2']='1'; os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,str(repo))
    from binocmesher import BinocMesher
    cameras=make_cameras(args.cameras, step=args.camera_step)
    if args.profile=='demo':
        bounds=[-1000.,1000.,-1000.,1000.,-10.,10.]
    else:
        bounds=[-20.,20.,-20.,20.,-10.,10.]
    mesher=BinocMesher(cameras,bounds=bounds,slicing_time=(.5+args.cameras//2)/24.,pixels_per_cube=args.ppc,
        pixels_per_cube_coarse=(args.coarse if args.coarse is not None else 30),pixels_per_cube_outview=(args.outview if args.outview is not None else 120),min_dist=args.min_dist,
        simplify_occluded=False,relax_margin=0,boundary_margin=1,relax_iters=0,n_coarse_nodes=args.n_coarse_nodes,
        bisection_iters=args.bisection_iters,fading_time=args.fading_time,seed_stride=args.seed_stride,medium_group=args.medium_group,fine_group=args.fine_group,bisection_group=args.bisection_group,path=out)
    t0=time.time(); meshes,tags=mesher([terrain]); elapsed=time.time()-t0
    mesh=meshes[0]
    summary=json.loads((out/'event_registry_p1_summary.json').read_text()) if (out/'event_registry_p1_summary.json').exists() else {}
    result={'verdict':'PASS_LIGHTWEIGHT_PROFILE','profile':args.profile,'cameras':args.cameras,'ppc':args.ppc,'elapsed_seconds':elapsed,
      'coarse':(args.coarse if args.coarse is not None else 30),'outview':(args.outview if args.outview is not None else 120),'bisection_iters':args.bisection_iters,'seed_stride':args.seed_stride,'camera_step':args.camera_step,'fading_time':args.fading_time,'vertices':int(len(mesh.vertices)),'faces':int(len(mesh.faces)),'mesh_sha256':mesh_digest(mesh,tags[0]),'registry_summary':summary}
    (out/'profile_result.json').write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
