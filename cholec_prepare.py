"""Trich frame 1fps tu video Cholec80, gan nhan phase (tu file timestamp).
Luu jpg de fine-tune 7-lop. video 25fps -> lay moi 25 frame."""
import os, glob, csv, cv2, numpy as np
VID='data/cholec80_raw/ext/videos'; ANN='data/cholec80_raw/ext/videos'
OUT='data/cholec_frames'; FPS_SAMPLE=25   # lay 1 frame moi 25 (1fps)
PHASES=['Preparation','CalotTriangleDissection','ClippingCutting','GallbladderDissection',
        'GallbladderPackaging','CleaningCoagulation','GallbladderRetraction']
PID={p:i for i,p in enumerate(PHASES)}
os.makedirs(OUT,exist_ok=True)
vids=sorted(glob.glob(VID+'/video*.mp4'))
print('so video:',len(vids))
for vp in vids:
    name=os.path.splitext(os.path.basename(vp))[0]
    ann=os.path.join(ANN,name+'-timestamp.txt')
    if not os.path.exists(ann): print('thieu nhan',name); continue
    # doc nhan: dong i (tu 0) = frame i
    phases=[]
    with open(ann) as f:
        next(f)
        for line in f:
            parts=line.rstrip('\n').split('\t')
            if len(parts)>=2: phases.append(parts[1])
    vd=os.path.join(OUT,name)
    if os.path.isdir(vd) and len(os.listdir(vd))>50: print('skip',name); continue
    os.makedirs(vd,exist_ok=True)
    cap=cv2.VideoCapture(vp); idx=0; saved=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if idx%FPS_SAMPLE==0 and idx<len(phases):
            ph=phases[idx]
            if ph in PID:
                cv2.imwrite(os.path.join(vd,'f%06d_p%d.jpg'%(idx,PID[ph])),
                            cv2.resize(fr,(250,250)))
                saved+=1
        idx+=1
    cap.release()
    print('%s: %d frame (1fps), %d frame goc'%(name,saved,idx),flush=True)
print('[*] xong ->',OUT)
