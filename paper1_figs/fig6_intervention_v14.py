"""MSBQA paper 1, Fig. 6 (v14 labels): paired per-spin (P) / uniform-max (U) detuning intervention.
From Jiapei's reproduction/draw_revision_v14.py: runs the plotting block of fig6_intervention.py with the v14
labels and the #514 marker. Reads run_v4/phase1d/per_instance_1d.csv. Run from msbqa_phase1/:
    python paper1_figs/fig6_intervention_v14.py
Writes paper1_figs/fig6_intervention_v14.pdf / .png.
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent          # msbqa_phase1/paper1_figs
phase = HERE.parent                             # msbqa_phase1
out = HERE
sys.path.insert(0, str(HERE))
# Reuse only the original plotting block with its existing CSV, grouping and jitter.
# No spectrum, dynamics, fitting or regression code is executed.
import p1common as pc
d=pd.read_csv(phase/'run_v4/phase1d/per_instance_1d.csv').set_index('inst')
gp=d[d['group'].notna() & d.Dmin_K.notna() & d.Dmin_U.notna()].copy();gp['lr']=np.log(gp.Dmin_U/gp.Dmin_K)
GROUPS=[('KO',pc.RED,'KO: crossing removed'),('MO',pc.BLUE,'MO: crossing created'),('BOTH',pc.PURPLE,'BOTH'),('CTRL',pc.GREY,'CTRL: no crossing')]
LAY=[('L2g','L2'),('L3','L3'),('L4','L4')]
plot=(phase/'paper1_figs/fig6_intervention.py').read_text().split('plt = pc.setup_mpl()',1)[1].split('pc.save(fig,',1)[0]
plot='plt = pc.setup_mpl()'+plot
plot=plot.replace('uniform detuning','uniform-max detuning').replace(r'\Delta_U/\Delta_K',r'\Delta_U/\Delta_P').replace('label="uniform"','label="uniform-max"')
plot=plot.replace('fig.tight_layout(w_pad=0.8)','')
exec(plot)
# Recover precisely the existing jittered position of #514, retaining its original point.
mo=gp[gp.group=='MO'];idx=list(mo.index).index(514);xy=ax[1].collections[1].get_offsets()[idx]
xy0=(gp.loc[514,'Dmin_K'],gp.loc[514,'Dmin_U'])
for aa,point,offset in [(ax[0],xy0,(-20,-15)),(ax[1],xy,(13,-8))]:
 aa.scatter([point[0]],[point[1]],s=37,facecolors='none',edgecolors='black',linewidths=.7,zorder=6)
 aa.annotate('#514',point,xytext=offset,textcoords='offset points',fontsize=6,arrowprops=dict(arrowstyle='-',lw=.5),zorder=7)
fig.tight_layout(w_pad=.8)
for ext in ['pdf','png']:fig.savefig(out/('fig6_intervention_v14.'+ext),dpi=300,bbox_inches='tight')
plt.close(fig)
