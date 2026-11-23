"""MSBQA paper 1, Fig. 7 (v15 labels): ablations, Ising-window holds, n = 4 vs n = 6.
From Jiapei's reproduction/draw_revision_v15.py: runs fig7_sb_n6.py with the v15 labels and the dotted separator
in (c). Run from msbqa_phase1/:
    python paper1_figs/fig7_sb_n6_v15.py
Writes paper1_figs/fig7_sb_n6_v15.pdf / .png.
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
# Existing CSV-to-figure aggregation only; no research simulation or fit.
plot=(phase/'paper1_figs/fig7_sb_n6.py').read_text(encoding='utf8')
plot=plot.replace('say = pc.Numbers("fig7_sb_n6")','say = lambda *args: None').split('pc.save(fig,',1)[0]
plot=plot.replace('L4 + one ingredient  |  bSB$^*$ $-$ one','Independent replacements: L4 + | bSB$^*$ -')
plot=plot.replace('("quantum",','("Q: propagation\\n/ LZ estimate",')
plot=plot.replace('uniform $\\\\bar\\\\Delta$','uniform-mean $\\\\bar\\\\Delta$')
plot=plot.replace('fig.tight_layout(w_pad=0.6)','ax[2].axvline(0.5, color=pc.GREY, lw=0.6, ls=":")\nfig.tight_layout(w_pad=0.8)')
exec(plot)
ax[1].set_xticklabels(['uniform-mean\n'+r'$\bar\Delta$'+' (N=17)','per-spin\n'+r'$\Delta_i$'+' (N=26)'],fontsize=6.5)
ax[2].set_xticklabels(['cross.\nfrac.',r'L2 $\varepsilon^*$',r'L4 $\varepsilon^*$',r'bSB$^*$','Q'],fontsize=6.2)
ax[2].set_xlabel('Q: n=4 propagation; n=6 LZ estimate',fontsize=5.8)
ax[2].lines[-1].set_ydata([0,.78])
fig.tight_layout(w_pad=.8)
for ext in ['pdf','png']:fig.savefig(out/('fig7_sb_n6_v15.'+ext),dpi=300,bbox_inches='tight')
plt.close(fig)
