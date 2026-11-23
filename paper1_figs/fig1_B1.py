"""MSBQA paper 1, Fig. 1 (schematic, no data): comparison framework and crossing correspondence.
From Jiapei's reproduction/draw_figure1.py (version B1), output moved to paper1_figs/. Run from anywhere:
    python paper1_figs/fig1_B1.py
Writes paper1_figs/fig1_B1.pdf / .png / .svg.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
ROOT=Path(__file__).parent
OUT=ROOT
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
purple='#644978';blue='#235a7b';green='#416b57';gold='#a46c2e';ink='#202833';muted='#586474'
def build(version):
 fig=plt.figure(figsize=(6.8,6.3));gs=fig.add_gridspec(2,2,height_ratios=[1.05,1],hspace=.38,wspace=.40,left=.115,right=.98,top=.96,bottom=.085)
 ax=fig.add_subplot(gs[0,:]);a1=fig.add_subplot(gs[1,0]);a2=fig.add_subplot(gs[1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 def box(x,y,w,h,title,sub,color,fill,ts=9,ss=8):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=.012',fc=fill,ec=color,lw=.9))
  ax.text(x+w/2,y+h*.75,title,ha='center',va='center',fontsize=ts,fontweight='bold',color=color)
  ax.text(x+w/2,y+h*.32,sub,ha='center',va='center',fontsize=ss,color=muted,linespacing=1.4)
 def arrow(p,q):ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,lw=.85,color=muted))
 ax.text(-.045,1.015,'(a)',transform=ax.transAxes,fontsize=13,fontweight='bold')
 ax.text(.5,.97,'Same Ising objective',ha='center',va='center',fontsize=11,fontweight='bold',color=ink)
 rows=[('Initialization','L0','Quantum reference','L1 / L2','Wigner / uniform',blue,'#edf3f7'),('Dynamical rules','L2','Classical KPO','L3 / L4','Modified forces / walls',green,'#eff5f1'),('Algorithmic\nperformance','L0','Quantum reference','bSB* / dSB*','Independent tuning',gold,'#faf4eb')]
 if version=='B1':
  ax.text(.015,.81,'Comparison',fontsize=8,color=muted)
  ax.text(.45,.81,'Reference',ha='center',fontsize=8,color=muted)
  ax.text(.80,.81,'Counterpart',ha='center',fontsize=8,color=muted)
  ax.plot([0,1],[.77,.77],c=ink,lw=.9)
  for i,(title,l,ls,r,rs,col,fill) in enumerate(rows):
   y=.64-i*.23
   ax.text(.015,y,title,va='center',fontsize=9,fontweight='bold',color=col,linespacing=1.2)
   ax.text(.45,y+.035,l,ha='center',va='center',fontsize=10,fontweight='bold',color=ink)
   ax.text(.45,y-.045,ls,ha='center',va='center',fontsize=8,color=muted)
   ax.text(.80,y+.035,r,ha='center',va='center',fontsize=10,fontweight='bold',color=ink)
   ax.text(.80,y-.045,rs,ha='center',va='center',fontsize=8,color=muted)
   ax.plot([0,1],[y-.12,y-.12],c='#b9bec4',lw=.6)
 elif version=='B2':
  for i,(title,l,ls,r,rs,col,fill) in enumerate(rows):
   y=.64-i*.255
   ax.add_patch(FancyBboxPatch((.005,y-.015),.99,.22,boxstyle='round,pad=0.003,rounding_size=.012',facecolor=fill,edgecolor='none'))
   ax.plot([.012,.012],[y+.005,y+.185],c=col,lw=3)
   ax.text(.035,y+.095,title,va='center',fontsize=9,fontweight='bold',color=col,linespacing=1.25)
   ax.text(.45,y+.13,l,ha='center',va='center',fontsize=10,fontweight='bold',color=ink)
   ax.text(.45,y+.045,ls,ha='center',va='center',fontsize=8,color=muted)
   ax.plot([.60,.65],[y+.09,y+.09],c=col,lw=1)
   ax.text(.81,y+.13,r,ha='center',va='center',fontsize=10,fontweight='bold',color=ink)
   ax.text(.81,y+.045,rs,ha='center',va='center',fontsize=8,color=muted)
 else:
  for i,(title,l,ls,r,rs,col,fill) in enumerate(rows):
   y=.65-i*.28
   title=title.replace('\n',' ')
   ax.text(.02,y+.22,title,ha='left',va='center',fontsize=9,fontweight='bold',color=ink)
   box(.025,y,.36,.16,l,ls,ink,'white',ss=8)
   box(.615,y,.36,.16,r,rs,ink,'white',ss=8)
   ax.plot([.40,.60],[y+.08,y+.08],c=ink,lw=.8)
   ax.text(.5,y+.08,'compare',ha='center',va='center',fontsize=7.5,color=ink,bbox=dict(facecolor='white',edgecolor='none',pad=2))
 ax.text(.5,.005,'Instance-resolved comparisons of final Ising success',ha='center',va='center',fontsize=8,color=muted)
 # Same illustrative functions as approved v6/v10 Figure 1, never measured data.
 p=np.linspace(0,4,400);pc=2.2;Et=np.where(p>=.5,-.25*(p-.5)**2,np.nan);Eg=np.where(p>=.8,Et+.3*(pc-p),np.nan)
 a1.plot(p,Et,c='#C0392B',lw=1.5,label='competing cfg.');a1.plot(p,Eg,c='black',lw=1.5,label='ground cfg.')
 a1.axvline(pc,c=ink,ls=':',lw=.9);a1.text(pc+.08,.23,r'$p_{\rm c}$',fontsize=11)
 a1.set(xlim=(0,4),xlabel=r'pump $p$',ylabel=r'classical $E_{\mathbf{s}}(p)$');a1.set_xticks([]);a1.set_yticks([])
 a1.legend(loc='lower left',frameon=False,fontsize=8,handlelength=1.5);a1.text(-.18,1.03,'(b)',transform=a1.transAxes,fontsize=13,fontweight='bold')
 for h,col in ((1.,purple),(.5,blue),(.25,green)):
  ps=pc+.8*h;gap=.5*np.exp(-.9/h);e=.3*(p-ps);lvl=np.sqrt(e**2+gap**2)
  a2.plot(p,-lvl,c=col,lw=1.4,label=fr'$h={h:g}$');a2.plot(p,+lvl,c=col,lw=1.4,ls='--')
 a2.axvline(pc,c=ink,ls=':',lw=.9);a2.set(xlim=(0,4),ylim=(-.8,.8),xlabel=r'pump $p$',ylabel=r'quantum $E_0,E_1$');a2.set_xticks([]);a2.set_yticks([])
 a2.text(-.18,1.03,'(c)',transform=a2.transAxes,fontsize=13,fontweight='bold');a2.text(pc+.12,-.7,r'$p_{\rm c}$',fontsize=10,color=ink)
 a2.legend(loc='lower center',bbox_to_anchor=(.52,1.04),ncol=3,frameon=False,fontsize=8,handlelength=1.05,columnspacing=.5,handletextpad=.25)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'fig1_{version}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
for v in ['B1']:build(v)
print(OUT)
