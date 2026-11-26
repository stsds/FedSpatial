#!/usr/bin/env python3
"""Fresh-native P-only Figure-1 alpha/range boxplots."""
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
SC=[(.5,.033),(.5,.1),(.5,.234),(1.5,.037),(1.5,.11),(1.5,.256),(2.5,.038),(2.5,.113),(2.5,.264)]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); rows=list(csv.DictReader((a.root/'native_p_estimates.csv').open()))
 fig,axes=plt.subplots(3,6,figsize=(18,9),constrained_layout=True)
 for s,(nu,beta) in enumerate(SC):
  sub=[x for x in rows if int(x['scenario_id'])==s]
  for c,(field,truth) in enumerate((('alpha',1.),('beta',beta))):
   ax=axes[s//3,(s%3)*2+c]; ax.boxplot([[float(x[field]) for x in sub]],labels=['P']); ax.axhline(truth,color='red',linewidth=1); ax.set_title(f'nu={nu}, beta={beta:g}: {field}')
 fig.savefig(a.root/'figure1_two_worker_lowrank.png',dpi=180); fig.savefig(a.root/'figure1_two_worker_lowrank.pdf')
 (a.root/'plot_manifest.json').write_text(json.dumps({'source':'native_p_estimates.csv','method':'P only','cases':180,'samples_per_scenario':20,'truth_lines':'alpha=1; beta=scenario truth'},indent=2)+'\n')
if __name__=='__main__': main()
