import csv
from pathlib import Path
rows=[]
for nu,betas in ((.5,(.033,.1,.234)),(1.5,(.037,.11,.256)),(2.5,(.038,.113,.264))):
 for beta in betas: rows.append([len(rows),nu,beta,1.,.25,1000,10,100,100,100,50,'IND;Low-rank'])
out=Path('results/figure1_cpp_gpu_faithful'); out.mkdir(parents=True,exist_ok=True)
with (out/'figure1_scenarios.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['scenario','nu','beta','alpha','delta','n','workers','n_local','knots','replications','iterations','methods']);w.writerows(rows)
print('SCENARIOS',len(rows))
