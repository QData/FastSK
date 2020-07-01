import glob
from fastsk import FastSK
datasets = []
for filepath in glob.iglob('../../data/*.train.fasta'):
    s = filepath[:filepath.find(".train")]
    datasets.append(s)
    #print(s)

for s in datasets:
    name = s[s.find("/")+1:]
    # Uncomment for confirmation for each dataset train
    '''
    s = input("about to train on " + name + " dataset.\ny/n?")
    if s.lower() != "y":
        continue
    '''
    fastsk = FastSK(g=7, m=3, t=16, approx=True, max_iters=75) 
    fastsk.compute_kernel(Xtrain=s+".train.fasta", Xtest="../../data/" + name  + ".test.fasta")
    fastsk.fit()
    fastsk.score(metric='auc')
