import json

best_auc = 0
with open('nr2c2.results.out', 'r') as f:
    for line in f:
        line = line.strip()
        result = json.loads(line.replace("'", "\""))
        auc = result['auc']
        best_auc = auc if auc > best_auc else best_auc
print("best auc = ", best_auc)
