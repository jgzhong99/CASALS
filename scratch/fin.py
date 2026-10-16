import json
q=r'notebooks/L1B/geolocation/vertical_datum_alignment.ipynb'
raw=open(q,encoding='utf8').read()
raw=raw.replace('H1_bias_as_is','H2_bias_as_is')
nb=json.loads(raw)
n=0
for c in nb['cells']:
    if c['cell_type']!='code': continue
    for o in c.get('outputs',[]):
        d=o.get('data')
        if d and 'text/html' in d and 'text/plain' in d:
            del d['text/html']; n+=1
json.dump(nb,open(q,'w',encoding='utf8'),indent=1,ensure_ascii=False)
import os;print(n,os.path.getsize(q))
