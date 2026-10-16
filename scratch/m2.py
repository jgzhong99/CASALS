import re
p='research/reference/frame_hypotheses.py'
s=open(p,encoding='utf8').read()
a=s.index("                             VDatum_height_shift_m=")
b=s.index("                             status=res[\"status\"]))")
s=s[:a]+s[b:]
s=s.replace('df = pd.DataFrame(rows).drop(columns="VDatum_height_shift_m")','df = pd.DataFrame(rows)')
open(p,'w',encoding='utf8').write(s)
