import numpy as np
from pyproj import CRS
from pyproj.transformer import TransformerGroup
lon,lat,h=-75.28,38.41,-25.5
for s,name in ((4979,'WGS84 ens'),(9057,'WGS84 G1762'),(7912,'ITRF2014'),(9990,'ITRF2020')):
    g=TransformerGroup(CRS.from_epsg(s),CRS.from_epsg(6319),always_xy=True)
    t=g.transformers[0]
    for ep in (2010.0,2024.87):
        x=t.transform(lon,lat,h,ep)
        print(name,ep,[f'{v:.7f}' for v in x], 'dh=%.4f'%(x[2]-h),'dE,dN(m)=%.3f %.3f'%((x[0]-lon)*111320*np.cos(np.radians(lat)),(x[1]-lat)*110574))
    print(t.definition[:600])
