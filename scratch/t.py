import numpy as np
from pyproj import datadir, network
from research.reference import vertical_datum_comparison as datum
from research.reference import frame_hypotheses as fh
grid = datum.OUTPUT_ROOT/"proj_data"/datum.GEOID18_GRID
network.set_network_enabled(False); datadir.set_data_dir(str(grid.parent))
lon=np.array([-75.28,-76.69]);lat=np.array([38.41,36.19]);h=np.array([-25.5,-32.4])
for n,(e,ep) in fh.HYPOTHESES.items():
    ep = fh.observation_epoch('casals_l1b_20241112T165718_001_02') if ep=='obs' else ep
    a,b,c,info=fh.apply_frame(e,ep,lon,lat,h)
    x,y,H,_=datum.transform_c7(a,b,c,source_epsg=6319)
    x0,y0,H0,_=datum.transform_c7(lon,lat,h)
    print(n,round(ep,4),info['operation'][:90],'dh',c-h,'dH vs noop',H-H0,'dxy',np.hypot(x-x0,y-y0))
