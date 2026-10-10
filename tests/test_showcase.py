import h5py
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from research.showcase import transect_coordinates, sample_raster, enu_frame, to_enu, from_enu, bin_to_xyz
from casals_l1b.waveform import build_record_index_grid, infer_waveform_record_axis, read_waveform_records


def test_transect_rotation_and_extent():
    s,d = transect_coordinates([[1,1],[0,2],[3,3],[-1,-1]], [0,0],[2,2])
    np.testing.assert_allclose(s,[np.sqrt(2),np.sqrt(2),3*np.sqrt(2),-np.sqrt(2)])
    np.testing.assert_allclose(d,[0,np.sqrt(2),0,0],atol=1e-14)
    with pytest.raises(ValueError):
        transect_coordinates([[0,0]],[0,0],[0,0])


def test_raster_crs_nodata_and_support(tmp_path):
    xy = np.array([[500005,4000015],[500015,4000015],[500005,4000005],[499999,4000005]])
    paths=[]
    for name,arr in [('surface',[[10,-9999],[30,40]]),('support',[[1,1],[0,1]])]:
        path=tmp_path/(name+'.tif');paths.append(path)
        with rasterio.open(path,'w',driver='GTiff',height=2,width=2,count=1,dtype='float32',
                           crs=32618,transform=from_origin(500000,4000020,10,10),nodata=-9999) as ds:
            ds.write(np.array(arr,dtype='float32'),1)
    lonlat=np.column_stack(Transformer.from_crs(32618,4326,always_xy=True).transform(*xy.T))
    v,valid=sample_raster(paths[0],lonlat,4326,paths[1])
    np.testing.assert_equal(valid,[True,False,False,False])
    np.testing.assert_allclose(v,[10,np.nan,np.nan,np.nan],equal_nan=True)


def test_reordered_pulse_identity_and_axis(tmp_path):
    with h5py.File(tmp_path/'x.h5','w') as f:
        f['sweep_num']=[1,0,1,0];f['track_num']=[0,1,1,0]
        f['rx_waveform']=np.arange(20).reshape(5,4)
        grid=build_record_index_grid(f)
        assert grid.record_index_grid[0,0]==3
        assert grid.record_index_grid[1,0]==0
        axis=infer_waveform_record_axis(f['rx_waveform'],4)
        assert axis==1
        np.testing.assert_equal(read_waveform_records(f['rx_waveform'],axis,[3,0]),np.arange(20).reshape(5,4)[:,[3,0]].T)


def test_h1_h2_and_ecef_enu_roundtrip():
    start=np.array([1e6,2e6,3e6]);stop=start+[40,60,-90]
    np.testing.assert_allclose(bin_to_xyz(start,stop,3,8,convention='H1'),start+3/7*(stop-start))
    np.testing.assert_allclose(bin_to_xyz(start,stop,3,8),start+3.5/8*(stop-start))
    origin,rotation=enu_frame(-76.69,36.19,-20)
    xyz=np.vstack([origin,origin+[150,210,1100],origin+[-120,440,-80]])
    np.testing.assert_allclose(from_enu(to_enu(xyz,origin,rotation),origin,rotation),xyz,rtol=0,atol=2e-9)
