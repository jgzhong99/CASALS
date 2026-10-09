import h5py
import numpy as np
import pytest

from research.geolocation.bin_georeferencing_audit.run_audit import (
    bin_to_xyz, inventory, read_values, statistics, xyz, TO_GEO, beam, angle,
    diagnostics, FIELDS,
)


def test_centers_edges_fractional_and_reverse():
    a, z = np.array([100.,200.,300.]), np.array([110.,220.,270.])
    np.testing.assert_equal(bin_to_xyz(a,z,[0,10],11,convention='H1'),[a,z])
    np.testing.assert_equal(bin_to_xyz(a,z,[0,10],11,convention='H3'),[z,a])
    np.testing.assert_allclose(bin_to_xyz(a,z,4.25,11,convention='H1'),a+.425*(z-a))
    np.testing.assert_allclose(bin_to_xyz(a,z,[0,10],11),[a+.5/11*(z-a),a+10.5/11*(z-a)])
    b=np.array([0.,.4,2.8,8.3,10.])
    points=bin_to_xyz(a,z,b,11)
    np.testing.assert_allclose(np.diff(points,axis=0)/np.diff(b)[:,None],np.broadcast_to((z-a)/11,(4,3)))


@pytest.mark.parametrize('bad',[-1,11,np.nan,np.inf])
def test_invalid_bin(bad):
    with pytest.raises(ValueError):bin_to_xyz([0,0,0],[1,2,3],bad,11)


@pytest.mark.parametrize('n',[0,1,2.5,True])
def test_invalid_size(n):
    with pytest.raises(ValueError):bin_to_xyz([0,0,0],[1,2,3],0,n)


def test_invalid_geometry_and_convention():
    for a,z in [([0,0,0],[0,0,0]),([np.nan,0,0],[1,2,3]),([0,0],[1,2])]:
        with pytest.raises(ValueError):bin_to_xyz(a,z,0,11)
    with pytest.raises(ValueError):bin_to_xyz([0,0,0],[1,2,3],0,11,convention='H7')


def test_geodetic_roundtrip_and_bounds():
    p=xyz(np.array([-76.]),np.array([39.]),np.array([25.]))
    np.testing.assert_allclose(np.array(TO_GEO.transform(*p.T)).ravel(),[-76,39,25],atol=1e-7)
    assert np.isnan(xyz(np.array([181.]),np.array([39.]),np.array([25.]))).all()


def test_nested_schema_fill_and_scale(tmp_path):
    with h5py.File(tmp_path/'test.h5','w') as h:
        d=h.create_dataset('nested/refh',data=[0.,-999.,2.])
        d.attrs['_FillValue']=-999.;d.attrs['scale_factor']=2.;d.attrs['add_offset']=1.
        inv,paths=inventory(h)
        assert paths['refh']=='nested/refh'
        assert inv['objects']['/nested/refh']['shape']==[3]
        np.testing.assert_allclose(read_values(d,slice(None)),[1,np.nan,5],equal_nan=True)
        h.create_dataset('other/refh',data=[1])
        with pytest.raises(ValueError,match='Ambiguous'):inventory(h)


def test_stats_do_not_hide_rejection():
    s=statistics(np.array([1.,2.,np.nan,3.]))
    assert s['n_total']==4 and s['n_valid']==3 and s['n_rejected']==1
    assert s['median']==2 and s['NMAD']==pytest.approx(1.4826)


def test_enu_sign_and_tiny_angles():
    d=beam(np.array([0.]),np.array([0.]),np.array([0.]),np.array([np.pi/2]))
    np.testing.assert_allclose(d,[[-1,0,0]],atol=1e-15)
    assert angle(d,-d)[0]==pytest.approx(180)


def test_geometry_separate_from_raw_peak_and_segment_domain():
    f={name:np.full(2,np.nan) for name in FIELDS}
    f.update(sweep_num=np.array([0,1]),track_num=np.array([0,0]),
             rwstart=np.array([100.,100.]),rwstop=np.array([0.,0.]),
             refh=np.array([50.,150.]),instrument_altitude=np.array([1000.,1000.]),
             refh_snr=np.array([20.,20.]),bin_size=np.array([20.,20.]),
             local_beam_azimuth=np.array([0.,0.]),local_beam_elevation=np.full(2,np.pi/2),
             rwstart_bounce_time_offset=np.zeros(2),rwstop_bounce_time_offset=np.full(2,2.5e-9),
             refh_bounce_time_offset=np.full(2,1.25e-9))
    for p in ['rwstart','rwstop','refh','instrument']:
        f[p+'_longitude']=np.zeros(2);f[p+'_latitude']=np.zeros(2)
    raw=np.array([[0,1,3,1,0],[0,1,3,3,0]],float)
    d=diagnostics(f,raw,np.array([0,1]),5,{'speed_of_light':299792458.})
    assert d.raw_argmax_bin.tolist()==[2,2]
    assert d.tie_count.tolist()==[1,2]
    assert d.high_quality.tolist()==[True,False]
    assert d.outside_segment.tolist()==[False,True]
    assert d.refh_line_distance_m.max()<1e-8
    assert d.refh_segment_distance_m.iloc[1]==pytest.approx(50,abs=1e-7)
    assert d.H2_3d_m.iloc[0]<1e-8
    assert d.H2_3d_m.iloc[1]==pytest.approx(100,abs=1e-7)
