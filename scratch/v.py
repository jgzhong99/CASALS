import requests, json
p={"s_x":"-75.28","s_y":"38.41","s_z":"-25.5","region":"chesapeak_delaware","s_h_frame":"ITRF2014","s_coor":"geo","s_v_frame":"ITRF2014","s_v_unit":"m","s_v_elevation":"height","s_v_geoid":"geoid18","t_h_frame":"NAD83_2011","t_coor":"geo","t_v_frame":"NAVD88","t_v_unit":"m","t_v_elevation":"height","t_v_geoid":"geoid18","epoch_in":"2024.8653","epoch_out":"2010.0"}
r=requests.get("https://vdatum.noaa.gov/vdatumweb/api/convert",params=p,timeout=30);print(r.status_code,r.text[:900])
p2=dict(p,s_h_frame="NAD83_2011",s_v_frame="NAD83_2011",epoch_in="2010.0")
r=requests.get("https://vdatum.noaa.gov/vdatumweb/api/convert",params=p2,timeout=30);print(r.text[:500])
