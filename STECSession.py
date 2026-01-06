from __future__ import annotations
import traceback
import pandas as pd
import gnx_py as gnx
import logging
from datetime import datetime
import numpy as np
import gnx_py.time
from gnx_py import TECSession
import json
import os


OBS_PATH='./sample_data'
NAV = './sample_data/BRDC00IGS_R_20240350000_01D_MN.rnx'
SP3_LIST=['./sample_data/COD0OPSFIN_20240340000_01D_05M_ORB.SP3',
'./sample_data/COD0OPSFIN_20240350000_01D_05M_ORB.SP3',
'./sample_data/COD0OPSFIN_20240360000_01D_05M_ORB.SP3'
]
ATX_PATH='./sample_data/igs20.atx'
DCB_PATH= './sample_data/COD0OPSFIN_20240350000_01D_01D_OSB.BIA'
GIM='./sample_data/COD0OPSFIN_20240350000_01D_01H_GIM.INX'
SINEX='./sample_data/IGS0OPSSNX_20240350000_01D_01D_CRD.SNX'
OUT ='./sample_data/output'
os.path.exists(OUT) or os.makedirs(OUT)
TLIM = [datetime(2024,2,4,0,0,0),
        datetime(2024,2,4,23,59,30)]
MIN_INTERVAL = 30/60

USE_SYS='G'
F1 = 1575.42e06
F2 = 1227.60e06
FE1a = F1
FE5a =1176.45e06
FE5b=1207.140e06
DOY = 35
if __name__ =='__main__':


    # root logger z INFO
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


    rnx_files = os.listdir(OBS_PATH)
    total_files = len(rnx_files)
    stations_processed=[f[:4] for f in os.listdir(OUT) if 'STEC' in f]
    print('Processed stations: ', sorted(stations_processed), len(stations_processed))
    names = [f[:4] for f in os.listdir(OBS_PATH) if f.endswith(('crx.gz','crx'))]
    # names = ['POTS','BRST','ZIM2','ZIM3','BOR1','BRUX']
    print('Selected stations: ', [name for name in names if name not in stations_processed])
    t_start, t_end = TLIM[0], TLIM[1]
    epochs = gnx_py.time.arange_datetime(t_start, t_end, MIN_INTERVAL)

    positions = None; sp3=None
    dcb_table = pd.read_csv('./sample_data/dcb_table.csv')
    for idx, RNX in enumerate(sorted(rnx_files, reverse=False), start=1):
        try:
            if RNX.endswith('crx.gz'):
                NAME = RNX[:4]

                GPS_DCB = dcb_table[dcb_table['Station'] == NAME]['gps_dcb'].values[0]
                GAL_DCB = dcb_table[dcb_table['Station'] == NAME]['gal_dcb'].values[0]
                OBS = os.path.join(OBS_PATH, RNX)
                print('Processing: ', NAME)

                config = gnx.TECConfig(obs_path=OBS,
                                       nav_path=NAV,
                                       sp3_path=SP3_LIST,
                                       gim_path=GIM,
                                       sys="G",
                                       gps_freq='L1L2',
                                       gal_freq='E1E5a',
                                       windup=False,
                                       rel_path=False,
                                       sat_pco=True,
                                       rec_pco=True,
                                       atx_path=ATX_PATH,
                                       interpolation_method='lagrange',
                                       ev_mask=30,

                                       add_dcb=True,
                                       add_sta_dcb=True,
                                       define_station_dcb=GPS_DCB,
                                       rcv_dcb_source='defined',

                                       screen=True,
                                       skip_sat=['G20','G02'],

                                       station_name=NAME,
                                       day_of_year=DOY,

                                       ionosphere_model='gim',
                                       compare_models=False,
                                       troposphere_model=False,
                                       use_gfz=True,
                                       leveling_ws=50,
                                       median_leveling_ws=50,
                                       min_arc_len=2
                                       )
                controller = TECSession(config=config)
                logging.getLogger("TEC").setLevel(logging.DEBUG)
                obs_tec = controller.run()

                print(f"GPS processed for {NAME}")


                obs_tec[['leveled_tec', 'median_leveled_tec', 'ev', 'az','L4','P4','code_tec','lat_ipp','lon_ipp','ion']].to_parquet(
                    os.path.join(OUT, f'{NAME}_G_STEC.parquet.gzip'),compression='gzip')
                print('NEGATIVE GPS: ')
                print(obs_tec[obs_tec['leveled_tec']<0],'\n',
                      obs_tec[obs_tec['leveled_tec']<0].index.get_level_values('sv').unique().tolist())
                print('\n')

                # GALILEO
                print('GALILEO: ')
                config.sys="E"
                config.define_station_dcb=GAL_DCB
                controller = TECSession(config=config)
                logging.getLogger("TEC").setLevel(logging.DEBUG)
                obs_tec = controller.run()
                print(f"GAL processed for {NAME}")
                obs_tec[['leveled_tec', 'median_leveled_tec', 'ev', 'az','L4','P4','code_tec','lat_ipp','lon_ipp','ion']].to_parquet(
                    os.path.join(OUT, f'{NAME}_E_STEC.parquet.gzip'),compression='gzip')
                print('NEGATIVE GAL: ')
                print(obs_tec[obs_tec['leveled_tec'] < 0])
                print('==='*30,'\n')
        except Exception as e:
            traceback.print_exc()
            continue
