from __future__ import annotations
import traceback
import gnx_py as gnx
import logging
from gnx_py import TECSession
import os

# Define input products
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

USE_SYS='G'
DOY = 35
if __name__ =='__main__':
    rnx_files = os.listdir(OBS_PATH)
    total_files = len(rnx_files)
    names = [f[:4] for f in os.listdir(OBS_PATH) if f.endswith(('crx.gz','crx'))]
    for idx, RNX in enumerate(sorted(rnx_files, reverse=False), start=1):
        try:
            if RNX.endswith('crx.gz'):
                NAME = RNX[:4]
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
                                       define_station_dcb=False,
                                       rcv_dcb_source='gim', # Both BRUX and BOR1 have their DCB's in GIM file header

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
                # Print output
                print(f"GPS processed for {NAME}")
                obs_tec[['leveled_tec', 'median_leveled_tec', 'ev', 'az','L4','P4','code_tec','lat_ipp','lon_ipp','ion']].to_parquet(
                    os.path.join(OUT, f'{NAME}_G_STEC.parquet.gzip'),compression='gzip')

                # Print negative TEC values - they sometimes appear mostly due to imperfect phase-to-code smoothing or inaccurately estimated station DCB
                print('NEGATIVE GPS: ')
                print(obs_tec[obs_tec['leveled_tec']<0],'\n',
                      obs_tec[obs_tec['leveled_tec']<0].index.get_level_values('sv').unique().tolist())
                print('\n')

                # Change to GALILEO
                print('GALILEO: ')
                config.sys="E"
                controller = TECSession(config=config)
                logging.getLogger("TEC").setLevel(logging.DEBUG)
                obs_tec = controller.run()
                # Print output
                print(f"GAL processed for {NAME}")
                obs_tec[['leveled_tec', 'median_leveled_tec', 'ev', 'az','L4','P4','code_tec','lat_ipp','lon_ipp','ion']].to_parquet(
                    os.path.join(OUT, f'{NAME}_E_STEC.parquet.gzip'),compression='gzip')
                # Print negative TEC values - they sometimes appear mostly due to imperfect phase-to-code smoothing or inaccurately estimated station DCB
                print('NEGATIVE GAL: ')
                print(obs_tec[obs_tec['leveled_tec'] < 0])
                print('==='*30,'\n')
        except Exception as e:
            traceback.print_exc()
            continue
