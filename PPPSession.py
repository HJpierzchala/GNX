from __future__ import annotations
import traceback
import gnx_py as gnx
import logging
from datetime import datetime
import os
import numpy as np
from gnx_py import PPPSession
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=3)


"""High‑level Precise Point Positioning (PPP) framework.

"""
# Define product paths
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

SYS= {'G'}

DOY = 35
if __name__ =='__main__':
    # grav rinex files from input dir
    rnx_files = os.listdir(OBS_PATH)
    total_files = len(rnx_files)
    for idx, RNX in enumerate(sorted(rnx_files, reverse=False), start=1):
        try:
            if RNX.endswith('crx.gz'):

                NAME = RNX[:4]
                start = datetime.now()
                OBS = os.path.join(OBS_PATH, RNX)
                print('Processing: ', OBS)
                config = gnx.PPPConfig(
                    obs_path=OBS,
                    nav_path=NAV,
                    atx_path=ATX_PATH,
                    dcb_path=DCB_PATH,
                    sp3_path=SP3_LIST,
                    gim_path=GIM,
                    sys=SYS,
                    gps_freq='L1L2',
                    gal_freq='E1',
                    orbit_type='precise',

                    positioning_mode='combined',
                    ionosphere_model='gim',
                    troposphere_model='niell',
                    use_iono_constr=True,
                    use_iono_rms=True,
                    t_end=15,


                    interpolation_method='lagrange',
                    time_limit=None,
                    day_of_year=DOY,
                    sinex_path=SINEX,
                    screen=False,
                    ev_mask=10,

                    use_gfz=True,
                    station_name=NAME,
                    min_arc_len=5,
                    trace_filter=True,
                    sat_pco='los'

                )
                controller = PPPSession(config)
                logging.getLogger("PPP").setLevel(logging.DEBUG)
                results = controller.run()
                print('Convergence <5mm time: ')
                print(results.convergence)
                print(f'Solution for {NAME}: ')
                print(results.solution.tail())
                ## simple plot:
                # results.solution[['de','dn','du']].plot(subplots=True)
                # plt.show()
                print('===='*30,'\n\n')
                ## Save solution and residuals:
              #   results.residuals_gps.to_parquet(f'{OUT}/residuals_gps_{NAME}_{config.gps_freq}_{config.positioning_mode}.parquet.gzip',
              # compression='gzip')
              #   if results.residuals_gal is not None:
              #       results.residuals_gal.to_parquet(f'{OUT}/residuals_gal_{NAME}_{config.gal_freq}_{config.positioning_mode}.parquet.gzip',
              # compression='gzip')
              #   results.solution.to_parquet(
              #       f'{OUT}/{NAME}_{config.gps_freq}_{config.gal_freq}_{config.positioning_mode}_{int(config.use_iono_constr)}_{config.reset_every}.parquet.gzip',
              # compression='gzip')

                end = datetime.now()
                total = end - start
                print('Finished in: ', total.total_seconds())
                print('===' * 50, '\n\n')
        except Exception as e:
            print('Error with: ', RNX)
            traceback.print_exc()
            continue
