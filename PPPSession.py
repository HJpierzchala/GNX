from __future__ import annotations
import traceback
import pandas as pd
import gnx_py as gnx
import logging
from datetime import datetime
import os
import numpy as np
import gnx_py.time
from gnx_py import PPPSession
import json
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=3)


"""High‑level Precise Point Positioning (PPP) framework.

"""

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

MIN_INTERVAL = 15/60
SYS= {'G'}
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
    t_start, t_end = TLIM[0], TLIM[1]
    epochs = gnx_py.time.arange_datetime(t_start, t_end, MIN_INTERVAL)

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


                # for gps_mode in ['L1']:
                #     for gal_mode in ['E1E5b']:
                #         config.gps_freq=gps_mode
                #         config.gal_freq=gal_mode
                controller = PPPSession(config)
                logging.getLogger("PPP").setLevel(logging.DEBUG)
                results = controller.run()
                # print('Pair: ', gps_mode, gal_mode)
                print('Convergence <5mm time: ')
                print(results.convergence)
                print(f'Solution for {NAME}: ')
                print(results.solution.tail())
                # results.solution[['de','dn','du']].plot(subplots=True)
                # plt.show()
                print('===='*30,'\n\n')




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

# cfg = PPPConfig(positioning_mode="combined")
# print(cfg.p_dt, cfg.clock_process)  # z presetu combined
#
# # ustawiasz override
# cfg.set_param(p_dt=5e9, q_tro=0.01)
#
# # zmieniasz tryb i zachowujesz override (default)
# cfg.set_mode("uncombined")
# # p_dt będzie 5e9 (override), a reszta wg uncombined
#
# # chcesz przełączyć tryb "na czysto"
# cfg.set_mode("single", keep_overrides=False)
#
# # chcesz cofnąć pojedynczy override
# cfg.set_param(p_dt=123)
# cfg.clear_override("p_dt")  # wraca preset
