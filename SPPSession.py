from __future__ import annotations
import traceback
from datetime import datetime
import os
import numpy as np
from gnx_py.spp import SPPSession, SPPConfig
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=3)


"""High‑level Single Point Positioning (PPP) framework.

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

# define systems and day of year
SYS= {'G','E'}
DOY = 35
if __name__ =='__main__':
    # for milti station processing - grab rinex files from input dir
    rnx_files = os.listdir(OBS_PATH)
    for idx, RNX in enumerate(sorted(rnx_files, reverse=False), start=1):
        try:
            if RNX.endswith('crx.gz'):

                NAME = RNX[:4]
                start = datetime.now()
                OBS = os.path.join(OBS_PATH, RNX)
                print('Processing: ', OBS)
                config = SPPConfig(
                    obs_path=OBS,
                    nav_path=NAV,
                    atx_path=ATX_PATH,
                    dcb_path=DCB_PATH,
                    gim_path=GIM,
                    sys=SYS,
                    sp3_path=SP3_LIST,
                    gps_freq='L1', 
                    gal_freq='E1',
                    orbit_type='precise',
                    ionosphere_model='gim',
                    troposphere_model='niell',
                    time_limit=None,
                    day_of_year=DOY,
                    sinex_path=SINEX,
                    screen=False,
                    ev_mask=5,

                    use_gfz=False,
                    station_name=NAME,
                    sat_pco=False,
                    trace_filter = True

                )
                controller= SPPSession(config)
                res = controller.run()
                sol = res.solution
                print(sol.tail())
                end = datetime.now()
                total = end - start
                print('Finished in: ', total.total_seconds())

                ## simple plot:
                # fig, ax = plt.subplots(4,1)
                # for num, col in enumerate(['de','dn','du','dtr_gps']):
                #     ax[num].plot(sol['time'], sol[col])
                plt.show()
        except Exception as e:
            print('Error with: ', RNX)
            traceback.print_exc()
            continue
