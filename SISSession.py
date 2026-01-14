import gnx_py as gnx
from datetime import datetime
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=3)

if __name__ =='__main__':
    # Define paths to orbit products we'd like to compare
    NAV = './sample_data/BRDC00CNE_R_20240350000_01D_GN.rnx'
    SP3 = './sample_data/GRG0MGXFIN_20240350000_01D_05M_ORB.SP3'
    # you can add sp3 from previous and next day to handle day-to-day overlap, but it's not necessary
    prev_sp3 = None
    next_sp3 = None

    prev_sp3_0=None
    next_sp3_0=None
    # satellite PCO path - needed for sp3 vs broadcast comparison or between-sp3 APC comparison
    ATX_PATH='./sample_data/igs20.atx'

    DCB_0_PATH=None
    DCB_1_PATH = None
    # Define processing configuration
    config = gnx.SISConfig(orb_path_0=NAV,orb_path_1=SP3,prev_sp3=prev_sp3,next_sp3=next_sp3,dcb_path_0=None,dcb_path_1=None,interval=120,gps_mode='L1L2',gal_mode='E1E5a',
                       atx_path=ATX_PATH,system='G',prev_sp3_0=prev_sp3_0,next_sp3_0=next_sp3_0, tlim=[datetime(2024,2,4,0,0,0),
                                                                                                              datetime(2024,2,5,0,0,0)],
                       clock_bias=True,clock_bias_function='mean',apply_eclipse=True,compare_dcb=False)
    #  run() comparison
    compared = gnx.SISController(config=config).run()
    # print and save the output
    print(compared[['dx','dy','dz','dt']])
    compared.to_csv(f'./sample_data/CNE_{config.system}_BRDC_SP3.csv')






