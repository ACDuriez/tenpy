import os
import numpy as np
import logging

from tenpy.algorithms.tebd_clifford import TEBDClifford

from utils.utils import get_date_postfix
from utils.args import get_config

def main(config):
    
    # Timestamp
    simlulation_time = get_date_postfix()
    config.saving_dir = os.path.join(config.saving_dir, simlulation_time)
    config.log_dir_path = os.path.join(config.saving_dir, 'log_dir')
    config.data_dir_path = os.path.join(config.saving_dir, 'data_dir')
    config.img_dir_path = os.path.join(config.saving_dir, 'img_dir')

    # Create directories if not exist
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.data_dir_path):
        os.makedirs(config.data_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    
    # Logger
    log_path = os.path.join(config.log_dir_path, simlulation_time+ '_logger.log')
    from importlib import reload
    reload(logging)
    logging.basicConfig(filename=log_path, level=logging.INFO,format='%(message)s \n\n')
    
    logger = logging.getLogger(log_path)
    logger.info(config)
    config.logger = logger
    
    #Running simulation
    solver = TEBDClifford(config)
    solver.run()

if __name__ == "__main__":

    config = get_config()

    # Physical System configurations
    config.L = 10
    config.Jxy = -0.1
    config.Jz = -0.5
    config.hz = 0.7
    config.bc = 'open'
    config.bc_MPS = 'finite'
    # Initial state (Bloch angles theta and phi)
    config.bloch_angles = (np.pi/2, 0.)

    # TEBD configurations
    config.dt = 0.1
    config.order = 4
    config.chi_max = 100
    config.dt = 0.1
    config.evol_time = 20


    # Saving directory
    config.saving_dir = 'results'

    print(config)

    main(config)