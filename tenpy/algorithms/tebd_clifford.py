
import numpy as np
import time
import typing
import warnings
import logging
logger = logging.getLogger(__name__)

from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS

from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from .truncation import svd_theta, decompose_theta_qr_based, TruncationError
from ..linalg import random_matrix
from ..tools.misc import consistency_check

class XXZChain(CouplingModel,MPOModel):
    def __init__(self, model_params):

        L = model_params['L']
        hz = model_params['hz']
        conserve = model_params['conserve']
        Jxy = model_params['Jxy']
        Jz = model_params['Jz']
        bc = model_params['bc']
        bc_MPS = model_params['bc_MPS']
        spin = SpinHalfSite(conserve=conserve)
        # the lattice defines the geometry
        lattice = Chain(L, 
                        spin, 
                        bc=bc, 
                        bc_MPS=bc_MPS)
        CouplingModel.__init__(self, lattice)
        # add terms of the Hamiltonian
        self.add_coupling(Jxy, 0, "Sigmax", 0, "Sigmax", 1)
        self.add_coupling(Jxy, 0, "Sigmay", 0, "Sigmay", 1)
        self.add_coupling(Jz, 0, "Sigmaz", 0, "Sigmaz", 1) # interaction
        
        self.add_onsite(hz, 0, "Sigmaz") # transverse field

        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, lattice, self.calc_H_bond())
        # finish initialization
        # generate MPO for DMRG
        MPOModel.__init__(self, lattice, self.calc_H_MPO())

class TEBDClifford():
    def __init__(self, config):
    # Physical System configurations
        self.L = config.L 
        self.Jxy = config.Jxy 
        self.Jz = config.Jz 
        self.hz = config.hz
        self.bc = config.bc
        self.bc_MPS = config.bc_MPS
        self.bloch_angles = config.bloch_angles

        # TEBD configurations
        self.dt = config.dt
        self.order = config.order
        self.chi_max = config.chi_max
        self.dt = config.dt
        self.evol_time = config.evol_time
        self.conserve = config.conserve

        # # Defining model
        self.model = XXZChain(dict(bc=self.bc,
                                   bc_mps=self.bc_MPS,
                                   conserve=config.conserve,
                                   L=self.L,
                                   hz=self.hz,
                                   Jxy=self.Jxy,
                                   Jz=self.Jz,
                                   bc_MPS=self.bc_MPS,))
        self.sites = self.model.lat.mps_sites()
        
        # Initial state
        self.psi = MPS.from_product_state(self.sites, [self.bloch_angles] * self.L, self.bc_MPS) 

        # Saving directory
        self.saving_dir = config.saving_dir
        # Image directory
        self.img_dir_path = config.img_dir_path
        # Log directory
        self.log_dir_path = config.log_dir_path
        
        #Logger
        self.logger = config.logger

    def run(self):
        self.logger.info('Testing logger...')

        time = 0.
        while time < self.evol_time:
            print(time)
            time += self.dt