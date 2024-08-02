import logging
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import my_tebd as tebd
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel,CouplingMPOModel
from tenpy.networks.site import SpinSite
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from src.utils import get_exact_mags
import matplotlib.pyplot as plt

logger = logging.getLogger('my_logger')
logging.basicConfig(level=logging.INFO, format='%(message)s',filename='logfile.log')

logger.info('testing logger')
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

        max_workers = 1
L = 10; Jxy = -0.1; Jz = -0.5; hz = 0.7
bc = 'open'
bc_MPS = 'finite'

model_params = dict(bc=bc,
                    bc_MPS =bc_MPS,
                    conserve='None',
                    L=L,
                    hz=hz,
                    Jxy=Jxy,
                    Jz=Jz)

model = XXZChain(model_params)
sites = model.lat.mps_sites()
theta, phi = np.pi/2, np.pi/2
bloch_sphere_state = np.array([np.cos(theta/2),np.sin(theta/2)])
psi = MPS.from_product_state(sites, [bloch_sphere_state] * L, bc_MPS) 

tebd_params = {
    'N_steps': 1,
    'dt': 0.1,
    'order': 4,
    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
}

cnot_gate = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])

# clifford = npc.Array.from_ndarray(cnot_gate,labels=['(p0.p1)', '(p0*.p1*)'],legcharges=[truth.get_leg(0).to_LegCharge(),truth.get_leg(1).to_LegCharge()])

# eng = tebd.TEBDEngine(psi, model, tebd_params)
eng = tebd.TEBDEngine(psi, model, tebd_params,clifford=cnot_gate,clifford_step=50)

def measurement(eng, data):
    keys = ['t', 'Sx', 'Sy', 'Sz', 'trunc_err','entropy']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    data['Sx'].append(eng.psi.expectation_value('Sigmax'))
    data['Sy'].append(eng.psi.expectation_value('Sigmay'))
    data['Sz'].append(eng.psi.expectation_value('Sigmaz'))
    data['entropy'].append(eng.psi.entanglement_entropy())
    data['trunc_err'].append(eng.trunc_err.eps)
    return data

data = measurement(eng, None)
while eng.evolved_time < 20.:
    eng.run()
    measurement(eng, data)

mx = []
my = []
mz = []
for i in range(len(data['Sx'])):
    mx.append(np.sum(data['Sx'][i]))
    my.append(np.sum(data['Sy'][i]))
    mz.append(np.sum(data['Sz'][i]))

times,exact_mx,exact_my,exact_mz = get_exact_mags(L=L)

f,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(data['t'], mx/np.array(L),'o',label='mx TEBD')
ax[0].plot(data['t'], my/np.array(L),'s',label='my TEBD')
ax[0].plot(data['t'], mz/np.array(L),'>',label='mz TEBD')
ax[0].plot(times,exact_mx,label='mx exact')
ax[0].plot(times,exact_my,label='y exact')
ax[0].plot(times,exact_mz,label='z exact')
ax[0].legend()
ax[0].set_xlabel('time $t$')

ax[1].plot(data['t'], np.array(data['entropy'])[:, L//2])

plt.savefig('plots.pdf',format='pdf')