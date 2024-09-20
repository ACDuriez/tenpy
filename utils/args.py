import argparse

def str2bool(v):
    return v.lower() in ['true']

def get_config():
    parser = argparse.ArgumentParser()

    # Physical System configurations
    parser.add_argument('--L', type=int, default=10, help='Number of sites in the lattice')
    parser.add_argument('--Jxy', type=float, default=-0.1, help='Coupling parameter in X and Y directions')
    parser.add_argument('--Jz', type=float, default=-0.5, help='Coupling parameter in the Z direction')
    parser.add_argument('--hz', type=float, default=0.7, help='Intensity of the transverse field')
    parser.add_argument('--bc', type=str, default='open', help='Boundary conditions of the lattice, can be periodic or open')
    parser.add_argument('--bc_MPS', type=str, default='finite', help='Boundary conditions of the MPS, can be infinite or finite')

    # TEBD configurations
    parser.add_argument('--dt', type=float, default=0.1, help='Trotterization time step')
    parser.add_argument('--order', type=int, default=4, help='Trotterization order, can be 1 to 4')
    parser.add_argument('--chi_max', type=int, default=100, help='Maximum bond dimension, where the cutoff is made')
    parser.add_argument('--svd_max', type=int, default=1.e-12, help='Precision for the SVD calculation')
    parser.add_argument('--evol_time', type=float, default=20., help='Total evolution time')
    parser.add_argument('--conserve', type=str, default='None', help='Conserved charge, can be None or Sz')


    # Saving directory
    parser.add_argument('--saving_dir', type=str, default='results', help='name of the saving directory')
    
    # Logger
    parser.add_argument('--logger', type=object, default=None, help='logger object')
    config = parser.parse_args()

    return config