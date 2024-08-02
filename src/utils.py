from qiskit.quantum_info import SparsePauliOp
import networkx as nx
import numpy as np
from qutip import Qobj
from qiskit.quantum_info import Statevector,Operator

def get_heisenberg_hamiltonian(graph,Jxy=1.,Jz=1.,hz=0.):
    """Returns the hamiltonian in the case where hl=hr. Here the value of the field is ap."""
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z field
    for qubit in graph.nodes():
        # Z field
        coeff = ('Z',[qubit],hz)
        sparse_list.append(coeff)

    # Z Interaction field (ZZ)
    for i,j in graph.edges():
        coeff = ('ZZ',[i,j],Jz)
        sparse_list.append(coeff)
        coeff = ('XX',[i,j],Jxy)
        sparse_list.append(coeff)
        coeff = ('YY',[i,j],Jxy)
        sparse_list.append(coeff)

    
    hamiltonian = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits).simplify()
    return hamiltonian

def get_line_graph(n_qubits):
    graph_line = nx.Graph()
    graph_line.add_nodes_from(range(n_qubits))

    edge_list = []
    for i in graph_line.nodes:
        if i < n_qubits-1:
            edge_list.append((i,i+1))

    # Generate graph from the list of edges
    graph_line.add_edges_from(edge_list)
    return graph_line

def get_mag_z_op(graph):
    
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        coeff = ('Z',[qubit],1)
        sparse_list.append(coeff)

    mag = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits)/num_qubits
    return mag

def get_mag_x_op(graph):
    
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        coeff = ('X',[qubit],1)
        sparse_list.append(coeff)

    mag = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits)/num_qubits
    return mag

def get_mag_y_op(graph):
    
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        coeff = ('Y',[qubit],1)
        sparse_list.append(coeff)

    mag = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits)/num_qubits
    return mag

def get_hamiltonian_magkink(graph,J=1.,hx=0.5,hz=0.,ap=0.):
    """Returns the hamiltonian in the case where hl=hr. Here the value of the field is ap."""
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        # X field
        coeff = ('X',[qubit],-1*hx)
        sparse_list.append(coeff)
        # Z field
        coeff = ('Z',[qubit],-1*hz)
        sparse_list.append(coeff)

    # Anti-paralel field at the borders
    coeff = ('Z',[0],ap) #this is the positive field (order reversed)
    sparse_list.append(coeff)
    coeff = ('Z',[num_qubits-1],-1*ap)
    sparse_list.append(coeff)

    #Interaction field (ZZ)
    for i,j in graph.edges():
        coeff = ('ZZ',[i,j],-1*J)
        sparse_list.append(coeff)
    
    hamiltonian = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits).simplify()
    return hamiltonian

def replace_below_threshold(arr, threshold=1e-12):
    """Takes off near-zero values of an array"""
    # Convert to a NumPy array if not already one
    arr = np.array(arr)
    
    # Find the indices of elements with absolute value below the threshold
    low_values_indices = np.abs(arr) < threshold
    
    # Replace these elements with zero
    arr[low_values_indices] = 0
    
    return arr

def get_eigen_decomposition(hamiltonian):
    ham_qutip = Qobj(hamiltonian.real)
    eigvals,eigvecs = ham_qutip.eigenstates()
    eig_vals = replace_below_threshold(eigvals)
    eig_matrix = replace_below_threshold(np.column_stack([e.full().real for e in eigvecs]))

    return eig_vals,eig_matrix

def get_exact_mags(L = 8, Jxy = -0.1,Jz = -0.5,hz = 0.7):
    graph=get_line_graph(L)
    heisenberg_hamiltonian = get_heisenberg_hamiltonian(graph,Jxy=Jxy,Jz=Jz,hz=hz)
    eig_vals,eig_matrix = get_eigen_decomposition(heisenberg_hamiltonian.to_matrix())
    initial_state = Statevector.from_label('+'* L)

    mag_z = get_mag_z_op(graph)
    mag_x = get_mag_x_op(graph)
    mag_y = get_mag_y_op(graph)

    mags_z = []
    mags_x = []
    mags_y = []
    times = np.linspace(0,20,100)
    for t in times:
        # U = Operator(expm(-1.j*t*heisenberg_hamiltonian.to_matrix()))
        U = Operator(eig_matrix @ np.diag(np.exp(-1.j*t*eig_vals)) @ eig_matrix.T)
        evolved_state = initial_state.evolve(U)
        
        mags_x.append(evolved_state.expectation_value(mag_x).real)
        mags_y.append(evolved_state.expectation_value(mag_y).real)
        mags_z.append(evolved_state.expectation_value(mag_z).real)
    return times,mags_x,mags_y,mags_z
