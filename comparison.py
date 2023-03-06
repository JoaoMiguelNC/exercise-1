import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Statevector


def convert_to_base_2(i: int, j: int):
    '''Converts the integers i and j to binary with a signed bit using the same number of bits for both.
    It Also returns the number of necessary bits.'''

    n_i = int(np.floor(np.log2(abs(i))) + 1) if i != 0 else 1
    n_j = int(np.floor(np.log2(abs(j))) + 1) if j != 0 else 1
    n = max(n_i, n_j)
    sign = np.heaviside([i, j], 1)  # get the signed bit

    return str(sign[0])[0] + bin(abs(i))[2:].zfill(n), str(sign[1])[0] + bin(abs(j))[2:].zfill(n), n + 1


def initialize(qc: QuantumCircuit, reg: QuantumRegister, binary: str, n: int):
    '''Initializes the qubits used for the binary representation of the state.

    Args:
        qc (QuantumCircuit): Quantum circuit being used.
        reg (QuantumRegister): Quantum register where the qubits will be initialized.
        binary (str): Binary represantation of the number.
        n (int): Number of bits in the binary represantation.'''

    for k in range(n):
        if binary[n - 1 - k] == '1':
            qc.x(reg[k])

    return qc


def greater_circuit(qc: QuantumCircuit, q_a, q_b, q_c):
    '''Constructs the single qubit 'greater than' operation circuit that computes a>b. 

    Args:
        qc (QuantumCircuit): Quantum circuit being used.
        q_a: Qubit corresponding to a in a>b.
        q_b: Qubit corresponding to b in a>b.
        q_c: Ancilla qubit used to store the value of a>b.'''

    # a>b <=> a /\ ~b
    qc.x(q_b)
    qc.toffoli(q_a, q_b, q_c)
    qc.x(q_b)
    return qc


def equal_circuit(qc: QuantumCircuit, q_a, q_b, q_e):
    '''Constructs the single qubit 'equal' operation circuit that computes a=b. 

    Args:
        qc (QuantumCircuit): Quantum circuit being used.
        q_a: Qubit corresponding to a in a=b.
        q_b: Qubit corresponding to b in a=b.
        q_c: Ancilla qubit used to store the value of a=b.'''

    # a=b <=> ~(a (+) b)
    qc.cx(q_a, q_b)
    qc.cx(q_b, q_e)
    qc.x(q_e)
    qc.cx(q_a, q_b)
    return qc


def greater_result(qc: QuantumCircuit, q_c: QuantumRegister, q_e: QuantumRegister, q_r: QuantumRegister, n: int):
    '''Constructs the quantum circuit to compute a>b given the result of the componentwise operations > and =. 

    Args:
        qc (QuantumCircuit): Quantum circuit being used.
        q_c (QuantumRegister): Quantum register containing the results of componentwise > operation.
        q_e (QuantumRegister): Quantum register containing the results of componentwise = operation.
        q_r (QuantumRegister): Quantum register where the result will be measured.'''

    for k in range(n):
        control = MCXGate(k + 1)
        qc.append(control, [q_e[n - 1 - m]
                  for m in range(k)] + [q_c[n - 1 - k], q_r])

    return qc


def comparison_circuit(i_bin: str, j_bin: str, n: int):
    '''Constructs the quantum circuit that performs the > comparison on two numbers in binary.

    Args:
        i_bin (str): Binary representation of i in i>j.
        j_bin (str): Binary representation of j in i>j.
        n (int): Number of bits in the binary represantation.'''

    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n, 'b')
    c = QuantumRegister(n, 'c')
    e = QuantumRegister(n, 'e')
    greater = QuantumRegister(1, 'greater')
    qc = QuantumCircuit(a, b, c, e, greater)

    initialize(qc, a, i_bin, n)
    initialize(qc, b, j_bin, n)
    qc.barrier()

    for k in range(n):
        greater_circuit(qc, a[k], b[k], c[k])
        # qc.barrier()
        equal_circuit(qc, a[k], b[k], e[k])
        qc.barrier()

    greater_result(qc, c, e, greater, n)

    return qc


def quantum_compare(i: int, j: int):
    '''Performs the i>j comparison between two integers using a quantum circuit.

    Args:
        i (int): Integer i in i>j.
        j (int): Integer j in i>j.'''

    i_bin, j_bin, n = convert_to_base_2(i, j)
    qc = comparison_circuit(i_bin, j_bin, n)

    if list(Statevector(qc).to_dict().keys())[0][0] == '1':
        return i
    else:
        return j
