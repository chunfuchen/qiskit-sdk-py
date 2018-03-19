# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Tools for working with Pauli Operators.

A simple pauli class and some tools.
"""
import random
import copy

import numpy as np
from scipy import sparse

class Pauli:
    """A simple class representing Pauli Operators.

    The form is P = (-i)^dot(v,w) Z^v X^w where v and w are elements of Z_2^n.
    That is, there are 4^n elements (no phases in this group).

    For example, for 1 qubit
    P_00 = Z^0 X^0 = I
    P_01 = X
    P_10 = Z
    P_11 = -iZX = (-i) iY = Y

    Multiplication is P1*P2 = (-i)^dot(v1+v2,w1+w2) Z^(v1+v2) X^(w1+w2)
    where the sums are taken modulo 2.

    Pauli vectors v and w are supposed to be defined as numpy arrays.

    Ref.
    Jeroen Dehaene and Bart De Moor
    Clifford group, stabilizer states, and linear and quadratic operations
    over GF(2)
    Phys. Rev. A 68, 042318 – Published 20 October 2003
    """

    def __init__(self, v, w):
        """Make the Pauli class."""

        if isinstance(v, list) and isinstance(v, list):
            result = Pauli.from_list(v, w)
            self.v = result.v
            self.w = result.w
            self.numberofqubits = result.v.size
            self.id = self.to_label()
            return

        if isinstance(v, np.ndarray) and isinstance(v, np.ndarray):
            self.v = v.astype(np.bool)
            self.w = w.astype(np.bool)
            self.numberofqubits = v.size
            self.id = self.to_label()

    @classmethod
    def from_list(cls, v, w):
        v = np.asarray(v).astype(np.bool)
        w = np.asarray(w).astype(np.bool)
        return cls(v, w)

    def __str__(self):
        """Output the Pauli as first row v and second row w."""
        stemp = 'v = '
        for i in self.v:
            stemp += str(int(i)) + '\t'
        stemp = stemp + '\nw = '
        for j in self.w:
            stemp += str(int(j)) + '\t'
        return stemp

    def __eq__(self, other):
        """Return True if all Pauli terms are equal."""
        bres = False
        if self.numberofqubits == other.numberofqubits:
            if np.all(self.v == other.v) and np.all(self.w == other.w):
                bres = True
        return bres

    def __mul__(self, other):
        """Multiply two Paulis."""
        if self.numberofqubits != other.numberofqubits:
            print('These Paulis cannot be multiplied - different number '
                  'of qubits')
        v_new = np.logical_xor(self.v, other.v)
        w_new = np.logical_xor(self.w, other.w)
        pauli_new = Pauli(v_new, w_new)
        return pauli_new

    def to_label(self):
        """Print out the labels in X, Y, Z format.

        Returns:
            str: pauli label
        """
        p_label = ''
        for j_index in range(self.numberofqubits):
            if not self.v[j_index] and not self.w[j_index]:
                p_label += 'I'
            elif not self.v[j_index] and self.w[j_index]:
                p_label += 'X'
            elif self.v[j_index] and self.w[j_index]:
                p_label += 'Y'
            elif self.v[j_index] and not self.w[j_index]:
                p_label += 'Z'
        return p_label

    def to_matrix(self):
        """Convert Pauli to a matrix representation.

        Order is q_n x q_{n-1} .... q_0

        Returns:
            numpy.array: a matrix that represnets the pauli.
        """
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        id_ = np.array([[1, 0], [0, 1]], dtype=complex)
        matrix = 1
        for k in range(self.numberofqubits):
            if not self.v[k] and not self.w[k]:
                new = id_
            elif self.v[k] and not self.w[k]:
                new = z
            elif not self.v[k] and self.w[k]:
                new = x
            elif self.v[k] and self.w[k]:
                new = y
            else:
                print('the string is not of the form 0 and 1')
            matrix = np.kron(new, matrix)

        return matrix

    def to_spmatrix(self):
        """Convert Pauli to a sparse matrix representation (CSR format).

        Order is q_n x q_{n-1} .... q_0

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represnets the pauli.
        """
        x = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
        y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
        z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        id_ = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
        matrix = 1
        for k in range(self.numberofqubits):
            if not self.v[k] and not self.w[k]:
                new = id_
            elif self.v[k] and not self.w[k]:
                new = z
            elif not self.v[k] and self.w[k]:
                new = x
            elif self.v[k] and self.w[k]:
                new = y
            else:
                print('the string is not of the form 0 and 1')
            matrix = sparse.kron(new, matrix, 'csr')
        return matrix

def random_pauli(number_qubits):
    """Return a random Pauli on numberofqubits."""
    v = np.random.choice(a=[False, True], size=number_qubits)
    w = np.random.choice(a=[False, True], size=number_qubits)
    return Pauli(v, w)


def sgn_prod(P1, P2):
    """Multiply two Paulis P1*P2 and track the sign.

    P3 = P1*P2: X*Y
    """

    if P1.numberofqubits != P2.numberofqubits:
        print('Paulis cannot be multiplied - different number of qubits')

    paulinew = P1 * P2
    phase = 1
    phase_change = 0
    for i in range(P1.v.size):
        if P1.v[i] and not P1.w[i]: #Z
            if not P2.w[i]:
                continue
            if P2.v[i]: #Y
                phase_change -= 1
            else: #X
                phase_change += 1
        elif not P1.v[i] and P1.w[i]: #X
            if not P2.v[i]:
                continue
            if not P2.w[i]: #Z
                phase_change -= 1
            else: #Y
                phase_change += 1
        elif P1.v[i] and P1.w[i]: #Y
            if not np.logical_xor(P2.v[i], P2.w[i]):
                continue
            if not P2.v[i]: #X
                phase_change -= 1
            elif P2.v[i]: #Z
                phase_change += 1
            # if not P2.v[i] and P2.w[i]: #X
            #     phase_change -= 1
            # elif P2.v[i] and not P2.w[i]: #Z
            #     phase_change += 1

    phase_change = phase_change % 4
    for i in range(phase_change):
        phase *= 1j

    return paulinew, phase


def inverse_pauli(other):
    """Return the inverse of a Pauli."""
    return copy.deepcopy(other)

def label_to_pauli(label):
    """Return the pauli of a string ."""
    v = np.zeros(len(label), dtype=np.bool)
    w = np.zeros(len(label), dtype=np.bool)
    for j, _ in enumerate(label):
        if label[j] == 'I':
            v[j] = False
            w[j] = False
        elif label[j] == 'Z':
            v[j] = True
            w[j] = False
        elif label[j] == 'Y':
            v[j] = True
            w[j] = True
        elif label[j] == 'X':
            v[j] = False
            w[j] = True
        else:
            print('something went wrong')
            return -1
    return Pauli(v, w)


def pauli_group(number_of_qubits, case=0):
    """Return the Pauli group with 4^n elements.

    The phases have been removed.
    case 0 is ordered by Pauli weights and
    case 1 is ordered by I,X,Y,Z counting last qubit fastest.

    Args:
        number_of_qubits (int): number of qubits
        case (int): determines ordering of group elements (0=weight, 1=tensor)

    Returns:
        list: list of Pauli objects

    Note:
        WARNING THIS IS EXPONENTIAL
    """
    if number_of_qubits < 5:
        temp_set = []
        if case == 0:
            tmp = pauli_group(number_of_qubits, case=1)
            # sort on the weight of the Pauli operator
            return sorted(tmp, key=lambda x: -np.count_nonzero(
                np.array(x.to_label(), 'c') == b'I'))

        elif case == 1:
            # the Pauli set is in tensor order II IX IY IZ XI ...
            for k_index in range(4 ** number_of_qubits):
                v = np.zeros(number_of_qubits, dtype=np.bool)
                w = np.zeros(number_of_qubits, dtype=np.bool)
                # looping over all the qubits
                for j_index in range(number_of_qubits):
                    # making the Pauli for each kindex i fill it in from the
                    # end first
                    element = int((k_index) / (4 ** (j_index))) % 4
                    if element == 0:
                        v[j_index] = False
                        w[j_index] = False
                    elif element == 1:
                        v[j_index] = False
                        w[j_index] = True
                    elif element == 2:
                        v[j_index] = True
                        w[j_index] = True
                    elif element == 3:
                        v[j_index] = True
                        w[j_index] = False
                temp_set.append(Pauli(v, w))
            return temp_set

    print('please set the number of qubits to less than 5')
    return -1


def pauli_singles(j_index, number_qubits):
    """Return the single qubit pauli in number_qubits."""
    # looping over all the qubits
    tempset = []
    v = np.zeros(number_qubits, dtype=np.bool)
    w = np.zeros(number_qubits, dtype=np.bool)
    v[j_index] = False
    w[j_index] = True
    tempset.append(Pauli(v, w))
    v = np.zeros(number_qubits, dtype=np.bool)
    w = np.zeros(number_qubits, dtype=np.bool)
    v[j_index] = True
    w[j_index] = True
    tempset.append(Pauli(v, w))
    v = np.zeros(number_qubits, dtype=np.bool)
    w = np.zeros(number_qubits, dtype=np.bool)
    v[j_index] = True
    w[j_index] = False
    tempset.append(Pauli(v, w))
    return tempset
