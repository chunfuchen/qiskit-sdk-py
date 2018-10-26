# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

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

        if isinstance(v, np.ndarray) and isinstance(w, np.ndarray):
            v = v.astype(np.bool)
            w = w.astype(np.bool)
            self._v = v
            self._w = w
            self.v = v.astype(np.int32)  # backward compatibility
            self.w = w.astype(np.int32)  # backward compatibility
            self.numberofqubits = v.size
            self.id = self.to_label()    # having a hashable id

        elif isinstance(v, list) and isinstance(w, list):
            v = np.asarray(v).astype(np.bool)
            w = np.asarray(w).astype(np.bool)
            self._v = v
            self._w = w
            self.v = v.astype(np.int32)  # backward compatibility
            self.w = w.astype(np.int32)  # backward compatibility
            self.numberofqubits = v.size
            self.id = self.to_label()    # having a hashable id

        else:
            raise TypeError("v and w must be either list or ndarray \
                    but {} are provided.".format(type(v)))

    def __str__(self):
        """Output the Pauli as first row v and second row w."""
        stemp = 'v = '
        for i in self._v:
            stemp += str(int(i)) + '\t'
        stemp += '\n'
        stemp += 'w = '
        for j in self._w:
            stemp += str(int(j)) + '\t'
        return stemp

    def __eq__(self, other):
        """Return True if all Pauli terms are equal."""
        bres = False
        if self.numberofqubits == other.numberofqubits:
            if np.all(self._v == other._v) and np.all(self._w == other._w):
                bres = True
        return bres

    def __mul__(self, other):
        """Multiply two Paulis."""
        if self.numberofqubits != other.numberofqubits:
            raise ValueError("These Paulis cannot be multiplied: \
                different number of qubits.")
        v_new = np.logical_xor(self._v, other._v)
        w_new = np.logical_xor(self._w, other._w)
        pauli_new = Pauli(v_new, w_new)
        return pauli_new

    def to_label(self):
        """Generate string representation of Pauli (IXYZ).

        Returns:
            str: pauli label
        """
        p_label = ''
        for j_index in range(self.numberofqubits):
            if not self._v[j_index] and not self._w[j_index]:
                p_label += 'I'
            elif not self._v[j_index] and self._w[j_index]:
                p_label += 'X'
            elif self._v[j_index] and self._w[j_index]:
                p_label += 'Y'
            elif self._v[j_index] and not self._w[j_index]:
                p_label += 'Z'
        return p_label

    def to_matrix(self):
        """Convert Pauli to a matrix representation.

        Order is q_n x q_{n-1} .... q_0

        Returns:
            numpy.array: a matrix that represents the pauli.
        """
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        id_ = np.array([[1, 0], [0, 1]], dtype=complex)
        matrix = 1
        for k in range(self.numberofqubits):
            if not self._v[k] and not self._w[k]:
                new = id_
            elif self._v[k] and not self._w[k]:
                new = z
            elif not self._v[k] and self._w[k]:
                new = x
            elif self._v[k] and self._w[k]:
                new = y
            else:
                raise ValueError("the string is not of the form 0 and 1")
            matrix = np.kron(new, matrix)

        return matrix

    def to_spmatrix(self):
        """Convert Pauli to a sparse matrix representation (CSR format).

        Order is q_n x q_{n-1} .... q_0

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represents the pauli.
        """
        x = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
        y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
        z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        id_ = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
        matrix = 1
        for k in range(self.numberofqubits):
            if not self._v[k] and not self._w[k]:
                new = id_
            elif self._v[k] and not self._w[k]:
                new = z
            elif not self._v[k] and self._w[k]:
                new = x
            elif self._v[k] and self._w[k]:
                new = y
            else:
                raise ValueError("the string is not of the form 0 and 1")
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
        raise ValueError(
            "Paulis cannot be multiplied - different number of qubits")
    p1_v = P1.v.astype(np.bool)
    p1_w = P1.w.astype(np.bool)
    p2_v = P2.v.astype(np.bool)
    p2_w = P2.w.astype(np.bool)

    v_new = np.logical_xor(p1_v, p2_v).astype(np.int)
    w_new = np.logical_xor(p1_w, p2_w).astype(np.int)

    paulinew = Pauli(v_new, w_new)
    phase_changes = 0

    for v1, w1, v2, w2 in zip(p1_v, p1_w, p2_v, p2_w):
        if v1 and not w1:  # Z
            if w2:
                phase_changes = phase_changes - 1 if v2 else phase_changes + 1
        elif not v1 and w1:  # X
            if v2:
                phase_changes = phase_changes + 1 if w2 else phase_changes - 1
        elif v1 and w1:  # Y
            if not v2 and w2:  # X
                phase_changes -= 1
            elif v2 and not w2:  # Z
                phase_changes += 1
    phase = (1j) ** (phase_changes % 4)
    return paulinew, phase


def inverse_pauli(other):
    """Return the inverse of a Pauli."""
    return copy.deepcopy(other)


def label_to_pauli(label):
    """Return the pauli of a string ."""
    v = np.zeros(len(label), dtype=np.bool)
    w = np.zeros(len(label), dtype=np.bool)
    for j in len(label):
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
            raise ValueError(
                'something went wrong in label, must only contain IXYZ')
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

    else:
        raise ValueError("Please set the number of qubits to less than 5 but \
         {} is set.".format(number_of_qubits))


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
