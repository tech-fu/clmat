# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:38:31 2014

@author: Nikola Karamanov
"""

import pytest
import numpy as np
import clmat as clm

has_gpu = clm.has_gpu()

A_shape_list = ['1x1', '3x4', '40x1', '1x50', '1024x1024', '1025x1101']
B_shape_list = ['x5'] #, 'x1', 'x50', 'x1024', 'x1101']
dtype_list = list([k.name for k in clm.ALGEBRAIC_DTYPES])

gpu_idx = 0
computer_keys = ['n']
if clm.HAS_PYOPENCL:
    if clm.has_cpu():
        computer_keys += ['c']
    if clm.has_gpu():
        computer_keys += ['g']
else:
    computer_keys = ['n']

logical_index_list = ['GOOD', 'OOB']  # OOB = out of bounds
slice_begin = [0, 1]
slice_end = [-2]
slice_step = [1, 3]

# py.test doesn't preserve resources when multiple session args are passed.
# so we have to put them in a tuple.
def A_session_args():
    for a in A_shape_list:
        for d in dtype_list:
            yield '%s %s' % (a, d)


def ALL_session_args():
    for a in A_shape_list:
        for d in dtype_list:
            for od in dtype_list:
                yield '%s %s %s' % (a, d, od)


#def AB_session_args():
#    for a in A_shape_list:
#        for ad in dtype_list:
#            for b in B_shape_list:
#                for bd in dtype_list:
#                    yield '%s %s %s %s' % (a, ad, b, bd)


def slice_args():
    for b in slice_begin:
        for e in slice_end:
            for s in slice_step:
                for b2 in slice_begin:
                    for e2 in slice_end:
                        for s2 in slice_step:
                            yield ((b, e, s), (b2, e2, s2))


def new_matrix(shape, dtype):
    if dtype.kind == 'f':
        result = np.random.random(shape)
    else:
#        if dtype.kind == 'i':
#            high = 2**(4*dtype.itemsize)
#        else:
#            high = 2**(8*dtype.itemsize)
#        # randint needs a c long
#        if high >= 2**63:
#            high = 2**63-1
        high = 64  # Completely avoid tested with overflow problems.
        result = np.random.randint(1, high, size=shape)
    result = result.astype(dtype)
    return result


@pytest.fixture(scope='session', params=A_session_args())
def A(request):
    a, ad = request.param.split()
    A_shape = tuple(map(int, a.split('x')))
    result = new_matrix(A_shape, np.dtype(ad))
    return result


@pytest.fixture(scope='session', params=ALL_session_args())
def ALL(request):
    a, ad, od = request.param.split()
    A_shape = tuple(map(int, a.split('x')))
    odtype = np.dtype(od)
    result = {}
    result['A'] = new_matrix(A_shape, np.dtype(ad))
    result['AA'] = new_matrix(A_shape, odtype)
    result['u'] = new_matrix((A_shape[0], 1), odtype)
    result['ut'] = result['u'].T.copy()
    result['vt'] = new_matrix((1, A_shape[1]), odtype)
    result['v'] = result['vt'].T.copy()
    for k in result:
        assert result[k].flags.c_contiguous

    result['AT'] = result['A'].flatten(order='C').reshape(A_shape, order='F')
    result['AT'] = result['AT'].copy(order='C')
    result['AF'] = new_matrix(A_shape, odtype)
    result['AF'] = result['AF'].copy(order='F')
    assert result['AF'].flags.f_contiguous

    if A_shape[0] > 1 and A_shape[1] > 1:
        # This is important for checking logical operations.
        result['scalar'] = result['A'][1, 1].astype(odtype)
    else:
        result['scalar'] = new_matrix((1, 1), odtype)[0, 0]
    assert result['scalar'].dtype == odtype
    assert not isinstance(result['scalar'], np.ndarray)
    Amax = np.max(result['A'])
    Amin = np.min(result['A'])
    lb = Amin + (Amax-Amin)/4
    ub = Amax - (Amax-Amin)/4
    result['lb'] = lb
    result['ub'] = ub

    return result


@pytest.fixture(scope='module', params=computer_keys)
def C(request):
    computer_key = request.param
    device_arg = None
    if computer_key == 'c':
        device_arg = clm.CPU
    if computer_key == 'g':
        device_arg = clm.get_gpu(gpu_idx)
        if device_arg is None:
            print('WARNING: Could not load requested GPU: %d' % gpu_idx)
            device_arg = clm.GPU
        else:
            print('Will use GPU #%d: %s' % (gpu_idx, device_arg.name.strip()))
    result = clm.Computer(device_arg)
#    print('C: %s' % object.__repr__(result))
    return result


@pytest.fixture(scope='class')
def MA(A, C, request):
    result = C.M(A)
#    print('MA: %s' % object.__repr__(result))
    return result


class NumericNamespace:
    def __init__(self, c):
        self._c = c
        self._ns = {}

    def add(self, k, NP, M):
        self._ns[k] = (NP, M)

    def get_np(self, k):
        return self._ns[k][0]

    def get_m(self, k):
        return self._ns[k][1]

    @property
    def c(self):
        return self._c


@pytest.fixture(scope='class')
def MALL(ALL, C):
    result = NumericNamespace(C)
    for k in ALL:
        NP = ALL[k]
        if isinstance(NP, np.ndarray):
            assert k != 'scalar'
            M = C.M(NP)
            assert NP.dtype == M.dtype
        else:
            M = NP
        result.add(k, NP, M)
#    print('MALL: %s' % result)
    return result


@pytest.fixture(scope='function', params=clm.OPS_PRECISE)
def precise_op(request):
    return request.param


@pytest.fixture(scope='function',
                params=['scalar', 'AT', 'AA', 'AF', 'vt', 'u'])
def op_other_var(request):
    return request.param


#@pytest.fixture(scope='function', params=['GOOD'])
#def logical_index(MALL, request):
#    v = request.param
#    A = MALL.get_m('A')
#    if v == 'GOOD':
#        shape = A.shape
#    elif v == 'OOB':
#        shape = (A.shape[0] + 1, A.shape[1] + 1)
#    return A.computer.M(np.random.random(shape) >= .5)


@pytest.fixture(scope='function', params=slice_args())
def slice_index(MALL, request):
    return request.param
