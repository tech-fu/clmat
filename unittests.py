# -*- coding: utf-8 -*-
"""
Created on Apr 2014

@author: Nikola Karamanov
"""

import pytest
import numpy as np
import pyopencl as cl
import clmat as clm
import warnings


def assert_np_equal(x, y):
    """Assert numpy array equality."""
    np.testing.assert_array_equal(x, y)


def assert_np_float_ok(x, y):
    if x.dtype.kind == 'f' and y.dtype.kind == 'f':
        atolx = np.finfo(x.dtype).eps
        atoly = np.finfo(y.dtype).eps
        atol = max(atolx, atoly)
        np.testing.assert_allclose(x, y, rtol=float('inf'), atol=atol)
    elif x.dtype.kind == 'f':
        # This case is hopeless because it isn't clear why these
        assert False  # DO NOT USE THIS for non-floats
#        y = y.astype(x.dtype)
#        atol = np.finfo(x.dtype).eps
#        print(max(abs(x-y).flatten()))
#        assert np.all(abs(x-y)<atol)
#        np.testing.assert_allclose(x, y, rtol=float('inf'), atol=atol)
    elif y.dtype.kind == 'f':
        assert False  # This is completely unexpected in these tests
#        x = x.astype(y.dtype)
#        atol = np.finfo(y.dtype).eps
#        np.testing.assert_allclose(x, y, rtol=float('inf'), atol=atol)
    else:  # One of them is an int so all of them should be.
        assert np.all(x == y)


def test_computer_constructor():
    testC = clm.Computer()
    assert not testC.use_opencl
    assert testC.device is None
    assert testC.context is None
    assert testC.queue is None

    testC = clm.Computer(None)
    assert not testC.use_opencl
    assert testC.device is None
    assert testC.context is None
    assert testC.queue is None

    if clm.HAS_PYOPENCL:
        testC = clm.Computer(clm.CPU)
        assert testC.use_opencl
        assert isinstance(testC.device, cl.Device)
        assert isinstance(testC.context, cl.Context)
        assert isinstance(testC.queue, cl.CommandQueue)
        assert testC.device.type == clm.CPU

        if clm.has_gpu():
            testC = clm.Computer(clm.GPU)
            assert testC.use_opencl
            assert isinstance(testC.device, cl.Device)
            assert isinstance(testC.context, cl.Context)
            assert isinstance(testC.queue, cl.CommandQueue)
            assert testC.device.type == clm.GPU
        else:
            with pytest.raises(ValueError):
                testC = clm.Computer(clm.GPU)

        device = clm.get_all_devices()[0]
        testC = clm.Computer(device)
        assert testC.use_opencl
        assert isinstance(testC.device, cl.Device)
        assert testC.device is device
        assert isinstance(testC.context, cl.Context)
        assert len(testC.context.devices) == 1
        assert device in testC.context.devices
        assert isinstance(testC.queue, cl.CommandQueue)
        assert testC.queue.context == testC.context  # "is" not working here.
    else:
        assert clm.CPU is None
        assert clm.GPU is None
        assert clm.DEVICE_TYPE_MAP is None
        assert not clm.has_fission_support()
        assert len(clm.get_all_devices()) == 0


class TestMatrix:
    def test_matrix_basic(self, A, C, MA, is_empty=False):
        assert MA.computer.use_opencl == C.use_opencl
        assert MA.use_opencl == C.use_opencl
        assert MA.computer is C
        assert MA.shape == A.shape
        assert MA.size == A.size
        assert MA.dtype == A.dtype
        assert MA.begin == 0
        ptr_strides = [A.strides[0]/A.itemsize, A.strides[1]/A.itemsize]
        if A.shape[0] == 1:
            ptr_strides[0] = 0
        if A.shape[1] == 1:
            ptr_strides[1] = 0
        assert MA.ptr_stride0 == ptr_strides[0]
        assert MA.ptr_stride1 == ptr_strides[1]
        if 1 in A.shape:
            c_cont = A.flags.c_contiguous or A.flags.f_contiguous
            f_cont = c_cont
        else:
            c_cont = A.flags.c_contiguous
            f_cont = A.flags.f_contiguous

        assert MA.c_contiguous == c_cont
        assert MA.f_contiguous == f_cont
        order = ''
        if c_cont:
            order = 'CF' if f_cont else 'C'
        elif f_cont:
            order = 'F'
        assert MA.order == order

        M_NP = MA.NP
        if MA.use_opencl or is_empty:
            assert M_NP is not A
        else:
            assert M_NP is A

        if not is_empty:
            assert_np_equal(M_NP, A)

        Mcopy = MA.copy()
        Mcopy_A = Mcopy.NP
        assert Mcopy_A is not M_NP
        assert Mcopy_A.dtype == M_NP.dtype
        assert_np_equal(Mcopy_A, M_NP)
        assert_np_equal(MA.T.NP, M_NP.T)

    def test_matrix_cast(self, A, C, MA):
        """"""
        MA_d = MA.astype(np.double).NP
        MA_NP_d = MA.NP.astype(np.double)
        assert_np_equal(MA_d, MA_NP_d)
        assert MA_d.dtype == MA_NP_d.dtype

    @pytest.mark.parametrize("copy_c_contiguous", [True, False])
    def test_matrix_copy(self, MA, copy_c_contiguous):
        """"""
        assert_np_equal(MA.copy(c_contiguous=copy_c_contiguous).NP,
                        MA.copy().NP)

    @pytest.mark.parametrize('reshape_order', ['rC', 'rF', 'rA'])
    def test_matrix_reshape(self, A, MA, reshape_order):
        """"""
        assert MA.shape == A.shape
        new_shape = (MA.shape[1], MA.shape[0])
        ro = reshape_order[1]
        new_M = MA.reshape(new_shape, order=ro)
        assert new_M.shape == new_shape
        assert_np_equal(new_M.NP, MA.NP.reshape(new_shape, order=ro))

        flat_M = MA.flatten(order=ro)
        assert flat_M.shape[1] == 1
        assert_np_equal(flat_M.NP.flatten(), MA.NP.flatten(order=ro))


def test_matrix_getitem_slice(MALL, slice_index):
    for var in ['A', 'AF']:
        M = MALL.get_m(var)
        shape = M.shape
        proper_index = list(slice_index)
        for i in range(2):
            si = list(proper_index[i])
            if si[1] is None:
                pass
            else:
                if si[1] < 0:
                    si[1] = shape[i] - si[1]
                if si[1] > shape[1]:
                    return
            proper_index[i] = tuple(si)
        proper_index = tuple(proper_index)
        M_NP = MALL.get_np(var)
        assert_np_equal(M[proper_index].NP, M_NP[proper_index])


#def test_matrix_logical_index(MALL, logical_index):
#    i = lambda x: (x if isinstance(x, clm.Mat) else
#                   x.reshape(x.shape[0], (1 if x.ndim == 1 else x.shape[1])))
#
#    if logical_index.sum() == 0:
#        # TODO test exception
#        assert MALL.get_m('A').li(logical_index) is None
#        assert MALL.get_m('AF').li(logical_index) is None
#    else:
#        logical_index_np = logical_index.NP.astype(bool)
#        for var in ['A', 'AF']:
#            M = MALL.get_m(var)
#            M_NP = MALL.get_np(var)
#            assert_np_equal(M.li(logical_index).NP, i(M_NP[logical_index_np]))


@pytest.mark.parametrize('reverse', ['normal', 'reverse'])
def test_matrix_op(MALL, precise_op, op_other_var, reverse, recwarn):
    rev = reverse == 'reverse'
    if op_other_var == 'AT' and precise_op not in clm.OPS_LOGICAL:
        pytest.skip('Not logical, A.T not checked.')
    m = [None, None]
    n = [None, None]
    m[rev] = MALL.get_m('A')
    n[rev] = MALL.get_np('A')
    m[1-rev] = MALL.get_m(op_other_var)
    n[1-rev] = MALL.get_np(op_other_var)
    assert m[0].dtype == n[0].dtype
    assert m[1].dtype == n[1].dtype

    M_cmd = "m[0] %s m[1]" % (precise_op)
    NP_cmd = "n[0] %s n[1]" % (precise_op)

    if op_other_var == 'scalar':
        tmp = np.empty(n[rev].shape, dtype=n[1-rev].dtype)
        tmp.fill(n[1-rev])
        n[1-rev] = tmp
        del tmp
        NP = eval(NP_cmd)

        if rev:
            if precise_op in clm.OPS_NON_REVERSABLE:
                with pytest.raises(clm.MissingOperationError):
                    M = eval(M_cmd)
                return
            else:
                warnings.simplefilter("always")
                M = eval(M_cmd)
                if precise_op in ['==', '!=']:
                    pass
                else:
                    w = recwarn.pop(clm.RtypeWarning)
                    assert issubclass(w.category, clm.RtypeWarning)
        else:
            M = eval(M_cmd)
        assert M.shape == NP.shape
    else:
        NP = eval(NP_cmd)
        M = eval(M_cmd)

    if (rev and op_other_var == 'scalar'):
        # The reverse will be tested separately for scalars.
        pass
    elif op_other_var == 'scalar':
        assert M.dtype == NP.dtype
        if M.dtype.kind == 'f':
            atol = 4*np.finfo(np.dtype('float32')).eps
            np.all(np.max(abs(M.NP - NP)) <= atol)
#            np.testing.assert_allclose(M, NP, rtol=0, atol=atol)
        else:
            #TODO Test various integer related problems.
            pass
    else:
        assert M.dtype == NP.dtype
        assert_np_equal(M.NP, NP)
