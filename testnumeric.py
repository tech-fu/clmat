#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Apr 2014

@author: Nikola Karamanov
"""

import site
site.addsitedir('.')
import sys
import argparse
from time import time
import numpy as np
from clmat import Computer, Mat, CPU, GPU, DEVICE_TYPE_MAP, REDUCTION_ENUM
import warnings

# Verbose levels vl must be >= to this
vl_error_summary = 0
vl_full_summary = 1
vl_times = 2
vl_cmds = 3
vl_data = 4

dt_names = {np.single: 's', np.double: 'd'}
dt_colors = {np.single: 36, np.double: 46}


def correct_nan_inf(x0, x1):
    nans = np.isnan(x0) * np.isnan(x1)
    posinfs = np.isposinf(x0) * np.isposinf(x1)
    neginfs = np.isneginf(x0) * np.isneginf(x1)
    if isinstance(x0, np.ndarray):
        x0 = x0.copy()
        x1 = x1.copy()
        x0[nans + posinfs + neginfs] = 0
        x1[nans + posinfs + neginfs] = 0
    elif nans or posinfs or neginfs:
        x0 *= 0
        x1 *= 0
    return x0, x1


def rel_err(x0, x1, x0_baseline=False):
    x0 = np.array(x0)
    x1 = np.array(x1)
    if x0_baseline:
        normalizer = np.linalg.norm(x0.flatten(), 2)
    else:
        normalizer = (np.linalg.norm(x0.flatten(), 2) +
                      np.linalg.norm(x1.flatten(), 2))/2
    if normalizer == 0:
        normalizer = 1
    return np.linalg.norm((x1-x0).flatten(), 2)/normalizer


def max_rel_diff(x0, x1, x0_baseline=False):
    x0 = np.array(x0).flatten()
    x1 = np.array(x1).flatten()
    if x0_baseline:
        normalizer = np.abs(x0)
    else:
        normalizer = (np.abs(x0)+np.abs(x1))/2
    if normalizer.ndim > 0:
        normalizer[normalizer == 0] = 1.0
    elif normalizer == 0:
        normalizer = 1
    return np.max(np.abs(x1-x0)/normalizer)


def color(s, color_int):
    """color_int: 30 Gray,31 Red,32 Green,33 Yellow,34 Blue,
                  35 Magenta,36 Cyan,37 White,38 Crimson,41-48 highlighted
    """
    return "\033[1;%dm%s\033[1;m" % (color_int, s)


def eps_status(val, dt, error_threshold=8):
    if np.isnan(val):
        return 1, float('nan'), color(' NAN', 31), color(' NAN', 31)
    MAX_DISP = 999
    val_str = '%.3e' % val
    eps_multiple = val/np.finfo(dt).eps
    if eps_multiple <= 1:
        status = -1
        status_str = ' OK '
        cid = 32
    else:
        status = int(eps_multiple > error_threshold)
        cid = 31 if status > 0 else 33
        if eps_multiple < MAX_DISP:
            status_str = '%3dx' % eps_multiple
        else:
            status_str = '>%d' % MAX_DISP
    status_str = color(status_str, cid)
    #if status==1: val_str = color(val_str,cid)
    val_str = color(val_str, cid)
    return status, eps_multiple, status_str, val_str


def parse_args(raw_args):
    parser = argparse.ArgumentParser(
        description='Test opencl code and the neural network computer.')
    parser.add_argument(
        '--list', dest='list_functions',
        required=False, action='count', default=0,
        help='List the functions currently testable.')
    parser.add_argument(
        '-f', '--function', dest='function_names', nargs='+',
        required=False, default='all',
        help='A list of functions/operations to test. Default is all.')
    parser.add_argument(
        '-t', '--time', dest='time', required=False, type=int, default=1,
        help='Print the time required to execute a single cl function'
             'the given number of times.'
             ' -t 100 will time 100 function calls.'
             'Must speficy the function name with -f.'
             ' Can only run with one computer type and one precision type.')
    parser.add_argument(
        '-m', '--memory', dest='memory',
        required=False, type=int, default=0,
        help='Print the time required to transfer memory to device'
             ' and back a given number of times.'
             ' Can only run with one computer type and one precision type.')
    parser.add_argument(
        '-s', '--single',
        dest='single', required=False, action='count', default=0,
        help='Use single precision for tests (rather than double).')
    parser.add_argument(
        '-n', '--np', dest='test_np',
        required=False, action='count', default=0,
        help='Test numpy based computations with Computer.')
    parser.add_argument(
        '-g', '--gpu', dest='test_gpu',
        required=False, action='count', default=0,
        help='Test gpu based computations.')
    parser.add_argument(
        '-c', '--cpu', dest='test_cpu',
        required=False, action='count', default=0,
        help='Test cpu based computations.')
    parser.add_argument(
        '--dist', dest='test_distributions',
        required=False, action='count', default=0,
        help='Test the random number generator statistics.')
    parser.add_argument(
        '-A', '--sizeA', dest='sizeA',
        required=False, default='1024x1024',
        help='The size of matrix A in ?x? format (default 1024x1024)')
    parser.add_argument(
        '-B', '--sizeB',
        dest='sizeB', required=False, default='1024x1024',
        help='The second dimension of matrix x? format (default x1024).'
             ' First dim of B will be set to match second dim of A.')
    parser.add_argument(
        '--gen', dest='generation_method',
        required=False, default='randn',
        help='The method to use to generate the matrices.'
             ' Possible are: random, randn, ones, arange.')
    parser.add_argument(
        '--gen_m', dest='generation_scalar',
        required=False, type=float, default=(1.0/64),
        help='A scalar to use to multiply all matrices at construction.'
             ' Default is 1/64, this eases exp usage and also removes '
             'roundoff noise from algebraic computations if you generate ones.'
             ' This is useful for control over randn and generating zeros.')
    parser.add_argument(
        '--epsm', dest='eps_error_multiple',
        required=False, type=float, default=8.0,
        help='Multiple of machine precision that should constitute an error.')
    parser.add_argument(
        '--time-sort', dest='time_sort', required=False, default='',
        help='Print results sorted by their times (after the usual printouts).'
             ' Use "bt"/"baseline" or "cl" to choose which time to sort by.')
    parser.add_argument(
        '-v', '--verbose', dest='verbose',
        required=False, action='count', default=0,
        help='Verbose level. 0 - summaries of problems, 1 - full summaries, '
             '2 - times and major steps, 3 - commands being executed, '
             '4 - data (numpy arrays).')
    parser.add_argument(
        '--debug', dest='debug',
        required=False, action='count', default=0,
        help='Will print tracebacks and/or exceptions.')
    parser.add_argument(
        '-w', '--won', dest='warnings_on',
        required=False, action='count', default=0,
        help='Will turn on warnings (disabled by default).')
    args = parser.parse_args(raw_args)
    return args


# HOW TO WRITE COMMANDS
# cmds[operation_name] = (clmat_cmd,true_cmd,max_precision_loss_cmd)
# A max_precision_loss_cmd of 1.0 implies no tolerance for numerical errors.
# A max_precision_loss_cmd on f(x) of abs(f(x))
# will lead to computing the max rel diff
# Symbols: A (mxn) , rA(special, see explanation), AA (mxn) , B (nxp) ,
#          u (mx1) , ut (1xm) , v (nx1) , vt (1xn) ,
#          a* is abs(*) , scalar (double)
#          AF (mxn) with order 'F' ,
#          {lb (double) , ub (double) lower/upper bounds for thresholding}
# ; in the command signifies that it is an in-place operation.
# rA is a copy of A only available for in-place operations,so A is not altered.


def constructor_cmds(cmds):
    cmds['copy'] = ('A.copy()', 'A.copy()', '1.0')
    cmds['copy'] = ('rA.copy();', 'A.copy()', '1.0')
    cmds['Ttranspose'] = ('A.transpose()', 'A.transpose()', '1.0')
    cmds['T'] = ('A.T', 'A.T', '1.0')
    cmds['NP'] = ('A.NP', 'A.copy()', '1.0')
    cmds['flatten'] = ('A.flatten()', 'A.flatten().reshape(A.size,1)', '1.0')
    cmds['reshape'] = ('A.reshape((A.shape[1],A.shape[0]))',
                       'A.reshape((A.shape[1],A.shape[0]))', '1.0')
    cmds['zeros'] = ('clc.zeros(A.shape,dtype=A.dtype)',
                     'np.zeros(A.shape,dtype=A.dtype)', '1.0')
    cmds['ones'] = ('clc.ones(A.shape,dtype=A.dtype)',
                    'np.ones(A.shape,dtype=A.dtype)', '1.0')
    cmds['randu'] = ('clc.randu((1024,1024)).size',
                     'np.random.random((1024,1024)).size', '1.0')
    cmds['randn'] = ('clc.randu((1024,1024)).size',
                     'np.random.randn(1024,1024).size', '1.0')
    cmds['fill'] = ('rA.fill(AA);', 'rA[:]=AA;', '1.0')
    cmds['fillc'] = ('rA.fill(scalar);', 'rA.fill(scalar);', '1.0')
    cmds['li'] = ('A.li(A>0)','A[A>0][:,None]','1.0')
    return cmds


def operator_cmds(cmds):
    cmds['m-'] = ('(-A)', '(-A)', '1.0')
    cmds['m+'] = ('(+A)', '(+A)', '1.0')
    cmds['mabs'] = ('abs(A)', 'abs(A)', '1.0')

    non_reservable_ops = ['<', '<=', '>', '>=']
    ops = ['+', '-', '*', '/', '==', '!=', '**'] + non_reservable_ops
    iops = ['+=', '-=', '*=', '/=', '**=']
    ops_var_legend = {'m': 'AA', 'c': 'scalar', 'r': 'AF', 'v': 'vt', 'u': 'u'}
    for o in ops:
        for suf in ops_var_legend:
            ocmd = 'A%s%s' % (o, ops_var_legend[suf])
            cmds['m'+o+suf] = (ocmd, ocmd, '1.0')
            # Check reversed order operation, especially c+m.
            # Reversed comparisons don't work, yet (python issue)
            if suf != 'm' and o not in non_reservable_ops:
                ocmd = '%s%sA' % (ops_var_legend[suf], o)
                cmds[suf+o+'m'] = (ocmd, ocmd, '1.0')
            ocmd = 'AF%s%s' % (o, ops_var_legend[suf])
            cmds['r'+o+suf] = (ocmd, ocmd, '1.0')
    for o in iops:
        for suf in ops_var_legend:
            ocmd = 'rA%s%s;' % (o, ops_var_legend[suf])
            cmds['m'+o+suf] = (ocmd, ocmd, '1.0')
            ocmd = 'rAF%s%s;' % (o, ops_var_legend[suf])
            cmds['r'+o+suf] = (ocmd, ocmd, '1.0')
    return cmds


def reduction_cmds(cmds):
    reductions = REDUCTION_ENUM
    reshapes = {'None': 'None', '0': '(1,A.shape[1])', '1': '(A.shape[0],1)'}
    for r in reductions:
        for a in reshapes:
            clcmd = 'A.%s(axis=%s)' % (r, a)
            npcmd = 'A.%s(axis=%s).reshape(%s)' % (r, a, reshapes[a])
            if r in ['sum', 'mean']:
                pcmd = 'aA.%s(axis=%s).reshape(%s)' % (r, a, reshapes[a])
            else:
                pcmd = '1.0'
            if a == 'None':
                cmds[r] = (clcmd, npcmd, pcmd)
            else:
                cmds[r+a] = (clcmd, npcmd, pcmd)
    return cmds


def map_cmds(cmds):
    cmds['round'] = ('A.round()', 'A.round(0)', '1.0')
    cmds['roundi'] = ('A.round(out=rA);', 'A.round(0,out=rA);', '1.0')
    cmds['clip'] = ('A.clip(lb,ub)', 'A.clip(lb,ub)', '1.0')
    cmds['clipi'] = ('A.clip(lb,ub,out=rA);', 'A.clip(lb,ub,out=rA);', '1.0')

    cmds['sqrt'] = ('clc.sqrt(A*A)', 'np.sqrt(A*A)', '1.0')
    cmds['exp'] = ('clc.exp(A)', 'np.exp(A)', '1.0')
    cmds['log'] = ('clc.log(abs(A))', 'np.log(abs(A))', '1.0')
    cmds['log1p'] = ('clc.log1p(abs(A))', 'np.log1p(abs(A))', '1.0')
    #cmds['xlogy'] = ('clc.xlogy(A,abs(AA))','A*np.log(abs(AA))','1.0')
    return cmds


def mmult_cmds(cmds):
    cmds['innerp'] = ('clc.innerprod(A,A)', 'np.dot(A.flat,A.flat)', 'np.dot(aA.flat,aA.flat)')
    cmds['innerpr'] = ('clc.innerprod(AF,AF)', 'np.dot(AF.flat,AF.flat)', 'np.dot(aAF.flat,aAF.flat)')
    cmds['innerpm'] = ('clc.innerprod(A,AF)', 'np.dot(A.flat,AF.flat)', 'np.dot(aA.flat,aAF.flat)')
    cmds['mvmult'] = ('clc.mmult(A,v)', 'np.dot(A,v)', 'np.dot(aA,av)')
    cmds['vmmult'] = ('clc.mmult(ut,A)', 'np.dot(ut,A)', 'np.dot(aut,aA)')
    cmds['mmult'] = ('clc.mmult(A,B)', 'np.dot(A,B)', 'np.dot(aA,aB)')
    cmds['mmult_m1t'] = ('clc.mmult(A.T,A)', 'np.dot(A.T,A)',
                         'np.dot(aA.T,aA)')
    cmds['mmult_m2t'] = ('clc.mmult(B,B.T)', 'np.dot(B,B.T)',
                         'np.dot(aB,aB.T)')
    cmds['mmult_m1t_m2t'] = ('clc.mmult(B.T,A.T)', 'np.dot(B.T,A.T)',
                             'np.dot(aA,aB).T')

    return cmds


def distribution_cmds(cmds):
    cmds['randum'] = (
        'np.mean(clc.randu((5120,5120),dtype=A.dtype).NP)', '.5',
        '1.0/abs(np.mean(np.random.random((5120,5120)).astype(A.dtype))-.5)')
    cmds['randus'] = (
        'np.std(clc.randu((5120,5120),dtype=A.dtype).NP)', 'np.sqrt(1.0/12)',
        '1.0/abs(np.std(np.random.random((5120,5120)).astype(A.dtype))-np.sqrt(1.0/12))')
    cmds['randnm'] = (
        'np.mean(clc.randn((5120,5120),dtype=A.dtype).NP)', '0.0',
        '1.0/abs(np.mean(np.random.randn(5120,5120).astype(A.dtype)))')
    cmds['randns'] = (
        'np.std(clc.randn((5120,5120),dtype=A.dtype).NP)', '1.0',
        '1.0/abs(np.std(np.random.randn(5120,5120).astype(A.dtype))-1.0)')
    return cmds


def get_computers(args):
    device_args = []
    if args.test_np > 0:
        device_args += [None]
    if args.test_cpu > 0:
        device_args += [CPU]
    if args.test_gpu > 0:
        device_args += [GPU]

    if len(device_args) == 0:
        print('No -n or -c or -g option used. Will test all.')
        device_args = [None, CPU, GPU]

    computers = {}
    for da in device_args:
        c_key = 'np' if da is None else DEVICE_TYPE_MAP[da]
        try:
            computers[c_key] = Computer(da)
        except Exception as e:
            print('Caught exception: %s' % e)
            if args.debug > 0:
                import traceback
                traceback.print_exc()
            print('Couldn\'t create the computer %s.' % (c_key))
    return computers


def get_vars(args):
    shapeA = tuple(map(int, args.sizeA.split('x')))
    shapeB = (shapeA[1], int(args.sizeB.split('x')[-1]))
    assert len(shapeA) == 2 and len(shapeB) == 2  # Only 2D matrices allowed.

    onlyA = args.memory > 0 and args.time == 0
    dtype = np.single if args.single > 0 else np.double

    if shapeA[1] != shapeB[0]:
        print('Bad sizes for A,B: %s,%s' % (shapeA, shapeB))
        exit(1)
    if args.generation_method == 'random':
        creator_f = lambda sz: np.random.random(sz)
    elif args.generation_method == 'randn':
        creator_f = lambda sz: np.random.randn(sz[0], sz[1])
    elif args.generation_method == 'ones':
        creator_f = lambda sz: args.generation_scalar*np.ones(sz)
    elif args.generation_method == 'arange':
        creator_f = lambda sz: \
            args.generation_scalar*np.arange(np.prod(sz)).reshape(sz)
    new_m = lambda sz: (args.generation_scalar*creator_f(sz)).astype(dtype)

    test_vars = {}
    test_vars['A'] = new_m(shapeA).astype(dtype)
    if not onlyA:
        test_vars['AA'] = new_m(shapeA).astype(dtype)
        test_vars['AF'] = new_m((shapeA[1], shapeA[0])).astype(dtype).T
        assert test_vars['AF'].flags.f_contiguous
        test_vars['B'] = new_m(shapeB).astype(dtype)
        test_vars['u'] = new_m((shapeA[0], 1)).astype(dtype)
        test_vars['ut'] = new_m((1, shapeA[0])).astype(dtype)
        test_vars['v'] = new_m((shapeA[1], 1)).astype(dtype)
        test_vars['vt'] = new_m((1, shapeA[1])).astype(dtype)
        # For logical comparisons it is best to use a scalar from A
        scalar = test_vars['A'][shapeA[0]//2, shapeA[1]//2]

        Amax = np.max(test_vars['A'])
        Amin = np.min(test_vars['A'])
        lb = Amin + (Amax-Amin)/4
        ub = Amax - (Amax-Amin)/4

        fresh_tmp_vars = {}
        fresh_tmp_vars['np'] = np
        fresh_tmp_vars['scalar'] = scalar
        fresh_tmp_vars['Amax'] = Amax
        fresh_tmp_vars['Amin'] = Amin
        fresh_tmp_vars['lb'] = lb
        fresh_tmp_vars['ub'] = ub
    return test_vars, fresh_tmp_vars


def run_operation(operation_name, args, cmds,
                  computers, test_vars, fresh_tmp_vars,
                  max_diffs, rel_errs, norm_diffs, max_rel_diffs,
                  max_precision_losses, baseline_times, speed_gains):

    dtype = np.single if args.single > 0 else np.double
    true_cmd = cmds[operation_name][1]
    if args.verbose >= vl_cmds:
        print('  true command: %s' % true_cmd)
    precision_cmd = cmds[operation_name][2]
    if args.verbose >= vl_cmds:
        print('  precision command: %s' % precision_cmd)

    tmp_vars = fresh_tmp_vars.copy()
    for vn in test_vars:
        tmp_vars[vn] = test_vars[vn]
        tmp_vars['a' + vn] = abs(test_vars[vn])

    if ';' in true_cmd:
        assert 'rA' in true_cmd  # Others not known to this loop
        tmp_vars['rA'] = tmp_vars['A'].copy()
        tmp_vars['rAF'] = tmp_vars['AF'].copy(order='F')
        t0 = time()
        for dummy in range(args.time):
            exec(true_cmd) in tmp_vars
        time_delta = time() - t0
        true_result = tmp_vars['rA']
        del tmp_vars['rA'], tmp_vars['rAF']
    else:
        t0 = time()
        for dummy in range(args.time):
            true_result = eval(true_cmd, tmp_vars)
        time_delta = time() - t0

    baseline_times[operation_name] = time_delta
    del time_delta

    if args.verbose >= vl_times:
        print('    time[%s,%s]: %.2e' %
              (operation_name, dt_names[dtype],
               baseline_times[operation_name]))
    assert ';' not in precision_cmd  # Not implemented
    max_loss = eval(precision_cmd, tmp_vars)
    max_precision_losses = max_loss

    del tmp_vars, true_cmd, precision_cmd

    clc_cmd = cmds[operation_name][0]
    if args.verbose >= vl_cmds:
        print('  clc command: %s' % clc_cmd)
    for computer_name in computers:
        # Computers loop last here to cycle devices
        # and allow then to cool off.
        clc = computers[computer_name]
        if args.verbose >= vl_times:
            print('   clc: %s' % computer_name)

        if args.verbose >= vl_cmds:
            print('   dtype: %s' % dtype)

        tmp_vars = fresh_tmp_vars.copy()
        tmp_vars['clc'] = clc
        for vn in test_vars:
            tmp_vars[vn] = clc.M(test_vars[vn])
            tmp_vars[vn].copy()

        if ';' in clc_cmd:
            assert 'rA' in clc_cmd  # Others not known to this loop
            tmp_vars['rA'] = tmp_vars['A'].copy()
            tmp_vars['rAF'] = tmp_vars['AF'].copy(c_contiguous=False)
            exec(clc_cmd) in tmp_vars
            tmp_vars['rA'] = tmp_vars['A'].copy()
            tmp_vars['rAF'] = tmp_vars['AF'].copy(c_contiguous=False)
            t0 = time()
            for dummy in range(args.time):
                exec(clc_cmd) in tmp_vars
            time_delta = time() - t0
            clc_result = tmp_vars['rA']
            del tmp_vars['rA'], tmp_vars['rAF']
        else:
            clc_result = eval(clc_cmd, tmp_vars)
            t0 = time()
            for dummy in range(args.time):
                clc_result = eval(clc_cmd, tmp_vars)
            time_delta = time() - t0

        del tmp_vars

        if isinstance(clc_result, Mat):
            if 'scalar' == clc_cmd[0:6]:
                pass  # No dtype checks for scalar first.
            elif true_result.dtype not in [np.bool]:
                assert clc_result.dtype == true_result.dtype
            clc_result = clc_result.NP
            if clc_result.shape != true_result.shape:
                print('clc-true shape mismatch %s != %s' %
                      (clc_result.shape, true_result.shape))
                exit(1)
        speed_gains[operation_name][computer_name] = \
            baseline_times[operation_name]/time_delta
        if args.verbose >= vl_times:
            print('     time[%s,%s]: %.2e' %
                  (operation_name, dt_names[dtype], time_delta))
        del time_delta

        if args.verbose >= vl_data:
            print('True:')
            print(true_result)
            print('ClC:')
            print(clc_result)

        mpl = max_precision_losses
        if isinstance(mpl, np.ndarray):
            mpl[mpl == 0] = 1.0
        tr, cr = correct_nan_inf(true_result, clc_result)
        # numpy does not support subtract for bool anymore
        if hasattr(cr, 'dtype') and np.issubdtype(cr.dtype, np.bool):
            cr = cr.astype(np.uint8)
        if hasattr(tr, 'dtype') and np.issubdtype(tr.dtype, np.bool):
            tr = tr.astype(np.uint8)
        diff = (cr-tr)
        rel_errs[operation_name][computer_name] = \
            rel_err(tr, cr, x0_baseline=True)
        max_rel_diffs[operation_name][computer_name] = \
            max_rel_diff(tr, cr, x0_baseline=True)
        norm_diffs[operation_name][computer_name] = \
            np.max(np.abs(diff)/mpl)
        max_diffs[operation_name][computer_name] = \
            np.max(np.abs(diff))


def summarize(computer_name, operation_name, args,
              error_summaries, summaries_by_time,
              max_diffs, rel_errs, norm_diffs, max_rel_diffs,
              max_precision_losses, baseline_times, speed_gains):

    dtype = np.single if args.single > 0 else np.double
    error_summaries[computer_name][operation_name] = []
    op_show = color(operation_name, 35)

    speed_gain = speed_gains[operation_name][computer_name]

    rel_err = rel_errs[operation_name][computer_name]
    max_rel_diff = max_rel_diffs[operation_name][computer_name]
    norm_diff = norm_diffs[operation_name][computer_name]
    max_diff = max_diffs[operation_name][computer_name]

    dt_show = color(dt_names[dtype], dt_colors[dtype])

    summary = color(' |', dt_colors[dtype])

    baseline_t = baseline_times[operation_name]
    speed_show = color('%7.2f [%.2e/%.2e]' %
                       (speed_gain, baseline_t,
                        baseline_t/speed_gain),
                       36 if speed_gain > 1 else 30)
    summary += speed_show

    es = lambda e, t: eps_status(e, dtype,
                                 error_threshold=args.eps_error_multiple)[t]

    summary += '[%s,%s,%s]' % (
        es(rel_err, 2), es(max_rel_diff, 2), es(norm_diff, 2))
    summary += ' %s[%s]: ' % (op_show, dt_show)
    summary += '[ relerr=%s , maxreldiff=%s ,' \
               ' normdiff=%s ] , maxdiff=%.3e' % \
               (es(rel_err, 3), es(max_rel_diff, 3),
                es(norm_diff, 3), max_diff)

    status = np.max([
        es(rel_err, 0), es(max_rel_diff, 0), es(norm_diff, 0)])
    if status > 0:
        error_summaries[computer_name][operation_name] += [summary]

    if args.verbose >= vl_full_summary:
        print(summary)
    if args.time_sort:
        t = baseline_t if args.time_sort in ['bt', 'baseline'] \
            else (baseline_t/speed_gain)
        if t not in summaries_by_time:
            summaries_by_time[t] = []
        summaries_by_time[t] += [computer_name + summary]
    return status


def main(raw_args):
    args = parse_args(raw_args)

    if not (args.warnings_on > 0):
        warnings.simplefilter('ignore')

    assert args.time_sort in ['', 'bt', 'baseline', 'cl']

    dtype = np.single if args.single > 0 else np.double

    ##### GET ALL THE COMMANDS #####

    cmds = {}
    constructor_cmds(cmds)
    operator_cmds(cmds)
    reduction_cmds(cmds)
    map_cmds(cmds)
    mmult_cmds(cmds)
    if args.test_distributions > 0:
        distribution_cmds(cmds)
    operations = sorted(cmds.keys())

    if args.list_functions > 0:
        print(operations)
        print('NOTE: operations may be contingent on flags (e.g. --dist).')
        exit(0)

    if args.function_names != 'all':
        for ff in args.function_names:
            if ff not in operations:
                print('Unrecognized function/operation name: ' + ff +
                      '. Use --list to see available.')
                exit(1)
        operations = args.function_names

    ##### CONSTRUCT COMPUTERS #####

    computers = get_computers(args)
    if len(computers) == 0:
        print('No computers were created.')
        exit(0)

    ##### CONSTRUCT MATRICES #####

    test_vars, fresh_tmp_vars = get_vars(args)

    if args.verbose >= vl_data:
        for vn in test_vars:
            print(vn+':')
            print(test_vars[vn])
        print('scalar: %e' % fresh_tmp_vars['scalar'])
        print('Amax = %.3e\nAmin = %.3e\nlb = %.3e\nub = %.3e' %
              (fresh_tmp_vars['Amax'], fresh_tmp_vars['Amin'],
               fresh_tmp_vars['lb'], fresh_tmp_vars['ub']))

    ##### MEMORY #####

    if args.memory > 0:
        assert len(computers) == 1  # Must use only one computer.
        c_key = computers.keys()[0]
        clc = computers[c_key]
        A = test_vars['A']
        sys.stdout.write('Computer %s transfered %d x  M(A[%s]).NP in  ' %
                        (c_key, args.memory, dt_names[dtype]))
        sys.stdout.flush()
        t0 = time()
        for i in range(args.memory):
            clc.M(A).NP
        time_delta = time() - t0
        decimal = color('%015.10f' % time_delta, 31)
        scientific = color('%.2e' % time_delta, 32)
        print('%s  %s  seconds.' % (decimal, scientific))
        if args.time == 0:
            exit(0)

    ##### TIME #####

    if False:
        assert len(computers) == 1  # Must use only one computer.
        c_key = computers.keys()[0]

        clc = computers[c_key]

        tmp_vars = fresh_tmp_vars.copy()
        tmp_vars['clc'] = clc
        for vn in test_vars:
            tmp_vars[vn] = clc.M(test_vars[vn])
            # Make sure data is on device for correct timing.
            tmp_vars[vn].copy()

        max_operation_len = max([len(o) for o in operations])

        for operation_name in operations:
            clc_cmd = cmds[operation_name][0]

            #op_format =
            summary_format = 'Computer %s executed   %d x ' + \
                             ('%%0%ds' % (max_operation_len+1)) + '[%s] in  '
            sys.stdout.write((summary_format) %
                             (c_key, args.time, operation_name,
                              dt_names[dtype]))
            sys.stdout.flush()

            if ';' in clc_cmd:
                assert 'rA' in clc_cmd  # Others not known to this loop
                tmp_vars['rA'] = tmp_vars['A'].copy()
                tmp_vars['rAF'] = tmp_vars['AF'].copy('F')
            else:
                clc_cmd = 'clc_result = %s' % clc_cmd

            # Run once to remove overhead from kernel compilation.
            exec(clc_cmd) in tmp_vars
            t0 = time()
            for i in range(args.time):
                exec(clc_cmd) in tmp_vars
            time_delta = time() - t0

            decimal = color('%015.10f' % time_delta, 31)
            scientific = color('%.2e' % time_delta, 32)
            print('%s  %s  seconds.' % (decimal, scientific))

            if ';' in clc_cmd:
                del tmp_vars['rA'], tmp_vars['rAF']

        del tmp_vars
        exit(0)

    ##### COMPUTE MEASUREMENTS #####

    max_diffs = {}
    rel_errs = {}
    norm_diffs = {}
    max_rel_diffs = {}
    speed_gains = {}
    baseline_times = {}
    for operation_name in operations:
        if args.verbose >= vl_times:
            print(' operation: %s' % operation_name)

        max_diffs[operation_name] = {}
        rel_errs[operation_name] = {}
        norm_diffs[operation_name] = {}
        max_rel_diffs[operation_name] = {}
        max_precision_losses = {}

        baseline_times[operation_name] = {}
        speed_gains[operation_name] = {}

        run_operation(operation_name, args, cmds,
                      computers, test_vars, fresh_tmp_vars,
                      max_diffs, rel_errs, norm_diffs, max_rel_diffs,
                      max_precision_losses, baseline_times, speed_gains)

    ##### PRINT ANALYSIS #####

    if args.verbose >= vl_full_summary:
        print(color('THE FULL ANALYSIS (based on %dxEPS):' %
                    args.eps_error_multiple, 44))
    error_summaries = {}
    summaries_by_time = {}
    problem_counter = 0
    for computer_name in computers:
        error_summaries[computer_name] = {}
        if args.verbose >= vl_full_summary:
            clc = computers[computer_name]
            print(color('Computer %s: %s' %
                  (computer_name, clc.summary()), 44))
            del clc
        for operation_name in operations:
            problem_counter += 0 < summarize(
                computer_name, operation_name, args,
                error_summaries, summaries_by_time,
                max_diffs, rel_errs, norm_diffs, max_rel_diffs,
                max_precision_losses, baseline_times, speed_gains)

    if problem_counter == 0:
        print(color('NO PROBLEMS DETECTED', 42))
    else:
        if args.verbose >= vl_error_summary:
            print('Machine epsilons: double=%s , single=%s' %
                  (color('%e' % np.finfo(np.double).eps, 33),
                   color('%e' % np.finfo(np.single).eps, 33)))
        if args.verbose >= vl_error_summary:
            print(color(
                'THESE ARE THE PROBLEMATIC RESULTS (based on %dxEPS):' %
                args.eps_error_multiple, 41))
        for computer_name in error_summaries:
            clc = computers[computer_name]
            total_clc_errs = np.sum(
                [len(el) for el in error_summaries[computer_name].values()])
            if total_clc_errs > 0:
                if args.verbose >= vl_error_summary:
                    print(color('Computer %s: %s' %
                          (computer_name, clc.summary()), 41))
                for operation_name in sorted(error_summaries[computer_name]):
                    for summary in \
                            error_summaries[computer_name][operation_name]:
                        if args.verbose >= vl_error_summary:
                            print(summary)
        print(color('TOTAL POSSIBLE PROBLEMS: %d' % problem_counter, 41))

    if args.time_sort:
        print(color('RESULTS SORTED BY %s TIME' % args.time_sort, 46))
        for bt in sorted(summaries_by_time):
            for s in summaries_by_time[bt]:
                print(s)


if __name__ == "__main__":
    main(sys.argv[1:])
