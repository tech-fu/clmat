#!/usr/bin/python -B
# -*- coding: utf-8 -*-
"""
Created on April 2014

@author: Nikola Karamanov
"""

import pyopencl as cl
import clmat as clm

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Obtain info about the clmat module and related tasks.')
    parser.add_argument(
        '--all-devices', dest='all_devices', action='count', default=0,
        help='Print a list of devices sorted by type and device id.'
             ' Also prints all device info.')
    parser.add_argument(
        '--all-dtypes', dest='all_dtypes', action='count', default=0,
        help='Print a sorted list of supported dtypes.')
    parser.add_argument(
        '--safe-casts', dest='safe_casts', action='count', default=0,
        help='Print a list of safe casts based on dtype pairs.')
    args = parser.parse_args(sys.argv[1:])

    if len(sys.argv) == 1:
        parser.print_usage()

    if args.all_devices > 0:
        all_devices = clm.get_all_devices()
        device_i = 0
        for device in all_devices:
            print('Device %d (%s) : %s' %
                  (device_i, clm.DEVICE_TYPE_MAP[device.type], device.name.strip()))
            for attr in sorted(dir(device)):
                if attr[0] != '_':
                    try:
                        value = getattr(device, attr)
                        if not callable(value):
                            print('\t%s\t%s' % (attr, value))
                    except cl.LogicError as logical_err:
                        pass
            device_i += 1

    if args.all_dtypes > 0:
        print(sorted(clm.ALGEBRAIC_DTYPES))

    if args.safe_casts > 0:
        C = clm.Computer()
        for dt1 in sorted(clm.ALGEBRAIC_DTYPES):
            for dt2 in sorted(clm.ALGEBRAIC_DTYPES):
                safe = C.safe_cast_non_logical(dt1, dt2)
                print('%s + %s -> %s' % (dt1.name, dt2.name, safe.name))
