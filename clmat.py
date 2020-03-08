# -*- coding: utf-8 -*-
"""
A module for numerical computations that supports
opencl and non-opencl computations.

Initial version created in 2013

@author: Nikola Karamanov
"""
import numpy as np
import warnings

CPU = None
"""pyopencl.device_type.CPU or None if pyopencl is not available"""

GPU = None
"""pyopencl.device_type.GPU or None if pyopencl is not available"""

DEVICE_TYPE_MAP = None
"""Useful mapping from pyopencl.device_type to string."""

HAS_PYOPENCL = False
"""True iff pyopencl was successfully loaded and can be used."""

try:
    import pyopencl as cl

    CPU = cl.device_type.CPU

    GPU = cl.device_type.GPU

    DEVICE_TYPE_MAP = {cl.device_type.CPU: 'cpu', cl.device_type.GPU: 'gpu'}

    HAS_PYOPENCL = True

except ImportError:
    warnings.warn('clmat could not import pyopencl, '
                  'all computation will be done in pure numpy. '
                  'pyopencl-dependent functionality will not be available.',
                  ImportWarning)


_DTYPE = np.dtype
"""A shorthand for numpy.dtype"""

BOOL_TYPE = _DTYPE('bool')
"""The dtype to use when the data is boolean in nature."""

ALGEBRAIC_DTYPES = {_DTYPE('float32'): 'float', _DTYPE('float64'): 'double',
                    _DTYPE('int8'): 'char', _DTYPE('int16'): 'short',
                    _DTYPE('int32'): 'int', _DTYPE('int64'): 'long',
                    _DTYPE('uint8'): 'uchar', _DTYPE('uint16'): 'ushort',
                    _DTYPE('uint32'): 'uint', _DTYPE('uint64'): 'ulong'}
"""Algebraic dtypes and their corresponding opencl api type names."""

SUPPORTED_DTYPES = ALGEBRAIC_DTYPES.copy()
"""Supported dtypes and their corresponding opencl api type names."""
SUPPORTED_DTYPES[BOOL_TYPE] = 'char'


# Note that argmax and argmin return int64 so SIZE_T must be at least uint32
def _size_t(any_num):
    """Casts a number to dtype SIZE_T which matches opencl KERNEL_SIZE_T."""
    return np.uint32(any_num)

# Using size_t or ulong is known to cause issues one some platforms.
KERNEL_SIZE_T = 'unsigned int'
"""The type that all kernels use for Mat size and shape."""

SIZE_T_MAX = _size_t(2**32-1)
"""The maximum supported number for sizes and indices."""

ZERO = _size_t(0)
"""@type: _size_t"""
ONE = _size_t(1)
"""@type: _size_t"""
TWO = _size_t(2)
"""@type: _size_t"""

OPS_LOGICAL = ['==', '!=', '>', '<', '>=', '<=']
"""Operators that return a BOOL_TYPE."""
OPS_ALGEBRAIC = ['+', '-', '*', '/', '**']
OPS_NON_REVERSABLE = ['<', '<=', '>', '>=']
"""Operators that don't support scalar-matrix operations in that order."""
OPS_PRECISE = OPS_LOGICAL + OPS_ALGEBRAIC[0:-2]
"""Operators that are expected to yield same results for numpy and opencl."""
OPS_NON_PRECISE = OPS_ALGEBRAIC[-2:]
"""Operators that are may yield different results for numpy and opencl."""
OPS = OPS_PRECISE + OPS_NON_PRECISE
"""Operators that DO NOT act in-place."""
IOPS = ['+=', '-=', '*=', '/=', '**=']
"""Operators that act in-place."""

REDUCTION_ENUM = {'all': 0, 'any': 1, 'sum': 5, 'prod': 6,
                  'max': 2, 'min': 3, 'ptp': 4, 'argmax': 12, 'argmin': 13,
                  'mean': 17, 'var': 18, 'std': 19}
"""Enum used in reduction kernels."""


def get_all_devices():
    """Return all available devices in a list."""
    result = []
    if not HAS_PYOPENCL:
        return result
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            assert device.type in DEVICE_TYPE_MAP  # DEVICE_TYPE_MAP improper.
            result += [device]
    return result


def has_cpu():
    """Return True if device with type GPU exists."""
    all_devices = get_all_devices()
    for device in all_devices:
        if device.type == CPU:
            return True
    return False


def has_gpu():
    """Return True if device with type GPU exists."""
    all_devices = get_all_devices()
    for device in all_devices:
        if device.type == GPU:
            return True
    return False


def get_gpu(gpu_idx):
    """Get a gpu device from the list of available gpu's.
    @type gpu_idx: int
    @param gpu_idx: The index of the gpu in the list of gpu devices found on the system,
    see: get_all_devices()
    @rtype: pyopencl.Device
    @returns: A gpu device if it exists or None if it doesn't
    """
    gpu = None
    gpu_count = 0
    for device in get_all_devices():
        if device.type != GPU:
            continue
        if gpu_idx == gpu_count:
            gpu = device
            break
        gpu_count += 1
    return gpu


def has_fission_support():
    """Returns True if pyopencl is loaded and known to have fission support."""
    if not HAS_PYOPENCL:
        return False
    cl_header_version = cl.get_cl_header_version()
    return cl_header_version[0] >= 1 and cl_header_version[1] >= 2


def get_sub_device(device, max_cores):
    """
    Use opencl fission to get a sub-device with a given number of cores.
    @type device: pyopencl.Device
    @param device: The device on which to base the sub-device.
    @type max_cores: int
    @param max_cores: The result will be limited to using this number of cores.
    @rtype: pyopencl.Device
    @returns: A device that uses the desired number of cores.
    @raises ValueError:
        If the base device has less cores than required for the sub-device.
    @raises RuntimeError:
        If opencl fission does not obtain a device
        with the desired number of cores.
    """
    if device.max_compute_units < max_cores:
        raise ValueError('device does not have %d compute units.' % max_cores)
    parts = [max_cores, device.max_compute_units - max_cores]
    dpp = cl.device_partition_property_ext
    subdevs = device.create_sub_devices_ext(
        [dpp.BY_COUNTS, parts[0], parts[1], dpp.PARTITION_BY_COUNTS_LIST_END])
    # subdevs may not be sorted properly
    result = None
    for proposed_device in subdevs:
        if proposed_device.max_compute_units == max_cores:
            result = proposed_device
            break
    if result is None:
        raise RuntimeError('opencl fission could not create the subdevice.')
    return result


class RtypeWarning(RuntimeWarning):
    """A warning that indicates an issue with a return type."""
    pass


class InefficiencyWarning(RuntimeWarning):
    """A warning that indicates something inefficient is happening."""
    pass


class MissingOperationError(NotImplementedError):
    """Exception that indicates an operation is missing/not implemented."""
    pass


class Computer(object):
    """
    A class that represents a numerical computer
    capable of using opencl or pure numpy.
    """

    LOADED_SOURCES = {}
    """Store all loaded sources (lists of line) by filename key."""

    KERNELS_PATH = 'kernels'
    """Path to directory containing the cl kernel files."""

    def __init__(self, device_arg=None):
        """
        Construct a Computer given specific option that define its regime.
        @type device_arg: None, pyopencl.Device, CPU, GPU
        @param device_arg: A device or a device type or None.
                       If None opencl will not be used.
                       If you specify a device type (CPU or GPU),
                       then the one with most compute units will be used.
        @raises ValueError: If the device type is not recognized.
        @raises ValueError: If no devices of the given type exist.
        """
        self._use_cl_native = True

        self._use_opencl = device_arg is not None

        self._ranlux_lux = 2
        self._random_state_buf = None  # Will get initialized on the first run.

        if self._use_opencl:
            if isinstance(device_arg, cl.Device):
                self._device = device_arg
            else:
                device_type = device_arg
                if device_type not in DEVICE_TYPE_MAP:
                    raise ValueError('Unrecognized device type: %s' %
                                     device_type)
                all_devices = get_all_devices()
                device = None
                for proposed_device in all_devices:
                    if proposed_device.type == device_type:
                        device = proposed_device
                        break
                if device is None:
                    raise ValueError('No devices with type: %d (%s)' %
                                     (device_type,
                                      DEVICE_TYPE_MAP[device_type]))
                self._device = device
            self._context = cl.Context(devices=[self._device],
                                       properties=None, dev_type=None)
            self._queue = cl.CommandQueue(self._context)
            self._preferred_vector_width_store = {}
            for dtype in SUPPORTED_DTYPES:
                self._preferred_vector_width_store[dtype] = \
                    self.compute_preferred_vector_width(dtype)

            self._ranlux_states = None
            max_mmult_block_size = np.sqrt(self._device.max_work_group_size)
            self._mmult_preferred_block_size = int(max_mmult_block_size)
            while self._mmult_preferred_block_size > max_mmult_block_size:
                self._mmult_preferred_block_size //= 2

            self._programs = {}
        else:
            self._device = None
            self._context = None
            self._queue = None

        self.reset_programs()

    @property
    def use_opencl(self):
        """
        True iff the Computer uses opencl for computations.
        @rtype: bool"""
        return self._use_opencl

    @property
    def device(self):
        """
        The device used for computations.
        @rtype: pyopencl.Device
        """
        return self._device

    @property
    def context(self):
        """
        The context used for computations.
        @rtype: cl.Context
        """
        return self._context

    @property
    def queue(self):
        """
        The queue used for computations.
        @rtype: cl.CommandQueue
        """
        return self._queue

    def summary(self):
        """Return a string summary of the main Computer properties."""
        result = ""
        result += "(OpenCl: %s)" % self.use_opencl
        if self.use_opencl:
            result += " (Device: %s)" % (DEVICE_TYPE_MAP[self.device.type])
            result += " (Max CU: %d)" % self.device.max_compute_units
        return result

    def reset_programs(self):
        """
        Reset/(Re)initialize the programs to their original state.
        Any kernels and programs added since the original state will
        be removed from the cache.
        """
        self._programs = {}

    @property
    def use_cl_native(self):
        """
        Whether kernels CAN use "native_" versions of opencl functions
        (such as native_log).
        Some kernels may not use native version even if this is enabled.
        This only affects single precision computations.
        Changing this will reset the programs cache and require rebuilding,
        it is advised that changes be made before any computations are done.
        @rtype: bool
        """
        return self._use_cl_native

    @use_cl_native.setter
    def use_cl_native(self, value):
        """
        Set the use_cl_native property.
        """
        self.reset_programs()
        self._use_cl_native = value

    @property
    def ranlux_lux(self):
        """
        The lux parameter of the ranlux random number generator
        (larger values mean better quality).
        Changing this will reset the programs cache and require rebuilding,
        it is advised that changes be made before any computations are done.
        @rtype: [1,2,3,4]"""
        return self._ranlux_lux

    @ranlux_lux.setter
    def ranlux_lux(self, value):
        """Set the ranlux_lux property."""
        if value not in [1, 2, 3, 4]:
            raise ValueError('Cannot set ranlux_lux to value %s' % value)
        self.reset_programs()
        self._ranlux_lux = value

    @property
    def ranlux_num_states(self):
        """The number of states to maintain for ranlux rng."""
        return self.device.max_work_group_size*self.device.max_compute_units

    def dtype_is_supported(self, dtype):
        """Return True if the Computer supports computations with the dtype."""
        dtype = _DTYPE(dtype)
        return dtype in SUPPORTED_DTYPES

    ##### Kernel Getters #####

    def get_kernel_path(self, kernel_name):
        """Return the kernel path corresponding to a kernel_name."""
        return '%s/%s.cl' % (self.KERNELS_PATH, kernel_name)

    def inject_kernel_source(self, kernel_name):
        """Prepend LOADED_SOURCES[kernel_name] with necessary sources."""
        pass

    def get_kernel_source(self, kernel_name, **kwargs):
        """
        Return the kernel source corresponding to a kernel_name and options.
        """
        if kernel_name not in self.LOADED_SOURCES:
            kernel_lines = []
            with open(self.get_kernel_path(kernel_name), 'r') as source_file:
                for line in source_file:
                    kernel_lines += [line]
            self.LOADED_SOURCES[kernel_name] = kernel_lines
            self.inject_kernel_source(kernel_name)
        kernel_lines = self.LOADED_SOURCES[kernel_name]

        kernel_source = ""
        for line in kernel_lines:
            if line[0:7] == 'UNROLL_':
                toks = line[7:].split(' ')
                reduced_line = ' '.join(toks[1:])
                unroll_tok = toks[0]
                replace_tok = unroll_tok + '_I'
                for unroll_i in range(kwargs[unroll_tok]):
                    kernel_source += reduced_line.replace(replace_tok,
                                                          '%d' % unroll_i)
            else:
                kernel_source += line
        if 'VECTOR_WIDTH' in kwargs and kwargs['VECTOR_WIDTH'] > 1:
            vw_str = '%d' % kwargs['VECTOR_WIDTH']
        else:
            vw_str = ''
        kernel_source = kernel_source.replace('SIZE_T', KERNEL_SIZE_T)
        kernel_source = kernel_source.replace('_VW', vw_str)
        for k in kwargs:
            assert k != 'DTYPE'
            if k[0:5] == 'DTYPE':
                dtype = kwargs[k]
                dtype_str = SUPPORTED_DTYPES[dtype]
                kernel_source = kernel_source.replace(k, dtype_str)

        header = "//"
        for k in kwargs:
            header += '(%s,%s)' % (k, kwargs[k])
        kernel_source = header + '\n' + kernel_source

        return kernel_source

    def get_include_opts(self):
        return '-I %s' % self.KERNELS_PATH

    def get_program(self, kernel_name, **kwargs):
        """
        Return the kernel corresponding to a kernel_name and kernel options.
        If the kernel is not in the cache it will be build and cached.
        """

        if kernel_name not in self._programs:
            self._programs[kernel_name] = {}
        kernel_opts = str(sorted(kwargs.items()))

        try:
            result = self._programs[kernel_name][kernel_opts]
        except KeyError:
            kernel_source = self.get_kernel_source(kernel_name, **kwargs)

            compile_opts = self.get_include_opts()
            for k in sorted(kwargs):
                if k[0:5] == 'DTYPE':
                    pass  # These are not compile opts.
                elif isinstance(kwargs[k], bool):
                    # These are pure defines
                    if kwargs[k]:
                        compile_opts += " -D%s " % k
                elif k == 'VECTOR_WIDTH' and kwargs[k] == 1:
                    pass
                else:
                    compile_opts += " -D%s=%s " % (k, kwargs[k])
                    if(isinstance(kwargs[k], str)):
                        kernel_source = kernel_source.replace(k, kwargs[k])

            try:
                kernel_program = cl.Program(self._context,
                                            kernel_source).build(compile_opts)
            except cl.RuntimeError as runtime_err:
                message = 'opencl could not build kernel. %s\n' \
                          'KERNEL_OPTS were %s\n %s' % \
                          ('Perhaps the function is not compatible'
                           ' with the types it is being called with.',
                           kernel_opts, runtime_err)
                raise RuntimeError(message)
            self._programs[kernel_name][kernel_opts] = kernel_program
            result = kernel_program
        return result

    ##### Kernel Call Helpers #####

    def safe_cast_non_logical(self, dt1, dt2, operator='+'):
        """
        Return a dtype which is safe(st) for two dtypes to be cast to.
        An attempt will be made to be consistent with numpy.
        """
        result = eval('(np.zeros(1,dt1) %s np.zeros(1,dt2)).dtype' % operator)
        if result not in SUPPORTED_DTYPES:
            warnings.warn('Not sure what dtype to use for operation %s '
                          'on dtypes %s %s. Will use float64.' %
                          (operator, dt1, dt2), RtypeWarning)
            result = _DTYPE('float64')
        return result

    def preferred_work_group_size_multiple(self, kernel):
        """Get the device's preferred work group size multiple for a kernel."""
        return kernel.get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            self._device)

    def _cl_elementwise_local_size(self, kernel, dim):
        """Gets the optimal local size for a given kernel and dimensionality.
        This is loosely based on some prior knowledge/benchmarks and
        the preferred and maximal work group sizes of the device.
        @type kernel: cl.Kernel
        @type dim: 1,2
        @param dim: The dimensionality of the kernel
        @rtype: [int,...]
        @returns: A list of integer type ready to pass to the kernel call
                  as a local work size.
        """
        if self.device.type == CPU:
            if dim == 1:
                result = [int(self.device.max_work_group_size)]
            else:  # Simply reverse order operation on two matrices
                # Take care of the most common scenario first.
                if self.device.max_work_group_size == 1024:
                    result = [32, 32]
                else:
                    local_size_major = 1
                    local_size_minor = self.device.max_work_group_size
                    while local_size_major < local_size_minor:
                        local_size_major *= 2
                        local_size_minor //= 2
                    result = [int(local_size_major),
                              int(local_size_minor)]
        else:
            if dim == 1:
                result = [int(
                          self.preferred_work_group_size_multiple(kernel))]
            else:
                result = [2,
                          int(self.device.max_work_group_size//2)]
        return result

    def _cl_elementwise_global_size(self, min_global_size, local_size):
        """
        Get the global size for the given local_size and matrix shape/size.
        @type min_global_size: [int] , [int,int]
        @param min_global_size: The minimum global size to return.
        @type local_size: [int] , [int,int]
        @param local_size: The local size that will be used.
        @rtype: [int,...]
        @return: The global size (ready for passing to kernel calls)
        """
        local_size = [l for l in local_size]
        dim = len(local_size)
        if local_size[0] > min_global_size[0]:
            local_size[0] = int(min_global_size[0])
        if dim == 2 and local_size[1] > min_global_size[1]:
            local_size[1] = int(min_global_size[1])

        gs0 = int((min_global_size[0]//local_size[0])*local_size[0])
        if gs0 < min_global_size[0]:
            gs0 += local_size[0]
        if dim == 1:
            global_size = [gs0]
        else:
            gs1 = int((min_global_size[1]//local_size[1])*local_size[1])
            if gs1 < min_global_size[1]:
                gs1 += local_size[1]
            global_size = [gs0, gs1]
        return global_size, local_size

    def _cl_buffer_copy(self, dst, src):
        """Copy from/to host and device.
        The arguments must conform to cl.enqueue_copy requirements.
        @type dst: cl.Buffer/np.ndarray
        @param dst: The object to copy to.
        @type src: np.ndarray/cl.Buffer
        @param src: The object to copy from.
        """
        cl.enqueue_copy(self._queue, dst, src)
        self._queue.finish()

    def _consistify_args(self, dim, *args):
        """
        Checks the arguments for 1d, 2d or other kernel consistency
        and returns their consistent versions if possible.
        Each arg is expected to be a Mat or a scalar.
        @type dim: int
        @param dim: 1 if kernel is 1d, 2 if kernel is 2d, -1 for mixed/other.
                    For mixed/other no shape checks will be performed.
        @raises ValueError: If any argument inconsistency cannot be rectified.
        """
        result = []
        out = args[0]
        for m_arg in args:
            if isinstance(m_arg, Mat):
                if m_arg.computer is not self:
                    raise ValueError('One of the arguments is not using'
                                     ' this computer: %s' % m_arg)
                if dim == 1 and m_arg.size != out.size:
                    raise ValueError('1d kernel arg and output size mismatch:'
                                     ' %d != %d' % (m_arg.size, out.size))
                elif dim == 2 and (m_arg.shape0 not in [out.shape0, 1] or
                                   m_arg.shape1 not in [out.shape1, 1]):
                    raise ValueError('2d kernel arg and output shape mismatch:'
                                     ' %s != %s' % (m_arg.shape, out.shape))
            elif not isinstance(m_arg, np.ndarray):
                new_m = np.array([m_arg])
                if new_m.size != 1:
                    raise ValueError('One of the arguments could not be '
                                     'converted to a scalar ndarray: %s' %
                                     m_arg)
                m_arg = new_m
            result += [m_arg]
        return tuple(result)

    def _call_dim(self, *args):
        """
        Return the lowest dimensionality of kernel
        that can be called with the given args.
        NOTE: Will not check the arguments for shape or other consistency.
        @rtype: 1,2
        @returns: 1 if all the Mat arguments are contiguous
                  and have identical shapes and compatible orders, 2 otherwise.
        @raises TypeError: If the first argument is not a Mat.
        """
        if not isinstance(args[0], Mat):
            raise TypeError('Expected first argument to be a Mat (was %s)' %
                            args[0])
        main = args[0]
        if not main.contiguous:
            return 2
        shape = main.shape
        for m_arg in args:
            if isinstance(m_arg, Mat):
                if m_arg.shape != shape:
                    return 2
                if not (m_arg.c_contiguous and main.c_contiguous) and \
                   not (m_arg.f_contiguous and main.f_contiguous):
                    return 2
        return 1

    def compute_preferred_vector_width(self, dtype):
        """Compute the preferred opencl vector width for a given dtype."""
        dtype_str = SUPPORTED_DTYPES[dtype]
        is_unsigned = dtype_str[0] == 'u'
        if is_unsigned:
            dtype_str = dtype_str[1:]
        result = getattr(self.device, 'preferred_vector_width_%s' % dtype_str)
        if is_unsigned and result > 1:
            result //= 2
        return result

    def get_preferred_vector_width(self, dtype):
        """Get the preferred opencl vector width for a given dtype."""
        return self._preferred_vector_width_store[dtype]

    def _optimal_vector_width(self, size, dtype):
        """
        Return the vector width most suited for kernel calls
        given buffers of the given size.
        @type size: SIZE_T
        @param size: The size in which the vector width must fit.
        @type dtype: numpy dtype
        @param dtype: The data type that will be vectorized.
        @rtype: int
        @returns: The smallest vector width that divides the size
                  and is not greater than the preferred for that dtype.
        """
        if size % 2:
            return 1
        result = self.get_preferred_vector_width(dtype)
        while size % result:
            result //= 2
        return max(result, 1)

    ##### Kernel Callers #####

    def _cl_random(self, out, normal=False):
        """Call the random kernel."""
        out, = self._consistify_args(-1, out)
        queue = self.queue
        if out.dtype.kind != 'f':
            raise TypeError('random expects Mat with float dtype.')
        is_exact = bool(out.size % 4 == 0)
        is_double = bool(out.dtype == np.double)
        program = self.get_program(
            'random', EXACT=is_exact, DOUBLE=is_double, NORMAL=bool(normal),
            DTYPE_OUT=out.dtype, RANLUXCL_LUX=self.ranlux_lux)

        gws = [int(self.ranlux_num_states)]
        lws = [int(self.device.max_work_group_size)]

        if self._ranlux_states is None:
            seed = np.uint32(np.random.randint(2**32))
            self._ranlux_states = cl.Buffer(
                self._context, cl.mem_flags.READ_WRITE,
                28*4*self.ranlux_num_states)
            program.random_states_init(queue, gws, lws, seed,
                                       self._ranlux_states)
            queue.finish()

        out_size = out.size
        program.random(queue, gws, lws,
                       out_size, out.buffer, _size_t(out_size % 4),
                       _size_t(out_size // 4), out.buffer,
                       self._ranlux_states)
        self._queue.finish()

    def _cl_map_1d(self, function, mat1, out=None):
        """Call the map_1d kernel."""
        if out is None:
            out = mat1
        out, mat1 = self._consistify_args(1, out, mat1)

        kvw = self._optimal_vector_width(
            out.size, self.safe_cast_non_logical(out.dtype, mat1.dtype))
        k_getter = lambda e: self.get_program(
            'map_1d', MAP_FUNCTION=function,
            DTYPE_OUT=out.dtype, DTYPE_IN=mat1.dtype, DTYPE_M1=mat1.dtype,
            OFFSET_M1=bool(mat1.begin != 0), OFFSET_OUT=bool(out.begin != 0),
            EXACT=bool(e), VECTOR_WIDTH=kvw).map_1d
        kernel = k_getter(True)

        call_size = _size_t(out.size/kvw)
        lws = self._cl_elementwise_local_size(kernel, 1)
        gws, lws = self._cl_elementwise_global_size([call_size], lws)
        if gws[0] != call_size:
            kernel = k_getter(False)

        kernel(self.queue, gws, lws, call_size,
               out.buffer, out.begin, mat1.buffer, mat1.begin)
        self.queue.finish()

    def _cl_map(self, function, mat1, out=None):
        """Call the map kernel"""
        if out is None:
            out = mat1
        out, mat1 = self._consistify_args(2, out, mat1)

        reverse_ws = out.order == 'C'
        k_getter = lambda e: self.get_program(
            'map', MAP_FUNCTION=function, REVERSE_WS=reverse_ws,
            DTYPE_OUT=out.dtype, DTYPE_IN=mat1.dtype, DTYPE_M1=mat1.dtype,
            EXACT=bool(e)).map
        kernel = k_getter(True)
        lws = self._cl_elementwise_local_size(kernel, 2)
        gws, lws = self._cl_elementwise_global_size(out.shape, lws)
        if tuple(gws) != out.shape:
            kernel = k_getter(False)
        if reverse_ws:
            lws = [lws[1], lws[0]]
            gws = [gws[1], gws[0]]

        kernel(self.queue, gws, lws, out.shape0, out.shape1,
               out.buffer, out.ptr_stride0, out.ptr_stride1, out.begin,
               mat1.buffer, mat1.ptr_stride0, mat1.ptr_stride1, mat1.begin)

        self.queue.finish()

    def _cl_op_logical_1d(self, function, out, mat1, mat2):
        """Call the op_logical_1d kernel"""
        m1_is_matrix = isinstance(mat1, Mat)
        m2_is_matrix = isinstance(mat2, Mat)
        out, mat1, mat2 = self._consistify_args(1, out, mat1, mat2)

        kvw = 1
        k_getter = lambda e: self.get_program(
            'op_1d', OPERATOR=function, LOGICAL=True, VECTOR_WIDTH=kvw,
            DTYPE_OUT=out.dtype,
            DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype,
            SCALAR_M1=not m1_is_matrix, SCALAR_M2=not m2_is_matrix,
            OFFSET_OUT=bool(out.begin != 0),
            OFFSET_M1=bool(m1_is_matrix and mat1.begin != 0),
            OFFSET_M2=bool(m2_is_matrix and mat2.begin != 0),
            EXACT=bool(e)).op_1d
        kernel = k_getter(True)

        call_size = _size_t(out.size/kvw)
        lws = self._cl_elementwise_local_size(kernel, 1)
        gws, lws = self._cl_elementwise_global_size([call_size], lws)
        if gws[0] != call_size:
            kernel = k_getter(False)

        args = [out.buffer, out.begin]
        if m1_is_matrix:
            args += [mat1.buffer, mat1.begin]
        else:
            args += [mat1, ZERO]
        if m2_is_matrix:
            args += [mat2.buffer, mat2.begin]
        else:
            args += [mat2, ZERO]
        kernel(self.queue, gws, lws, call_size, *args)
        self.queue.finish()

    def _cl_op_logical(self, function, out, mat1, mat2):
        """Call the op_logical kernel"""
        out, mat1, mat2 = self._consistify_args(2, out, mat1, mat2)

        m1_is_matrix = isinstance(mat1, Mat)
        m2_is_matrix = isinstance(mat2, Mat)
        reverse_ws = out.order == 'C'
        k_getter = lambda e: self.get_program(
            'op', OPERATOR=function, LOGICAL=True,
            DTYPE_OUT=out.dtype,
            DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype,
            SCALAR_M1=not m1_is_matrix, SCALAR_M2=not m2_is_matrix,
            REVERSE_WS=reverse_ws, EXACT=bool(e)).op
        kernel = k_getter(True)

        lws = self._cl_elementwise_local_size(kernel, 2)
        gws, lws = self._cl_elementwise_global_size(out.shape, lws)
        if tuple(gws) != out.shape:
            kernel = k_getter(False)
        if reverse_ws:
            lws = [lws[1], lws[0]]
            gws = [gws[1], gws[0]]

        args = [out.buffer, out.ptr_stride0, out.ptr_stride1, out.begin]
        if m1_is_matrix:
            args += [mat1.buffer, mat1.ptr_stride0, mat1.ptr_stride1,
                     mat1.begin]
        else:
            args += [mat1, ZERO, ZERO, ZERO]
        if m2_is_matrix:
            args += [mat2.buffer, mat2.ptr_stride0, mat2.ptr_stride1,
                     mat2.begin]
        else:
            args += [mat2, ZERO, ZERO, ZERO]
        kernel(self.queue, gws, lws, out.shape0, out.shape1, *args)
        self.queue.finish()

    def _cl_op_1d(self, function, out, mat1, mat2):
        """Call the op_1d kernel"""
        m1_is_matrix = isinstance(mat1, Mat)
        m2_is_matrix = isinstance(mat2, Mat)
        out, mat1, mat2 = self._consistify_args(1, out, mat1, mat2)

        kvw = self._optimal_vector_width(
            out.size, self.safe_cast_non_logical(mat1.dtype, mat2.dtype,
                                                 operator=function))
        dtype_in = self.safe_cast_non_logical(mat1.dtype, mat2.dtype,
                                              operator=function)
        k_getter = lambda e: self.get_program(
            'op_1d', OPERATOR=function, EXACT=bool(e), VECTOR_WIDTH=kvw,
            DTYPE_OUT=out.dtype, DTYPE_IN=dtype_in,
            DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype,
            SCALAR_M1=not m1_is_matrix, SCALAR_M2=not m2_is_matrix,
            OFFSET_OUT=bool(out.begin != 0),
            OFFSET_M1=bool(m1_is_matrix and mat1.begin != 0),
            OFFSET_M2=bool(m2_is_matrix and mat2.begin != 0)).op_1d
        kernel = k_getter(True)

        call_size = _size_t(out.size/kvw)
        lws = self._cl_elementwise_local_size(kernel, 1)
        gws, lws = self._cl_elementwise_global_size([call_size], lws)
        if gws[0] != call_size:
            kernel = k_getter(False)

        args = [out.buffer, out.begin]
        if m1_is_matrix:
            args += [mat1.buffer, mat1.begin]
        else:
            args += [mat1, ZERO]
        if m2_is_matrix:
            args += [mat2.buffer, mat2.begin]
        else:
            args += [mat2, ZERO]
        kernel(self.queue, gws, lws, call_size, *args)
        self.queue.finish()

    def _cl_op(self, function, out, mat1, mat2):
        """Call the op kernel"""
        out, mat1, mat2 = self._consistify_args(2, out, mat1, mat2)

        m1_is_matrix = isinstance(mat1, Mat)
        m2_is_matrix = isinstance(mat2, Mat)
        dtype_in = self.safe_cast_non_logical(mat1.dtype, mat2.dtype,
                                              operator=function)
        reverse_ws = out.order == 'C'
        k_getter = lambda e: self.get_program(
            'op', OPERATOR=function,
            DTYPE_OUT=out.dtype, DTYPE_IN=dtype_in,
            DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype,
            SCALAR_M1=not m1_is_matrix, SCALAR_M2=not m2_is_matrix,
            REVERSE_WS=reverse_ws, EXACT=bool(e)).op
        kernel = k_getter(True)

        lws = self._cl_elementwise_local_size(kernel, 2)
        gws, lws = self._cl_elementwise_global_size(out.shape, lws)
        if tuple(gws) != out.shape:
            kernel = k_getter(False)
        if reverse_ws:
            lws = [lws[1], lws[0]]
            gws = [gws[1], gws[0]]

        args = [out.buffer, out.ptr_stride0, out.ptr_stride1, out.begin]
        if m1_is_matrix:
            args += [mat1.buffer, mat1.ptr_stride0, mat1.ptr_stride1,
                     mat1.begin]
        else:
            args += [mat1, ZERO, ZERO, ZERO]
        if m2_is_matrix:
            args += [mat2.buffer, mat2.ptr_stride0, mat2.ptr_stride1,
                     mat2.begin]
        else:
            args += [mat2, ZERO, ZERO, ZERO]
        kernel(self.queue, gws, lws, out.shape0, out.shape1, *args)
        self.queue.finish()

    def _cl_iop_1d(self, function, out, mat1):
        """Call the iop_1d kernel"""
        if out.computer is not self:
            raise ValueError('out is not using this computer.')

        out, mat1 = self._consistify_args(1, out, mat1)
        m1_is_matrix = isinstance(mat1, Mat)

        kvw = self._optimal_vector_width(
            out.size, self.safe_cast_non_logical(out.dtype, mat1.dtype))
        k_getter = lambda e: self.get_program(
            'iop_1d', OPERATOR=function, EXACT=bool(e), VECTOR_WIDTH=kvw,
            DTYPE_OUT=out.dtype, DTYPE_M1=mat1.dtype,
            OFFSET_OUT=bool(out.begin != 0),
            OFFSET_M1=bool(m1_is_matrix and mat1.begin != 0),
            SCALAR_M1=not m1_is_matrix).iop_1d
        kernel = k_getter(True)

        call_size = _size_t(out.size/kvw)
        lws = self._cl_elementwise_local_size(kernel, 1)
        gws, lws = self._cl_elementwise_global_size([call_size], lws)
        if gws[0] != call_size:
            kernel = k_getter(False)

        args = [out.buffer, out.begin]
        if m1_is_matrix:
            args += [mat1.buffer, mat1.begin]
        else:
            args += [mat1, ZERO]
        kernel(self.queue, gws, lws, call_size, *args)
        self.queue.finish()

    def _cl_iop(self, function, out, mat1):
        """Call the iop kernel"""
        out, mat1 = self._consistify_args(2, out, mat1)

        m1_is_matrix = isinstance(mat1, Mat)
        reverse_ws = out.order == 'C'
        k_getter = lambda e: self.get_program(
            'iop', OPERATOR=function, DTYPE_OUT=out.dtype, DTYPE_M1=mat1.dtype,
            SCALAR_M1=not m1_is_matrix, REVERSE_WS=reverse_ws,
            EXACT=bool(e)).iop
        kernel = k_getter(True)

        lws = self._cl_elementwise_local_size(kernel, 2)
        gws, lws = self._cl_elementwise_global_size(out.shape, lws)
        if tuple(gws) != out.shape:
            kernel = k_getter(False)
        if reverse_ws:
            lws = [lws[1], lws[0]]
            gws = [gws[1], gws[0]]

        args = [out.buffer, out.ptr_stride0, out.ptr_stride1, out.begin]
        if isinstance(mat1, Mat):
            args += [mat1.buffer, mat1.ptr_stride0, mat1.ptr_stride1,
                     mat1.begin]
        else:
            args += [mat1, ZERO, ZERO, ZERO]
        kernel(self.queue, gws, lws, out.shape0, out.shape1, *args)
        self.queue.finish()

    def _cl_reduce_1d(self, function, mat1):
        """Call the reduce_1d kernel"""
        mat1, = self._consistify_args(1, mat1)
        out_dtype = BOOL_TYPE if function in ['any', 'all'] else mat1.dtype
        kernel = self.get_program(
            'reduce_1d', REDUCTION=REDUCTION_ENUM[function],
            DTYPE_OUT=out_dtype, DTYPE_M1=mat1.dtype).reduce_1d

        if False:  # TODO function in ['min', 'max', 'sum', 'prod']:
            num_blocks = _size_t(4*self.device.max_compute_units)
            gws = [int(num_blocks)]
            lws = [4]
            partial_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                       size=num_blocks*out_dtype.itemsize)
            block_size = mat1.size//num_blocks
            kernel(self.queue, gws, lws, ONE, partial_buffer, ZERO,
                   block_size, mat1.size, mat1.buffer, mat1.begin)
        else:
            num_blocks = mat1.size
            partial_buffer = mat1.buffer

        gws = [1]
        lws = [1]
        out_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                               size=out_dtype.itemsize)
        kernel(self.queue, gws, lws, ONE, out_buffer, ZERO,
               num_blocks, num_blocks, partial_buffer, ZERO)
        result = np.empty((1, ), dtype=out_dtype)
        self._cl_buffer_copy(result, out_buffer)
        self.queue.finish()

        return result[0]

    def _cl_reduce_2d(self, function, out, mat1, axis):
        """Call the reduce_2d kernel"""

        out, mat1 = self._consistify_args(-1, out, mat1)
        if out.shape[axis] != 1:
            raise ValueError('The output axis dimension must be 1.')
        if out.shape[1-axis] != mat1.shape[1-axis]:
            raise ValueError('The output-input non-axis dimensions mismatch.')

        kernel = self.get_program(
            'reduce_2d', REDUCTION=REDUCTION_ENUM[function], AXIS=axis,
            DTYPE_OUT=out.dtype, DTYPE_M1=mat1.dtype).reduce_2d

        gws = list(map(int, out.shape))
        lws = [1, 1]
        if self.device.type == CPU:
            pass
        elif axis == 0 and mat1.c_contiguous:
            lws[1] = self.preferred_work_group_size_multiple(kernel)
        else:
            lws[0] = self.device.max_work_group_size // \
                self.preferred_work_group_size_multiple(kernel)
        gws[0] = lws[0] * int(np.ceil(gws[0]/lws[0]))
        gws[1] = lws[1] * int(np.ceil(gws[1]/lws[1]))

        kernel(self.queue, gws, lws,
               out.buffer, out.ptr_stride0, out.ptr_stride1, out.begin,
               mat1.shape0, mat1.shape1,
               mat1.buffer, mat1.ptr_stride0, mat1.ptr_stride1, mat1.begin)
        self.queue.finish()

    def _cl_f(self, function, out, *args):
        """Call the f kernel"""
        if len(args) > 3:
            raise TypeError('f only support up to 3 input arguments'
                            ' (was given %d).' % len(args))
        cons_args = self._consistify_args(2, out, *args)

        gkargs = {}
        dtype_in = cons_args[1].dtype
        for args_i in range(1, len(cons_args)):
            m_arg = cons_args[args_i]
            gkargs['DTYPE_M%d' % args_i] = m_arg.dtype
            if args_i != 1:
                gkargs['M%d' % args_i] = True
            gkargs['SCALAR_M%d' % args_i] = bool(not isinstance(m_arg, Mat))
            if args_i > 1:
                dtype_in = self.safe_cast_non_logical(dtype_in, m_arg.dtype)

        reverse_ws = out.order == 'C'
        k_getter = lambda e: self.get_program(
            'f', EXACT=bool(e), MAP_FUNCTION=function, REVERSE_WS=reverse_ws,
            DTYPE_OUT=out.dtype, DTYPE_IN=dtype_in,
            **gkargs).f
        kernel = k_getter(True)

        lws = self._cl_elementwise_local_size(kernel, 2)
        gws, lws = self._cl_elementwise_global_size(out.shape, lws)
        if tuple(gws) != out.shape:
            kernel = k_getter(False)
        if reverse_ws:
            lws = [lws[1], lws[0]]
            gws = [gws[1], gws[0]]

        kargs = []
        for args_i in range(4):  # Includes out
            if args_i < len(cons_args):
                m_arg = cons_args[args_i]
            else:
                m_arg = ZERO
            if isinstance(m_arg, Mat):
                kargs += [m_arg.buffer,
                          m_arg.ptr_stride0, m_arg.ptr_stride1, m_arg.begin]
            else:
                kargs += [m_arg, ZERO, ZERO, ZERO]
        kernel(self.queue, gws, lws, out.shape0, out.shape1, *kargs)
        self.queue.finish()

    def _cl_mmult(self, out, mat1, mat2):
        """Call the mmult kernel"""
        out, mat1, mat2 = self._consistify_args(-1, out, mat1, mat2)
        if mat1.shape1 != mat2.shape0:
            raise ValueError('m1 and m2 have inconsistent dimensions: %s %s' %
                             (mat1.shape, mat2.shape))
        if out.shape0 != mat1.shape0:
            raise ValueError('out and m1 have inconsistent dimensions: %s %s' %
                             (out.shape, mat1.shape))
        if out.shape1 != mat2.shape1:
            raise ValueError('out and m2 have inconsistent dimensions: %s %s' %
                             (out.shape, mat2.shape))

        block_size = self._mmult_preferred_block_size
        m1_mod = mat1.shape0 % block_size
        m2_mod = mat2.shape1 % block_size
        common_mod = mat1.shape1 % block_size
        if common_mod == 0 and mat1.dtype == mat2.dtype:
            kvw = self._optimal_vector_width(
                mat1.shape1 // block_size,
                self.safe_cast_non_logical(out.dtype, mat1.dtype))
            kvw = min(kvw, 4)  # dot does not support kvw > 4
        else:
            kvw = 1
        m1_local_size = kvw * block_size * block_size * mat1.dtype.itemsize
        m2_local_size = kvw * block_size * block_size * mat2.dtype.itemsize
        while kvw > 1 and \
                m1_local_size + m2_local_size > self.device.local_mem_size:
            block_size //= 2
            m1_local_size = kvw * block_size * block_size * mat1.dtype.itemsize
            m2_local_size = kvw * block_size * block_size * mat2.dtype.itemsize

        is_exact = (m1_mod == 0) and \
                   (common_mod == 0) and \
                   (m2_mod == 0)
        kernel = self.get_program(
            'mmult', EXACT=bool(is_exact),
            DTYPE_OUT=out.dtype, DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype,
            BLOCK_SIZE=block_size, VECTOR_WIDTH=kvw).mmult

        lws = [block_size, block_size]
        # Note the global sizes are reversed on purpose as in kernel.
        gws = [int(out.shape1), int(out.shape0)]
        if m1_mod:
            gws[1] += (block_size-m1_mod)
        if m2_mod:
            gws[0] += (block_size-m2_mod)

        m1_block = cl.LocalMemory(m1_local_size)
        m2_block = cl.LocalMemory(m2_local_size)

        common_dim = _size_t(mat1.shape1/kvw)
        kernel(self.queue, gws, lws, out.shape0, common_dim, out.shape1,
               out.buffer, out.ptr_stride0, out.ptr_stride1, out.begin,
               mat1.buffer, mat1.ptr_stride0, mat1.ptr_stride1, mat1.begin,
               mat2.buffer, mat2.ptr_stride0, mat2.ptr_stride1, mat2.begin,
               m1_block, m2_block)
        self.queue.finish()

    def _cl_innerprod(self, mat1, mat2):
        """Call the innerprod kernel"""
        mat1, mat2 = self._consistify_args(2, mat1, mat2)

        num_blocks = int(4*self.device.max_compute_units)
        reverse_ws = mat1.order == 'C'
        if reverse_ws:
            gws = (num_blocks, 1)
            lws = [1, 1]
        else:
            gws = (1, num_blocks)
            lws = [1, 1]

        out_dtype = self.safe_cast_non_logical(mat1.dtype, mat2.dtype,
                                               operator='*')

        partial = self.empty(gws, dtype=out_dtype,
                             c_contiguous=mat1.c_contiguous)

        kernel = self.get_program(
            'innerprod', REVERSE=reverse_ws, DTYPE_OUT=out_dtype,
            DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype).innerprod

        shape0 = mat1.shape0
        shape1 = mat1.shape1
        block_size_r = shape0//gws[reverse_ws]
        if block_size_r*gws[reverse_ws] < shape0:
            block_size_r += 1
        block_size_c = shape1//gws[1-reverse_ws]
        if block_size_c*gws[1-reverse_ws] < shape1:
            block_size_c += 1
        kernel(self.queue, gws, lws, partial.buffer, shape0, shape1,
               mat1.buffer, mat1.begin, mat1.ptr_stride0, mat1.ptr_stride1,
               mat2.buffer, mat2.begin, mat2.ptr_stride0, mat2.ptr_stride1,
               _size_t(block_size_r), _size_t(block_size_c))
        self.queue.finish()

        return partial.sum()

    def _cl_li(self, out, mat1, mat2, mat3):
        """Call the li kernel."""
        # TODO consistify args
#        raise NotImplementedError('This method is not finished.')

        LI_OUT = bool(out is not mat1)
        is_get = bool(mat3 is None)
        m3_is_scalar = bool(is_get or isinstance(mat3, Mat))
        if is_get or m3_is_scalar:
            mat3_dtype = out.dtype
        else:
            mat3_dtype = mat3.dtype
        m2_is_row = bool(mat2.shape0 == 1)
        m2_is_col = bool(mat2.shape1 == 1)
        kernel = self.get_program(
            'li', DTYPE_OUT=out.dtype, LI_GET=is_get, LI_OUT=LI_OUT,
            DTYPE_M1=mat1.dtype, DTYPE_M2=mat2.dtype, DTYPE_M3=mat3_dtype,
            M2_ROW=m2_is_row, M2_COL=m2_is_col, SCALAR_M3=m3_is_scalar
            ).li
        gws = [1]
        lws = [1]

        kargs = [out.buffer, out.begin, out.ptr_stride0, out.ptr_stride1]
        kargs += [mat1.buffer, mat1.shape0, mat1.shape1]
        kargs += [mat1.begin, mat1.ptr_stride0, mat1.ptr_stride1]
        kargs += [mat2.buffer, mat2.begin, mat2.ptr_stride0, mat2.ptr_stride1]
        if is_get:
            kargs += [np.zeros((1, 1), dtype=mat3_dtype), ZERO, ZERO, ZERO]
        elif m3_is_scalar:
            kargs += [mat3, ZERO, ZERO, ZERO]
        else:
            kargs += [mat3.buffer, mat3.begin,
                      mat3.ptr_stride0, mat3.ptr_stride1]

        kernel(self.queue, gws, lws, *kargs)
        self.queue.finish()

    ##### Functions #####

    def f(self, out, function, *args):
        """Fill a Mat by applying a function to at most 3 inputs."""
        if self.use_opencl:
            self._cl_f(function, out, *args)
        else:
            npf = getattr(np, function)
            np_args = []
            for arg in args:
                if isinstance(arg, Mat):
                    np_args += [arg.NP]
                else:
                    np_args += [arg]
            out.NP[:] = npf(*np_args)

    def map(self, function, mat, out=None):
        """Apply a function to one input."""
        if out is None:
            out = self.empty_like(mat)
        self.f(out, function, mat)
        return out

    def sqrt(self, mat, out=None):
        """Apply sqrt to each element of the Mat."""
        return self.map('sqrt', mat, out=out)

    def exp(self, mat, out=None):
        """Apply exp to each element of the Mat."""
        return self.map('exp', mat, out=out)

    def log(self, mat, out=None):
        """Apply log to each element of the Mat."""
        return self.map('log', mat, out=out)

    def log1p(self, mat, out=None):
        """Apply log1p to each element of the Mat."""
        return self.map('log1p', mat, out=out)

    def mmult(self, mat1, mat2, out=None):
        """Perform matrix multiplication with two Mat instances."""
        if self.use_opencl:
            if out is None:
                rdtype = self.safe_cast_non_logical(mat1.dtype, mat2.dtype,
                                                    operator='+')
                out = self.empty((mat1.shape0, mat2.shape1),
                                 dtype=rdtype, c_contiguous=mat1.c_contiguous)
            self._cl_mmult(out, mat1, mat2)
        else:
            if out is None:
                np_arr = np.dot(mat1.NP, mat2.NP).reshape(mat1.shape0,
                                                          mat2.shape1)
                out = self.M(np_arr)
            else:
                np.dot(mat1.NP, mat2.NP, out=out.NP)
        return out

    def innerprod(self, v1, v2):
        """Computes the inner-product of v1 and v2, i.e. (v1*v2).sum()"""
        if self.use_opencl:
#            result = (v1*v2).sum()
            result = self._cl_innerprod(v1, v2)
        else:
            result = np.dot(v1.NP.flat, v2.NP.flat)
        return result

    ##### Matrix Constructors #####

    def M(self, *args, **kwargs):
        """Call Mat constructor with self as first argument."""
        return Mat(self, *args, **kwargs)

    def empty(self, *args, **kwargs):
        """Return an empty Mat with self as its computer."""
        result = self.M(*args, **kwargs)
        return result

    def zeros(self, *args, **kwargs):
        """Return a Mat of zeros with self as its computer."""
        result = self.M(*args, **kwargs)
        result.fill(0)
        return result

    def ones(self, *args, **kwargs):
        """Return a Mat of ones with self as its computer."""
        result = self.M(*args, **kwargs)
        result.fill(1)
        return result

    def empty_like(self, mat):
        """Return an empty Mat similar to the one given."""
        return self.empty(mat.shape,
                          dtype=mat.dtype, c_contiguous=mat.c_contiguous)

    def zeros_like(self, mat):
        """Return a Mat of zeros similar to the one given."""
        return self.zeros(mat.shape,
                          dtype=mat.dtype, c_contiguous=mat.c_contiguous)

    def ones_like(self, mat):
        """Return a Mat of ones similar to the one given."""
        return self.ones(mat.shape,
                         dtype=mat.dtype, c_contiguous=mat.c_contiguous)

    def _random(self, normal, *args, **kwargs):
        """
        Random number generator helper.
        @type normal: bool
        @param normal: Whether to use normal distribution.
        args and kwargs have same meaning as those for Computer.empty.
        """
        if self.use_opencl:
            result = self.empty(*args, **kwargs)
            self._cl_random(result, normal=normal)
        else:
            shape = args[0]
            if normal:
                result = self.M(np.random.randn(shape[0], shape[1]))
            else:
                result = self.M(np.random.random(shape))
            if 'dtype' in kwargs:
                result = result.astype(kwargs['dtype'])
            if 'c_contiguous' in kwargs:
                order = 'C' if kwargs['c_contigous'] else 'F'
                result = result.copy(order=order)
        return result

    def randu(self, *args, **kwargs):
        """
        Generate uniform random numbers.
        args and kwargs have same meaning as those for Computer.empty.
        """
        return self._random(False, *args, **kwargs)

    def randn(self, *args, **kwargs):
        """
        Generate normal random numbers.
        args and kwargs have same meaning as those for Computer.empty.
        """
        return self._random(True, *args, **kwargs)

    def logical_copy(self, result, index, valiftrue, valiffalse):
        # TODO
        return NotImplemented


class Mat(object):
    """A matrix class that supports opencl or pure numpy."""

    def __init__(self, computer, nparr_or_shape,
                 dtype=None, c_contiguous=None):
        """
        Mat constructor.
        Be advised that Mat object can be constructed through Computer methods,
        such as M, zeros, ones, empty, etc.

        @type computer: Computer
        @param computer: The object that handles opencl/numpy computations.
        @type nparr_or_shape: np.ndarray , 2-tuple
        @param nparr_or_shape:
            Numpy data that the matrix should be based on
            or the shape of the matrix.
            Raises a ValueError if shape has length!=2.
            If this is an np.ndarray it will be used to initialize the data.
            If the computer does not use opencl then this object
            will have its internal data be a reference to the nparr argument.
            This means that if you change the nparr externally
            it will also change this object internally.
            This is to keep memory overhead to a minimum,
            if you want to be sure that this object has its own data,
            then pass in a .copy() of the numpy data as the nparr argument.
        @type dtype: numpy.dtype arg
        @param dtype:
            The type of the data (e.g. np.double, np.single, 'float', 'f8').
            Any argument to numpy.dtype function can be passed.
            A ValueError will be raised if the data type is not supported.
            If you specify nparr, then this must be left unset.
            The default is np.double.
        @type c_contiguous: bool
        @param c_contiguous:
            Whether the internal cl.Buffer should be interpreted
            as C-contiguous / has row-major order.
            Otherwise it will have F-contiguous / column-major order.
            If you specify nparr, then this must be left unset.
            The default is True.
            NOTE: Mat object constructed through other methods
            may have neither order.
        @returns: A Mat instance that is contiguous andhas begin=0.
        @raises TypeError: if nparr_or_shape is not numpy.ndarray or tuple.
        @raises ValueError: if setting keyword args when using numpy data.
        @raises ValueError: if using shape and shape has length!=2.
        @raises ValueError: if the dtype is not supported.
        """
        if computer is None:
            assert nparr_or_shape is None  # Empty constructor mistake.
            return
        self._computer = computer
        init_from_np = isinstance(nparr_or_shape, np.ndarray)
        if init_from_np:
            if nparr_or_shape.ndim != 2:
                raise ValueError('The numpy array must have ndim 2 '
                                 '(was %s)' % nparr_or_shape.ndim)
            if dtype is not None:
                raise ValueError('Cannot specify dtype when '
                                 'constructing based on numpy data.')
            if c_contiguous is not None:
                raise ValueError('Cannot specify c_contiguous when '
                                 'constructing based on numpy data.')

        elif isinstance(nparr_or_shape, tuple):
            if len(nparr_or_shape) != 2:
                raise ValueError('The shape of the matrix must have 2 values '
                                 '(was %s)' % nparr_or_shape)

            nparr_or_shape = \
                _size_t(nparr_or_shape[0]), _size_t(nparr_or_shape[1])
            if dtype is None:
                dtype = _DTYPE('float64')
            else:
                dtype = _DTYPE(dtype)
                if not self.computer.dtype_is_supported(dtype):
                    raise ValueError('The dtype %s is not supported.' % dtype)
            if c_contiguous is None:
                c_contiguous = True

        else:
            raise TypeError('nparr_or_shape must be an np.ndarray or a tuple.')

        if not computer.use_opencl:
            order = 'C' if c_contiguous else 'F'
            if not init_from_np:
                nparr_or_shape = \
                    np.empty(nparr_or_shape, dtype=dtype, order=order)
            self._init_np_from_np(computer, nparr_or_shape)
        elif init_from_np:
            self._init_cl_from_np(computer, nparr_or_shape)
        else:
            self._init_cl_empty(computer, nparr_or_shape, dtype, c_contiguous)

    def _init_cl_empty(self, computer, shape, dtype, c_contiguous,
                       init_buffer=True):
        """
        Initialize an empty Mat that uses opencl.
        No type checks are performed on args.
        @raises ValueError: If the size exceeds the maximum allowed.
        """
        use_opencl = computer.use_opencl
        shape0 = shape[0]
        shape1 = shape[1]
        size = shape0*shape1
        if size > SIZE_T_MAX:
            raise ValueError('size of matrix exceeds the maximum allowed.')
        itemsize = dtype.itemsize
        nbytes = int(size*itemsize)

        ptr_stride0 = shape1 if c_contiguous else ONE
        ptr_stride1 = ONE if c_contiguous else shape0
        if shape0 == 1:
            ptr_stride0 = ZERO
        if shape1 == 1:
            ptr_stride1 = ZERO

        self._computer = computer
        self._use_opencl = use_opencl
        self._shape0 = shape0
        self._shape1 = shape1
        self._shape = shape0, shape1
        self._dtype = dtype
        self._size = size
        self._ndim = TWO
        self._begin = ZERO
        self._itemsize = itemsize
        self._nbytes = nbytes

        self._set_ptr_strides(ptr_stride0, ptr_stride1)
        if init_buffer:
            self._buffer = cl.Buffer(computer.context, cl.mem_flags.READ_WRITE,
                                     size=nbytes)

    def _init_cl_from_np(self, computer, ndarray):
        """
        Initialize object that uses opencl using an np.ndarray instance.
        No type checks are performed on args.
        """
        c_contiguous = ndarray.flags.c_contiguous
        if not c_contiguous and not ndarray.flags.f_contiguous:
            c_contiguous = True
            ndarray = ndarray.copy(order='C')

        shape = _size_t(ndarray.shape[0]), _size_t(ndarray.shape[1])
        self._init_cl_empty(computer, shape, ndarray.dtype, c_contiguous)
        computer._cl_buffer_copy(self._buffer, ndarray)

    def _init_np_from_np(self, computer, ndarray):
        """
        Initialize object that uses numpy using an np.ndarray instance.
        No type checks are performed on args.
        """
        self._use_opencl = False

        size = _size_t(ndarray.size)
        if size > SIZE_T_MAX:
            raise ValueError('size of matrix exceeds the maximum allowed.')
        itemsize = _size_t(ndarray.itemsize)
        shape = ndarray.shape
        shape0 = _size_t(shape[0])
        shape1 = _size_t(shape[1])

        strides = ndarray.strides
        ptr_stride0 = _size_t(strides[0]/itemsize)
        ptr_stride1 = _size_t(strides[1]/itemsize)
        if shape0 == 1:
            ptr_stride0 = ZERO
        if shape1 == 1:
            ptr_stride1 = ZERO

        self._computer = computer
        self._ndarray = ndarray
        self._dtype = ndarray.dtype
        self._shape0 = shape0
        self._shape1 = shape1
        self._shape = shape0, shape1
        self._size = size
        self._ndim = _size_t(ndarray.ndim)
        self._begin = ZERO
        self._itemsize = itemsize
        self._nbytes = int(size*itemsize)

        self._set_ptr_strides(ptr_stride0, ptr_stride1)

    def _set_begin(self, begin):
        """FOR INTERNAL USE ONLY.
        Set the _begin property.
        No type checks are performed on args.
        """
        self._begin = begin

    def _set_ptr_strides(self, ptr_stride0, ptr_stride1):
        """FOR INTERNAL USE ONLY.
        Set/Initialize all the properties dependent on ptr_strides.
        No type checks are performed on args.
        """
        self._ptr_stride0 = ptr_stride0
        self._ptr_stride1 = ptr_stride1

        if ptr_stride0 != 0 and ptr_stride1 != 0:
            c_contiguous = ptr_stride0 == self.shape1 and ptr_stride1 == 1
            f_contiguous = ptr_stride1 == self.shape0 and ptr_stride0 == 1
        elif ptr_stride0 == 0:
            contiguous = ptr_stride1 <= 1
            c_contiguous = contiguous
            f_contiguous = contiguous
        else:
            contiguous = ptr_stride0 <= 1
            c_contiguous = contiguous
            f_contiguous = contiguous

        self._c_contiguous = c_contiguous
        self._f_contiguous = f_contiguous
        self._contiguous = c_contiguous or f_contiguous

        order = ''
        if c_contiguous:
            order += 'C'
        if f_contiguous:
            order += 'F'
        self._order = order

    def _set_buffer(self, buf):
        """FOR INTERNAL USE ONLY.
        Will set the _buffer property of the object.
        No type checks are performed on args.
        """
        self._buffer = buf

    @property
    def computer(self):
        """
        @rtype: Computer
        """
        return self._computer

    @property
    def shape(self):
        """
        @rtype: (SIZE_T,SIZE_T)
        """
        return self._shape

    @property
    def shape0(self):
        """
        @rtype: SIZE_T
        """
        return self._shape0

    @property
    def shape1(self):
        """
        @rtype: SIZE_T
        """
        return self._shape1

    @property
    def size(self):
        """
        @rtype: SIZE_T
        """
        return self._size

    @property
    def dtype(self):
        """
        @rtype: numpy dtype
        """
        return self._dtype

    @property
    def ndim(self):
        """
        @rtype: SIZE_T
        """
        return self._ndim

    @property
    def itemsize(self):
        """
        @rtype: SIZE_T
        """
        return self._itemsize

    @property
    def nbytes(self):
        """
        @rtype: int
        """
        return self._nbytes

    @property
    def buffer(self):
        """
        @rtype: cl.Buffer
        """
        return self._buffer

    @property
    def begin(self):
        """
        The beginning index of the data with respect to its internal data.
        This is used only for opencl data because sub-buffer creation
        with arbitrary offsets is not guaranteed to be supported.
        @rtype: SIZE_T
        """
        return self._begin

    @property
    def ptr_stride0(self):
        """
        The number of elements to skip to get to the next row.
        Will be 0 when the number of rows is 1,
        this will allow easy matrix-vector operations.
        This is used internally in cl kernels.
        @rtype: SIZE_T
        """
        return self._ptr_stride0

    @property
    def ptr_stride1(self):
        """
        The number of elements to skip to get to the next column.
        Will be 0 when the number of columns is 1,
        this will allow easy matrix-vector operations.
        This is used internally in cl kernels.
        @rtype: SIZE_T
        """
        return self._ptr_stride1

    @property
    def c_contiguous(self):
        """
        True iff the data can be accessed in C order.
        @rtype: bool
        """
        return self._c_contiguous

    @property
    def f_contiguous(self):
        """
        True iff the data can be accessed in F/Fortran order.
        @rtype: bool
        """
        return self._f_contiguous

    @property
    def contiguous(self):
        """
        True iff the data is stored contiguously.
        @rtype: bool
        """
        return self._contiguous

    @property
    def order(self):
        """
        The type of ordering of the data.
        ''  if data is not contiguous,
        'C' if only c_contiguous,
        'F' if only f_contiguous,
        'CF' if both.
        @rtype: '','C','F','CF'
        """
        return self._order

    @property
    def use_opencl(self):
        """
        True iff the Computer uses opencl for computations.
        @rtype: bool
        """
        return self._use_opencl

    @property
    def NP(self):
        """
        The numpy/host representation of the matrix.
        If the Mat uses numpy data then this will be a reference
        to the internal data.
        If the Mat uses cl.Buffer data then this will be a copy of that
        and will be computed on each call.
        @rtype: numpy.ndarray
        """
        # TODO Consider removing or documenting the inefficiency warning.
        if self.use_opencl:
            if self.begin != 0 and self.order == '':
                warnings.warn(
                    'Copying cl data with data_begin!=0 from larger'
                    'into smaller buffer. '
                    'Consider calling .NP on the larger buffer '
                    'if you intend to also access other parts of it.',
                    InefficiencyWarning)
            if self.begin != 0 or \
               (not self.c_contiguous and not self.f_contiguous):
                proper_self = self.copy()
                assert proper_self.contiguous
            else:
                proper_self = self
            result = np.empty(proper_self.shape, proper_self.dtype,
                              order='C' if proper_self.c_contiguous else 'F')
            self.computer._cl_buffer_copy(result, proper_self.buffer)
        else:
            result = self._ndarray
        return result

    @property
    def T(self):
        """
        Shorthand for self.transpose()
        @rtype: Mat
        """
        return self.transpose()

    ##### Reshaping Methods #####

    def transpose(self):
        """
        Compute the transpose.
        @rtype: Mat
        @returns: The transpose of the Mat.
        """
        if self.use_opencl:
            if self.contiguous:
                proper_self = self
            else:
                proper_self = self.copy()
            result = Mat(None, None)
            result._init_cl_empty(self.computer,
                                  (proper_self.shape1, proper_self.shape0),
                                  dtype=self.dtype,
                                  c_contiguous=not proper_self.c_contiguous,
                                  init_buffer=False)
            result._set_buffer(proper_self.buffer)
        else:
            result = Mat(self.computer, self.NP.transpose())
        return result

    def torow(self):
        """
        Constructs a row version of the Mat
        where loop column index changes faster than row index.
        """
        return self.flatten(order='C').T

    def tocol(self):
        """
        Constructs a column version of the Mat
        where loop row index changes faster than column index."""
        return self.flatten(order='F')

    def reshape(self, shape, order='C'):
        """
        Create a new instance based on self with a given shape.
        @param shape: The new shape.
        @type order: 'C','F','A'
        @param order:
            The order in which values should be taken from self.
            'C' means that column index with change faster than row index.
            'F' means that row index with change column index.
            'A' means that self.order will be used.
            NOTE: The "order" property of the result is independent from this.
        @rtype: Mat
        @returns: A Mat with the given shape.
        @raises ValueError: If the shape implies a different size.
        """
        if self.use_opencl:
            shape = (_size_t(shape[0]), _size_t(shape[1]))
            if shape[0]*shape[1] != self.size:
                raise ValueError('New shape (%s) doesn\'t match size (%d)' %
                                 (shape, self.size))
            if order in self.order or order == 'A':
                proper_self = self
            else:
                proper_self = self.copy(c_contiguous=not self.c_contiguous)
            result = Mat(None, None)
            result._init_cl_empty(self.computer, shape, dtype=self.dtype,
                                  c_contiguous=proper_self.c_contiguous,
                                  init_buffer=False)
            result._set_buffer(proper_self.buffer)
        else:
            result = Mat(self.computer,
                         self.NP.reshape(shape, order=order))
        return result

    def flatten(self, order='C'):
        """
        @type order: 'C','F','A'
        @returns: self.reshape((self.size, 1), order=order)
        """
        return self.reshape((self.size, 1), order=order)

    ##### Assignment Methods #####

    def copy(self, c_contiguous=None):
        """
        Copy the object.
        @param c_contiguous:
            True if the result should be C-contiguous. False otherwise.
        @rtype: Mat
        @returns: A copy that is contiguous (c or f) and has begin 0.
        """
        if self.use_opencl:
            result = self._map('', c_contiguous=c_contiguous)
        else:
            result = Mat(self.computer,
                         self.NP.copy(order='C' if c_contiguous else 'F'))
        return result

    def empty_copy(self):
        """Return an empty version of self."""
        if self.use_opencl:
            result = Mat(None, None)
            result._init_cl_empty(self.computer, self.shape, self.dtype,
                                  self.c_contiguous)
        else:
            result = Mat(None, None)
            order = 'C' if self.c_contiguous else 'F'
            ndarr = np.empty(self.shape, dtype=self.dtype, order=order)
            result._init_np_from_np(self.computer, ndarr)
        return result

    def astype(self, dtype):
        """Cast Mat to the given dtype.
        @type dtype: numpy dtype
        @param dtype: The dtype of the resulting array.
        @rtype: Mat
        @returns: A newly created array (even if the dtype doesn't change).
        """
        if self.use_opencl:
            result = Mat(self.computer, self.shape, dtype=dtype,
                         c_contiguous=self.c_contiguous)
            # The map functions are faster than fill.
            if self.contiguous:
                self.computer._cl_map_1d('', self, result)
            else:
                self.computer._cl_map('', self, result)
        else:
            result = Mat(self.computer, self.NP.astype(dtype))
        return result

    def fill(self, other):
        """
        Fill the Mat with a scalar or with corresponding values from other Mat.
        When using numpy data and a scalar it will call numpy.ndarray.fill.
        Otherwise this function is more general
        than the numpy.ndarray.fill function,
        because it allows the argument to be non-scalar.
        @type other: Mat/scalar
        @param other: A scalar or a Mat of the same shape.
        """
        if self.use_opencl:
            self._iop('=', other)
        else:
            if isinstance(other, Mat):
                self.NP[:] = other.NP[:]
            else:
                self.NP.fill(other)

    ##### Access Methods #####
    def count_nonzero(self):
        """Return the number of non-zero elements in the Mat."""
        # TODO implement this in kernel, improve speed here.
        return (self != 0).astype(np.uint64).sum()

    # def getl(self, index):
        # """Get elements using a logical index of the same size."""
        # if index.shape != self.shape:
            # raise ValueError('Index shape mismatch %s != %s',
                             # index.shape, self.shape)

    # def setl(self, index, value):
        # """Set elements using a logical index of the same size"""
        # if index.shape != self.shape:
            # raise ValueError('Index shape mismatch %s != %s',
                             # index.shape, self.shape)
        # if value.shape != self.shape:
            # raise ValueError('Value shape mismatch %s != %s',
                             # value.shape, self.shape)
    # def getl2(self, index0, index1):
        # """Get elements using two logical indexes."""
    # def setl2(self, index0, index1, value):
        # """Set elements using two logical indexes."""

    def li(self, index, value=None):
        """
        Logical index operations get/set with Mat index.
        @type index: Mat
        @param index:
            A Mat containing info on which values to get/set.
            index[i,j] != 0 indicates self[i,j] should be used.
            Each index dimension must equal 1 or match self.
            At least one index dimension must match self.
        @param value: If not None will set instead of get.
        @returns:
            None if logical index is empty and value is None.
            Otherwise a new Mat if value is None, otherwise self.
            If 1 not in index.shape, will return a Mat with shape1==1
        @raises ValueError:
            If index.shape does not conform to self.shape.
            If value.size does not conform to
            number of values referred to by index.
        """
        if index.shape0 != self.shape0:
            incompatible = index.shape0 != 1 or index.shape1 != self.shape1
        elif index.shape1 != self.shape1 and index.shape0 != 1:
            incompatible = True
        else:
            incompatible = False
        if incompatible:
            raise ValueError('index shape %s is incompatible with %s' %
                             (index.shape, self.shape))
        if self.use_opencl:
            if value is None:  # Get operation
                nnz = index.count_nonzero()
                if nnz == 0:
                    return None
                if index.shape0 == 1:
                    rshape = (self.shape0, nnz)
                elif index.shape1 == 1:
                    rshape = (nnz, self.shape1)
                else:
                    rshape = (nnz, 1)

                result = Mat(self.computer, rshape,
                             dtype=self.dtype,
                             c_contiguous=self.c_contiguous)
            else:  # Set operation
                result = self
            self.computer._cl_li(result, self, index, value)
        else:
            index_np = index.NP
            if index_np.dtype != np.bool:
                index_np = index_np.astype(np.bool)
            if index.shape == self.shape:
                index_arg = index.NP
            elif index.shape0 == self.shape0:
                index_arg = (index.NP.squeeze(), slice(None))
            else:
                index_arg = (slice(None), index.NP.squeeze())
            if value is None:
                result_np = self.NP[index_arg]
                if result_np.size == 0:
                    return None
                if result_np.ndim == 1:
                    result_np = result_np[:, np.newaxis]
                result = Mat(self.computer, result_np)
            else:
                if isinstance(value, Mat):
                    value = value.NP
                self.NP[index_arg] = value
                result = self
        return result

    def _slice_helper(self, slice_arg, axis_size, axis):
        """Checks and converts slice to appropriate begin,step,end."""
        begin = slice_arg.start
        if begin is None:
            begin = 0
        elif begin < 0:
            raise IndexError('Negative indices are not allowed (yet).')
        end = slice_arg.stop
        if end is None:
            end = axis_size
        elif end > axis_size:
            raise IndexError('index %s is out of bounds for axis'
                             '%d with size %s' % (end, axis, axis_size))
        if end < begin:
            raise IndexError('Non-increasing slice %s:%s not supported (yet).'
                             % (begin, end))
        step = slice_arg.step
        if step is None:
            step = 1
        elif step <= 0:
            raise IndexError('Non-positive steps are not allowed (yet).')
        new_axis_size = end-begin
        if step > new_axis_size:
            raise IndexError('Slice step size is too large %s:%s:%s '
                             % (begin, end, step))
        if step != 1:
            new_axis_size += 1
            new_axis_size //= step
        return (_size_t(begin), _size_t(step), _size_t(end),
                _size_t(new_axis_size))

    def _getitem_by_slice(self, slice0, slice1):
        """
        Obtain a submatrix of self (assumes use_opencl).
        If possible the submatrix will reuse the data.
        Negative values in slices are not allowed.
        @type slice0: slice
        @param slice0: Defines the rows to choose from self.
        @type slice1: slice
        @param slice1: Defines the columns to choose from self.
        """
        begin0, step0, end0, shape0 = \
            self._slice_helper(slice0, self.shape0, 0)
        begin1, step1, end1, shape1 = \
            self._slice_helper(slice1, self.shape1, 1)

        stride0 = self.ptr_stride0
        stride1 = self.ptr_stride1
        begin = self.begin + begin0*stride0 + begin1*stride1
        stride0 *= step0
        stride1 *= step1

        result = Mat(None, None)
        result._init_cl_empty(self.computer, (shape0, shape1),
                              dtype=self.dtype, c_contiguous=self.c_contiguous,
                              init_buffer=False)
        result._set_buffer(self.buffer)
        result._set_begin(begin)
        result._set_ptr_strides(stride0, stride1)
        return result

    def __getitem__(self, key):
        """
        self[key] for restricted set of slice/int tuples.
        Also see Mat.li (logical indexing).
        For arbitrary keys you can also use self.computer.M(self.NP[key]),
        as long as the key works with numpy.ndarray.__getitem__.
        @type key: (slice/int, slice/int)
        @param key:
            Not all slices are allowed, will raise exception.
        @raises TypeError: If key is not a tuple of (slice, slice)
        @returns:
            For opencl, guaranteed to return a Mat that uses the same data.
            For numpy this will most likely also be the case,
            but depends on the numpy implementation.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('Expected key to be 2-tuple.')
        for k in key:
            if not (isinstance(k, slice) or isinstance(k, int)):
                raise TypeError('Expected slice or int.')
        k0, k1 = key
        if self.use_opencl:
            if isinstance(k0, int):
                if k0 < 0:
                    k0 = self.shape0 - k0
                k0 = slice(k0, k0 + 1, 1)
            if isinstance(k1, int):
                if k1 < 0:
                    k1 = self.shape1 - k1
                k1 = slice(k1, k1 + 1, 1)

            result = self._getitem_by_slice(k0, k1)
        else:
            result = Mat(self.computer, self.NP[key])
        return result

#    def __setitem__(self, key, value):
#        """self[key]=value
#        @type value: Mat, scalar
#        @param value: anything that can be passed to fill."""
#        if isinstance(key, Mat):
#            key = key.NP
#        altered_arr = self.NP
#        altered_arr[key] = value
#        self.fill(self.computer.M(altered_arr))

#        if self.use_opencl:
#            submat = self.__getitem__(key)
#            success = False
#            if submat.buffer is self.buffer:
#                try:
#                    submat.fill(value)
#                    success = True
#                except Exception:
#                    pass
#            if not success:
#                warnings.warn('__setitem__ has not implemented support '
#                              'for this type of key (yet). '
#                              'You may want to use slices and/or '
#                              'a Mat value instead.',
#                              InefficiencyWarning)
#                if isinstance(value, Mat):
#                    value = value.NP
#                new_np = self.NP
#                if isinstance(key, Mat):
#                    key = key.NP
#                new_np[key] = value
#                self.fill(Mat(self.computer, new_np))
#        else:
#            if isinstance(key, Mat):
#                key = key.NP
#            if isinstance(value, Mat):
#                value = value.NP
#            self._ndarray[key] = value

#    def __contains__(self, item):
#        """Whether item is in self"""
#        return NotImplemented

    ##### Object Methods #####

    def __bool__(self):
        """Needed to allow scalar==self and scalar!=self."""
        # TODO change this once you know where it fails.
        raise MissingOperationError(
            '__nonzero__ is disabled because it is called innapriopriately '
            'for some scalar-matrix operators. '
            'This may be failing for that reason OR OTHERWISE. '
            'NOTE: The operation c <op> M is not available for <op>= %s. '
            'You have to use M <op> c instead.' % OPS_NON_REVERSABLE)
        return self

    def __str__(self):
        """Returns self.NP.__str__()"""
        return self.NP.__str__()

    def __repr__(self):
        """Returns self.NP.__repr__()"""
        return self.NP.__repr__()

    ##### Arithmetic Methods #####

    def __pos__(self):
        """+self"""
        if self.use_opencl:
            result = self._map('+')
        else:
            result = Mat(self.computer, +self.NP)
        return result

    def __neg__(self):
        """-self"""
        if self.use_opencl:
            result = self._map('-')
        else:
            result = Mat(self.computer, -self.NP)
        return result

    def __abs__(self):
        """abs(self)"""
        if self.use_opencl:
            result = self._map('abs')
        else:
            result = Mat(self.computer, abs(self.NP))
        return result

    def __invert__(self, other):
        """~self"""
        if self.use_opencl:
            result = self._map('~')
        else:
            result = Mat(self.computer, ~self.NP)
        return result

    def __eq__(self, other):
        """self==other"""
        return self._op_logical('==', other)

    def __req__(self, other):
        """other==self"""
        return self._op_logical('==', other)

    def __ne__(self, other):
        """self!=other"""
        return self._op_logical('!=', other)

    def __rne__(self, other):
        """other!=self"""
        return self._op_logical('!=', other)

    def __gt__(self, other):
        """self>other"""
        return self._op_logical('>', other)

    def __ge__(self, other):
        """self>=other"""
        return self._op_logical('>=', other)

    def __lt__(self, other):
        """self<other"""
        return self._op_logical('<', other)

    def __le__(self, other):
        """self<=other"""
        return self._op_logical('<=', other)

    def __add__(self, other):
        """self+other"""
        return self._op('+', other)

    def __radd__(self, other):
        """other+self"""
        return self._op('+', other, reverse=True)

    def __sub__(self, other):
        """self-other"""
        return self._op('-', other)

    def __rsub__(self, other):
        """other-self"""
        return self._op('-', other, reverse=True)

    def __mul__(self, other):
        """self*other"""
        return self._op('*', other)

    def __rmul__(self, other):
        """other*self"""
        return self._op('*', other, reverse=True)

    def __truediv__(self, other):
        """self//other"""
        return self._op('/', other)

    def __rtruediv__(self, other):
        """other//self"""
        return self._op('/', other, reverse=True)

    def __mod__(self, other):
        """self%other"""
        return self._op('%', other)

    def __rmod__(self, other):
        """other%self"""
        return self._op('%', other, reverse=True)

    def __and__(self, other):
        """self&other"""
        return self._op('&', other)

    def __rand__(self, other):
        """other&self"""
        return self._op('&', other, reverse=True)

    def __or__(self, other):
        """self|other"""
        return self._op('|', other)

    def __ror__(self, other):
        """other|self"""
        return self._op('|', other, reverse=True)

    def __xor__(self, other):
        """self^other"""
        return self._op('^', other)

    def __rxor__(self, other):
        """other^self"""
        return self._op('^', other, reverse=True)

    def __lshift__(self, other):
        """self<<other"""
        return self._op('<<', other)

    def __rshift__(self, other):
        """self>>other"""
        return self._op('>>', other)

    def __pow__(self, other):
        """self**other"""
        if self.use_opencl:
            shape = self.shape
            if isinstance(other, Mat) and self.size < other.size:
                shape = other.shape
            result = Mat(None, None)
            result._init_cl_empty(self.computer, shape, self.dtype,
                                  self.c_contiguous)
            self.computer.f(result, 'pow', self, other)
        else:
            if isinstance(other, Mat):
                result = Mat(self.computer, self.NP**other.NP)
            else:
                result = Mat(self.computer, self.NP**other)
        return result

    def __rpow__(self, other):
        """other**self"""
        if self.use_opencl:
            shape = self.shape
            if isinstance(other, Mat) and self.size < other.size:
                shape = other.shape
            result = Mat(None, None)
            result._init_cl_empty(self.computer, shape, self.dtype,
                                  self.c_contiguous)
            self.computer.f(result, 'pow', other, self)
        else:
            if isinstance(other, Mat):
                result = Mat(self.computer, other.NP**self.NP)
            else:
                result = Mat(self.computer, other**self.NP)
        return result

    def __iadd__(self, other):
        """self+=other"""
        self._iop('+=', other)
        return self

    def __isub__(self, other):
        """self-=other"""
        self._iop('-=', other)
        return self

    def __imul__(self, other):
        """self*=other"""
        self._iop('*=', other)
        return self

    def __idiv__(self, other):
        """self/=other"""
        self._iop('/=', other)
        return self

    def __imod__(self, other):
        """self%=other"""
        self._iop('%=', other)
        return self

    def __iand__(self, other):
        """self&=other"""
        self._iop('&=', other)
        return self

    def __ior__(self, other):
        """self|=other"""
        self._iop('|=', other)
        return self

    def __ixor__(self, other):
        """self^=other"""
        self._iop('^=', other)
        return self

    def __ilshift__(self, other):
        """self<<=other"""
        self._iop('<<=', other)
        return self

    def __irshift__(self, other):
        """self>>=other"""
        self._iop('>>=', other)
        return self

    def __ipow__(self, other):
        """self**=other"""
        if self.use_opencl:
            self.computer.f(self, 'pow', self, other)
        else:
            if isinstance(other, Mat):
                self._ndarray **= other.NP
            else:
                self._ndarray **= other
        return self

    def clip(self, a_min, a_max, out=None):
        """
        Clip the values from self to lie in the interval [a_min,a_max].
        @type a_min: Mat/scalar
        @type a_max: Mat/scalar
        @type out: Mat
        """
        if out is None:
            out = self.empty_copy()
        if self.use_opencl:
            self.computer.f(out, 'clamp', self, a_min, a_max)
        else:
            if out.use_opencl:
                raise ValueError('out cannot be using opencl if self isn''t.')
            # numpy clip has a bug when order is different for out
            self_np = self.NP
            out_np = out.NP
            self_order = self.order
            if self_order is None or self.order != out.order:
                self_np = self_np.copy(
                    order='C' if out_np.flags.c_contiguous else 'F')
            self_np.clip(a_min, a_max, out=out_np)
        return out

    def round(self, out=None):
        """
        round using numpy.round or the opencl builtin round function.
        Note that results may differ for borderline cases.
        If you want to make sure they don't
        you have to do the rounding using numpy arrays, e.g. something like
        out[:] = self.computer.M(self.NP.round())
        """
        if out is None:
            out = self.empty_copy()
        if self.use_opencl:
            self.computer.f(out, 'round', self)
        else:
            if out.use_opencl:
                raise ValueError('out cannot be using opencl if self isn''t.')
            self.NP.round(decimals=0, out=out.NP)
        return out

    ##### Reduction Methods #####

    def all(self, axis=None, out=None):
        """all"""
        return self._reduce('all', axis=axis, out=out)

    def any(self, axis=None, out=None):
        """any"""
        return self._reduce('any', axis=axis, out=out)

    def max(self, axis=None, out=None):
        """max"""
        return self._reduce('max', axis=axis, out=out)

    def argmax(self, axis=None, out=None):
        """argmax"""
        return self._reduce('argmax', axis=axis, out=out)

    def min(self, axis=None, out=None):
        """min"""
        return self._reduce('min', axis=axis, out=out)

    def argmin(self, axis=None, out=None):
        """argmin"""
        return self._reduce('argmin', axis=axis, out=out)

    def ptp(self, axis=None, out=None):
        """ptp"""
        return self._reduce('ptp', axis=axis, out=out)

    def sum(self, axis=None, out=None):
        """sum"""
        if axis is None:
            first_axis = int(self.c_contiguous)
            result = self.sum(axis=first_axis).sum(axis=1-first_axis).NP[0, 0]
        else:
            result = self._reduce('sum', axis=axis, out=out)
        return result

#    def cumsum(self, axis=None, out=None):
#        return self._reduce('cumsum', axis=axis, out=out)

    def prod(self, axis=None, out=None):
        """prod"""
        return self._reduce('prod', axis=axis, out=out)

#    def cumprod(self, axis=None, out=None):
#        return self._reduce('cumprod', axis=axis, out=out)

    def mean(self, axis=None, out=None):
        """mean"""
        if axis is None:
            first_axis = int(self.c_contiguous)
            result = self.mean(axis=first_axis).mean(axis=1-first_axis).NP[0, 0]
        else:
            result = self._reduce('mean', axis=axis, out=out)
        return result

    def var(self, axis=None, out=None, ddof=0):
        """var"""
        result = self._reduce('var', axis=axis, out=out)
        if ddof != 0:
            n = float(self.size/result.size)
            result *= (n/(n-ddof))
        return result

    def std(self, axis=None, out=None, ddof=0):
        """std"""
        result = self._reduce('std', axis=axis, out=out)
        if ddof != 0:
            n = float(self.size/result.size)
            result *= np.sqrt(n/(n-ddof))
        return result

    ##### Special Methods #####

#    def nonzero(self):
#        return result
    #def repeat(self):
    #def sort(self):
    #def diagonal(self):
    #def trace([offset, axis1, axis2, dtype, out]):
        #"""Return the sum along diagonals of the array."""

    ##### Kernel Methods #####

    def _map(self, function_name, c_contiguous=None):
        """Maps a function into a result (not in-place)."""
        # Note here that any non-contiguous self
        # will produce an f_contiguous result,
        # this is optimal in terms of memory access
        # since work items loop order is column major.
        if c_contiguous is None:
            c_contiguous = self.c_contiguous
        result = Mat(self.computer, self.shape, dtype=self.dtype,
                     c_contiguous=c_contiguous)
        if self.computer._call_dim(result, self) == 1:
            self.computer._cl_map_1d(function_name, self, result)
        else:
            self.computer._cl_map(function_name, self, result)
        return result

    def _imap(self, function_name):
        """Maps a function in-place."""
        if self.contiguous:
            self.computer._cl_map_1d(function_name, self)
        else:
            self.computer._cl_map(function_name, self)

    def _op_logical(self, function_name, other):
        """Applies a logical operator into a result (not in-place)"""
        rdtype = BOOL_TYPE
        shape = self.shape
        if isinstance(other, Mat) and self.size < other.size:
            shape = other.shape

        if self.use_opencl:
            result = Mat(self.computer, shape,
                         dtype=rdtype,
                         c_contiguous=self.c_contiguous)
            if self.computer._call_dim(self, other) == 1:
                op_logical_f = self.computer._cl_op_logical_1d
            else:
                op_logical_f = self.computer._cl_op_logical
            op_logical_f(function_name, result, self, other)
        else:
            other_arg = other
            if isinstance(other_arg, Mat):
                other_arg = other.NP
            nparr = eval('self.NP %s other_arg' % function_name)
            result = Mat(self.computer, nparr)
            if result.shape != shape:
                raise MissingOperationError(
                    'Trying to return Mat with unexpected shape.')
        return result

    def _op(self, function_name, other, reverse=False):
        """
        Applies an operator into a result (not in-place).
        NOT TO BE USED WITH LOGICAL OPERATORS
        (may still work but result is not guaranteed to be consistent)
        @rtype: Mat
        @returns: The resulting Mat will have a dtype determined by
        the safe_cast_non_logical method.
        """
        is_scalar = not isinstance(other, Mat)

        if is_scalar:
            rdtype = self.computer.safe_cast_non_logical(
                self.dtype, np.array(other).dtype, operator=function_name)
        else:
            rdtype = self.computer.safe_cast_non_logical(
                self.dtype, other.dtype, operator=function_name)
        if is_scalar and reverse:
            # The exception below gets caught by python and has no effect.
            warnings.warn(
                'When calling c %s M it may return a different dtype. '
                'Consider calling M %s c instead to match numpy behavior.'
                '' % (function_name, function_name),
                RtypeWarning)

        if self.use_opencl:
            shape = self.shape
            if not is_scalar and self.size < other.size:
                shape = other.shape
            result = Mat(self.computer, shape,
                         dtype=rdtype,
                         c_contiguous=self.c_contiguous)
            if self.computer._call_dim(self, other) == 1:
                op_f = self.computer._cl_op_1d
            else:
                op_f = self.computer._cl_op
            if reverse:
                op_f(function_name, result, other, self)
            else:
                op_f(function_name, result, self, other)
        else:
            other_arg = other
            if isinstance(other_arg, Mat):
                other_arg = other.NP
            if reverse:
                nparr = eval('other_arg %s self.NP' % function_name)
            else:
                nparr = eval('self.NP %s other_arg' % function_name)

            if nparr.dtype != rdtype:
                nparr = nparr.astype(rdtype)
            result = Mat(self.computer, nparr)
        return result

    def _iop(self, function_name, other):
        """Applies an in-place operator."""
        if self.use_opencl:
            if self.computer._call_dim(self, other) == 1:
                self.computer._cl_iop_1d(function_name, self, other)
            else:
                self.computer._cl_iop(function_name, self, other)
        else:
            other_arg = other
            if isinstance(other_arg, Mat):
                other_arg = other.NP
            exec('self._ndarray %s other_arg' % function_name)

    def _reduce(self, function_name, axis=None, out=None):
        """
        Apply a reduction to self along a given axis.
        out argument only works for axis in [0, 1]
        """
        if axis is None:
            pass
        elif axis == 0:
            shape = (1, self.shape1)
        elif axis == 1:
            shape = (self.shape0, 1)
        if self.use_opencl:
            if axis is None:
                if not self.contiguous:
                    raise InefficiencyWarning(
                        'Reduce is calling copy due to non-contiguous data.'
                        ' You may want to reimplement the callee.')
                    proper_self = self.copy()
                else:
                    proper_self = self
                result = self.computer._cl_reduce_1d(function_name,
                                                     proper_self)
            else:
                if out is None:
                    out_dtype = self.dtype
                    if function_name in ['all', 'any']:
                        out_dtype = BOOL_TYPE
                    elif 'arg' in function_name:
                        out_dtype = _DTYPE('int64')
                    out = Mat(self.computer, shape, dtype=out_dtype)
                self.computer._cl_reduce_2d(function_name, out, self, axis)
                result = out
        else:
            result = getattr(self.NP, function_name)(axis, out=out)
            if axis is not None:
                result = result.reshape(shape)
                result = Mat(self.computer, result)
        return result
