# clmat
A python module/package that allows numerical computations with numpy or opencl using the same code.

It will run on machines that do not have opencl.

clmat runs in the following regimes:
    [pure numpy]
        In this case clmat.matrix is basically a wrapper around numpy.ndarray, with a few differences.
        The main differences are:
            - all data types are matrices.
            - reductions like sum(axis=0) yield matrices.
    [opencl on any pyopencl.Device object]
        Any single opencl device can be used for the computations.
        opencl fission can help you limit the number of device cores being used.

IMPORTANT TARGETS
'make doc/index.html'
    Will generate the api from clmat.py docstrings using epydoc.
'make test'
    Will run all unit tests using py.test


DESIGN PRIORITIES
When there is a trade-off in code design, the following priorities will *generally* hold.
- Portability - try to make the code execute in all regimes.
- Consistency - try to make the different regimes yield the same results (except for known numerical errors).
- Device constancy - keep all computations on the same device, avoid conversion to and from host/numpy.
- GPU computation speed - optimize opencl computation on gpu devices.
- CPU computation speed - optimize opencl computation on cpu devices.
- floating point computations - optimize for float32, then float64, then integer types.
- numpy consistency - mimic numpy behaviour.

DEPENDENCIES:
python2 - Any version >=2.7 should work.
numpy
pyopencl (optional) - Needed for any opencl related functionality.
epydoc (optional) - Needed for generating api.
py.test (optional) - Needed for unit tests. Prefer version 2.5.2. Other versions may fail or use too much memory.

EXAMPLES:
clc = clmat.Computer(clmat.CPU)  # Construct a Computer instance that uses the CPU for opencl computations.
a = clc.randn((1024,1024),dtype='float32')  # Constructs a Mat full of random normal numbers.
b = clc.randu((1024,1024))  # Constructs a Mat full of random uniform numbers.
c = a+b  # Add the values in a and b into a new Mat c.
cnp = c.NP  # Get an numpy.ndarray that has the same values as c.
clc.mmult(a,b)  # Matrix multiplication.

Use clmat-info.py (a script) to get device and other information.
Use 'python clmat-info.py -h' to get help.
# 
