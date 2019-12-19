#ifndef RANLUXCL_CL
#define RANLUXCL_CL

/**** RANLUXCL v1.3.1 MODIFIED *************************************************

Implements the RANLUX generator of Matrin Luscher, based on the Fortran 77
implementation by Fred James. This OpenCL code is a complete implementation
which should perfectly replicate the numbers generated by the original Fortran
77 implementation (if using the legacy initialization routine).

***** QUICK USAGE DESCRIPTION **************************************************

1. Create an OpenCL buffer with room for at least 28 32-bit variables (112 byte)
per work-item. I.e., in C/C++: size_t buffSize = numWorkitems * 112;

2. Pass the buffer and an unsigned integer seed <ins> to a kernel that launches
the ranluxcl_initialization function. The seed <ins> can be any unsigned 32-bit
integer, and must be different on different OpenCL devices/NDRanges to ensure
different sequences. As long as the number of work-items on each device/NDRange
is less than 2^32 = 4294967296 all sequences will be different.
An examle initialization kernel would be:
	#include "ranluxcl.cl"
	kernel void Kernel_Ranluxcl_Init(private uint ins,
		global ranluxcl_state_t *ranluxcltab)
	{
		ranluxcl_initialization(ins, ranluxcltab);
	}

3. Now the generator is ready for use. Remember to download the seeds first,
and upload them again when done. Example kernel that downloads seeds, generates
a float4 where each component is uniformly distributed between 0 and 1, end
points not included, then uploads the seeds again:
	#include "ranluxcl.cl"
	kernel void Kernel_Example(global ranluxcl_state_t *ranluxcltab)
	{
		//ranluxclstate stores the state of the generator.
		ranluxcl_state_t ranluxclstate;

		//Download state into ranluxclstate struct.
		ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

		//Generate a float4 with each component on (0,1),
		//end points not included. We can call ranluxcl as many
		//times as we like until we upload the state again.
		float4 randomnr = ranluxcl32(&ranluxclstate);

		//Upload state again so that we don't get the same
		//numbers over again the next time we use ranluxcl.
		ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
	}

***** MACROS *******************************************************************

The following macros can optionally be defined:

RANLUXCL_LUX:
Sets the luxury level of the generator. Should be 0-4, or if it is 24 or larger
it sets the p-value of the generator (generally not needed). If this macro is
not set then lux=4 is the default (highest quality). For many applications the
high quality of lux=4 may not be needed. Indeed if two values (each value
having 24 random bits) are glued together to form a 48-bit value the generator
passes all tests in the TestU01 suite already with lux=2. See
"TestU01: A C Library for Empirical Testing of Random Number Generators" by
PIERRE LAeECUYER and RICHARD SIMARD. SWB(224, 10, 24)[24, l] is RANLUX with
two values glued together to create 48-bit numbers, and we see that it passes
all tests already at luxury value 2.

RANLUXCL_NO_WARMUP:
Turns off the warmup functionality in ranluxcl_initialization. This macro
should generally not be used, since the generators will initially be correlated
if it is defined. The only advantage is that the numbers generated will exactly
correspond to those of the original Fortran 77 implementation.

RANLUXCL_SUPPORT_DOUBLE:
Enables double precision functions. Please enable the OpenCL double precision
extension yourself, usually by "#pragma OPENCL EXTENSION cl_khr_fp64 : enable".

RANLUXCL_USE_LEGACY_INITIALIZATION
Uses exactly the same initialization routine as in the original Fortran 77 code,
leading to the same sequences. If using legacy initialization there are some
restrictions on what the seed <ins> can be, and it may also be necessary to
define RANLUXCL_MAXWORKITEMS if several sequences are to be run in parallel.

RANLUXCL_MAXWORKITEMS:
When RANLUXCL_USE_LEGACY_INITIALIZATION is defined we may need this macro.
If several OpenCL NDRanges will be running in parallel and the parallel
sequences should be different then this macro should have a value equal or
larger than the
largest number of work-items in any of the parallel runs. The default is to
use the current global size, so if all NDRanges are of the same size this need
not be defined.
	Each parallel instance must also have different seeds <ins>. For example if
we are launching 5120 work-items on GPU1 and 10240 work-items on GPU2 we would
use different seeds for the two generators, and RANLUXCL_MAXWORKITEMS must be
defined to be at least 10240. If GPU1 and GPU2 had the same number of work-items
this would not be necessary. 
	An underestimate of the highest permissible seed <ins> is given by the
smallest of:
(<maxins> = 10^9 / <numWorkitems>) or (<maxins> = 10^9 / RANLUXCL_MAXWORKITEMS).
Please make sure that <ins> is never higher than this since it could cause
undetected problems. For example with 10240 work-items the highest permissible
<ins> is about 100 000.
	Again note that this is only relevant when using the legacy initialization
function enabled by RANLUXCL_USE_LEGACY_INITIALIZATION. When not using the
legacy initialization this macro is effectively set to a very high value of
2^32-1.

***** FUNCTIONS: INITIALIZATION ************************************************

The initialization function is defined as:
void ranluxcl_initialization(uint ins, global ranluxcl_state_t *ranluxcltab)
Run once at the very beginning. ranluxcltab should be a buffer with space for
112 byte per work-item in the NDRange. <ins> is the seed to the generator.
For a given <ins> each work-item in the NDRange will generate a different
sequence. If more than one NDRange is used in parallel then <ins> must be
different for each NDRange to avoid identical sequences.

***** FUNCTIONS: SEED UPLOAD/DOWNLOAD ******************************************

The following two functions should be launced at the beginning and end of a
kernel that uses ranluxcl to generate numbers, respectively:

void ranluxcl_download_seed(ranluxcl_state_t *rst,
	global ranluxcl_state_t *ranluxcltab)
Run at the beginning of a kernel to download ranluxcl state data

void ranluxcl_upload_seed(ranluxcl_state_t *rst,
	global ranluxcl_state_t *ranluxcltab)
Run at the end of a kernel to upload state data

***** FUNCTIONS: GENERATION AND SYNCHRONIZATION ********************************

float4 ranluxcl32(ranluxcl_state_t *rst)
Run to generate a pseudo-random float4 where each component is a number between
0 and 1, end points not included (meaning the number will never be exactly 0 or
1).

double4 ranluxcl64(ranluxcl_state_t *rst)
Double precision version of the above function. The preprocessor macro
RANLUXCL_SUPPORT_DOUBLE must be defined for this function to be available.
This function "glues" together two single-precision numbers to make one double
precision number. Most of the work is still done in single precision, so the
performance will be roughly halved regardless of the double precision
performance of the hardware.

float4 ranluxcl32norm(ranluxcl_state_t *rst)
Run to generate a pseudo-random float4 where each component is normally
distributed with mean 0 and standard deviation 1.

double4 ranluxcl64norm(ranluxcl_state_t *rst)
Double precision version of the above function. The preprocessor macro
RANLUXCL_SUPPORT_DOUBLE must be defined for this function to be available.

void ranluxcl_synchronize(ranluxcl_state_t *rst)
Run to synchronize execution in case different work-items have made a different
number of calls to ranluxcl. On SIMD machines this could lead to inefficient
execution. ranluxcl_synchronize allows us to make sure all generators are
SIMD-friendly again. Not needed if all work-items always call ranluxcl the same
number of times.

***** PERFORMANCE **************************************************************

For luxury setting 4, performance on AMD Cypress should be ~4.5*10^9 pseudo-
random values per second, when not downloading values to host memory (i.e. the
values are just generated, but not used for anything in particular).

***** DESCRIPTION OF THE IMPLEMENTATION ****************************************

This code closely follows the original Fortran 77 code (see credit section).
Here the differences (and similarities) between RANLUXCL (this implementation)
and the original RANLUX are discussed.

The Fortran 77 implementation uses a simple LCG to initialize the generator, and
so the same approach is taken here. If RANLUXCL is initialized with <ins> = 0 as
seed, the first work-item behaves like the original RANLUX with seed equal 1,
the second work-item as if with seed equal 2 and so on. If <ins> = 1 then the
first work-item behaves like the original RANLUX with seed equal to
<numWorkitems> + 1, and so on for higher <ins> so that we never have overlapping
sequences. This is why the RANLUXCL_MAXWORKITEMS macro must be set if we have
different NDRanges with a different number of work-items.

RANLUX is based on chaos theory, and what we are actually doing when selecting
a luxury value is setting how many values to skip over (causing decorrelation).
The number of values to skip is controlled by the so-called p-value of the
generator. After generating 24 values we skip p - 24 values until again
generating 24 values.

This implementation is somewhat modified from the original fortran
implementation by F. James. Because of the way the OpenCL code is optimized with
4-component 32-bit float vectors, it is most convenient to always throw away
some multiple of 24 values (i.e. p is always a multiple of 24).

However, there might be some resonances if we always throw away a multiple of
the seeds table size. Therefore the implementation is slightly more intricate
where p can be a multiple of 4 instead, at a cost to performance (only about 10%
lower than the cleaner 24 values approach on AMD Cypress). These two approaches
are termed planar and planar shift respectively. The idea for the planar
approach comes from the following paper:
Vadim Demchik, Pseudo-random number generators for Monte Carlo simulations on
Graphics Processing Units, arXiv:1003.1898v1 [hep-lat]

Below the p-values for the original reference implementation are listed along
with those of the planar shift implementation. Suggested values for the planar
approach are also presented. When this function is called with RANLUXCL_LUX
set to 0-4, the planar shift values are used. To use the pure planar approach
(for some extra performance with likely undetectable quality decrease), set lux
equal to the specific p-value.

Luxury setting (RANLUXCL_LUX):                   0   1   2   3   4
Original fortran77 implementation by F. James:  24  48  97  223 389
Planar (suggested):                             24  48  120 240 408
Planar shift:                                   24  48  100 224 404

Note that levels 0 and 1 are the same as in the original implementation for both
planar and planar shift. Level 4 of planar shift where p=404 is the same as
chosen for luxury level 1 by Martin Luescher for his v3 version of RANLUX.
Therefore if it is considered important to only use "official" values, luxury
settings 0, 1 or 4 of planar shift should be used. It is however unlikely that
the other values are bad, they just haven't been as extensively used and tested
by others.

Variable names are generally the same as in the fortran77 implementation,
however because of the way the generator is implemented, the i24 and j24
variables are no longer needed.

***** CREDIT *******************************************************************

I have been told by Fred James (the coder) that the original Fortran 77
implementation (which is the subject of the second paper below) is free to use
and share. Therefore I am using the MIT license (below). But most importantly
please always remember to give credit to the two articles by Martin Luscher and
Fred James, describing the generator and the Fortran 77 implementation on which
this implementation is based, respectively:

Martin Luescher, A portable high-quality random number generator for lattice
field theory simulations, Computer Physics Communications 79 (1994) 100-110

F. James, RANLUX: A Fortran implementation of the high-quality pseudorandom
number generator of Luescher, Computer Physics Communications 79 (1994) 111-114

***** LICENSE ******************************************************************

Copyright (c) 2011 Ivar Ursin Nikolaisen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*******************************************************************************/

typedef struct{
	float
		s01, s02, s03, s04,
		s05, s06, s07, s08,
		s09, s10, s11, s12,
		s13, s14, s15, s16,
		s17, s18, s19, s20,
		s21, s22, s23, s24;
	float carry;
	float dummy; //Causes struct to be a multiple of 128 bits
	int in24;
	int stepnr;
} ranluxcl_state_t;

//Initial prototypes makes Apple's compiler happy
void ranluxcl_download_seed(ranluxcl_state_t *, global ranluxcl_state_t *);
void ranluxcl_upload_seed(ranluxcl_state_t *, global ranluxcl_state_t *);
float ranluxcl_os(float, float, float *, float *);
float4 ranluxcl32(ranluxcl_state_t *);
void ranluxcl_synchronize(ranluxcl_state_t *);
void ranluxcl_initialization(uint, global ranluxcl_state_t *);
float4 ranluxcl32norm(ranluxcl_state_t *);

#ifdef RANLUXCL_SUPPORT_DOUBLE
double4 ranluxcl64(ranluxcl_state_t *);
double4 ranluxcl64norm(ranluxcl_state_t *);
#endif

#define RANLUXCL_TWOM24 0.000000059604644775f
#define RANLUXCL_TWOM12 0.000244140625f

#ifdef RANLUXCL_LUX
#if RANLUXCL_LUX < 0
#error ranluxcl: lux must be zero or positive.
#endif
#else
#define RANLUXCL_LUX 4 //Default to high quality
#endif //RANLUXCL_LUX

//Here the luxury values are defined
#if RANLUXCL_LUX == 0
#define RANLUXCL_NSKIP 0
#elif RANLUXCL_LUX == 1
#define RANLUXCL_NSKIP 24
#elif RANLUXCL_LUX == 2
#define RANLUXCL_NSKIP 76
#elif RANLUXCL_LUX == 3
#define RANLUXCL_NSKIP 200
#elif RANLUXCL_LUX == 4
#define RANLUXCL_NSKIP 380
#else
#define RANLUXCL_NSKIP (RANLUXCL_LUX - 24)
#endif //RANLUXCL_LUX == 0

//Check that nskip is a permissible value
#if RANLUXCL_NSKIP % 4 != 0 
#error nskip must be divisible by 4!
#endif
#if RANLUXCL_NSKIP < 24 && RANLUXCL_NSKIP != 0
#error nskip must be either 0 or >= 24!
#endif
#if RANLUXCL_NSKIP < 0
#error nskip is negative!
#endif

//Check if planar scheme is recovered
#if RANLUXCL_NSKIP % 24 == 0
#define RANLUXCL_PLANAR
#endif

//Check if we will skip at all
#if RANLUXCL_NSKIP == 0
#define RANLUXCL_NOSKIP
#endif

//Single-value global size and id
#define RANLUXCL_NUMWORKITEMS \
	(get_global_size(0) * get_global_size(1) * get_global_size(2))
#define RANLUXCL_MYID \
	(get_global_id(0) + get_global_id(1) * get_global_size(0) + \
	 get_global_id(2) * get_global_size(0) * get_global_size(1))

void ranluxcl_download_seed(ranluxcl_state_t *rst,
	global ranluxcl_state_t *ranluxcltab)
{
	(*rst) = ranluxcltab[RANLUXCL_MYID];
}

void ranluxcl_upload_seed(ranluxcl_state_t *rst,
	global ranluxcl_state_t *ranluxcltab)
{
	ranluxcltab[RANLUXCL_MYID] = (*rst);
}

/*
 * Performs one "step" (generates a single value or skip). Only used internally,
 * not intended to be called from user code.
 */
float ranluxcl_os(float sj24m1, float sj24, float *si24, float *carry)
{
	float uni, out;
	uni = sj24 - (*si24) - (*carry);
	if(uni < 0.0f){
		uni += 1.0f;
		(*carry) = RANLUXCL_TWOM24;
	} else (*carry) = 0.0f;
	out = ((*si24) = uni);

	if(uni < RANLUXCL_TWOM12){
		out += RANLUXCL_TWOM24 * sj24m1;
		if(out == 0.0f) out = RANLUXCL_TWOM24 * RANLUXCL_TWOM24;
	}
	return out;
}

/*
 * Return a float4 where each component is a uniformly distributed pseudo-
 * random value between 0 and 1, end points not included.
 */
float4 ranluxcl32(ranluxcl_state_t *rst)
{
	float4 out;

	if(rst->stepnr == 0){
		out.x = ranluxcl_os(rst->s09, rst->s10, &(rst->s24), &(rst->carry));
		out.y = ranluxcl_os(rst->s08, rst->s09, &(rst->s23), &(rst->carry));
		out.z = ranluxcl_os(rst->s07, rst->s08, &(rst->s22), &(rst->carry));
		out.w = ranluxcl_os(rst->s06, rst->s07, &(rst->s21), &(rst->carry));
		rst->stepnr += 4;
	}

	else if(rst->stepnr == 4){
		out.x = ranluxcl_os(rst->s05, rst->s06, &(rst->s20), &(rst->carry));
		out.y = ranluxcl_os(rst->s04, rst->s05, &(rst->s19), &(rst->carry));
		out.z = ranluxcl_os(rst->s03, rst->s04, &(rst->s18), &(rst->carry));
		out.w = ranluxcl_os(rst->s02, rst->s03, &(rst->s17), &(rst->carry));
		rst->stepnr += 4;
	}

	else if(rst->stepnr == 8){
		out.x = ranluxcl_os(rst->s01, rst->s02, &(rst->s16), &(rst->carry));
		out.y = ranluxcl_os(rst->s24, rst->s01, &(rst->s15), &(rst->carry));
		out.z = ranluxcl_os(rst->s23, rst->s24, &(rst->s14), &(rst->carry));
		out.w = ranluxcl_os(rst->s22, rst->s23, &(rst->s13), &(rst->carry));
		rst->stepnr += 4;
	}

	else if(rst->stepnr == 12){
		out.x = ranluxcl_os(rst->s21, rst->s22, &(rst->s12), &(rst->carry));
		out.y = ranluxcl_os(rst->s20, rst->s21, &(rst->s11), &(rst->carry));
		out.z = ranluxcl_os(rst->s19, rst->s20, &(rst->s10), &(rst->carry));
		out.w = ranluxcl_os(rst->s18, rst->s19, &(rst->s09), &(rst->carry));
		rst->stepnr += 4;
	}

	else if(rst->stepnr == 16){
		out.x = ranluxcl_os(rst->s17, rst->s18, &(rst->s08), &(rst->carry));
		out.y = ranluxcl_os(rst->s16, rst->s17, &(rst->s07), &(rst->carry));
		out.z = ranluxcl_os(rst->s15, rst->s16, &(rst->s06), &(rst->carry));
		out.w = ranluxcl_os(rst->s14, rst->s15, &(rst->s05), &(rst->carry));
		rst->stepnr += 4;
	}

	else if(rst->stepnr == 20){
		out.x = ranluxcl_os(rst->s13, rst->s14, &(rst->s04), &(rst->carry));
		out.y = ranluxcl_os(rst->s12, rst->s13, &(rst->s03), &(rst->carry));
		out.z = ranluxcl_os(rst->s11, rst->s12, &(rst->s02), &(rst->carry));
		out.w = ranluxcl_os(rst->s10, rst->s11, &(rst->s01), &(rst->carry));
		rst->stepnr = 0;

// The below preprocessor directives are here to recover the simpler planar
// scheme when nskip is a multiple of 24. For the most general planar shift
// approach, just ignore all #if's below.
#ifndef RANLUXCL_PLANAR
	}

	(*&(rst->in24)) += 4;
	if((*&(rst->in24)) == 24){
		(*&(rst->in24)) = 0;
#endif //RANLUXCL_PLANAR

		int initialskips = (rst->stepnr) ? (24 - rst->stepnr) : 0;
		int bulkskips = ((RANLUXCL_NSKIP - initialskips)/24) * 24;
		int remainingskips = RANLUXCL_NSKIP - initialskips - bulkskips;

//We know there won't be any initial skips in the planar scheme
#ifndef RANLUXCL_PLANAR
		//Do initial skips (lack of breaks in switch is intentional).
		switch(initialskips){
			case(20):
				ranluxcl_os(rst->s05, rst->s06, &(rst->s20), &(rst->carry));
				ranluxcl_os(rst->s04, rst->s05, &(rst->s19), &(rst->carry));
				ranluxcl_os(rst->s03, rst->s04, &(rst->s18), &(rst->carry));
				ranluxcl_os(rst->s02, rst->s03, &(rst->s17), &(rst->carry));
			case(16):
				ranluxcl_os(rst->s01, rst->s02, &(rst->s16), &(rst->carry));
				ranluxcl_os(rst->s24, rst->s01, &(rst->s15), &(rst->carry));
				ranluxcl_os(rst->s23, rst->s24, &(rst->s14), &(rst->carry));
				ranluxcl_os(rst->s22, rst->s23, &(rst->s13), &(rst->carry));
			case(12):
				ranluxcl_os(rst->s21, rst->s22, &(rst->s12), &(rst->carry));
				ranluxcl_os(rst->s20, rst->s21, &(rst->s11), &(rst->carry));
				ranluxcl_os(rst->s19, rst->s20, &(rst->s10), &(rst->carry));
				ranluxcl_os(rst->s18, rst->s19, &(rst->s09), &(rst->carry));
			case(8):
				ranluxcl_os(rst->s17, rst->s18, &(rst->s08), &(rst->carry));
				ranluxcl_os(rst->s16, rst->s17, &(rst->s07), &(rst->carry));
				ranluxcl_os(rst->s15, rst->s16, &(rst->s06), &(rst->carry));
				ranluxcl_os(rst->s14, rst->s15, &(rst->s05), &(rst->carry));
			case(4):
				ranluxcl_os(rst->s13, rst->s14, &(rst->s04), &(rst->carry));
				ranluxcl_os(rst->s12, rst->s13, &(rst->s03), &(rst->carry));
				ranluxcl_os(rst->s11, rst->s12, &(rst->s02), &(rst->carry));
				ranluxcl_os(rst->s10, rst->s11, &(rst->s01), &(rst->carry));
		}
#endif //RANLUXCL_PLANAR

//Also check if we will ever need to skip at all
#ifndef RANLUXCL_NOSKIP
		for(int i=0; i<bulkskips/24; i++){
			ranluxcl_os(rst->s09, rst->s10, &(rst->s24), &(rst->carry));
			ranluxcl_os(rst->s08, rst->s09, &(rst->s23), &(rst->carry));
			ranluxcl_os(rst->s07, rst->s08, &(rst->s22), &(rst->carry));
			ranluxcl_os(rst->s06, rst->s07, &(rst->s21), &(rst->carry));
			ranluxcl_os(rst->s05, rst->s06, &(rst->s20), &(rst->carry));
			ranluxcl_os(rst->s04, rst->s05, &(rst->s19), &(rst->carry));
			ranluxcl_os(rst->s03, rst->s04, &(rst->s18), &(rst->carry));
			ranluxcl_os(rst->s02, rst->s03, &(rst->s17), &(rst->carry));
			ranluxcl_os(rst->s01, rst->s02, &(rst->s16), &(rst->carry));
			ranluxcl_os(rst->s24, rst->s01, &(rst->s15), &(rst->carry));
			ranluxcl_os(rst->s23, rst->s24, &(rst->s14), &(rst->carry));
			ranluxcl_os(rst->s22, rst->s23, &(rst->s13), &(rst->carry));
			ranluxcl_os(rst->s21, rst->s22, &(rst->s12), &(rst->carry));
			ranluxcl_os(rst->s20, rst->s21, &(rst->s11), &(rst->carry));
			ranluxcl_os(rst->s19, rst->s20, &(rst->s10), &(rst->carry));
			ranluxcl_os(rst->s18, rst->s19, &(rst->s09), &(rst->carry));
			ranluxcl_os(rst->s17, rst->s18, &(rst->s08), &(rst->carry));
			ranluxcl_os(rst->s16, rst->s17, &(rst->s07), &(rst->carry));
			ranluxcl_os(rst->s15, rst->s16, &(rst->s06), &(rst->carry));
			ranluxcl_os(rst->s14, rst->s15, &(rst->s05), &(rst->carry));
			ranluxcl_os(rst->s13, rst->s14, &(rst->s04), &(rst->carry));
			ranluxcl_os(rst->s12, rst->s13, &(rst->s03), &(rst->carry));
			ranluxcl_os(rst->s11, rst->s12, &(rst->s02), &(rst->carry));
			ranluxcl_os(rst->s10, rst->s11, &(rst->s01), &(rst->carry));
		}
#endif //RANLUXCL_NOSKIP

//There also won't be any remaining skips in the planar scheme
#ifndef RANLUXCL_PLANAR
		//Do remaining skips
		if(remainingskips){
			ranluxcl_os(rst->s09, rst->s10, &(rst->s24), &(rst->carry));
			ranluxcl_os(rst->s08, rst->s09, &(rst->s23), &(rst->carry));
			ranluxcl_os(rst->s07, rst->s08, &(rst->s22), &(rst->carry));
			ranluxcl_os(rst->s06, rst->s07, &(rst->s21), &(rst->carry));

			if(remainingskips > 4){
				ranluxcl_os(rst->s05, rst->s06, &(rst->s20), &(rst->carry));
				ranluxcl_os(rst->s04, rst->s05, &(rst->s19), &(rst->carry));
				ranluxcl_os(rst->s03, rst->s04, &(rst->s18), &(rst->carry));
				ranluxcl_os(rst->s02, rst->s03, &(rst->s17), &(rst->carry));
			}

			if(remainingskips > 8){
				ranluxcl_os(rst->s01, rst->s02, &(rst->s16), &(rst->carry));
				ranluxcl_os(rst->s24, rst->s01, &(rst->s15), &(rst->carry));
				ranluxcl_os(rst->s23, rst->s24, &(rst->s14), &(rst->carry));
				ranluxcl_os(rst->s22, rst->s23, &(rst->s13), &(rst->carry));
			}

			if(remainingskips > 12){
				ranluxcl_os(rst->s21, rst->s22, &(rst->s12), &(rst->carry));
				ranluxcl_os(rst->s20, rst->s21, &(rst->s11), &(rst->carry));
				ranluxcl_os(rst->s19, rst->s20, &(rst->s10), &(rst->carry));
				ranluxcl_os(rst->s18, rst->s19, &(rst->s09), &(rst->carry));
			}

			if(remainingskips > 16){
				ranluxcl_os(rst->s17, rst->s18, &(rst->s08), &(rst->carry));
				ranluxcl_os(rst->s16, rst->s17, &(rst->s07), &(rst->carry));
				ranluxcl_os(rst->s15, rst->s16, &(rst->s06), &(rst->carry));
				ranluxcl_os(rst->s14, rst->s15, &(rst->s05), &(rst->carry));
			}
		}
#endif //RANLUXCL_PLANAR

		// Initial skips brought stepnr down to 0. The bulk skips did only
		// full cycles. Therefore stepnr is now equal to remainingskips.
		rst->stepnr = remainingskips;
	}

	return out;
}

/*
 * Perform the necessary operations to set the generator to the "beginning",
 * i.e., ready to generate 24 numbers before the next skipping sequence. This
 * is useful if different work-items have called ranluxcl a different number
 * of times. Since that would lead to out of sync execution on different work-
 * items it could be rather inefficient on SIMD architectures (like current
 * GPUs). This function thus allows us to resynchronize execution across work-
 * items.
 */
void ranluxcl_synchronize(ranluxcl_state_t *rst)
{
	// Do necessary number of calls to ranluxcl so that stepnr == 0 at the end.
	if(rst->stepnr == 4)
		ranluxcl32(rst);
	if(rst->stepnr == 8)
		ranluxcl32(rst);
	if(rst->stepnr == 12)
		ranluxcl32(rst);
	if(rst->stepnr == 16)
		ranluxcl32(rst);
	if(rst->stepnr == 20)
		ranluxcl32(rst);
}

/*
 * Uses a 64-bit xorshift PRNG by George Marsaglia to initialize the generator.
 *
 * This function can be used instead of ranluxcl_initialization if manual
 * control of the seed of each generator is desired. x must be unique for each
 * time this function is called, and *ranluxcltab should point to the specific
 * entry in the table to be initialized. Compare this to ranluxcl_initialization
 * where ins needs only be unique for each NDRange, and *ranluxcltab points
 * to the base address of the table for the entire NDRange. Also note that
 * depending on what you are doing the ranluxcl_upload_seed and
 * ranluxcl_download_seed functions may not do what you want, so make sure
 * you know what you are doing!
 */

void ranluxcl_init(ulong x, global ranluxcl_state_t *ranluxcltab)
{
	ranluxcl_state_t rst;

	#define RANLUXCL_POW2_24 16777216
	#define RANLUXCL_56 0x00FFFFFFFFFFFFFF
	#define RANLUXCL_48 0x0000FFFFFFFFFFFF
	#define RANLUXCL_40 0x000000FFFFFFFFFF
	#define RANLUXCL_32 0x00000000FFFFFFFF
	#define RANLUXCL_24 0x0000000000FFFFFF
	#define RANLUXCL_16 0x000000000000FFFF
	#define RANLUXCL_8  0x00000000000000FF

	ulong x1, x2, x3;

	//Logical shifts used so that all 64 bits of output are used (24 bits
	//per float), to be certain that all initial states are different.
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x1=x;
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x2=x;
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x3=x;
	rst.s01 = (float)  (x1 >> 40)
		/ (float)RANLUXCL_POW2_24;
	rst.s02 = (float) ((x1 & RANLUXCL_40) >> 16)
		/ (float)RANLUXCL_POW2_24;
	rst.s03 = (float)(((x1 & RANLUXCL_16) << 8) + (x2 >> 56))
		/ (float)RANLUXCL_POW2_24;
	rst.s04 = (float) ((x2 & RANLUXCL_56) >> 32)
		/ (float)RANLUXCL_POW2_24;
	rst.s05 = (float) ((x2 & RANLUXCL_32) >> 8)
		/ (float)RANLUXCL_POW2_24;
	rst.s06 = (float)(((x2 & RANLUXCL_8 ) << 16) + (x3 >> 48))
		/ (float)RANLUXCL_POW2_24;
	rst.s07 = (float) ((x3 & RANLUXCL_48) >> 24)
		/ (float)RANLUXCL_POW2_24;
	rst.s08 = (float)  (x3 & RANLUXCL_24)
		/ (float)RANLUXCL_POW2_24;

	x^=(x<<13);x^=(x>>7);x^=(x<<17);x1=x;
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x2=x;
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x3=x;
	rst.s09 = (float)  (x1 >> 40)
		/ (float)RANLUXCL_POW2_24;
	rst.s10 = (float) ((x1 & RANLUXCL_40) >> 16)
		/ (float)RANLUXCL_POW2_24;
	rst.s11 = (float)(((x1 & RANLUXCL_16) << 8) + (x2 >> 56))
		/ (float)RANLUXCL_POW2_24;
	rst.s12 = (float) ((x2 & RANLUXCL_56) >> 32)
		/ (float)RANLUXCL_POW2_24;
	rst.s13 = (float) ((x2 & RANLUXCL_32) >> 8)
		/ (float)RANLUXCL_POW2_24;
	rst.s14 = (float)(((x2 & RANLUXCL_8 ) << 16) + (x3 >> 48))
		/ (float)RANLUXCL_POW2_24;
	rst.s15 = (float) ((x3 & RANLUXCL_48) >> 24)
		/ (float)RANLUXCL_POW2_24;
	rst.s16 = (float)  (x3 & RANLUXCL_24)
		/ (float)RANLUXCL_POW2_24;

	x^=(x<<13);x^=(x>>7);x^=(x<<17);x1=x;
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x2=x;
	x^=(x<<13);x^=(x>>7);x^=(x<<17);x3=x;
	rst.s17 = (float)  (x1 >> 40)
		/ (float)RANLUXCL_POW2_24;
	rst.s18 = (float) ((x1 & RANLUXCL_40) >> 16)
		/ (float)RANLUXCL_POW2_24;
	rst.s19 = (float)(((x1 & RANLUXCL_16) << 8) + (x2 >> 56))
		/ (float)RANLUXCL_POW2_24;
	rst.s20 = (float) ((x2 & RANLUXCL_56) >> 32)
		/ (float)RANLUXCL_POW2_24;
	rst.s21 = (float) ((x2 & RANLUXCL_32) >> 8)
		/ (float)RANLUXCL_POW2_24;
	rst.s22 = (float)(((x2 & RANLUXCL_8 ) << 16) + (x3 >> 48))
		/ (float)RANLUXCL_POW2_24;
	rst.s23 = (float) ((x3 & RANLUXCL_48) >> 24)
		/ (float)RANLUXCL_POW2_24;
	rst.s24 = (float)  (x3 & RANLUXCL_24)
		/ (float)RANLUXCL_POW2_24;

	#undef RANLUXCL_POW2_24
	#undef RANLUXCL_56
	#undef RANLUXCL_48
	#undef RANLUXCL_40
	#undef RANLUXCL_32
	#undef RANLUXCL_24
	#undef RANLUXCL_16
	#undef RANLUXCL_8

	rst.in24 = 0;
	rst.stepnr = 0;
	rst.carry = 0.0f;
	if(rst.s24 == 0.0f)
		rst.carry = RANLUXCL_TWOM24;

	#ifndef RANLUXCL_NO_WARMUP
	//Warming up the generator, ensuring there are no initial correlations.
	//16 is a "magic number". It is the number of times we must generate
	//a batch of 24 numbers to ensure complete decorrelation, however it
	//seems like it is necessary to double this for the special case when
	//the generator is initialized to all zeros.
	for(int i=0; i<16 * 2; i++){
		ranluxcl_os(rst.s09, rst.s10, &(rst.s24), &(rst.carry));
		ranluxcl_os(rst.s08, rst.s09, &(rst.s23), &(rst.carry));
		ranluxcl_os(rst.s07, rst.s08, &(rst.s22), &(rst.carry));
		ranluxcl_os(rst.s06, rst.s07, &(rst.s21), &(rst.carry));
		ranluxcl_os(rst.s05, rst.s06, &(rst.s20), &(rst.carry));
		ranluxcl_os(rst.s04, rst.s05, &(rst.s19), &(rst.carry));
		ranluxcl_os(rst.s03, rst.s04, &(rst.s18), &(rst.carry));
		ranluxcl_os(rst.s02, rst.s03, &(rst.s17), &(rst.carry));
		ranluxcl_os(rst.s01, rst.s02, &(rst.s16), &(rst.carry));
		ranluxcl_os(rst.s24, rst.s01, &(rst.s15), &(rst.carry));
		ranluxcl_os(rst.s23, rst.s24, &(rst.s14), &(rst.carry));
		ranluxcl_os(rst.s22, rst.s23, &(rst.s13), &(rst.carry));
		ranluxcl_os(rst.s21, rst.s22, &(rst.s12), &(rst.carry));
		ranluxcl_os(rst.s20, rst.s21, &(rst.s11), &(rst.carry));
		ranluxcl_os(rst.s19, rst.s20, &(rst.s10), &(rst.carry));
		ranluxcl_os(rst.s18, rst.s19, &(rst.s09), &(rst.carry));
		ranluxcl_os(rst.s17, rst.s18, &(rst.s08), &(rst.carry));
		ranluxcl_os(rst.s16, rst.s17, &(rst.s07), &(rst.carry));
		ranluxcl_os(rst.s15, rst.s16, &(rst.s06), &(rst.carry));
		ranluxcl_os(rst.s14, rst.s15, &(rst.s05), &(rst.carry));
		ranluxcl_os(rst.s13, rst.s14, &(rst.s04), &(rst.carry));
		ranluxcl_os(rst.s12, rst.s13, &(rst.s03), &(rst.carry));
		ranluxcl_os(rst.s11, rst.s12, &(rst.s02), &(rst.carry));
		ranluxcl_os(rst.s10, rst.s11, &(rst.s01), &(rst.carry));
	}
	#endif //RANLUXCL_NO_WARMUP

	//Upload the state
	*ranluxcltab = rst;
}

void ranluxcl_init_legacy(uint ins, global ranluxcl_state_t *ranluxcltab)
{
	//Using legacy initialization from original Fortan 77 implementation

	//ins is scaled so that if the user makes another call somewhere else
	//with ins + 1 there should be no overlap. Also adding one
	//allows us to use ins = 0.
	int k, maxWorkitems;
	ranluxcl_state_t rst;

	#ifdef RANLUXCL_MAXWORKITEMS
	maxWorkitems = RANLUXCL_MAXWORKITEMS;
	#else
	maxWorkitems = RANLUXCL_NUMWORKITEMS;
	#endif //RANLUXCL_MAXWORKITEMS

	int scaledins = ins * maxWorkitems + 1;

	int js = scaledins + RANLUXCL_MYID;

	//Make sure js is not too small (should really be an error)
	if(js < 1)
		js = 1;

	#define IC 2147483563
	#define ITWO24 16777216

	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s01=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s02=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s03=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s04=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s05=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s06=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s07=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s08=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s09=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s10=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s11=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s12=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s13=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s14=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s15=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s16=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s17=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s18=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s19=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s20=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s21=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s22=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s23=(js%ITWO24)*RANLUXCL_TWOM24;
	k = js/53668; js=40014*(js-k*53668)-k*12211; if(js<0)js=js+IC;
		rst.s24=(js%ITWO24)*RANLUXCL_TWOM24;

	#undef IC
	#undef ITWO24

	rst.in24 = 0;
	rst.stepnr = 0;
	rst.carry = 0.0f;
	if(rst.s24 == 0.0f)
		rst.carry = RANLUXCL_TWOM24;

	#ifndef RANLUXCL_NO_WARMUP
	//Warming up the generator, ensuring there are no initial correlations.
	//16 is a "magic number". It is the number of times we must generate
	//a batch of 24 numbers to ensure complete decorrelation.
	for(int i=0; i<16; i++){
		ranluxcl_os(rst.s09, rst.s10, &(rst.s24), &(rst.carry));
		ranluxcl_os(rst.s08, rst.s09, &(rst.s23), &(rst.carry));
		ranluxcl_os(rst.s07, rst.s08, &(rst.s22), &(rst.carry));
		ranluxcl_os(rst.s06, rst.s07, &(rst.s21), &(rst.carry));
		ranluxcl_os(rst.s05, rst.s06, &(rst.s20), &(rst.carry));
		ranluxcl_os(rst.s04, rst.s05, &(rst.s19), &(rst.carry));
		ranluxcl_os(rst.s03, rst.s04, &(rst.s18), &(rst.carry));
		ranluxcl_os(rst.s02, rst.s03, &(rst.s17), &(rst.carry));
		ranluxcl_os(rst.s01, rst.s02, &(rst.s16), &(rst.carry));
		ranluxcl_os(rst.s24, rst.s01, &(rst.s15), &(rst.carry));
		ranluxcl_os(rst.s23, rst.s24, &(rst.s14), &(rst.carry));
		ranluxcl_os(rst.s22, rst.s23, &(rst.s13), &(rst.carry));
		ranluxcl_os(rst.s21, rst.s22, &(rst.s12), &(rst.carry));
		ranluxcl_os(rst.s20, rst.s21, &(rst.s11), &(rst.carry));
		ranluxcl_os(rst.s19, rst.s20, &(rst.s10), &(rst.carry));
		ranluxcl_os(rst.s18, rst.s19, &(rst.s09), &(rst.carry));
		ranluxcl_os(rst.s17, rst.s18, &(rst.s08), &(rst.carry));
		ranluxcl_os(rst.s16, rst.s17, &(rst.s07), &(rst.carry));
		ranluxcl_os(rst.s15, rst.s16, &(rst.s06), &(rst.carry));
		ranluxcl_os(rst.s14, rst.s15, &(rst.s05), &(rst.carry));
		ranluxcl_os(rst.s13, rst.s14, &(rst.s04), &(rst.carry));
		ranluxcl_os(rst.s12, rst.s13, &(rst.s03), &(rst.carry));
		ranluxcl_os(rst.s11, rst.s12, &(rst.s02), &(rst.carry));
		ranluxcl_os(rst.s10, rst.s11, &(rst.s01), &(rst.carry));
	}
	#endif //RANLUXCL_NO_WARMUP

	//Upload the state
	ranluxcl_upload_seed(&rst, ranluxcltab);
}

void ranluxcl_initialization(uint ins, global ranluxcl_state_t *ranluxcltab)
{
	#ifdef RANLUXCL_USE_LEGACY_INITIALIZATION
	ranluxcl_init_legacy(ins, ranluxcltab);

	#else // Not RANLUXCL_USE_LEGACY_INITIALIZATION

	// We scale ins by 2^32. As long as we never use more than (2^32)-1
	// work-items per NDRange the initial states should never be the same.

	ulong x = (ulong)RANLUXCL_MYID + (ulong)ins * ((ulong)UINT_MAX + 1);
	ranluxcl_init(x, ranluxcltab + RANLUXCL_MYID);

	#endif // RANLUXCL_USE_LEGACY_INITIALIZATION
}

float4 ranluxcl32norm(ranluxcl_state_t *rst)
{
	//Returns a vector where each component is a normally
	//distributed PRN centered on 0, with standard deviation 1.

	//Roll our own since M_PI_F does not exist in OpenCL 1.0.
	#define RANLUXCL_PI_F 3.1415926535f

	float4 U = ranluxcl32(rst);

	float4 Z;
	float R, phi;

	R = sqrt(-2 * log(U.x));
	phi = 2 * RANLUXCL_PI_F * U.y;
	Z.x = R * cos(phi);
	Z.y = R * sin(phi);

	R = sqrt(-2 * log(U.z));
	phi = 2 * RANLUXCL_PI_F * U.w;
	Z.z = R * cos(phi);
	Z.w = R * sin(phi);

	return Z;

	#undef RANLUXCL_PI_F
}

#ifdef RANLUXCL_SUPPORT_DOUBLE
double4 ranluxcl64(ranluxcl_state_t *rst)
{
	double4 out;
	float4 randvec;

	//We know this value is caused by the never-zero part
	//of the original algorithm, but we want to allow zero for
	//the most significant bits in the double precision result.
	randvec = ranluxcl32(rst);
	if(randvec.x == RANLUXCL_TWOM24 * RANLUXCL_TWOM24)
		randvec.x = 0.0f;
	if(randvec.z == RANLUXCL_TWOM24 * RANLUXCL_TWOM24)
		randvec.z = 0.0f;

	out.x = (double)(randvec.x) + (double)(randvec.y) / 16777216;
	out.y = (double)(randvec.z) + (double)(randvec.w) / 16777216;

	randvec = ranluxcl32(rst);
	if(randvec.x == RANLUXCL_TWOM24 * RANLUXCL_TWOM24)
		randvec.x = 0.0f;
	if(randvec.z == RANLUXCL_TWOM24 * RANLUXCL_TWOM24)
		randvec.z = 0.0f;

	out.z = (double)(randvec.x) + (double)(randvec.y) / 16777216;
	out.w = (double)(randvec.z) + (double)(randvec.w) / 16777216;

	return out;
}

double4 ranluxcl64norm(ranluxcl_state_t *rst)
{
	//Returns a vector where each component is a normally
	//distributed PRN centered on 0, with standard deviation
	//1.

	double4 U = ranluxcl64(rst);

	double4 Z;
	double R, phi;

	R = sqrt(-2 * log(U.x));
	phi = 2 * M_PI * U.y;
	Z.x = R * cos(phi);
	Z.y = R * sin(phi);

	R = sqrt(-2 * log(U.z));
	phi = 2 * M_PI * U.w;
	Z.z = R * cos(phi);
	Z.w = R * sin(phi);

	return Z;
}
#endif //RANLUXCL_SUPPORT_DOUBLE

#undef RANLUXCL_TWOM24
#undef RANLUXCL_TWOM12
#undef RANLUXCL_NUMWORKITEMS
#undef RANLUXCL_MYID
#undef RANLUXCL_PLANAR
#undef RANLUXCL_NOSKIP

#endif //RANLUXCL_CL
