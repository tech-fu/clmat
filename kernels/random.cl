/* Author: Nikola Karamanov */

#ifdef DOUBLE
#define RANLUXCL_SUPPORT_DOUBLE
#endif

#include "ranluxcl.cl"

#ifdef NORMAL

#ifdef DOUBLE
#define RANDOM(ss) ranluxcl64norm(ss)
#else
#define RANDOM(ss) ranluxcl32norm(ss)
#endif

#else

#ifdef DOUBLE
#define RANDOM(ss) ranluxcl64(ss)
#else
#define RANDOM(ss) ranluxcl32(ss)
#endif

#endif

/**\brief Initialize ranlux rng states.
 */
__kernel void random_states_init( const unsigned int seed , __global ranluxcl_state_t *states){
    ranluxcl_initialization(seed , states);
}

/**\brief Generate random floating point numbers.
 * \param vw1_size The size of the output buffer.
 * \param outvw1 The output buffer cast as having vector width 1. (Only used when not EXACT)
 * \param vw1_mod_4 The remainder when the vw1_size is divided by 4.
 * \param vw4_size The size of the output buffer divided by 4 (integer division).
 * \param outvw4 The output buffer cast as having vector width 4.
 * \param states Must have length get_global_size(0) (with 28*4 bytes per state).
 */
__kernel void random( const SIZE_T vw1_size, __global DTYPE_OUT*  outvw1, const SIZE_T vw1_mod_4,
                      const SIZE_T vw4_size, __global DTYPE_OUT4* outvw4,
                      __global ranluxcl_state_t* states ){
    
    ranluxcl_state_t state;
    ranluxcl_download_seed(&state, states);
    
    const SIZE_T num_states = get_global_size(0);
    
    SIZE_T i=get_global_id(0);
    for(  ; i<vw4_size ; i+=num_states ){
        outvw4[i] = RANDOM(&state);
    }
    
#ifndef EXACT
    if( (vw1_mod_4 != 0) && (get_global_id(0)==num_states-1)){
        const DTYPE4 extra = RANDOM(&state);
        
        i = 4*num_elems4;
        outvw1[i++] = extra.x; // This one we know exists.
        if(vw1_mod_4>1) outvw1[i++] = extra.y;
        if(vw1_mod_4>2) outvw1[i  ] = extra.z;
        // There can't be a fourth one.
    }
#endif
    
	ranluxcl_upload_seed(&state, states);
}
