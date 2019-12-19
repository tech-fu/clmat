/* Author: Nikola Karamanov */

// TODO prod is different for floats (use log/exp trick) and non-floats (simply multiply or use log2)
// cumsum, cumprod should probably be elsewhere
// reduction_enum = {'all': 0, 'any': 1, 'max': 2, 'min': 3, 'ptp': 4, 'sum': 5, 'prod': 6,
//                   'argmax': 12, 'argmin': 13, 'mean': 17, 'var': 18, 'std': 19}
__kernel void reduce_1d(const SIZE_T out_size, __global DTYPE_OUT* out, const SIZE_T out_offset,
                        const SIZE_T block_size,
                        const SIZE_T m1_size, const __global DTYPE_M1* m1,
                        const SIZE_T m1_offset
                        //,const DTYPE_OUT* extra_array, const DTYPE_OUT extra_var, const SIZE_T extra_index,
                        ){
    
    SIZE_T i = get_global_id(0)*block_size;
    SIZE_T end = i + block_size;
    if(end > m1_size) end = m1_size;
    i += m1_offset;
    end += m1_offset;
    
#if REDUCTION == 0
    bool result = 1;
#elif REDUCTION == 1
    bool result = 0;
#elif REDUCTION == 2 || REDUCTION == 3
    DTYPE_OUT result = m1[i++];
#elif REDUCTION == 4
    // Init both values to the same.
    DTYPE_OUT resultM = m1[i];
    DTYPE_OUT resultm = m1[i++];
#elif REDUCTION == 12 || REDUCTION == 13
    DTYPE_OUT result = i;
    DTYPE_M1 extremum = m1[i++];
#elif REDUCTION == 17 || REDUCTION == 18 || REDUCTION == 19
    DTYPE_OUT mean = 0;
    SIZE_T n = 0;
#if REDUCTION == 18 || REDUCTION == 19
    DTYPE_OUT sum_of_squares = 0;
#endif
#else
    DTYPE_OUT result = 0;
#endif

    for(  ; i < end ; i++ ){
        
#if REDUCTION == 0
        if( !m1[i] ){
            result = 0;
            break;
        }
#elif REDUCTION == 1
        if( m1[i] ){
            result = 1;
            break;
        }
#elif REDUCTION == 2
        result = fmax(result, m1[i]);
#elif REDUCTION == 3
        result = fmin(result, m1[i]);
#elif REDUCTION == 4
        const DTYPE_OUT val = m1[i];
        if(val > resultM){
            resultM = val;
        }else if(val < resultm){
            resultm = val;
        }
#elif REDUCTION == 5
        result += m1[i];
#elif REDUCTION == 6
        result *= m1[i];

#elif REDUCTION == 12 || REDUCTION == 13
        const DTYPE_M1 val = m1[i];
#if REDUCTION == 12
        if( val > extremum ){
#elif REDUCTION == 13
        if( val < extremum ){
#endif
            result = i;
            extremum = val;
        }

#elif REDUCTION == 17
        mean += (m1[i] - mean)/(++n);
#elif REDUCTION == 18 || REDUCTION == 19
        const DTYPE_M1 val = m1[i];
        const DTYPE_OUT old_mean = mean;
        mean += (val - mean)/(++n);
        sum_of_squares += (val-old_mean)*(val-mean);
#else
        REDUCTION(result, CONVERT_INPUT(m1[i]));
#endif

    }

out[get_global_id(0)+out_offset] =
#if REDUCTION == 4
    resultM - resultm;
#elif REDUCTION == 17
    mean;
#elif REDUCTION == 18
    sum_of_squares/n;
#elif REDUCTION == 19
    sqrt(sum_of_squares/n);
#else
    result;
#endif

}
