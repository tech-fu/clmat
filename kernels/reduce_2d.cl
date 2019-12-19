/* Author: Nikola Karamanov */

// global size must be the same as the shape of the output.
#define OUTAT(r,c) (out[out_offset + (r)*out_ptr_stride_r + (c)*out_ptr_stride_c])
#define M1AT(r,c) (m1[m1_offset + (r)*m1_ptr_stride_r + (c)*m1_ptr_stride_c])
__kernel void reduce_2d(__global DTYPE_OUT* out,
                        const SIZE_T out_ptr_stride_r, const SIZE_T out_ptr_stride_c, const SIZE_T out_offset,
                        const SIZE_T nrows, const SIZE_T ncols,
                        const __global DTYPE_M1* m1,
                        const SIZE_T m1_ptr_stride_r, const SIZE_T m1_ptr_stride_c, const SIZE_T m1_offset
                        ){
    
    SIZE_T r = get_global_id(0);
    SIZE_T c = get_global_id(1);
    
#if REDUCTION == 0
    bool result = 1;
#elif REDUCTION == 1
    bool result = 0;
#elif REDUCTION == 2
    DTYPE_OUT result = M1AT(r,c);
#elif REDUCTION == 3
    DTYPE_OUT result = M1AT(r,c);
#elif REDUCTION == 4
    // Init both values to the same.
    DTYPE_OUT resultM = M1AT(r,c);
    DTYPE_OUT resultm = resultM;
#elif REDUCTION == 12 || REDUCTION == 13
    DTYPE_OUT result = 0;
    DTYPE_M1 extremum = M1AT(r,c);

#elif REDUCTION == 17 || REDUCTION == 18 || REDUCTION == 19
    DTYPE_OUT mean = 0;
    SIZE_T n = 0;
#if REDUCTION == 18 || REDUCTION == 19
    DTYPE_OUT sum_of_squares = 0;
#endif

#else
    DTYPE_OUT result = 0;
#endif


#if AXIS == 0
    for(  ; r < nrows ; r++ ){
#elif AXIS == 1
    for(  ; c < ncols ; c++ ){
#endif
        
#if REDUCTION == 0
        if( !M1AT(r,c) ){
            result = 0;
            break;
        }
#elif REDUCTION == 1
        if( M1AT(r,c) ){
            result = 1;
            break;
        }
#elif REDUCTION == 2
        result = fmax(result, M1AT(r,c));
#elif REDUCTION == 3
        result = fmin(result, M1AT(r,c));
#elif REDUCTION == 4
        const DTYPE_OUT val = M1AT(r,c);
        if(val > resultM){
            resultM = val;
        }else if(val < resultm){
            resultm = val;
        }
#elif REDUCTION == 5
        result += M1AT(r,c);
#elif REDUCTION == 6
        result *= M1AT(r,c);

#elif REDUCTION == 12 || REDUCTION == 13
        const DTYPE_M1 val = M1AT(r,c);
#if REDUCTION == 12
        if( val > extremum ){
#elif REDUCTION == 13
        if( val < extremum ){
#endif
#if AXIS == 0
            result = r;
#elif AXIS == 1
            result = c;
#endif
            extremum = val;
        }

#elif REDUCTION == 17
        mean += (M1AT(r,c) - mean)/(++n);
#elif REDUCTION == 18 || REDUCTION == 19
        const DTYPE_M1 val = M1AT(r,c);
        const DTYPE_OUT old_mean = mean;
        mean += (val - mean)/(++n);
        sum_of_squares += (val-old_mean)*(val-mean);
#else
        REDUCTION(result, CONVERT_INPUT(M1AT(r,c)));
#endif


    }

OUTAT( get_global_id(0) , get_global_id(1) ) =
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
