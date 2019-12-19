/* Author: Nikola Karamanov */

#define M1AT(r,c) (m1[m1_offset + (r)*m1_ptr_stride_r + (c)*m1_ptr_stride_c])
#define M2AT(r,c) (m2[m2_offset + (r)*m2_ptr_stride_r + (c)*m2_ptr_stride_c])

#ifndef REVERSE
#define GLR get_global_id(0)
#define GLC get_global_id(1)
#else
#define GLR get_global_id(1)
#define GLC get_global_id(0)
#endif

// out[0] will be the result, but out should have global_size.
__kernel void innerprod(__global DTYPE_OUT* out, const SIZE_T nrows, const SIZE_T ncols,
    const __global DTYPE_M1* m1, const SIZE_T m1_offset, const SIZE_T m1_ptr_stride_r, const SIZE_T m1_ptr_stride_c,
    const __global DTYPE_M2* m2, const SIZE_T m2_offset, const SIZE_T m2_ptr_stride_r, const SIZE_T m2_ptr_stride_c,
    const SIZE_T block_size_r, const SIZE_T block_size_c
                        ){
    
    SIZE_T r = GLR*block_size_r;
    SIZE_T r_end = r + block_size_r;
    if(r_end>nrows) r_end=nrows;
    
    SIZE_T c = GLC*block_size_c;
    SIZE_T c_end = c + block_size_c;
    if(c_end>ncols) c_end=ncols;
    
    DTYPE_OUT result = 0;
    SIZE_T i = 0;
#ifndef REVERSE
    for( ; c<c_end ; c++ ){
        for( r = GLR*block_size_r ; r<r_end ; r++ ){
#else
    for(  ; r<r_end ; r++ ){
        for( c = GLC*block_size_c ; c<c_end ; c++ ){
#endif
            result += M1AT(r,c) * M2AT(r,c);
        }
    }
    out[get_global_id(0)*get_global_size(1) + get_global_id(1)] = result;
}
