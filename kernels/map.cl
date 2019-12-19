/* Author: Nikola Karamanov */

#define abs(x) (x>=0?x:-x)
#define OUTAT(r,c) (out[out_offset + (r)*out_ptr_stride_r + (c)*out_ptr_stride_c])
#define M1AT(r,c) (m1[m1_offset + (r)*m1_ptr_stride_r + (c)*m1_ptr_stride_c])
__kernel void map( const SIZE_T nrows , const SIZE_T ncols
                   ,__global DTYPE_OUT* out
                   ,const SIZE_T out_ptr_stride_r, const SIZE_T out_ptr_stride_c ,const SIZE_T out_offset
                   ,const __global DTYPE_M1* m1
                   ,const SIZE_T m1_ptr_stride_r, const SIZE_T m1_ptr_stride_c ,const SIZE_T m1_offset
                  ){

#ifndef REVERSE_WS
    const SIZE_T r=get_global_id(0);
    const SIZE_T c=get_global_id(1);
#else
    const SIZE_T r=get_global_id(1);
    const SIZE_T c=get_global_id(0);
#endif

#ifndef EXACT
    if(r<nrows && c<ncols){
#endif
        OUTAT(r,c) = (DTYPE_OUT) MAP_FUNCTION((DTYPE_IN) M1AT(r,c));
#ifndef EXACT
    }
#endif
}
