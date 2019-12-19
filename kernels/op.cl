/* Author: Nikola Karamanov */

#define OUTAT(r,c) (out[out_offset + r*out_ptr_stride_r + c*out_ptr_stride_c])

#ifdef SCALAR_M1
#define M1AT(r,c) (m1)
#else
#define M1AT(r,c) (m1[m1_offset + (r)*m1_ptr_stride_r + (c)*m1_ptr_stride_c])
#endif

#ifdef SCALAR_M2
#define M2AT(r,c) (m2)
#else
#define M2AT(r,c) (m2[m2_offset + (r)*m2_ptr_stride_r + (c)*m2_ptr_stride_c])
#endif
__kernel void op( const SIZE_T nrows , const SIZE_T ncols , __global DTYPE_OUT* out ,
        const SIZE_T out_ptr_stride_r, const SIZE_T out_ptr_stride_c,const SIZE_T out_offset,
#ifdef SCALAR_M1
        const DTYPE_M1 m1,
#else
        const __global DTYPE_M1* m1,
#endif
        const SIZE_T m1_ptr_stride_r, const SIZE_T m1_ptr_stride_c,const SIZE_T m1_offset,
#ifdef SCALAR_M2
        const DTYPE_M2 m2,
#else
        const __global DTYPE_M2* m2,
#endif
        const SIZE_T m2_ptr_stride_r, const SIZE_T m2_ptr_stride_c,const SIZE_T m2_offset
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

#ifdef LOGICAL
        OUTAT(r,c) = (M1AT(r,c) OPERATOR M2AT(r,c));
#else
        OUTAT(r,c) = ( (DTYPE_IN) M1AT(r,c) ) OPERATOR ( (DTYPE_IN) M2AT(r,c) );
#endif

#ifndef EXACT
    }
#endif
}
