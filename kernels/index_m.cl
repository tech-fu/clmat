/* Author: Nikola Karamanov */

// Matrix index kernel (logical or integer).
// global size is size of matrix.
// TODO this kernel is not ready

__kernel void li(__global DTYPE_OUT* out,
                 const SIZE_T out_ptr_stride_r, const SIZE_T out_ptr_stride_c, const SIZE_T out_offset,
                 __global DTYPE_M1* m1,
                 const SIZE_T m1_rows, const SIZE_T m1_cols,
                 const SIZE_T m1_ptr_stride_r, const SIZE_T m1_ptr_stride_c, const SIZE_T m1_offset,
                 const __global DTYPE_M2* m2,
                 const SIZE_T m2_ptr_stride_r, const SIZE_T m2_ptr_stride_c, const SIZE_T m2_offset,
#ifdef SCALAR_M3
                 const DTYPE_M3 m3,
#else
                 const __global DTYPE_M3* m3,
#endif
                 const SIZE_T m3_ptr_stride_r, const SIZE_T m3_ptr_stride_c, const SIZE_T m3_offset
                 ){
#ifdef M2_ROW
    SIZE_T r=0, count_c=0;
    for(SIZE_T c=0 ; c<m1_cols ; c++){{
#else
    #ifdef M2_COL
    SIZE_T c=0, count_c=0;
    for(SIZE_T r=0 ; r<m1_rows ; r++){{
    #else
    SIZE_T count=0;
    for(SIZE_T c=0 ; c<m1_cols ; c++){
        for(SIZE_T r=0 ; r<m1_rows ; r++){
    #endif
#endif
            if(M2AT(r,c) != 0){
#ifdef LI_GET
    #ifdef M2_ROW
                for(SIZE_T rr=0 ; rr<m1_rows ; rr++){
                    OUTAT(rr,count_c) = M1AT(rr,c);
                }
                count_c++;
    #else
        #ifdef M2_COL
                for(SIZE_T cc=0 ; cc<m1_cols ; cc++){
                    OUTAT(count_r,cc) = M1AT(r,cc);
                }
                count_r++;
        #else
        printf("%d %d %d %f\n", r, c, count, M1AT(r,c));
                OUTAT(count, 0) = M1AT(r,c);
                count++;
        #endif
    #endif
#else
    #ifdef LI_OUT // Set into out (if true take m3, else take m1, since this is what the m1 would be without LI_OUT)
                OUTAT(r,c) = M3AT(r,c);
            }else{
                OUTAT(r,c) = M1AT(r,c);
    #else // Set into m1
                M1AT(r,c) = M3AT(r,c);
    #endif
#endif
            }
        }
    }
}
