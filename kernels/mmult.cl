/* Author: Nikola Karamanov */

#ifndef VECTOR_WIDTH
#define VECTOR_WIDTH 1
#endif

#define M1AT(r,c) (m1[ (r) * m1_ptr_stride0  + (c) * m1_ptr_stride1  + m1_offset ])
#define M2AT(r,c) (m2[ (r) * m2_ptr_stride0  + (c) * m2_ptr_stride1  + m2_offset ])
#if VECTOR_WIDTH == 1
#define VM1AT(r,c) M1AT(r,c)
#define VM2AT(r,c) M2AT(r,c)
#elif VECTOR_WIDTH == 2
#define VM1AT(r,c) (DTYPE_M1_VW) (M1AT(r,2*(c)),M1AT(r,2*(c)+1))
#define VM2AT(r,c) (DTYPE_M2_VW) (M2AT(2*(r),c),M2AT(2*(r)+1,c))
#elif VECTOR_WIDTH == 4
#define VM1AT(r,c) (DTYPE_M1_VW) (M1AT(r,4*(c)),M1AT(r,4*(c)+1),M1AT(r,4*(c)+2),M1AT(r,4*(c)+3))
#define VM2AT(r,c) (DTYPE_M2_VW) (M2AT(4*(r),c),M2AT(4*(r)+1,c),M2AT(4*(r)+2,c),M2AT(4*(r)+3,c))
//#elif VECTOR_WIDTH == 8
//#define VM1AT(r,c) (DTYPE_M1_VW) (M1AT(r,8*(c)),M1AT(r,8*(c)+1),M1AT(r,8*(c)+2),M1AT(r,8*(c)+3),M1AT(r,8*(c)+4),M1AT(r,8*(c)+5),M1AT(r,8*(c)+6),M1AT(r,8*(c)+7))
//#define VM2AT(r,c) (DTYPE_M2_VW) (M2AT(8*(r),c),M2AT(8*(r)+1,c),M2AT(8*(r)+2,c),M2AT(8*(r)+3,c),M2AT(8*(r)+4,c),M2AT(8*(r)+5,c),M2AT(8*(r)+6,c),M2AT(8*(r)+7,c))
#endif

#define OUTAT(r,c) (out[(r) * out_ptr_stride0 + (c) * out_ptr_stride1 + out_offset])
#define result_c get_global_id(0)
#define result_r get_global_id(1)
#define lc get_local_id(0)
#define lr get_local_id(1)

#define B1AT(c) m1_block[block_ri + (c)]
#define B2AT(r) m2_block[(r)*BLOCK_SIZE+lc]

__kernel void mmult(const SIZE_T nrows, const SIZE_T common_dim, const SIZE_T ncols,
    __global DTYPE_OUT* out, const SIZE_T out_ptr_stride0, const SIZE_T out_ptr_stride1, const SIZE_T out_offset,
    const __global DTYPE_M1* m1, const SIZE_T m1_ptr_stride0, const SIZE_T m1_ptr_stride1, const SIZE_T m1_offset,
    const __global DTYPE_M2* m2, const SIZE_T m2_ptr_stride0, const SIZE_T m2_ptr_stride1, const SIZE_T m2_offset,
    __local DTYPE_M1_VW* m1_block, __local DTYPE_M2_VW* m2_block
    ){
    
    //const SIZE_T result_c = get_global_id(0);
    //const SIZE_T result_r = get_global_id(1);
    
    //const SIZE_T lc = get_local_id(0);
    //const SIZE_T lr = get_local_id(1);
    const SIZE_T block_ri = lr*BLOCK_SIZE;
    const SIZE_T block_i = block_ri + lc; // The index of the block element that this thread handles.
    
    DTYPE_OUT result = 0;
    
#ifndef EXACT
    const bool pass_r = result_r<nrows;
    const bool pass_c = result_c<ncols;
#endif

    for( SIZE_T block_common_rc=0 ; block_common_rc<common_dim ; block_common_rc+=BLOCK_SIZE ){

#ifndef EXACT
        if( pass_r && (block_common_rc+lc<common_dim) ){
#endif
            m1_block[block_i] = VM1AT(result_r , block_common_rc + lc);
#ifndef EXACT
        }else{
            m1_block[block_i] = 0.0;
        }
#endif

#ifndef EXACT
        if( pass_c && (block_common_rc+lr<common_dim) ){
#endif
            m2_block[block_i] = VM2AT(block_common_rc + lr , result_c);
#ifndef EXACT
        }else{
            m2_block[block_i] = 0.0;
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

#ifndef EXACT
        if( pass_r && pass_c ){
#endif

#if VECTOR_WIDTH == 1
UNROLL_BLOCK_SIZE  result += B1AT(BLOCK_SIZE_I) * B2AT(BLOCK_SIZE_I);
#elif VECTOR_WIDTH < 8
UNROLL_BLOCK_SIZE  result += dot( B1AT(BLOCK_SIZE_I) , B2AT(BLOCK_SIZE_I) );
//#elif VECTOR_WIDTH == 8
//UNROLL_BLOCK_SIZE  result += TODO;
#endif

#ifndef EXACT
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#ifndef EXACT
    if( result_r<nrows && result_c<ncols ){
#endif
                OUTAT(result_r,result_c) = result;
#ifndef EXACT
    }
#endif

}
