/* Author: Nikola Karamanov */

#ifdef VECTOR_WIDTH
#define CAST_OUTPUT(t) (convert_DTYPE_OUT_VW(t))
#define CAST_INPUT(t) (convert_DTYPE_IN_VW(t))
#else
#define CAST_OUTPUT(t) ((DTYPE_OUT) (t))
#define CAST_INPUT(t) ((DTYPE_IN) (t))
#endif

#define abs(x) (x>=0?x:-x)
#ifdef OFFSET_M1
#define M1AT(i) (m1[m1_offset+(i)])
#else
#define M1AT(i) (m1[(i)])
#endif
#ifdef OFFSET_OUT
#define OUTAT(i) (out[out_offset+(i)])
#else
#define OUTAT(i) (out[(i)])
#endif
__kernel void map_1d( const SIZE_T size ,
                     __global DTYPE_OUT_VW* out,const SIZE_T out_offset
                     ,const __global DTYPE_M1_VW* m1,const SIZE_T m1_offset
                    ){
    const SIZE_T i=get_global_id(0);
#ifndef EXACT
    if( i < size ){
#endif
#ifdef VECTOR_WIDTH
        OUTAT(i) = CAST_OUTPUT( MAP_FUNCTION( CAST_INPUT((DTYPE_M1_VW) M1AT(i)) ) );
#else
        OUTAT(i) = CAST_OUTPUT( MAP_FUNCTION( CAST_INPUT(M1AT(i)) ) );
#endif
#ifndef EXACT
    }
#endif
}
