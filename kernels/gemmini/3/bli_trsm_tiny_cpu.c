/*

BLIS
An object-based framework for developing high-performance BLAS-like
libraries.

Copyright (C) 2018-2019, Advanced Micro Devices, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of The University of Texas at Austin nor the names
of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM

// XA = B; A is lower-traingular; No transpose; double precision; non-unit diagonal
static  err_t bli_dtrsm_small_XAlB(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{

    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B

    dim_t i, j, k;
    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;

    for(k = N;k--;)
    {
        double lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M;i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k;j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(k)+(j)*lda];
            }
        }
    }

    return BLIS_SUCCESS;
}



//XA = B; A is lower triabgular; No transpose; double precision; unit-diagonal
static  err_t bli_dtrsm_small_XAlB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{

    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;
    double A_k_j;
     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(k)+(j)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;
            }
        }
    }


    return BLIS_SUCCESS;
}



//XA = B; A is lower-triangular; A is transposed; double precision; non-unit-diagonal
static  err_t bli_dtrsm_small_XAltB(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{

    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

    for(k = 0; k < N; k++)
    {
        double lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
    return BLIS_SUCCESS;

}

//XA = B; A is lower-triangular; A is transposed; double precision; unit-diagonal
static  err_t bli_dtrsm_small_XAltB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
    return BLIS_SUCCESS;
}

// XA = B; A is upper triangular; No transpose; double presicion; non-unit diagonal
static err_t bli_dtrsm_small_XAuB
     (
       side_t  side,
       obj_t*  AlphaObj,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B

    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

     for(k = 0; k < N; k++)
     {
        double lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }

    }
    return BLIS_SUCCESS;

}

//XA = B; A is upper triangular; No transpose; double precision; unit-diagonal
static  err_t bli_dtrsm_small_XAuB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B

    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }
    }
    return BLIS_SUCCESS;

}

//XA = B; A is upper-triangular; A is transposed; double precision; non-unit diagonal
static  err_t bli_dtrsm_small_XAutB(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *=alpha;

    for(k = N; k--;)
    {
        double lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M; i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k; j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(j)+(k)*lda];
            }
        }
    }
    return BLIS_SUCCESS;

}

//XA = B; A is upper-triangular; A is transposed; double precision; unit diagonal
static  err_t bli_dtrsm_small_XAutB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;
    double A_k_j;

    for(j = 0; j< N; j++)
        for(i = 0; i< M; i++)
            B[i+j*ldb] *= alpha;

     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(j)+(k)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;

            }
        }
    }
    return BLIS_SUCCESS;

}

//AX = B; A is lower triangular; No transpose; double precision; non-unit diagonal
static err_t bli_dtrsm_small_AlXB(
                side_t side,
                obj_t* AlphaObj,
                obj_t* a,
                obj_t* b,
                cntx_t* cntx,
                cntl_t* cntl
                )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


  dim_t i, j, k;

  for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

  for (k = 0; k < M; k++)
  {
    double lkk_inv = 1.0/A[k+k*lda];
    for (j = 0; j < N; j++)
    {
        B[k + j*ldb] *= lkk_inv;
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
    }
  }// k -loop
 return BLIS_SUCCESS;

}


//AX = B; A is lower triangular; No transpose; double precision; unit diagonal
static err_t bli_dtrsm_small_AlXB_unitDiag(
                side_t side,
                obj_t* AlphaObj,
                obj_t* a,
                obj_t* b,
                cntx_t* cntx,
                cntl_t* cntl
                )
{
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    double alpha = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict A = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B


  dim_t i, j, k;

  for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

  for (k = 0; k < M; k++)
  {
      for (j = 0; j < N; j++)
      {
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
     }
  }
  return BLIS_SUCCESS;

}

///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

// XA = B; A is lower-traingular; No transpose; single precision; non-unit diagonal
static  err_t bli_strsm_small_XAlB(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm1\n");

    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B

    dim_t i, j, k;
    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;

    for(k = N;k--;)
    {
        float lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M;i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k;j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(k)+(j)*lda];
            }
        }
    }

    return BLIS_SUCCESS;
}



//XA = B; A is lower triabgular; No transpose; single precision; unit-diagonal
static  err_t bli_strsm_small_XAlB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm2\n");

    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;
    float A_k_j;
     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(k)+(j)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;
            }
        }
    }


    return BLIS_SUCCESS;
}



//XA = B; A is lower-triangular; A is transposed; single precision; non-unit-diagonal
static  err_t bli_strsm_small_XAltB(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm3\n");

    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

    for(k = 0; k < N; k++)
    {
        float lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
    return BLIS_SUCCESS;

}

//XA = B; A is lower-triangular; A is transposed; single precision; unit-diagonal
static  err_t bli_strsm_small_XAltB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm4\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
    return BLIS_SUCCESS;
}

// XA = B; A is upper triangular; No transpose; single presicion; non-unit diagonal
static err_t bli_strsm_small_XAuB
     (
       side_t  side,
       obj_t*  AlphaObj,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
    printf("smalltrsm5\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B

    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

     for(k = 0; k < N; k++)
     {
        float lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }

    }
    return BLIS_SUCCESS;

}

//XA = B; A is upper triangular; No transpose; single precision; unit-diagonal
static  err_t bli_strsm_small_XAuB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm0\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B

    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }
    }
    return BLIS_SUCCESS;

}

//XA = B; A is upper-triangular; A is transposed; single precision; non-unit diagonal
static  err_t bli_strsm_small_XAutB(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm6\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;

    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *=alpha;

    for(k = N; k--;)
    {
        float lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M; i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k; j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(j)+(k)*lda];
            }
        }
    }
    return BLIS_SUCCESS;

}

//XA = B; A is upper-triangular; A is transposed; single precision; unit diagonal
static  err_t bli_strsm_small_XAutB_unitDiag(
            side_t side,
            obj_t* AlphaObj,
            obj_t* a,
            obj_t* b,
            cntx_t* cntx,
            cntl_t* cntl
            )
{
    printf("smalltrsm7\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


    dim_t i, j, k;
    float A_k_j;

    for(j = 0; j< N; j++)
        for(i = 0; i< M; i++)
            B[i+j*ldb] *= alpha;

     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(j)+(k)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;

            }
        }
    }
    return BLIS_SUCCESS;

}

//AX = B; A is lower triangular; No transpose; single precision; non-unit diagonal
static err_t bli_strsm_small_AlXB(
                side_t side,
                obj_t* AlphaObj,
                obj_t* a,
                obj_t* b,
                cntx_t* cntx,
                cntl_t* cntl
                )
{
    printf("smalltrsm8\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


  dim_t i, j, k;

  for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

  for (k = 0; k < M; k++)
  {
    float lkk_inv = 1.0/A[k+k*lda];
    for (j = 0; j < N; j++)
    {
        B[k + j*ldb] *= lkk_inv;
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
    }
  }// k -loop
 return BLIS_SUCCESS;

}


//AX = B; A is lower triangular; No transpose; single precision; unit diagonal
static err_t bli_strsm_small_AlXB_unitDiag(
                side_t side,
                obj_t* AlphaObj,
                obj_t* a,
                obj_t* b,
                cntx_t* cntx,
                cntl_t* cntl
                )
{
    printf("smalltrsm9\n");
    dim_t M = bli_obj_length(b);  //number of rows
    dim_t N = bli_obj_width(b);   //number of columns

    dim_t lda = bli_obj_col_stride(a); //column stride of matrix A
    dim_t ldb = bli_obj_col_stride(b); //column stride of matrix B


    float alpha = *(float *)AlphaObj->buffer;    //value of Alpha
    float* restrict A = a->buffer;      //pointer to matrix A
    float* restrict B = b->buffer;      //pointer to matrix B


  dim_t i, j, k;

  for(j = 0 ; j < N; j++)
      for(i = 0; i < M; i++)
          B[i+j*ldb] *= alpha;

  for (k = 0; k < M; k++)
  {
      for (j = 0; j < N; j++)
      {
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
     }
  }
  return BLIS_SUCCESS;

}


/*
* The bli_trsm_small implements unpacked version of TRSM 
* Currently only column-major is supported, A & B are column-major
* Input: A: MxM (triangular matrix)
*        B: MxN matrix
* Output: X: MxN matrix such that AX = alpha*B or XA = alpha*B or A'X = alpha*B or XA' = alpha*B 
* Here the output X is stored in B
* The custom-kernel will be called only when M*(M+N)* sizeof(Matrix Elements) < L3 cache
*/
err_t bli_trsm_small
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING
    return BLIS_NOT_YET_IMPLEMENTED;
#endif

    dim_t m = bli_obj_length(b);
    dim_t n = bli_obj_width(b);

    if(!(m && n))
        return BLIS_SUCCESS;

    if ( m > BLIS_SMALL_MATRIX_THRES_TRSM || n > BLIS_SMALL_MATRIX_THRES_TRSM )
    {
      return BLIS_NOT_YET_IMPLEMENTED;
    }

    printf("entered small matrix\n");


    // If alpha is zero, B matrix will become zero after scaling & hence solution is also zero matrix 
    if (bli_obj_equals(alpha, &BLIS_ZERO))
    {
        return BLIS_NOT_YET_IMPLEMENTED; // scale B by alpha
    }
    // We have to call matrix scaling if alpha != 1.0
    
    // if row major format return. Check this again.
/*
    if ((bli_obj_row_stride(a) != 1) ||
        (bli_obj_row_stride(b) != 1))
    {
	printf("hit a small matrix row-major\n");
        return BLIS_INVALID_ROW_STRIDE;
    }
*/
    num_t dt = ((*b).info & (0x7 << 0));

    // only float and double datatypes are supported as of now.
    if (dt != BLIS_DOUBLE && dt != BLIS_FLOAT)
    {
    return BLIS_EXPECTED_REAL_DATATYPE;
    }

    // A is expected to be triangular in trsm
    if (!bli_obj_is_upper_or_lower (a))
    {
    return BLIS_EXPECTED_TRIANGULAR_OBJECT;
    }

    // can use other control structs - even can use array of function pointers,
    // indexed by a number with bits formed by f('side', 'uplo', 'transa', dt).
    // In the below implementation, based on the number of finally implemented
    // cases, can move the checks with more cases higher up.

    if(side == BLIS_LEFT)
    {
        if(bli_obj_has_trans(a))
        {
            if(dt == BLIS_DOUBLE)
            {
                if(bli_obj_is_upper(a))
                {
                    //return bli_dtrsm_small_AutXB(side, alpha, a, b, cntx, cntl);
                    return BLIS_NOT_YET_IMPLEMENTED;
                }
                else
                {
                    //return bli_dtrsm_small_AltXB(side, alpha, a, b, cntx, cntl);
                    return BLIS_NOT_YET_IMPLEMENTED;
                }
            }
            else
            {
                if(bli_obj_is_upper(a))
                {
                    //return bli_strsm_small_AutXB(side, alpha, a, b, cntx, cntl);
                    return BLIS_NOT_YET_IMPLEMENTED;
                }
                else
                {
                    //return bli_strsm_small_AltXB(side, alpha, a, b, cntx, cntl);
                    return BLIS_NOT_YET_IMPLEMENTED;
                }

            }
        }
        else
        {
            if(dt == BLIS_DOUBLE)
            {
                if(bli_obj_is_upper(a))
                {
                    //return bli_dtrsm_small_AuXB(side, alpha, a, b, cntx, cntl);
                    return BLIS_NOT_YET_IMPLEMENTED;
                }
                else
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_dtrsm_small_AlXB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_dtrsm_small_AlXB(side, alpha, a, b, cntx, cntl);
                }
            }
            else
            {
                if(bli_obj_is_upper(a))
                {
                    //return bli_strsm_small_AuXB(side, alpha, a, b, cntx, cntl);
                    return BLIS_NOT_YET_IMPLEMENTED;
                }
                else
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_strsm_small_AlXB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_strsm_small_AlXB(side, alpha, a, b, cntx, cntl);
                }

            }

        }
    }
    else
    {
        if(bli_obj_has_trans(a))
        {
            if(dt == BLIS_DOUBLE)
            {
                if(bli_obj_is_upper(a))
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_dtrsm_small_XAutB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_dtrsm_small_XAutB(side, alpha, a, b, cntx, cntl);
                }
                else
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_dtrsm_small_XAltB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_dtrsm_small_XAltB(side, alpha, a, b, cntx, cntl);
                }
            }
            else
            {
                if(bli_obj_is_upper(a))
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_strsm_small_XAutB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_strsm_small_XAutB(side, alpha, a, b, cntx, cntl);
                }
                else
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_strsm_small_XAltB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_strsm_small_XAltB(side, alpha, a, b, cntx, cntl);
                }

            }
        }
        else
        {
            if(dt == BLIS_DOUBLE)
            {
                if(bli_obj_is_upper(a))
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_dtrsm_small_XAuB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_dtrsm_small_XAuB(side, alpha, a, b, cntx, cntl);
                }
                else
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_dtrsm_small_XAlB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_dtrsm_small_XAlB(side, alpha, a, b, cntx, cntl);
                }
            }
            else
            {
                if(bli_obj_is_upper(a))
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_strsm_small_XAuB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_strsm_small_XAuB(side, alpha, a, b, cntx, cntl);
                }
                else
                {
                    if(bli_obj_has_unit_diag(a))
                        return bli_strsm_small_XAlB_unitDiag(side, alpha, a, b, cntx, cntl);
                    else
                        return bli_strsm_small_XAlB(side, alpha, a, b, cntx, cntl);
                }

            }

        }
    }
    return BLIS_NOT_YET_IMPLEMENTED;
};



/* TRSM scalar code for the case AX = alpha * B
 * A is lower-triangular, non-unit-diagonal, no transpose
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
/*
static err_t dtrsm_small_AlXB (
                  double *A,
                  double *B,
                  dim_t M,
                  dim_t N,
                  dim_t lda,
                  dim_t ldb
                )
{

  dim_t i, j, k;

  for (k = 0; k < M; k++)
  {
    double lkk_inv = 1.0/A[k+k*lda];
    for (j = 0; j < N; j++)
    {
        B[k + j*ldb] *= lkk_inv;
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
    }
  }// k -loop
 return BLIS_SUCCESS;
}// end of function
*/
/* TRSM scalar code for the case AX = alpha * B
 * A is lower-triangular, unit-diagonal, no transpose
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
/*
static err_t dtrsm_small_AlXB_unitDiag (
                  double *A,
                  double *B,
                  dim_t M,
                  dim_t N,
                  dim_t lda,
                  dim_t ldb
                )
{

  dim_t i, j, k;

  for (k = 0; k < M; k++)
  {
      for (j = 0; j < N; j++)
      {
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
     }
  }
 return BLIS_SUCCESS;
}// end of function
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, non-unit-diagonal no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAuB (
            double *A,
            double *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

     dim_t i, j, k;
     for(k = 0; k < N; k++)
     {
        double lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }

    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, non-unit triangular, no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAlB (
            double *A,
            double *B,
            double alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;
    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;

    for(k = N;k--;)
    {
        double lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M;i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k;j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(k)+(j)*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/

/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, unit-diagonal, no transpose
 *Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAlB_unitDiag(
            double *A,
            double *B,
            double alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;
    double A_k_j;
     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(k)+(j)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;
            }
        }
    }


return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 *A is upper-triangular, non-unit-diagonal, A is transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAutB (
            double *A,
            double *B,
            double alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *=alpha;

    for(k = N; k--;)
    {
        double lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M; i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k; j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(j)+(k)*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, unit-diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAutB_unitDiag(
            double *A,
            double *B,
            double alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;
    double A_k_j;

    for(j = 0; j< N; j++)
        for(i = 0; i< M; i++)
            B[i+j*ldb] *= alpha;

     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(j)+(k)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;

            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, non-unit-diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAltB (
            double *A,
            double *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(k = 0; k < N; k++)
    {
        double lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for XA = alpha * B
 * A is lower-triangular, unit-diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAltB_unitDiag(
            double *A,
            double *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, unit-diagonal,  no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t dtrsm_small_XAuB_unitDiag (
            double *A,
            double *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////


/* TRSM scalar code for the case AX = alpha * B
 * A is lower-triangular, non-unit-diagonal, no transpose
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
/*
static err_t strsm_small_AlXB (
                  float *A,
                  float *B,
                  dim_t M,
                  dim_t N,
                  dim_t lda,
                  dim_t ldb
                )
{

  dim_t i, j, k;

  for (k = 0; k < M; k++)
  {
    float lkk_inv = 1.0/A[k+k*lda];
    for (j = 0; j < N; j++)
    {
        B[k + j*ldb] *= lkk_inv;
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
    }
  }// k -loop
 return BLIS_SUCCESS;
}// end of function
*/
/* TRSM scalar code for the case AX = alpha * B
 * A is lower-triangular, unit-diagonal, no transpose
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
/*
static err_t strsm_small_AlXB_unitDiag (
                  float *A,
                  float *B,
                  dim_t M,
                  dim_t N,
                  dim_t lda,
                  dim_t ldb
                )
{

  dim_t i, j, k;

  for (k = 0; k < M; k++)
  {
      for (j = 0; j < N; j++)
      {
        for (i = k+1; i < M; i++)
        {
            B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
        }
     }
  }
 return BLIS_SUCCESS;
}// end of function
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, non-unit-diagonal no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAuB (
            float *A,
            float *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

     dim_t i, j, k;
     for(k = 0; k < N; k++)
     {
        float lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }

    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, non-unit triangular, no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAlB (
            float *A,
            float *B,
            float alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;
    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;

    for(k = N;k--;)
    {
        float lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M;i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k;j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(k)+(j)*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, unit-diagonal, no transpose
 *Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAlB_unitDiag(
            float *A,
            float *B,
            float alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(j = 0 ; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *= alpha;
    float A_k_j;
     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(k)+(j)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;
            }
        }
    }


return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 *A is upper-triangular, non-unit-diagonal, A is transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAutB (
            float *A,
            float *B,
            float alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(j = 0; j < N; j++)
        for(i = 0; i < M; i++)
            B[i+j*ldb] *=alpha;

    for(k = N; k--;)
    {
        float lkk_inv = 1.0/A[(k)+(k)*lda];
        for(i = M; i--;)
        {
            B[(i)+(k)*ldb] *= lkk_inv;
            for(j = k; j--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A[(j)+(k)*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, unit-diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAutB_unitDiag(
            float *A,
            float *B,
            float alpha,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;
    float A_k_j;

    for(j = 0; j< N; j++)
        for(i = 0; i< M; i++)
            B[i+j*ldb] *= alpha;

     for(k = N; k--;)
     {
        for(j = k; j--;)
        {
            A_k_j = A[(j)+(k)*lda];
            for(i = M; i--;)
            {
                B[(i)+(j)*ldb] -= B[(i)+(k)*ldb] * A_k_j;

            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, non-unit-diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAltB (
            float *A,
            float *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(k = 0; k < N; k++)
    {
        float lkk_inv = 1.0/A[k+k*lda];
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] *= lkk_inv;
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for XA = alpha * B
 * A is lower-triangular, unit-diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAltB_unitDiag(
            float *A,
            float *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/
/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, unit-diagonal,  no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
/*
static err_t strsm_small_XAuB_unitDiag (
            float *A,
            float *B,
            dim_t M,
            dim_t N,
            dim_t lda,
            dim_t ldb
)
{

    dim_t i, j, k;

    for(k = 0; k < N; k++)
    {
        for(i = 0; i < M; i++)
        {
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }
    }
return BLIS_SUCCESS;
}
*/

#endif
