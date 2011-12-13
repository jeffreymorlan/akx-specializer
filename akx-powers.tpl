//#define TRACE
#include <Python.h>
#include <numpy/arrayobject.h>

// C headers
#include <stdlib.h> // for NULL
#include <stdio.h>  // for fprintf

#ifdef __SSE3__ // will be defined when compiling, but not when checking dependencies
#include <pmmintrin.h> // for SSE
#endif

#include "akx.h"

#ifdef _OPENMP
#include <omp.h>
#else
#include <pthread.h> // for pthreads stuff
pthread_barrier_t barrier;
#endif

## Load a tile-sized group of variables from y
<%def name="load_y(y, ib, b_m, b_n, b_transpose)">
  %if (not b_transpose) and (b_n % 2 == 0):
    %for i in xrange(b_m):
      __m128d ${y}${i} = _mm_load_sd(&y[${ib}*${b_m} + ${i}]);
    %endfor
  %elif b_transpose and (b_m % 2 == 0):
    %for i in xrange(0, b_m, 2):
      __m128d ${y}${i} = _mm_load_pd(&y[${ib}*${b_m} + ${i}]);
    %endfor
  %else:
    %for i in xrange(b_m):
      double ${y}${i} = y[${ib}*${b_m} + ${i}];
    %endfor
  %endif
</%def>

<%def name="load_y_zero(y, b_m, b_n, b_transpose)">
  %if (not b_transpose) and (b_n % 2 == 0):
    %for i in xrange(b_m):
      __m128d ${y}${i} = _mm_setzero_pd();
    %endfor
  %elif b_transpose and (b_m % 2 == 0):
    %for i in xrange(0, b_m, 2):
      __m128d ${y}${i} = _mm_setzero_pd();
    %endfor
  %else:
    %for i in xrange(b_m):
      double ${y}${i} = 0.0;
    %endfor
  %endif
</%def>

## Store a tile-sized group of variables to y
<%def name="store_y(y, ib, b_m, b_n, b_transpose)">
  %if (not b_transpose) and (b_n % 2) == 0:
    %for i in xrange(b_m):
      _mm_store_sd(&y[${ib}*${b_m} + ${i}], _mm_hadd_pd(${y}${i}, ${y}${i}));
    %endfor
  %elif b_transpose and (b_m % 2) == 0:
    %for i in xrange(0, b_m, 2):
      _mm_store_pd(&y[${ib}*${b_m} + ${i}], ${y}${i});
    %endfor
  %else:
    %for i in xrange(b_m):
      y[${ib}*${b_m} + ${i}] = ${y}${i};
    %endfor
  %endif
</%def>

<%def name="load_x(x, jb, b_m, b_n, b_transpose)">
  %if (not b_transpose) and (b_n % 2 == 0):
    %for j in xrange(0, b_n, 2):
      __m128d ${x}${j} = _mm_load_pd(&x[${jb}*${b_n} + ${j}]);
    %endfor
  %elif b_transpose and (b_m % 2 == 0):
    %for j in xrange(b_n):
      __m128d ${x}${j} = _mm_load1_pd(&x[${jb}*${b_n} + ${j}]);
    %endfor
  %else:
    %for j in xrange(b_n):
      double ${x}${j} = x[${jb}*${b_n} + ${j}];
    %endfor
  %endif
</%def>

<%def name="do_tile(y, x, b_m, b_n, b_transpose)">
  %if not b_transpose:
    %if b_n % 2 == 0:
      %for i in xrange(b_m):
        %for j in xrange(0, b_n, 2):
          ${y}${i} = _mm_add_pd(${y}${i}, _mm_mul_pd(${x}${j}, _mm_load_pd(&A->bvalues[jb*${b_m*b_n} + ${i*b_n + j}])));
        %endfor
      %endfor
    %else:
      %for i in xrange(b_m):
        %for j in xrange(b_n):
          ${y}${i} += A->bvalues[jb*${b_m*b_n} + ${i*b_n + j}] * ${x}${j};
        %endfor
      %endfor
    %endif
  %else:
    %if b_m % 2 == 0:
      %for j in xrange(b_n):
        %for i in xrange(0, b_m, 2):
          ${y}${i} = _mm_add_pd(${y}${i}, _mm_mul_pd(${x}${j}, _mm_load_pd(&A->bvalues[jb*${b_m*b_n} + ${j*b_m + i}])));
        %endfor
      %endfor
    %else:
      %for j in xrange(b_n):
        %for i in xrange(b_m):
          ${y}${i} += A->bvalues[jb*${b_m*b_n} + ${j*b_m + i}] * ${x}${j};
        %endfor
      %endfor
    %endif
  %endif
</%def>

<%def name="do_tilerow(b_m, b_n, b_transpose)">
  ${load_y_zero("y", b_m, b_n, b_transpose)}
  for (jb = A->browptr[ib]; jb < A->browptr[ib+1]; ++jb)
  {
    index_t j = A->bcolidx[jb];
    ${load_x("x", "j", b_m, b_n, b_transpose)}
    ${do_tile("y", "x", b_m, b_n, b_transpose)}
  }
  ${store_y("y", "ib", b_m, b_n, b_transpose)}
  %if usecoeffs:
    // TODO: use SSE here too
    %for i in xrange(b_m):
      y[ib*${b_m} + ${i}] += x[ib*${b_m} + ${i}] * coeff;
    %endfor
  %endif
</%def>

%for b_m, b_n, b_transpose in variants:

void bcsr_spmv_${b_m}_${b_n}_${b_transpose}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb)
{
  index_t ib, jb;
  for (ib = 0; ib < mb; ++ib)
  {
    ${do_tilerow(b_m, b_n, b_transpose)}
  }
}

void bcsr_spmv_rowlist_${b_m}_${b_n}_${b_transpose}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  const index_t *__restrict__ computation_seq,
  index_t seq_len)
{
  index_t q, ib, jb;
  for (q = 0; q < seq_len; q++)
  {
    ib = computation_seq[q];
    ${do_tilerow(b_m, b_n, b_transpose)}
  }
}

void bcsr_spmv_stanzas_${b_m}_${b_n}_${b_transpose}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  const index_t *__restrict__ computation_seq,
  index_t seq_len)
{
  index_t q, ib, jb;
  for (q = 0; q < seq_len; q += 2)
  {
    for (ib = computation_seq[q]; ib < computation_seq[q+1]; ib++)
    {
      ${do_tilerow(b_m, b_n, b_transpose)}
    }
  }
}

void bcsr_spmv_symmetric_${b_m}_${b_n}_${b_transpose}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb)
{
  index_t ib, jb;
  for (ib = 0; ib < mb; ++ib)
  {
    ${load_y_zero("yi", b_m, b_n, b_transpose)}
    ${store_y("yi", "ib", b_m, b_n, b_transpose)}
  }
  for (ib = 0; ib < mb; ++ib)
  {
    ${load_y("yi", "ib", b_m, b_n, b_transpose)}
    ${load_x("xi", "ib", b_m, b_n, not b_transpose)}
    for (jb = A->browptr[ib]; jb < A->browptr[ib+1]; ++jb)
    {
      index_t j = A->bcolidx[jb];
      ${load_x("xj", "j", b_m, b_n, b_transpose)}
      ${do_tile("yi", "xj", b_m, b_n, b_transpose)}
      if (j > ib && j < mb)
      {
        ${load_y("yj", "j", b_m, b_n, not b_transpose)}
        ${do_tile("yj", "xi", b_m, b_n, not b_transpose)}
        ${store_y("yj", "j", b_m, b_n, not b_transpose)}
      }
    }
    ${store_y("yi", "ib", b_m, b_n, b_transpose)}
    %if usecoeffs:
      // TODO: use SSE here too
      %for i in xrange(b_m):
        y[ib*${b_m} + ${i}] += x[ib*${b_m} + ${i}] * coeff;
      %endfor
    %endif
  }
}

%endfor

struct bcsr_funcs {
  index_t b_m;
  index_t b_n;
  int b_transpose;
  void (*bcsr_spmv)(
    const struct bcsr_t *__restrict__ A,
    const value_t *__restrict__ x,
    value_t *__restrict__ y,
%if usecoeffs:
    value_t coeff,
%endif
    index_t mb);
  void (*bcsr_spmv_rowlist)(
    const struct bcsr_t *__restrict__ A,
    const value_t *__restrict__ x,
    value_t *__restrict__ y,
%if usecoeffs:
    value_t coeff,
%endif
    const index_t *__restrict__ computation_seq,
    index_t seq_len);
  void (*bcsr_spmv_stanzas)(
    const struct bcsr_t *__restrict__ A,
    const value_t *__restrict__ x,
    value_t *__restrict__ y,
%if usecoeffs:
    value_t coeff,
%endif
    const index_t *__restrict__ computation_seq,
    index_t seq_len);  
  void (*bcsr_spmv_symmetric)(
    const struct bcsr_t *__restrict__ A,
    const value_t *__restrict__ x,
    value_t *__restrict__ y,
%if usecoeffs:
    value_t coeff,
%endif
    index_t mb);
} bcsr_funcs_table[] = {
%for b_m, b_n, b_transpose in variants:
  { ${b_m}, ${b_n}, ${b_transpose},
    bcsr_spmv_${b_m}_${b_n}_${b_transpose},
    bcsr_spmv_rowlist_${b_m}_${b_n}_${b_transpose},
    bcsr_spmv_stanzas_${b_m}_${b_n}_${b_transpose},
    bcsr_spmv_symmetric_${b_m}_${b_n}_${b_transpose} },
%endfor
};

void * do_akx ( void *__restrict__ input )
{
  struct akx_data *data = (struct akx_data*) input;
  struct akx_thread_block *__restrict__ tb = data->thread_block;

  level_t glevel = 0;
  while (glevel < data->steps)
  {
    // On the last iteration, we may do fewer than k steps.
    // To minimize redundancy, we should do the later levels, [k-#steps, k),
    // rather than the earlier levels, [0, #steps).
    level_t start = tb->orig_eb.k - (data->steps - glevel);
    if (start < 0)
      start = 0;
    glevel -= start;

    part_id_t block;
    for (block = 0; block < tb->explicit_blocks; block++) {
      struct akx_explicit_block *eb = &tb->eb[block];
#define V_LOCAL(l)  (&eb->V[(l)*eb->V_size])
#define V_GLOBAL(l) (&data->V_global[(glevel+(l))*data->V_global_m])
      index_t i;
      // copy vector to local data using perm
      value_t *local = V_LOCAL(start);
      value_t *global = V_GLOBAL(start);
      for (i = 0; i < eb->size / ${orig_b_n}; ++i)
      {
%       for j in xrange(orig_b_n):
          local[i*${orig_b_n} + ${j}] = global[eb->perm[i]*${orig_b_n} + ${j}];
%       endfor
      }

      struct bcsr_funcs *bf = bcsr_funcs_table;
      while (bf->b_m != eb->A_part->b_m ||
             bf->b_n != eb->A_part->b_n ||
             bf->b_transpose != eb->A_part->b_transpose)
      {
        bf++;
        if (bf == &bcsr_funcs_table[sizeof bcsr_funcs_table / sizeof *bcsr_funcs_table])
          abort();
      }

      level_t l;
      if (eb->implicit_blocks)
      {
        part_id_t block;
        for (block = 0; block < eb->implicit_blocks; block++) {
          for (l = start; l < eb->k; l++) {
            index_t lev_start = eb->level_start[block * eb->k + l];
            index_t lev_end   = eb->level_start[block * eb->k + l + 1];
            //printf("thread %d block %d level %d (%d,%d)\n", pthread_self(), block, l, lev_start, lev_end);
            (eb->implicit_stanza ? bf->bcsr_spmv_stanzas : bf->bcsr_spmv_rowlist)(
              eb->A_part,
              V_LOCAL(l),
              V_LOCAL(l+1),
%if usecoeffs:
              data->coeffs[glevel + l],
%endif
              &eb->computation_seq[lev_start],
              lev_end - lev_start);
          }
        }
        for (l = start; l < eb->k; l++)
        {
          // copy vector to global data using perm
          local = V_LOCAL(l+1);
          global = V_GLOBAL(l+1);
          for (i = 0; i < eb->schedule[eb->k-1] / ${orig_b_m}; ++i)
          {
%           for j in xrange(orig_b_m):
              global[eb->perm[i]*${orig_b_m} + ${j}] = local[i*${orig_b_m} + ${j}];
%           endfor
          }
        }
      }
      else
      {
        // Perform k SpMVs
        for (l = start; l < eb->k; ++l)
        {
          // Monomial basis:
          (eb->symmetric_opt ? bf->bcsr_spmv_symmetric : bf->bcsr_spmv)(
            eb->A_part,
            V_LOCAL(l),
            V_LOCAL(l+1),
%if usecoeffs:
            data->coeffs[glevel + l],
%endif
            (eb->schedule[l] + eb->A_part->b_m - 1) / eb->A_part->b_m);
          // Newton basis will be the spmv plus a _scal by the chosen shift:
          //   x_{i+1} = Ax_i - \lambda*x_i
          // Chebyshev basis will _scal the result of the spmv by 2 and _axpy with the prev. vector:
          //   x_{i+1} = 2Ax_i - x_{i-1},
          //   where x_{-1} = ones()

          // copy vector to global data using perm
          local = V_LOCAL(l+1);
          global = V_GLOBAL(l+1);
          for (i = 0; i < eb->schedule[eb->k-1] / ${orig_b_m}; ++i)
          {
%           for j in xrange(orig_b_m):
              global[eb->perm[i]*${orig_b_m} + ${j}] = local[i*${orig_b_m} + ${j}];
%           endfor
          }
        }
      }
#undef V_GLOBAL
#undef V_LOCAL
    }

#ifdef _OPENMP
    #pragma openmp barrier
#else
    pthread_barrier_wait(&barrier);
#endif
    glevel += tb->orig_eb.k;
  }

  return NULL;
}

static PyObject *
AkxPowers_powers(PyObject *self, PyObject *args)
{
  AkxObjectC *akxobj;
%if usecoeffs:
  PyArrayObject *vecs, *coeffs;
  if (!PyArg_ParseTuple(args, "OO!O!", &akxobj, &PyArray_Type, &vecs, &PyArray_Type, &coeffs))
    return NULL;
%else:
  PyArrayObject *vecs;
  if (!PyArg_ParseTuple(args, "OO!", &akxobj, &PyArray_Type, &vecs, &PyArray_Type))
    return NULL;
%endif

  if (vecs->nd != 2
      || vecs->dimensions[1] != akxobj->A.mb * akxobj->A.b_m
      || vecs->strides[1] != sizeof(value_t))
  {
    PyErr_SetString(PyExc_ValueError, "vector array has wrong shape");
    return NULL;
  }

%if usecoeffs:
  if (coeffs->nd != 1
      || coeffs->dimensions[0] != (vecs->dimensions[0] - 1)
      || coeffs->strides[0] != sizeof(value_t))
  {
    PyErr_SetString(PyExc_ValueError, "coefficient array has wrong shape");
    return NULL;
  }
%endif

  if (akxobj->p.n_parts == 0)
  {
    PyErr_SetString(PyExc_ValueError, "thread blocks not yet created");
    return NULL;
  }

#ifndef _OPENMP
#ifdef TRACE
  fprintf (stderr, "== initializing thread data ...\n");
#endif

  // TODO: Initialize pthreads
  pthread_attr_t attr;
  P( pthread_attr_init( &attr ) );
  P( pthread_barrier_init( &barrier, NULL, akxobj->p.n_parts ) );

  pthread_t *threads;
  _ALLOC_ ((void**) &threads, akxobj->p.n_parts * sizeof( pthread_t ) );
#endif
    
  struct akx_data * td;
  _ALLOC_ ( (void**) &td, akxobj->p.n_parts * sizeof (struct akx_data));

  part_id_t pp;
  for (pp = 0; pp < akxobj->p.n_parts; ++pp)
  {
    // TODO: sched. affinity stuff
    td[pp].V_global = (value_t *)vecs->data;
    td[pp].V_global_m = vecs->strides[0] / sizeof(value_t);
    td[pp].thread_block = &akxobj->tb[pp];
    td[pp].steps = vecs->dimensions[0] - 1;
%if usecoeffs:
    td[pp].coeffs = (value_t *)coeffs->data;
%endif
  }

#ifdef _OPENMP
  omp_set_num_threads(akxobj->p.n_parts);
  #pragma omp parallel
  {
    do_akx(&td[omp_get_thread_num()]);
  }
#else
  for (pp = 1; pp < akxobj->p.n_parts; ++pp)
    P( pthread_create( &threads[pp], &attr, &do_akx, (void*) &td[pp] ) );

  do_akx (&td[0]);

  for( pp = 1; pp < akxobj->p.n_parts; ++pp ) 
    P( pthread_join( threads[pp], NULL ) );

  P( pthread_barrier_destroy( &barrier ) );
  P( pthread_attr_destroy( &attr ) );
  _FREE_ (threads);
#endif

  _FREE_ ((void*) td );
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
	{ "powers", AkxPowers_powers, METH_VARARGS },
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
init_akx_powers(void)
{
  PyObject *module = Py_InitModule("_akx_powers", methods);
  if (!module)
    return;

  import_array();
}
