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

<%def name="do_tilerow(format, b_m, b_n, b_transpose)">
  %if format == '':
    ${load_y_zero("y", b_m, b_n, b_transpose)}
    for (jb = browptr[ib]; jb < browptr[ib+1]; ++jb)
    {
      index_t j = bcolidx[jb];
      ${load_x("x", "j", b_m, b_n, b_transpose)}
      ${do_tile("y", "x", b_m, b_n, b_transpose)}
    }
    ${store_y("y", "ib", b_m, b_n, b_transpose)}
  %else: ## Symmetric
    ${load_y("yi", "ib", b_m, b_n, b_transpose)}
    ${load_x("xi", "ib", b_m, b_n, not b_transpose)}
    for (jb = browptr[ib]; jb < browptr[ib+1]; ++jb)
    {
      index_t j = bcolidx[jb];
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
  %endif
  %if usecoeffs:
    // TODO: use SSE here too
    %for i in xrange(b_m):
      y[ib*${b_m} + ${i}] += x[ib*${b_m} + ${i}] * coeff;
    %endfor
  %endif
</%def>

%for b_m, b_n, b_transpose, browptr_comp, bcolidx_comp in variants:
%for format in ('', '_symmetric'):

<%def name="init(browptr_comp, bcolidx_comp)">
%if browptr_comp == 0:
  index_t *__restrict__ browptr = A->browptr;
%else:
  uint16_t *__restrict__ browptr = A->browptr16;
%endif
%if bcolidx_comp == 0:
  index_t *__restrict__ bcolidx = A->bcolidx;
%else:
  uint16_t *__restrict__ bcolidx = A->bcolidx16;
%endif
</%def>

void bcsr_spmv${format}_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb)
{
  index_t ib, jb;
  ${init(browptr_comp, bcolidx_comp)}
  for (ib = 0; ib < mb; ++ib)
  {
    ${do_tilerow(format, b_m, b_n, b_transpose)}
  }
}

void bcsr_spmv${format}_rowlist_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb,
  const index_t *__restrict__ computation_seq,
  index_t seq_len)
{
  index_t q, ib, jb;
  ${init(browptr_comp, bcolidx_comp)}
  for (q = 0; q < seq_len; q++)
  {
    ib = computation_seq[q];
    ${do_tilerow(format, b_m, b_n, b_transpose)}
  }
}

void bcsr_spmv${format}_stanzas_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp}(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb,
  const index_t *__restrict__ computation_seq,
  index_t seq_len)
{
  index_t q, ib, jb;
  ${init(browptr_comp, bcolidx_comp)}
  for (q = 0; q < seq_len; q += 2)
  {
    for (ib = computation_seq[q]; ib < computation_seq[q+1]; ib++)
    {
      ${do_tilerow(format, b_m, b_n, b_transpose)}
    }
  }
}

%endfor
%endfor

typedef void (*bcsr_func_noimplicit)(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb);
typedef void (*bcsr_func_implicit)(
  const struct bcsr_t *__restrict__ A,
  const value_t *__restrict__ x,
  value_t *__restrict__ y,
%if usecoeffs:
  value_t coeff,
%endif
  index_t mb,
  const index_t *__restrict__ computation_seq,
  index_t seq_len);

struct bcsr_funcs {
  index_t b_m;
  index_t b_n;
  int b_transpose;
  int browptr_comp;
  int bcolidx_comp;
  struct {
    bcsr_func_noimplicit noimplicit;
    bcsr_func_implicit implicit[2];
  } funcs[2];
} bcsr_funcs_table[] = {
%for b_m, b_n, b_transpose, browptr_comp, bcolidx_comp in variants:
  { ${b_m}, ${b_n}, ${b_transpose}, ${browptr_comp}, ${bcolidx_comp},
    { { bcsr_spmv_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp},
        { bcsr_spmv_rowlist_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp},
          bcsr_spmv_stanzas_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp} }
      },
      { bcsr_spmv_symmetric_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp},
        { bcsr_spmv_symmetric_rowlist_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp},
          bcsr_spmv_symmetric_stanzas_${b_m}_${b_n}_${b_transpose}_${browptr_comp}_${bcolidx_comp} }
      }
    }
  },
%endfor
};

void * do_akx ( void *__restrict__ input )
{
  struct akx_data *data = (struct akx_data*) input;

  level_t glevel = 0;
  while (glevel < data->steps)
  {
    // On the last iteration, we may do fewer than k steps.
    // To minimize redundancy, we should do the later levels, [k-#steps, k),
    // rather than the earlier levels, [0, #steps).
    level_t start = data->k - (data->steps - glevel);
    if (start < 0)
      start = 0;
    glevel -= start;

    part_id_t block;
    for (block = 0; block < data->nblocks; block++) {
      AkxBlock *__restrict__ eb = data->blocks[block];
#define V_LOCAL(l)  (&eb->V[(l)*eb->V_size])
#define V_GLOBAL(l) (&data->V_global[(glevel+(l))*data->V_global_m])
      index_t i;
      // copy vector to local data using perm
      value_t *__restrict__ local = V_LOCAL(start);
      value_t *__restrict__ global = V_GLOBAL(start);
      for (i = 0; i < eb->perm_size; ++i)
        local[i] = global[eb->perm[i]];

      struct bcsr_funcs *bf = bcsr_funcs_table;
      while (bf->b_m != eb->A_part.b_m ||
             bf->b_n != eb->A_part.b_n ||
             bf->b_transpose != eb->A_part.b_transpose ||
             bf->browptr_comp != eb->browptr_comp ||
             bf->bcolidx_comp != eb->bcolidx_comp)
      {
        bf++;
        if (bf == &bcsr_funcs_table[sizeof bcsr_funcs_table / sizeof *bcsr_funcs_table])
          abort();
      }

      level_t l;
      if (eb->implicit_blocks)
      {
        bcsr_func_implicit func = bf->funcs[eb->symmetric_opt].implicit[eb->implicit_stanza];
        part_id_t block;

        if (eb->symmetric_opt)
          memset(V_LOCAL(start+1), 0, sizeof(value_t) * eb->V_size * (eb->k - start));
        for (block = 0; block < eb->implicit_blocks; block++)
        {
          for (l = start; l < eb->k; l++)
          {
            index_t mb = (eb->schedule[l] + eb->A_part.b_m - 1) / eb->A_part.b_m;
            index_t lev_start = eb->level_start[block * eb->k + l];
            index_t lev_end   = eb->level_start[block * eb->k + l + 1];
            //printf("thread %d block %d level %d (%d,%d)\n", pthread_self(), block, l, lev_start, lev_end);
            func(
              &eb->A_part,
              V_LOCAL(l),
              V_LOCAL(l+1),
%if usecoeffs:
              data->coeffs[glevel + l],
%endif
              mb,
              &eb->computation_seq[lev_start],
              lev_end - lev_start);
          }
        }
        for (l = start; l < eb->k; l++)
        {
          // copy vector to global data using perm
          local = V_LOCAL(l+1);
          global = V_GLOBAL(l+1);
          for (i = 0; i < eb->schedule[eb->k-1]; ++i)
            global[eb->perm[i]] = local[i];
        }
      }
      else
      {
        bcsr_func_noimplicit func = bf->funcs[eb->symmetric_opt].noimplicit;
        // Perform k SpMVs
        for (l = start; l < eb->k; ++l)
        {
          if (eb->symmetric_opt)
            memset(V_LOCAL(l+1), 0, sizeof(value_t) * eb->V_size);
          // Monomial basis:
          func(
            &eb->A_part,
            V_LOCAL(l),
            V_LOCAL(l+1),
%if usecoeffs:
            data->coeffs[glevel + l],
%endif
            (eb->schedule[l] + eb->A_part.b_m - 1) / eb->A_part.b_m);
          // Newton basis will be the spmv plus a _scal by the chosen shift:
          //   x_{i+1} = Ax_i - \lambda*x_i
          // Chebyshev basis will _scal the result of the spmv by 2 and _axpy with the prev. vector:
          //   x_{i+1} = 2Ax_i - x_{i-1},
          //   where x_{-1} = ones()

          // copy vector to global data using perm
          local = V_LOCAL(l+1);
          global = V_GLOBAL(l+1);
          for (i = 0; i < eb->schedule[eb->k-1]; ++i)
            global[eb->perm[i]] = local[i];
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
    glevel += data->k;
  }

  return NULL;
}

static PyObject *
AkxObjectC_powers(AkxObjectC *akxobj, PyObject *args)
{
%if usecoeffs:
  PyArrayObject *vecs, *coeffs;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &vecs, &PyArray_Type, &coeffs))
    return NULL;
%else:
  PyArrayObject *vecs;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &vecs, &PyArray_Type))
    return NULL;
%endif

  if (vecs->nd != 2
      || vecs->dimensions[1] != akxobj->matrix_size
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

  struct akx_data *td = _ALLOC_ (akxobj->nthreads * sizeof (struct akx_data));

  part_id_t pp;
  for (pp = 0; pp < akxobj->nthreads; ++pp)
  {
    // TODO: sched. affinity stuff
    td[pp].k = akxobj->k;
    td[pp].V_global = (value_t *)vecs->data;
    td[pp].V_global_m = vecs->strides[0] / sizeof(value_t);
    td[pp].nblocks = akxobj->thread_offset[pp+1] - akxobj->thread_offset[pp];
    td[pp].blocks = &akxobj->blocks[akxobj->thread_offset[pp]];
    td[pp].steps = vecs->dimensions[0] - 1;
%if usecoeffs:
    td[pp].coeffs = (value_t *)coeffs->data;
%endif
  }

#ifdef _OPENMP
  omp_set_num_threads(akxobj->nthreads);
  #pragma omp parallel
  {
    do_akx(&td[omp_get_thread_num()]);
  }
#else
  pthread_attr_t attr;
  P( pthread_attr_init( &attr ) );
  P( pthread_barrier_init( &barrier, NULL, akxobj->nthreads ) );
  pthread_t *threads = _ALLOC_ (akxobj->nthreads * sizeof (pthread_t));

  for (pp = 1; pp < akxobj->nthreads; ++pp)
    P( pthread_create( &threads[pp], &attr, &do_akx, (void*) &td[pp] ) );

  do_akx (&td[0]);

  for( pp = 1; pp < akxobj->nthreads; ++pp ) 
    P( pthread_join( threads[pp], NULL ) );

  _FREE_ (threads);
  P( pthread_barrier_destroy( &barrier ) );
  P( pthread_attr_destroy( &attr ) );
#endif

  _FREE_ ((void*) td );
  Py_RETURN_NONE;
}

static PyObject *
AkxObjectC_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
  // Note: this does not do proper error checking (and can't; we don't have
  // access to &AkxBlock_Type here as we would need to prevent arbitrary
  // objects from getting passed off as blocks).
  // We rely on the Python code to ensure parameters are sane.

  int k, matrix_size;
  PyObject *list;
  if (!PyArg_ParseTuple(args, "iiO", &k, &matrix_size, &list))
    return NULL;

  AkxObjectC *self = PyObject_New(AkxObjectC, subtype);
  if (!self)
    return NULL;

  self->k = k;
  self->matrix_size = matrix_size;

  self->nthreads = PyList_GET_SIZE(list);
  self->thread_offset = _ALLOC_((self->nthreads + 1) * sizeof(part_id_t));
  part_id_t total_blocks = 0;
  part_id_t thread;
  for (thread = 0; thread < self->nthreads; thread++)
  {
    self->thread_offset[thread] = total_blocks;
    total_blocks += PyList_GET_SIZE(PyList_GET_ITEM(list, thread));
  }
  self->thread_offset[thread] = total_blocks;

  self->blocks = _ALLOC_(total_blocks * sizeof(AkxBlock *));
  for (thread = 0; thread < self->nthreads; thread++)
  {
    PyObject *sublist = PyList_GET_ITEM(list, thread);
    part_id_t j;
    for (j = 0; j < PyList_GET_SIZE(sublist); j++)
    {
      AkxBlock *block = (AkxBlock *)PyList_GET_ITEM(sublist, j);
      Py_INCREF(block);
      assert(block->k == k);
      self->blocks[self->thread_offset[thread] + j] = block;
    }
  }
  
  return (PyObject *)self;
}

static void
AkxObjectC_dealloc(AkxObjectC *akxobj)
{
  index_t i;
  for (i = 0; i < akxobj->thread_offset[akxobj->nthreads]; i++)
    Py_DECREF(akxobj->blocks[i]);
  PyObject_Del(akxobj);
}

#define METHOD(name, flags) { #name, (PyCFunction)AkxObjectC_##name, flags },
static PyMethodDef AkxObjectC_methods[] = {
  METHOD(powers, METH_VARARGS)
  { NULL, NULL, 0, NULL }
};
#undef METHOD

static PyTypeObject AkxObjectC_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                          /*tp_size*/
	"AkxObjectC",               /*tp_name*/
	sizeof(AkxObjectC),         /*tp_basicsize*/
	0,                          /*tp_itemsize*/
	/* methods */
	(destructor)AkxObjectC_dealloc,     /*tp_dealloc*/
	0,                          /*tp_print*/
	0,                          /*tp_getattr*/
	0,                          /*tp_setattr*/
	0,                          /*tp_compare*/
	0,                          /*tp_repr*/
	0,                          /*tp_as_number*/
	0,                          /*tp_as_sequence*/
	0,                          /*tp_as_mapping*/
	0,                          /*tp_hash*/
	0,                          /*tp_call*/
	0,                          /*tp_str*/
	0,                          /*tp_getattro*/
	0,                          /*tp_setattro*/
	0,                          /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,         /*tp_flags*/
	0,                          /*tp_doc*/
	0,                          /*tp_traverse*/
	0,                          /*tp_clear*/
	0,                          /*tp_richcompare*/
	0,                          /*tp_weaklistoffset*/
	0,                          /*tp_iter*/
	0,                          /*tp_iternext*/
	AkxObjectC_methods,          /*tp_methods*/
	0,                          /*tp_members*/
	0,                          /*tp_getset*/
	0,                          /*tp_base*/
	0,                          /*tp_dict*/
	0,                          /*tp_descr_get*/
	0,                          /*tp_descr_set*/
	0,                          /*tp_dictoffset*/
	0,                          /*tp_init*/
	0,                          /*tp_alloc*/
	AkxObjectC_new,              /*tp_new*/
};

static PyMethodDef methods[] = {
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
init_akx_powers(void)
{
  PyObject *module = Py_InitModule("_akx_powers", methods);
  if (!module)
    return;

  if (PyType_Ready(&AkxObjectC_Type) < 0)
    return;

  Py_INCREF(&AkxObjectC_Type);
  PyModule_AddObject(module, "AkxObjectC", (PyObject *)&AkxObjectC_Type);

  import_array();
}
