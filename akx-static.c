#undef NDEBUG // for assertions
#include <Python.h>
#include <numpy/arrayobject.h>

// C headers
#include <stddef.h> // for NULL
#include <string.h> // for memset
#include <stdio.h>  // for fprintf
#include <assert.h> // for assert

#ifdef __SSE3__ // will be defined when compiling, but not when checking dependencies
#include <pmmintrin.h>
#endif

// PaToH hypergraph partitioning library
#include "patoh.h"

#include "akx.h"

static void *copy_array(void *array, size_t size)
{
  void *out = _ALLOC_(size);
  memcpy(out, array, size);
  return out;
}

void bcsr_structure_transpose(
    struct bcsr_t *__restrict__ AT,
    const struct bcsr_t *__restrict__ A,
    index_t rows)
{
  AT->mb = A->nb;
  AT->nb = rows;
  AT->b_m = 0;
  AT->b_n = 0;
  AT->b_transpose = 0;
  AT->browptr_comp = 0;
  AT->bcolidx_comp = 0;
  AT->nnzb = A->browptr[rows];
  AT->browptr = _ALLOC_ ((AT->mb + 1) * sizeof(index_t));
  AT->bcolidx = _ALLOC_ (AT->nnzb * sizeof(index_t));
  AT->bvalues = NULL;

  index_t i, j;
  for (i = 0; i <= AT->mb; i++)
    AT->browptr[i] = 0;
  for (i = 0; i < A->browptr[rows]; i++)
    AT->browptr[A->bcolidx[i]]++;
  for (i = 0; i < AT->mb; i++)
    AT->browptr[i+1] += AT->browptr[i];
  for (i = rows; --i >= 0; )
    for (j = A->browptr[i+1]; --j >= A->browptr[i]; )
      AT->bcolidx[--AT->browptr[A->bcolidx[j]]] = i;
}

void bcsr_upper_triangle(
    struct bcsr_t *__restrict__ U,
    const struct bcsr_t *__restrict__ A)
{
  U->mb = A->mb;
  U->nb = A->nb;
  U->b_m = A->b_m;
  U->b_n = A->b_n;
  U->b_transpose = A->b_transpose;
  U->browptr_comp = 0;
  U->bcolidx_comp = 0;
  U->nnzb = 0;
  U->browptr = _ALLOC_ (sizeof(index_t) * (A->mb + 1));
  U->bcolidx = _ALLOC_ (sizeof(index_t) * A->nnzb);
  U->bvalues = _ALLOC_ (sizeof(value_t) * A->nnzb * A->b_m * A->b_n);

  index_t i, j;
  nnz_t nnzb = 0;
  for (i = 0; i < A->mb; i++)
  {
    // Skip nonzeros left of diagonal
    for (j = A->browptr[i]; j < A->browptr[i+1]; j++)
      if (A->bcolidx[j] >= i)
        break;
    // Copy upper-triangle part only
    index_t count = A->browptr[i+1] - j;
    U->browptr[i] = nnzb;
    memcpy(&U->bcolidx[nnzb], &A->bcolidx[j], count * sizeof(index_t));
    memcpy(&U->bvalues[nnzb * A->b_m * A->b_n],
           &A->bvalues[j * A->b_m * A->b_n],
           count * sizeof(value_t) * A->b_m * A->b_n);
    nnzb += count;
  }
  U->browptr[i] = nnzb;
  U->nnzb = nnzb;
}

void bcsr_free(struct bcsr_t *A)
{
  _FREE_ (A->browptr);
  _FREE_ (A->bcolidx);
  _FREE_ (A->bvalues);
}

void workspace_init(struct set *workspace, index_t capacity)
{
  workspace->capacity = capacity;
  workspace->elements = _ALLOC_ (capacity * sizeof(index_t));
  workspace->flags    = _ALLOC_ (capacity * sizeof(flag_t));
  memset(workspace->flags, 0, capacity * sizeof(flag_t));
}

void workspace_free(struct set *workspace)
{
  _FREE_ (workspace->elements);
  _FREE_ (workspace->flags);
}

index_t extend_net(
    struct set *__restrict__ workspace,
    const struct bcsr_t *__restrict__ A,
    index_t frontier_begin,
    index_t frontier_end)
{
  index_t n_elements = frontier_end;
  index_t f;
  for (f = frontier_begin; f < frontier_end; ++f)
  {
    // Add col-indices of some row of A to workspace
    index_t row_to_add   = workspace->elements[f];
    index_t colidx_start = A->browptr[row_to_add];
    index_t colidx_end   = A->browptr[row_to_add + 1];
    index_t i;
    for (i = colidx_start; i < colidx_end; ++i)
    {
      index_t new_pin = A->bcolidx[i];
      if (!workspace->flags[new_pin])
      {
        workspace->flags[new_pin] = 1;
        workspace->elements[n_elements++] = new_pin;
      }
    }
  }
  return n_elements;
}

void build_net(
    const struct bcsr_t *__restrict__ A,
    struct level_net *__restrict__ n,
    level_t k,
    index_t n_pins,
    index_t *first_level,
    struct set *__restrict__ workspace)
{
  assert(workspace->capacity >= A->nb);
  
  // Manually add 0-level vertices
  n->n_levels = k;
  n->levels = _ALLOC_ ((k + 2) * sizeof (index_t));
  n->levels[0] = 0;
  n->levels[1] = n_pins;

  // Load first level into workspace (flags)
  index_t i;
  for (i = 0; i < n_pins; ++i)
  {
    workspace->elements[i] = first_level[i];
    workspace->flags[first_level[i]] = 1;
  }

  // Extend closure levels 1 through k
  level_t l;
  for (l = 1; l <= k; l++)
    n->levels[l+1] = extend_net(workspace, A, n->levels[l-1], n->levels[l]);

  // Save result and clear workspace flags for next time
  n->n_pins = n->levels[k+1];
  n->pins = _ALLOC_ (n->n_pins * sizeof (index_t));
  for (i = 0; i < n->n_pins; ++i)
  {
    n->pins[i] = workspace->elements[i];
    workspace->flags[workspace->elements[i]] = 0;
  }
}

void build_net_2x(
    const struct bcsr_t *__restrict__ A1,
    const struct bcsr_t *__restrict__ A2,
    struct level_net *__restrict__ n,
    level_t k,
    index_t n_pins,
    index_t *first_level,
    struct set *__restrict__ workspace)
{
  assert(workspace->capacity >= A1->nb);
  assert(workspace->capacity >= A2->nb);
  
  // Manually add 0-level vertices:
  n->n_levels = k;
  n->levels = _ALLOC_ ((k + 2) * sizeof (index_t));
  n->levels[0] = 0;
  n->levels[1] = n_pins;

  // Load first level into workspace (flags)
  index_t i;
  for (i = 0; i < n_pins; ++i)
  {
    workspace->elements[i] = first_level[i];
    workspace->flags[first_level[i]] = 1;
  }

  // Extend closure levels 1 through k
  index_t prev = 0;
  level_t l;
  for (l = 1; l <= k; l++) {
    index_t next = extend_net(workspace, A1, n->levels[l-1], n->levels[l]);
    n->levels[l+1] = extend_net(workspace, A2, prev, next);
    prev = next;
  }

  // Save result and clear workspace flags for next time
  n->n_pins = n->levels[k+1];
  n->pins = _ALLOC_ (n->n_pins * sizeof (index_t));
  for (i = 0; i < n->n_pins; ++i)
  {
    n->pins[i] = workspace->elements[i];
    workspace->flags[workspace->elements[i]] = 0;
  }
}

void partition_matrix_hypergraph(
  const struct bcsr_t *A,
  index_t rows,
  level_t k,
  part_id_t n_parts,
  part_id_t *row_to_part,
  index_t *part_sizes,
  count_t *cut)
{
  index_t i;

  // Compute k-level column nets on A (row nets on A^T)
  struct level_net *nets = _ALLOC_(rows * sizeof(struct level_net));
  size_t n_pins = 0;
  struct bcsr_t AT;
  struct set workspace;
  bcsr_structure_transpose(&AT, A, rows);
  workspace_init(&workspace, rows);
  for (i = 0; i < rows; ++i)
  {
    build_net(&AT, &nets[i], k, 1, &i, &workspace);
    n_pins += nets[i].n_pins;
    if (n_pins > INT_MAX)
    {
      fprintf(stderr, "too many pins\n");
      abort();
    }
  }
  workspace_free(&workspace);
  bcsr_free(&AT);
 
  // Convert nets to hypergraph
  index_t *pins = _ALLOC_(n_pins * sizeof(index_t));
  pin_t *netptr = _ALLOC_((rows + 1) * sizeof(pin_t));
  pin_t cur_pin = 0;
  for (i = 0; i < rows; ++i)
  {
    netptr[i] = cur_pin;
    memcpy(pins + cur_pin, nets[i].pins, nets[i].n_pins * sizeof(index_t));
    cur_pin += nets[i].n_pins;
    _FREE_(nets[i].levels);
    _FREE_(nets[i].pins);
  }
  netptr[i] = cur_pin;
  _FREE_(nets);

  // Call PaToH
  PaToH_Parameters args;
  PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);
  args._k = n_parts;
  PaToH_Part(&args, rows, rows, 0, 0, NULL, NULL, netptr, pins, NULL, row_to_part, part_sizes, cut);
  PaToH_Free();

  _FREE_(netptr);
  _FREE_(pins);
}

void build_explicit_block(
  const struct bcsr_t *A,
  index_t *part_rows,
  index_t part_size,
  struct set *workspace,
  level_t k,
  AkxBlock *block)
{
  index_t i, j;

  // Workspace for (block) column permutation
  index_t *__restrict__ perm = _ALLOC_ (A->nb * sizeof(index_t));

  block->k = k;

  block->schedule = _ALLOC_ (k * sizeof(index_t));

  struct level_net n;
  build_net(A, &n, k, part_size, part_rows, workspace);

  // Copy and permute bcsr_t matrix into block->A_part
  // Copy schedule into some structure inside akx_thead_block
  // Write driver for do_akx (pthread call)

  // Explicitly partition A in block rows according to the permutation defined by the level_net
  struct bcsr_t *A_part = &block->A_part;
  A_part->mb = n.levels[k] - n.levels[0];
  A_part->nb = n.levels[k+1] - n.levels[0];
  A_part->b_m = A->b_m;
  A_part->b_n = A->b_n;
  A_part->b_transpose = A->b_transpose;

  // Count nnz and identify permutation
  A_part->nnzb = 0;
  for (i = n.levels[0]; i < n.levels[k]; ++i)
  {
    index_t browidx = n.pins[i];
    perm[browidx] = i;
    A_part->nnzb += A->browptr[browidx + 1] - A->browptr[browidx];
  }
  for (; i < n.levels[k+1]; ++i)
  {
    index_t browidx = n.pins[i];
    perm[browidx] = i;
  }

  A_part->browptr = _ALLOC_ ((A_part->mb + 1) * sizeof (index_t));
  A_part->bcolidx = _ALLOC_ (A_part->nnzb * sizeof (index_t) );
  A_part->bvalues = _ALLOC_ (A_part->nnzb * A_part->b_m * A_part->b_n * sizeof (value_t) );
  A_part->browptr_comp = 0;
  A_part->bcolidx_comp = 0;

  // Relabel bcolidx and permute values accordingly
  // TODO: consider sparse->dense->permute->sparse
  nnz_t cur_nnzb = 0;
  nnz_t cur_brow_begin = 0;
  level_t l;
  for (l = 0; l < k; ++l)
  {
    for (i = n.levels[l]; i < n.levels[l+1]; ++i)
    {
      index_t browidx = n.pins[i];
      A_part->browptr[ i ] = cur_brow_begin;

      for (j = A->browptr[browidx]; j < A->browptr[browidx + 1]; ++j)
      {
        // Use insertion sort to apply the symmetric permutation to the columns
        index_t tmp = cur_nnzb - 1;
        while ( tmp >= cur_brow_begin && A_part->bcolidx[ tmp ] > perm [ A->bcolidx[j]] )
        {
          A_part->bcolidx [tmp + 1] = A_part->bcolidx[ tmp ];
          memcpy(&A_part->bvalues[(tmp+1) * A_part->b_m * A_part->b_n ],
            &A_part->bvalues[ tmp * A_part->b_m * A_part->b_n ],
            A_part->b_m * A_part->b_n * sizeof(value_t));
          --tmp;
        }
        A_part->bcolidx [tmp + 1] = perm [ A->bcolidx[j]];
        memcpy((void*) &A_part->bvalues[(tmp+1)* A_part->b_m * A_part->b_n ],
            (void*) &A->bvalues[j * A->b_m * A->b_n ],
            A_part->b_m * A_part->b_n * sizeof (value_t) );
        ++cur_nnzb;
      }
      cur_brow_begin = cur_nnzb;
    }
    block->schedule[k - l - 1] = i * A->b_m;
  }
  A_part->browptr[i] = cur_nnzb;
  A_part->nnzb = cur_nnzb;
  block->perm_size = n.levels[k+1] - n.levels[0];
  block->perm = n.pins;

  _FREE_ (n.levels);
  block->symmetric_opt = 0;

  _FREE_ (perm);
}

void make_implicit_blocks (
    AkxImplicitSeq *__restrict__ imp,
    AkxBlock *__restrict__ block,
    struct set *workspace,
    struct partition_data *p,
    part_id_t nblocks,
    int stanza)
{
  index_t i, j;
  level_t l;

  level_t k = block->k;

  assert(block->A_part.b_m == block->A_part.b_n);

  imp->k = k;
  imp->mb = block->A_part.mb;
  imp->nblocks = nblocks;
  imp->stanza = stanza;

  // Count number of computations done in this thread block,
  // and make room for worst-case computation sequence array
  i = 0;
  for (l = 0; l < k; l++)
    i += (block->schedule[l] + block->A_part.b_m - 1) / block->A_part.b_m;
  imp->level_start = _ALLOC_ ((nblocks * k + 1) * sizeof(index_t));
  imp->computation_seq = _ALLOC_ (i*2 * sizeof(index_t));

  level_t *computed_level = _ALLOC_ (block->A_part.mb * sizeof(level_t));
  memset(computed_level, 0, block->A_part.mb * sizeof(level_t));

  // Make a copy of the thread block (structure only) with outside dependencies removed.
  // Register tiling causes the set of dependencies to grow faster at each level, and
  // we don't want this to result in going outside the thread block
  struct bcsr_t A_temp;
  A_temp.mb = block->A_part.mb;
  A_temp.nb = block->A_part.mb;
  A_temp.b_m = 0;
  A_temp.b_n = 0;
  A_temp.b_transpose = block->A_part.b_transpose;
  A_temp.browptr = _ALLOC_ (sizeof(index_t) * (A_temp.mb + 1));
  A_temp.bcolidx = _ALLOC_ (sizeof(nnz_t) * block->A_part.nnzb);
  A_temp.bvalues = NULL;
  nnz_t n = 0;
  i = 0;
  for (i = 0; i < A_temp.mb; i++)
  {
    nnz_t j;
    A_temp.browptr[i] = n;
    for (j = block->A_part.browptr[i]; j != block->A_part.browptr[i+1]; j++)
      if (block->A_part.bcolidx[j] < A_temp.nb)
        A_temp.bcolidx[n++] = block->A_part.bcolidx[j];
  }
  A_temp.browptr[i] = n;
  A_temp.nnzb = n;

  struct bcsr_t AT_temp;
  if (block->symmetric_opt)
    bcsr_structure_transpose(&AT_temp, &A_temp, A_temp.mb);

  part_id_t ibn;
  i = 0;
  for (ibn = 0; ibn < nblocks; ibn++)
  {
    // Compute dependencies of cache block
    struct level_net cbn;
    if (block->symmetric_opt)
    {
      // Computation of row i at level l+1 depends on computation of row j at level l
      // iff i and j share any element in common, so build net of A^T * A
      build_net_2x(&A_temp, &AT_temp, &cbn, k,
                   p->ptr[ibn + 1] - p->ptr[ibn], &p->part_to_row[p->ptr[ibn]],
                   workspace);
    }
    else
    {
      build_net(&A_temp, &cbn, k,
                p->ptr[ibn + 1] - p->ptr[ibn], &p->part_to_row[p->ptr[ibn]],
                workspace);
    }

    for (l = 0; l < k; l++)
    {
      // It is still not necessary to compute values outside the thread block,
      // even though register tiling can make it look like they're needed
      index_t limit = (block->schedule[l] + block->A_part.b_m - 1) / block->A_part.b_m;

      imp->level_start[ibn*k+l] = i;
      if (stanza)
      {
        // Stanza encoding - array of (start, end) pairs
        index_t next = -1;
        for (j = 0; j < cbn.levels[k-l]; j++)
        {
          index_t localindex = cbn.pins[j];
          if (localindex < limit && computed_level[localindex] == l)
          {
            if (localindex != next)
            {
              imp->computation_seq[i++] = localindex;
              imp->computation_seq[i++] = localindex;
            }
            imp->computation_seq[i-1]++;
            next = localindex + 1;
            computed_level[localindex]++;
          }
        }
      }
      else
      {
        for (j = 0; j < cbn.levels[k-l]; j++)
        {
          index_t localindex = cbn.pins[j];
          if (localindex < limit && computed_level[localindex] == l)
          {
            imp->computation_seq[i++] = localindex;
            computed_level[localindex]++;
          }
        }
      }
    }
    _FREE_ (cbn.levels);
    _FREE_ (cbn.pins);
  }

  if (block->symmetric_opt)
    bcsr_free(&AT_temp);
  bcsr_free(&A_temp);

  imp->level_start[nblocks*k] = i;
  _FREE_ (computed_level);
}

#if 0
void print_sp_matrix (const struct bcsr_t *A, index_t max_brow)
{
  fprintf (stderr, "bcsr_t matrix:\n\t(mb, nb) = (%d, %d), (b_m, b_n) = (%d, %d), nnzb = %d\n", A->mb, A->nb, A->b_m, A->b_n, A->nnzb);
  index_t a, b, i, j;
  if (max_brow <= 0)
    max_brow = A->mb;
  fprintf (stderr, "\tbrowptr: (");
  for (i = 0; i < max_brow; ++i)
    fprintf (stderr, " %d", A->browptr[i]);
  fprintf(stderr, " )\n\tbcolidx: (");
  for (i = 0; i < max_brow; ++i)
  {
    for (j = A->browptr[i]; j < A->browptr[i+1]; ++j)
      fprintf (stderr, " %d", A->bcolidx[j]);
    fprintf(stderr, (i == max_brow - 1 ? " )\n\tbvalues: (" : "\n\t          ")); 
  }
  for (i = 0; i < max_brow; ++i)
    for (b = 0; b < A->b_m; ++b)
    {
      for (j = A->browptr[i]; j < A->browptr[i+1]; ++j)
	for (a = 0; a < A->b_n; ++a)
	  fprintf (stderr, " %.2g", A->bvalues[j*A->b_m*A->b_n + b*A->b_m + a]);
      fprintf (stderr, (i == max_brow - 1 ? " )\n" : "\n\t          "));
    }  
}
#endif

static PyTypeObject AkxBlock_Type;
static PyTypeObject AkxImplicitSeq_Type;

void matrix_from_arrays(struct bcsr_t *A, PyArrayObject *indptr, PyArrayObject *indices, PyArrayObject *data)
{
  A->mb = A->nb = indptr->dimensions[0] - 1;
  A->b_m = data->nd > 2 ? data->dimensions[1] : 1;
  A->b_n = data->nd > 2 ? data->dimensions[2] : 1;
  A->b_transpose = 0;
  A->browptr_comp = 0;
  A->bcolidx_comp = 0;
  A->nnzb = data->dimensions[0];
  A->browptr = (index_t *)indptr->data;
  A->bcolidx = (index_t *)indices->data;
  A->bvalues = (value_t *)data->data;
}

int make_partition_data(struct partition_data *p, PyObject *partition, index_t rows, int n_parts)
{
  // ptr [ <part id> ] = <offset within part_to_row array corresponding to the beginning of part id's rows>
  // part_to_row [ ptr[ <part id> ] + <local row index>] = <global row index>

  p->part_to_row = _ALLOC_ (rows * sizeof (index_t));
  p->ptr = _ALLOC_ ((n_parts + 1) * sizeof (index_t));

  part_id_t pp;
  index_t i;

  if (partition == Py_None)
  {
    // Just divide into equally-sized contiguous partitions
    for (pp = 0; pp < n_parts; pp++)
    {
      index_t start = pp * rows / n_parts;
      index_t end = (pp+1) * rows / n_parts;
      for (i = start; i < end; i++)
        p->part_to_row[i] = i;
      p->ptr[pp] = start;
    }
    p->ptr[n_parts] = rows;
    return 1;
  }

  PyObject *parray = PyArray_ContiguousFromAny(partition, NPY_INT, 1, 1);
  if (parray)
  {
    // row_to_part [ <global row index> ] = <part id who owns that row> (this is filled in by PaToH)
    int *row_to_part = PyArray_DATA(parray);
    if (PyArray_DIM(parray, 0) != rows)
    {
      PyErr_Format(PyExc_IndexError, "partition array has wrong size, got %d expected %d", (int)PyArray_DIM(parray, 0), rows);
      goto fail;
    }

    // Count partition sizes
    memset(p->ptr, 0, (n_parts + 1) * sizeof(index_t));
    for (i = 0; i < rows; i++) {
      pp = row_to_part[i];
      if (pp < 0 || pp >= n_parts)
      {
        PyErr_SetString(PyExc_IndexError, "partition array has out-of-bounds value");
        goto fail;
      }
      p->ptr[pp]++;
    }

    // Make cumulative sum so ptr[pp] points to the end of partition pp
    for (pp = 1; pp <= n_parts; pp++)
      p->ptr[pp] += p->ptr[pp-1];

    // Build part_to_row (in the process, ptr[pp] is moved from the end of pp to the start)
    // The global ordering 0,1,2,...,n is preserved within each part
    for (i = rows; --i >= 0; )
      p->part_to_row[--p->ptr[row_to_part[i]]] = i;

    Py_DECREF(parray);
    return 1;
  }

  PyErr_SetString(PyExc_TypeError, "partition argument should be None or array");
fail:
  p->part_to_row = _ALLOC_ (rows * sizeof (index_t));
  p->ptr = _ALLOC_ ((n_parts + 1) * sizeof (index_t));
  return 0;
}

void dest_partition_data ( struct partition_data * p )
{
  _FREE_ (p->part_to_row);
  _FREE_ (p->ptr);
}

PyDoc_STRVAR(tb_partition_doc,
"tb_partition(indptr, indices, data, k, n_parts) -> (partition, sizes, cut)\n"
"\n"
"Call PaToH to compute an approximately optimal partitioning of this\n"
"matrix into thread blocks.");
static PyObject *
Akx_tb_partition(PyObject *self, PyObject *args)
{
  PyArrayObject *indptr, *indices, *data;
  int k, n_parts;
  if (!PyArg_ParseTuple(args, "O!O!O!ii",
      &PyArray_Type, &indptr, &PyArray_Type, &indices, &PyArray_Type, &data,
      &k, &n_parts))
  {
    return NULL;
  }

  struct bcsr_t A;
  matrix_from_arrays(&A, indptr, indices, data);

  npy_intp dim = A.mb;
  PyObject *partition = PyArray_SimpleNew(1, &dim, NPY_INT);
  dim = n_parts;
  PyObject *sizes = PyArray_SimpleNew(1, &dim, NPY_INT);
  count_t cut;
  partition_matrix_hypergraph(&A, A.mb, k, n_parts, PyArray_DATA(partition), PyArray_DATA(sizes), &cut);

  return Py_BuildValue("OOi", partition, sizes, cut);
}

PyDoc_STRVAR(threadblocks_doc,
"threadblocks(indptr, indices, data, k, n_parts, partition)\n\
\n\
Splits the matrix into n_parts thread blocks, optionally using the\n\
provided partitioning. Returns a list of AkxBlock objects, one\n\
representing each thread block.");
static PyObject *
Akx_threadblocks(PyObject *self, PyObject *args)
{
  PyArrayObject *indptr, *indices, *data;
  int k, n_parts;
  PyObject *partition;
  if (!PyArg_ParseTuple(args, "O!O!O!iiO",
      &PyArray_Type, &indptr, &PyArray_Type, &indices, &PyArray_Type, &data,
      &k, &n_parts, &partition))
  {
    return NULL;
  }

  PyObject *ret = PyList_New(n_parts);
  if (ret == NULL)
    return NULL;

  struct bcsr_t A;
  matrix_from_arrays(&A, indptr, indices, data);

  struct partition_data p;
  if (!make_partition_data(&p, partition, A.mb, n_parts))
    return NULL;

  // Workspace for transitive closure
  struct set workspace;
  workspace_init(&workspace, A.nb);

  part_id_t pp;
  for (pp = 0; pp < n_parts; ++pp)
  {
    AkxBlock *block = PyObject_New(AkxBlock, &AkxBlock_Type);
    build_explicit_block(
      &A,
      &p.part_to_row[p.ptr[pp]],
      p.ptr[pp + 1] - p.ptr[pp],
      &workspace,
      k,
      block);
    PyList_SET_ITEM(ret, pp, (PyObject *)block);
  }
  dest_partition_data(&p);
  workspace_free(&workspace);
  return ret;
}


PyDoc_STRVAR(shape_doc,
"block.shape() -> (height, width) of block");
static PyObject *
AkxBlock_shape(AkxBlock *block, PyObject *args)
{
  struct bcsr_t *A = &block->A_part;
  return Py_BuildValue("ii", A->mb * A->b_m, A->nb * A->b_n);
}

PyDoc_STRVAR(nnzb_doc,
"block.nnzb() -> number of nonzero tiles in block");
static PyObject *
AkxBlock_nnzb(AkxBlock *block, PyObject *args)
{
  return PyInt_FromLong(block->A_part.nnzb);
}

PyDoc_STRVAR(schedule_doc,
"block.schedule() -> number of rows used at each level (k integers)");
static PyObject *
AkxBlock_schedule(AkxBlock *block, PyObject *args)
{
  npy_intp size = block->k;
  // TODO: this assumes that index_t is same size as NPY_INT
  PyObject *obj = PyArray_SimpleNewFromData(1, &size, NPY_INT, block->schedule);
  PyArray_FLAGS(obj) &= ~NPY_WRITEABLE;
  return obj;
}

PyDoc_STRVAR(flopcount_doc,
"block.flopcount() -> number of floating point operations\n");
static PyObject *
AkxBlock_flopcount(AkxBlock *block, PyObject *args)
{
  if (block->symmetric_opt)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do flop count after symmetric optimization");
    return NULL;
  }

  long nnz = 0;
  level_t level;
  for (level = 0; level < block->k; level++)
    nnz += block->A_part.browptr[(block->schedule[level] + block->A_part.b_m - 1) / block->A_part.b_m];
  return PyInt_FromLong(2 * nnz * block->A_part.b_m * block->A_part.b_n);
}

PyDoc_STRVAR(variant_doc,
"block.variant() -> tuple describing block format");
static PyObject *
AkxBlock_variant(AkxBlock *block, PyObject *args)
{
  struct bcsr_t *A = &block->A_part;
  return Py_BuildValue("iiiii", A->b_m, A->b_n, A->b_transpose, A->browptr_comp, A->bcolidx_comp);
}

PyDoc_STRVAR(tilecount_doc,
"block.tilecount(b_m, b_n, samples) -> approximate # nonzero tiles\n"
"\n"
"Estimate the number of nonzero tiles the block would contain given a\n"
"tile size of b_m x b_n. More samples gets a more accurate estimate.\n");
static PyObject *
AkxBlock_tilecount(AkxBlock *block, PyObject *args)
{
  int b_m, b_n, samples;
  if (!PyArg_ParseTuple(args, "iii", &b_m, &b_n, &samples))
    return NULL;

  if (block->symmetric_opt)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do tiling after symmetric optimization");
    return NULL;
  }
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do tiling after index compression");
    return NULL;
  }

  struct bcsr_t *A = &block->A_part;
  if (A->b_m != 1 || A->b_n != 1)
  {
    PyErr_SetString(PyExc_ValueError, "block is already tiled");
    return NULL;
  }

  int sampno;
  double tile_count = 0.0;
  index_t row = 0;
  for (sampno = 0; sampno < samples; sampno++)
  {
    nnz_t sample = ((long long)sampno * A->nnzb) / samples;

    // get the row number of this sample
    while (A->browptr[row+1] <= sample)
      row++;

    // get the bounds of the tile containing it
    index_t top = row - (row % b_m);
    index_t bottom = top + b_m;
    if (bottom > A->mb)
      bottom = A->mb;
    index_t left = A->bcolidx[sample] - (A->bcolidx[sample] % b_n);
    index_t right = left + b_n;

    // count nonzeros inside the tile
    nnz_t i = 0;
    nnz_t tile_nnz = 0;
    for (i = A->browptr[top]; i != A->browptr[bottom]; i++)
      if (A->bcolidx[i] >= left && A->bcolidx[i] < right)
        tile_nnz++;

    // the sample accounts for this fraction of a tile
    tile_count += 1.0 / tile_nnz;
  }
  return Py_BuildValue("d", tile_count * A->nnzb / samples);
}

#define MAX_TILE_HEIGHT 16

PyDoc_STRVAR(tile_doc,
"block.tile(b_m, b_n, b_transpose) -> tiled block\n"
"\n"
"Create a copy of the block with a tile height of b_m and tile width\n"
"of b_n. Tiles may be either row-major (b_transpose = False) or\n"
"column-major (b_transpose = True). Input block must be untiled.\n");
static PyObject *
AkxBlock_tile(AkxBlock *block, PyObject *args)
{
  int b_m, b_n, b_transpose;
  if (!PyArg_ParseTuple(args, "iii", &b_m, &b_n, &b_transpose))
    return NULL;

  if (b_m > MAX_TILE_HEIGHT)
  {
    PyErr_SetString(PyExc_ValueError, "tile size too large");
    return NULL;
  }
  if (block->symmetric_opt)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do tiling after symmetric optimization");
    return NULL;
  }
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do tiling after index compression");
    return NULL;
  }

  struct bcsr_t *A = &block->A_part;
  if (A->b_m != 1 || A->b_n != 1)
  {
    PyErr_SetString(PyExc_ValueError, "block is already tiled");
    return NULL;
  }

  index_t top, bottom, left, right;
  index_t curptr[MAX_TILE_HEIGHT];
  nnz_t nnz = 0;

  // Count exact number of nonzero tiles first
  for (top = 0; top < A->mb; top = bottom)
  {
    bottom = top + b_m;
    if (bottom > A->mb)
      bottom = A->mb;
    int height = bottom - top;

    memcpy(curptr, &A->browptr[top], height * sizeof(index_t));
    index_t *endptr = &A->browptr[top + 1];

    while (1) {
      index_t next = A->nb;
      index_t row;

      for (row = 0; row < height; row++)
        if (curptr[row] != endptr[row] && A->bcolidx[curptr[row]] < next)
          next = A->bcolidx[curptr[row]];
      if (next == A->nb)
        break;

      left  = next - (next % b_n);
      right = left + b_n;

      for (row = 0; row < height; row++)
        while (curptr[row] != endptr[row] && A->bcolidx[curptr[row]] < right)
          curptr[row]++;

      nnz++;
    }
  }

  struct bcsr_t Anew;
  Anew.mb = (A->mb + b_m - 1) / b_m;
  Anew.b_m = b_m;
  Anew.nb = (A->nb + b_n - 1) / b_n;
  Anew.b_n = b_n;
  Anew.b_transpose = b_transpose;
  Anew.nnzb = nnz;
  Anew.browptr = _ALLOC_ ((Anew.mb + 1) * sizeof(index_t));
  Anew.bcolidx = _ALLOC_ (Anew.nnzb * sizeof(index_t));
  Anew.bvalues = _ALLOC_ (Anew.nnzb * (b_m * b_n) * sizeof(value_t));
  Anew.browptr_comp = 0;
  Anew.bcolidx_comp = 0;

  index_t *browptr = Anew.browptr;
  nnz = 0;
  for (top = 0; top < A->mb; top = bottom)
  {
    *browptr++ = nnz;

    bottom = top + b_m;
    if (bottom > A->mb)
      bottom = A->mb;
    index_t height = bottom - top;

    memcpy(curptr, &A->browptr[top], height * sizeof(index_t));
    index_t *endptr = &A->browptr[top + 1];

    while (1) {
      index_t next = A->nb;
      index_t row, col;
      value_t *tile;

      for (row = 0; row < height; row++)
        if (curptr[row] != endptr[row] && A->bcolidx[curptr[row]] < next)
          next = A->bcolidx[curptr[row]];
      if (next == A->nb)
        break;

      Anew.bcolidx[nnz] = next / b_n;
      left  = Anew.bcolidx[nnz] * b_n;
      right = left + b_n;

      tile = &Anew.bvalues[nnz * b_m * b_n];
      memset(tile, 0, b_m * b_n * sizeof(value_t));
      for (row = 0; row < height; row++)
      {
        while (curptr[row] != endptr[row] && A->bcolidx[curptr[row]] < right)
        {
          col = A->bcolidx[curptr[row]] - left;
          if (!b_transpose)
            tile[row * b_n + col] = A->bvalues[curptr[row]];
          else
            tile[col * b_m + row] = A->bvalues[curptr[row]];
          curptr[row]++;
        }
      }

      nnz++;
    }
  }
  *browptr = nnz;

  AkxBlock *newblock = PyObject_New(AkxBlock, &AkxBlock_Type);
  newblock->k = block->k;
  newblock->A_part = Anew;
  newblock->schedule = copy_array(block->schedule, block->k * sizeof(index_t));
  newblock->perm_size = block->perm_size;
  newblock->perm = copy_array(block->perm, block->perm_size * sizeof(index_t));
  newblock->symmetric_opt = 0;

  return (PyObject *)newblock;
}

PyDoc_STRVAR(partition_doc,
"block.partition(k, n_parts) -> (partition, sizes, cut)\n"
"\n"
"Call PaToH to compute an approximately optimal partitioning of this\n"
"block's rows into sub-blocks.");
static PyObject *
AkxBlock_partition(AkxBlock *block, PyObject *args)
{
  int k, n_parts;
  if (!PyArg_ParseTuple(args, "ii", &k, &n_parts))
    return NULL;

  if (block->A_part.b_m != block->A_part.b_n)
  {
    PyErr_SetString(PyExc_ValueError, "block tile size not square");
    return NULL;
  }
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do cache blocking after index compression");
    return NULL;
  }

  index_t b_m = block->A_part.b_m;
  index_t rows = (block->schedule[block->k-1] + b_m - 1) / b_m; // Non-ghost entries only

  npy_intp dim = rows;
  PyObject *partition = PyArray_SimpleNew(1, &dim, NPY_INT);
  dim = n_parts;
  PyObject *sizes = PyArray_SimpleNew(1, &dim, NPY_INT);
  count_t cut;
  partition_matrix_hypergraph(&block->A_part, rows, k, n_parts, PyArray_DATA(partition), PyArray_DATA(sizes), &cut);

  return Py_BuildValue("OOi", partition, sizes, cut);
}

PyDoc_STRVAR(split_doc,
"block.split(n_parts, partition) -> list of blocks\n"
"\n"
"Split a block into multiple smaller blocks. A partitioning may be\n"
"given to specify which sub-block to put each row into.");
static PyObject *
AkxBlock_split(AkxBlock *block, PyObject *args)
{
  int n_parts;
  PyObject *partition;
  if (!PyArg_ParseTuple(args, "iO", &n_parts, &partition))
    return NULL;

  if (n_parts < 2)
  {
    PyErr_SetString(PyExc_ValueError, "n_parts < 2");
    return NULL;
  }

  if (block->A_part.b_m != 1 || block->A_part.b_n != 1)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do explicit cache blocking after tiling");
    return NULL;
  }
  if (block->symmetric_opt)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do explicit cache blocking after symmetric optimization");
    return NULL;
  }
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do cache blocking after index compression");
    return NULL;
  }

  PyObject *ret = PyList_New(n_parts);
  if (ret == NULL)
    return NULL;

  index_t b_m = block->A_part.b_m;
  index_t rows = (block->schedule[block->k-1] + b_m - 1) / b_m; // Non-ghost entries only
  struct partition_data p;
  if (!make_partition_data(&p, partition, rows, n_parts))
  {
    Py_DECREF(ret);
    return NULL;
  }

  // Workspace for transitive closure
  struct set workspace;
  workspace_init(&workspace, block->A_part.nb);

  part_id_t pp;
  for (pp = 0; pp < n_parts; ++pp)
  {
    AkxBlock *newblock = PyObject_New(AkxBlock, &AkxBlock_Type);
    build_explicit_block(
      &block->A_part,                // matrix to take a partition of
      &p.part_to_row[p.ptr[pp]], // partition rows array
      p.ptr[pp + 1] - p.ptr[pp], // partition rows array size
      &workspace,
      block->k,
      newblock);

    // Translate permutation from local to global indices
    index_t i;
    for (i = 0; i < newblock->perm_size; i++)
      newblock->perm[i] = block->perm[newblock->perm[i]];
    PyList_SET_ITEM(ret, pp, (PyObject *)newblock);
  }
  workspace_free(&workspace);
  dest_partition_data(&p);
  return ret;
}

PyDoc_STRVAR(symm_opt_doc,
"block.symm_opt() -> new block\n"
"\n"
"Create a copy of the block in upper-triangular representation (all\n"
"tiles below the diagonal removed). Tile size must be square.\n");
static PyObject *
AkxBlock_symm_opt(AkxBlock *block, PyObject *args)
{
  if (block->A_part.b_m != block->A_part.b_n)
  {
    PyErr_SetString(PyExc_ValueError, "block tile size not square");
    return NULL;
  }
  if (block->symmetric_opt)
  {
    PyErr_SetString(PyExc_ValueError, "symmetric optimization already done");
    return NULL;
  }
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do symmetric optimization after index compression");
    return NULL;
  }

  AkxBlock *newblock = PyObject_New(AkxBlock, &AkxBlock_Type);
  newblock->k = block->k;
  bcsr_upper_triangle(&newblock->A_part, &block->A_part);
  newblock->symmetric_opt = 1;
  newblock->schedule = copy_array(block->schedule, block->k * sizeof(index_t));
  newblock->perm_size = block->perm_size;
  newblock->perm = copy_array(block->perm, block->perm_size * sizeof(index_t));

  return (PyObject *)newblock;
}

PyDoc_STRVAR(implicitblocks_doc,
"block.implicitblocks(n_parts, partition, stanza) -> AkxImplicitSeq\n"
"\n"
"Create an object representing a reordering of computation within\n"
"the block (implicit cache blocks). Tile size must be square.");
static PyObject *
AkxBlock_implicitblocks(AkxBlock *block, PyObject *args)
{
  int n_parts, stanza;
  PyObject *partition;
  if (!PyArg_ParseTuple(args, "iOi", &n_parts, &partition, &stanza))
    return NULL;

  if (block->A_part.b_m != block->A_part.b_n)
  {
    PyErr_SetString(PyExc_ValueError, "block tile size not square");
    return NULL;
  }
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "cannot do cache blocking after index compression");
    return NULL;
  }

  // Partition the thread block into cache blocks
  index_t b_m = block->A_part.b_m;
  index_t rows = (block->schedule[block->k-1] + b_m - 1) / b_m; // Non-ghost entries only
  struct partition_data p;
  if (!make_partition_data(&p, partition, rows, n_parts))
    return NULL;

  // Workspace for transitive closure
  struct set workspace;
  workspace_init(&workspace, block->A_part.nb);

  AkxImplicitSeq *imp = PyObject_New(AkxImplicitSeq, &AkxImplicitSeq_Type);
  make_implicit_blocks(imp, block, &workspace, &p, n_parts, stanza);

  workspace_free(&workspace);

  dest_partition_data(&p);

  return (PyObject *)imp;
}

static uint16_t *
index_comp(index_t *array, size_t size)
{
  size_t i;
  uint16_t *out = _ALLOC_(sizeof(uint16_t) * size);
  for (i = 0; i < size; i++)
  {
    assert(array[i] >= 0 && array[i] < 65536);
    out[i] = array[i];
  }
  return out;
}

PyDoc_STRVAR(index_comp_doc,
"block.index_comp() -> new block\n"
"\n"
"Create a copy of the block with index arrays compressed to 16 bits\n"
"per index if possible. If not, the original block is returned.\n");
static PyObject *
AkxBlock_index_comp(AkxBlock *block, PyObject *args)
{
  if (block->A_part.browptr_comp || block->A_part.bcolidx_comp)
  {
    PyErr_SetString(PyExc_ValueError, "index compression already done");
    return NULL;
  }
  if (block->A_part.nnzb >= 65536 && block->A_part.nb > 65536)
  {
    Py_INCREF(block);
    return (PyObject *)block;
  }

  AkxBlock *newblock = PyObject_New(AkxBlock, &AkxBlock_Type);
  newblock->k = block->k;
  newblock->A_part.mb = block->A_part.mb;
  newblock->A_part.nb = block->A_part.nb;
  newblock->A_part.b_m = block->A_part.b_m;
  newblock->A_part.b_n = block->A_part.b_n;
  newblock->A_part.b_transpose = block->A_part.b_transpose;
  newblock->A_part.nnzb = block->A_part.nnzb;
  if (block->A_part.nnzb >= 65536)
  {
    newblock->A_part.browptr = copy_array(block->A_part.browptr, sizeof(index_t) * (block->A_part.mb + 1));
    newblock->A_part.browptr_comp = 0;
  }
  else
  {
    newblock->A_part.browptr16 = index_comp(block->A_part.browptr, block->A_part.mb + 1);
    newblock->A_part.browptr_comp = 1;
  }
  if (block->A_part.nb > 65536)
  {
    newblock->A_part.bcolidx = copy_array(block->A_part.bcolidx, sizeof(index_t) * block->A_part.nnzb);
    newblock->A_part.bcolidx_comp = 0;
  }
  else
  {
    newblock->A_part.bcolidx16 = index_comp(block->A_part.bcolidx, block->A_part.nnzb);
    newblock->A_part.bcolidx_comp = 1;
  }
  newblock->A_part.bvalues = copy_array(block->A_part.bvalues, 
    sizeof(value_t) * block->A_part.b_m * block->A_part.b_n * block->A_part.nnzb);
  newblock->schedule = copy_array(block->schedule, block->k * sizeof(index_t));
  newblock->perm_size = block->perm_size;
  newblock->perm = copy_array(block->perm, block->perm_size * sizeof(index_t));
  newblock->symmetric_opt = block->symmetric_opt;

  return (PyObject *)newblock;
}

static void
AkxBlock_dealloc(AkxBlock *block)
{
  bcsr_free(&block->A_part);
  _FREE_(block->schedule);
  _FREE_(block->perm);
  PyObject_Del(block);
}

static void
AkxImplicitSeq_dealloc(AkxImplicitSeq *imp)
{
  _FREE_(imp->level_start);
  _FREE_(imp->computation_seq);
  PyObject_Del(imp);
}

#define METHOD(name, flags) { #name, (PyCFunction)AkxBlock_##name, flags, name##_doc },
static PyMethodDef AkxBlock_methods[] = {
  METHOD(shape, METH_VARARGS)
  METHOD(nnzb, METH_VARARGS)
  METHOD(schedule, METH_VARARGS)
  METHOD(flopcount, METH_VARARGS)
  METHOD(variant, METH_VARARGS)
  METHOD(tilecount, METH_VARARGS)
  METHOD(tile, METH_VARARGS)
  METHOD(partition, METH_VARARGS)
  METHOD(split, METH_VARARGS)
  METHOD(symm_opt, METH_VARARGS)
  METHOD(implicitblocks, METH_VARARGS)
  METHOD(index_comp, METH_VARARGS)
  { NULL, NULL, 0, NULL }
};
#undef METHOD

PyDoc_STRVAR(AkxBlock_doc,
"The AkxBlock object represents a portion of the matrix powers\n"
"computation, and contains all data necessary to complete that portion\n"
"independent of any other AkxBlock.");
static PyTypeObject AkxBlock_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                          /*tp_size*/
	"akx.AkxBlock",             /*tp_name*/
	sizeof(AkxBlock),           /*tp_basicsize*/
	0,                          /*tp_itemsize*/
	/* methods */
	(destructor)AkxBlock_dealloc,     /*tp_dealloc*/
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
	AkxBlock_doc,               /*tp_doc*/
	0,                          /*tp_traverse*/
	0,                          /*tp_clear*/
	0,                          /*tp_richcompare*/
	0,                          /*tp_weaklistoffset*/
	0,                          /*tp_iter*/
	0,                          /*tp_iternext*/
	AkxBlock_methods,           /*tp_methods*/
};

PyDoc_STRVAR(AkxImplicitSeq_doc,
"Represents an implicit cache blocking of a block.");
static PyTypeObject AkxImplicitSeq_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                          /*tp_size*/
	"akx.AkxImplicitSeq",       /*tp_name*/
	sizeof(AkxImplicitSeq),     /*tp_basicsize*/
	0,                          /*tp_itemsize*/
	/* methods */
	(destructor)AkxImplicitSeq_dealloc, /*tp_dealloc*/
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
	AkxImplicitSeq_doc,         /*tp_doc*/
};

static PyMethodDef methods[] = {
	{ "tb_partition", Akx_tb_partition, METH_VARARGS, tb_partition_doc },
	{ "threadblocks", Akx_threadblocks, METH_VARARGS, threadblocks_doc },
	{ NULL, NULL, 0, NULL }
};

PyDoc_STRVAR(module_doc,
"Module for matrix data manipulation");
PyMODINIT_FUNC
init_akx_static(void)
{
  PyObject *module = Py_InitModule3("_akx_static", methods, module_doc);
  if (!module)
    return;

  if (PyType_Ready(&AkxBlock_Type) < 0)
    return;
  Py_INCREF(&AkxBlock_Type);
  PyModule_AddObject(module, "AkxBlock", (PyObject *)&AkxBlock_Type);

  if (PyType_Ready(&AkxImplicitSeq_Type) < 0)
    return;
  Py_INCREF(&AkxImplicitSeq_Type);
  PyModule_AddObject(module, "AkxImplicitSeq", (PyObject *)&AkxImplicitSeq_Type);

  import_array();
}
