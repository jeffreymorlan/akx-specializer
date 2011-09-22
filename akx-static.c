#undef NDEBUG // for assertions
//#define TRACE
#include <Python.h>
#include <numpy/arrayobject.h>

// C headers
#include <stdlib.h> // for NULL, atoi
#include <string.h> // for memset
#include <stdio.h>  // for fprintf
#include <assert.h> // for assert

#ifdef __SSE3__ // will be defined when compiling, but not when checking dependencies
#include <pmmintrin.h>
#endif

// PaToH hypergraph partitioning library
#include "patoh.h"

#include "akx.h"

void dest_hypergraph ( struct hypergraph * h)
{
  _FREE_ (h->netptr);
  _FREE_ (h->pins);
}

void extend_net ( const struct bcsr_t *__restrict__ A, struct level_net *__restrict__ n, level_t depth, struct set *__restrict__ workspace )
{
  if (depth <= 0)
    return;

  // Load n into workspace (flags)
  index_t i;
  for (i = 0; i < n->n_pins; ++i)
    workspace->flags[ n->pins[i] ] = 1;
  workspace->n_elements = 0;

  level_t l;

  index_t frontier_begin = n->levels[n->cur_level - 1];
  index_t frontier_end   = n->levels[n->cur_level];
  index_t f;
  for (l = 0; l < depth; ++l)
  {
    for (f = frontier_begin; f < frontier_end; ++f)
    {
      // Add col-indices of some row of A to workspace
      index_t row_to_add = ( l == 0 ? n->pins[f] : workspace->elements[f] );
      index_t colidx_start = A->browptr[row_to_add];
      index_t colidx_end   = A->browptr[row_to_add + 1];
      index_t i;
      for (i = colidx_start; i < colidx_end; ++i)
      {
	index_t new_pin = A->bcolidx[i];
	if ( workspace->flags[new_pin] != 1 )
	{
	  workspace->flags[new_pin] = 1;
	  workspace->elements[workspace->n_elements] = new_pin;
	  workspace->n_elements++;
	}
      }
    }
    frontier_begin = ( l == 0 ? 0 : frontier_end );
    frontier_end   = workspace->n_elements;
    n->levels[++(n->cur_level)] = n->n_pins + workspace->n_elements;
  }

  // Now update n:
  index_t * new_pins;
  _ALLOC_ ( (void**) &new_pins, (n->n_pins + workspace->n_elements) * sizeof (index_t) );

  // NOTE: aligned copy
  _COPY_AL_ ( n->pins, new_pins, n->n_pins * sizeof (index_t) );

  // NOTE: unaligned copy
  _COPY_UAL_ ( workspace->elements, new_pins + n->n_pins, workspace->n_elements * sizeof (index_t));

  _FREE_ ( n->pins );
  n->pins = new_pins;
  n->n_pins += workspace->n_elements;

  for (i = 0; i < n->n_pins; ++i)
    workspace->flags[ n->pins[i] ] = 0;
}

void build_net ( const struct bcsr_t *__restrict__ A, struct level_net *__restrict__ n, level_t k,
  index_t n_pins, index_t *first_level, struct set *__restrict__ workspace )
{
  // Manually add 0-level vertices:
  n->n_pins = n_pins;
  _ALLOC_ ( (void**) &n->pins, n_pins * sizeof (index_t) );
  _COPY_UAL_ ( first_level, n->pins, n_pins * sizeof (index_t) );
  n->n_levels = k;
  _ALLOC_ ( (void**) &n->levels, (k + 2) * sizeof (index_t) );
  n->levels[0] = 0;
  n->levels[1] = n_pins;
  n->cur_level = 1;

  // Extend closure levels 1 through k
  extend_net(A, n, k, workspace);
}

// A must be a square matrix
void compute_closure ( const struct bcsr_t* A, struct level_net** nets, level_t k )
{
  // Build nets
  _ALLOC_ ( (void**) nets, A->nb * sizeof (struct level_net) );

  // Workspace:
  struct set workspace;
  workspace.capacity = A->nb;
  _ALLOC_ ((void**) &workspace.elements, A->nb * sizeof (index_t));
  _ALLOC_ ((void**) &workspace.flags,    A->nb * sizeof (flag_t));
  memset(workspace.flags, 0, A->nb * sizeof (flag_t));

  index_t i;
  for (i = 0; i < A->nb; ++i)
  {
#ifdef TRACE
    if (!(i & 1023))
    fprintf (stderr, "\r = computing %d-level closure for row-net %d of %d ...",
      k, i, A->nb);
#endif
    build_net (A, *nets + i, k, 1, &i, &workspace);
  }
#ifdef TRACE
  fprintf (stderr, "\n");
#endif

  _FREE_ ( workspace.elements );
  _FREE_ ( workspace.flags );
}

void nets_to_netlist ( const struct level_net* nets, index_t n_nets, struct hypergraph* netlist )
{
  pin_t n_pins = 0;
  index_t i;

  for (i = 0; i < n_nets; ++i)
    n_pins += nets[i].n_pins;

  netlist->n_nets = n_nets;
  netlist->n_pins = n_pins;

  if (n_pins <= 0)
  {
    fprintf (stderr, "jesus christ.");
    // this is a super error condition. shouldn't just
    return;
  }

  _ALLOC_ ((void**) &netlist->pins, n_pins * sizeof (index_t));
  _ALLOC_ ((void**) &netlist->netptr, (n_nets + 1) * sizeof (pin_t) );

  // NOTE: unaligned copy ... could do the first one aligned
  // TODO: levels too!
  netlist->netptr[0] = 0; // TODO: This is a PaToH hack
  pin_t cur_pin = 0;
  for (i = 0; i < n_nets; ++i)
  {
    _COPY_UAL_ ((void*) nets[i].pins, (void*) (netlist->pins + cur_pin), nets[i].n_pins * sizeof (index_t));
    cur_pin += nets[i].n_pins;
    netlist->netptr[i + 1] = cur_pin;
  }
}

// cant be a const hypergraph b/c of PaToH fn prototype
void compute_partition ( struct hypergraph *__restrict__ netlist, part_id_t n_parts, struct partition_data *__restrict__ p )
{
  index_t i;
  p->n = netlist->n_nets;
  p->n_parts = n_parts;
  _ALLOC_ ((void**) &p->row_to_part, netlist->n_nets * sizeof (part_id_t));
  _ALLOC_ ((void**) &p->part_to_row, netlist->n_nets * sizeof (index_t));
  _ALLOC_ ((void**) &p->row_to_part_row, netlist->n_nets * sizeof (index_t));
  _ALLOC_ ((void**) &p->ptr, (n_parts + 1) * sizeof (index_t));

  p->ptr[0] = 0;
  if (n_parts > 1)
  {
  PaToH_Parameters args;
  PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);
  args._k = n_parts;
  count_t cut;
#ifdef TRACE
  fprintf (stderr, " = calling PaToH_Part () ...\n");
#endif
  /*fprintf (stderr, ""
      "netptr: [%p, %p]\n"
      "pins  : [%p, %p]\n"
      "rtp:    [%p, %p]\n"
      "ptr:    [%p, %p]\n"
      "nnets: %d\n",
      netlist->netptr, netlist->netptr + netlist->n_nets,
      netlist->pins, netlist->pins + netlist->netptr[netlist->n_nets + 1] - 1,
      p->row_to_part, p->row_to_part + netlist->n_nets,
      p->ptr, p->ptr + n_parts + 1, netlist->n_nets);*/
  PaToH_Part ( &args, netlist->n_nets, netlist->n_nets, 0, 0, NULL, NULL, netlist->netptr, netlist->pins, NULL, p->row_to_part, p->ptr + 1, &cut );

  PaToH_Free ();
  }
  else
  {
    p->ptr[1] = netlist->n_nets;
    memset(p->row_to_part, 0, netlist->n_nets * sizeof(part_id_t));
  }

  part_id_t pp;
  // Assume PaToH saved partweights into p->ptr[1:n_parts]
#ifdef DEBUG
  index_t sum = 0;
  for (pp = 0; pp <= n_parts; ++pp)
    sum += p->ptr[pp];
  assert (sum == netlist->n_nets);
#endif

#ifdef TRACE
  fprintf (stderr, " = processing PaToH partition ...\n");
#endif
  // Auxiliary array, TODO: keep around
  index_t *__restrict__ offset;
  _ALLOC_ ((void**) &offset, n_parts * sizeof (index_t));
  memset(offset, 0, n_parts * sizeof(index_t));

  // 4 confusing arrays:
  // row_to_part [ <global row index> ] = <part id who owns that row> (this is filled in by PaToH)
  // row_to_part_row[ <global row index> ] = <local row index (within the responsible part)>
  // ptr [ <part id> ] = <offset within part_to_row array corresponding to the beginning of part id's rows>
  // part_to_row [ ptr[ <part id> ] + <local row index>] = <global row index>

  // Build ptr by integrating the values filled in by PaToH's partweights, so that ptr records cumulative weight rather than weight-per-part
  for ( pp = 1; pp <= n_parts; ++pp )
    p->ptr[pp] += p->ptr[pp-1];

  // Build row_to_part_row and part_to_row
  // The global ordering 0,1,2,...,n is preserved within each part because of this for loop
  for (i = 0; i < p->n; ++i)
  {
    // TODO: unroll
    index_t part_tmp = p->row_to_part [i];
    index_t offset_tmp = offset[ part_tmp ]++;
    p->row_to_part_row [ i ] = offset_tmp;
    p->part_to_row [ p->ptr[part_tmp] + offset_tmp ] = i;
  }

  _FREE_ (offset);
}

void partition_matrix_naive(index_t rows, part_id_t n_parts, struct partition_data *p)
{
  p->n = rows;
  p->n_parts = n_parts;
  _ALLOC_ ((void**) &p->row_to_part, p->n * sizeof (part_id_t));
  _ALLOC_ ((void**) &p->part_to_row, p->n * sizeof (index_t));
  _ALLOC_ ((void**) &p->row_to_part_row, p->n * sizeof (index_t));
  _ALLOC_ ((void**) &p->ptr, (n_parts + 1) * sizeof (index_t));

  // Just divide into equally-sized contiguous partitions
  part_id_t part;
  for (part = 0; part < n_parts; part++)
  {
    index_t start = part*p->n / n_parts;
    index_t end = (part+1)*p->n / n_parts;
    index_t i;
    for (i = start; i < end; i++)
    {
      p->row_to_part[i] = part;
      p->part_to_row[i] = i;
      p->row_to_part_row[i] = i - start;      
    }
    p->ptr[part] = start;
  }
  p->ptr[n_parts] = p->n;
}

void partition_matrix(struct bcsr_t *A, index_t rows, level_t k, part_id_t n_parts, struct partition_data *p)
{
  index_t i, j;

  if (n_parts == 1)
  {
    // Don't waste time with trivial partitionings
    partition_matrix_naive(rows, n_parts, p);
    return;
  }

  /*********************\
   * COMPUTE TRANSPOSE *
   \*********************/
  // Form transpose -- we are assuming that we are NOT going to compute y=(A^T)^k*x though
  //  --- in this case, we would simultaneously compute closure on A, and then our thread blocking routine
  // will be different in order to reuse the nets
  struct bcsr_t AT;
  AT.mb = A->nb;
  AT.nb = rows;
  AT.b_m = A->b_n;
  AT.b_n = A->b_m;
  AT.b_transpose = 0;
  AT.nnzb = A->browptr[rows];

  _ALLOC_ ((void**) &AT.browptr, (AT.mb + 1)*sizeof(index_t) );
  _ALLOC_ ((void**) &AT.bcolidx, AT.nnzb * sizeof(index_t) );
  AT.bvalues = NULL;
#ifdef TRACE
  fprintf (stderr, "== csr_matrix_transpose_kernel_double_real () ...\n");
#endif

  for (i = 0; i <= AT.mb; i++)
    AT.browptr[i] = 0;
  for (i = 0; i < A->browptr[rows]; i++)
    AT.browptr[A->bcolidx[i]]++;
  for (i = 0; i < AT.mb; i++)
    AT.browptr[i+1] += AT.browptr[i];
  for (i = rows; --i >= 0; )
    for (j = A->browptr[i+1]; --j >= A->browptr[i]; )
      AT.bcolidx[--AT.browptr[A->bcolidx[j]]] = i;

#ifdef TRACE
  fprintf (stderr, "== compute closure () ...\n");
#endif
 
  // Compute closure on transpose
  struct level_net *nets;
  compute_closure (&AT, &nets, k);

  _FREE_ ( AT.browptr );
  _FREE_ ( AT.bcolidx );

#ifdef TRACE
  fprintf (stderr, "== nets_to_netlist () ...\n");
#endif
 
  // Convert nets to hypergraph
  struct hypergraph h;
  nets_to_netlist ( nets, AT.nb, &h );
  for (i = 0; i < AT.nb; ++i)
  {
    _FREE_ ( nets[i].levels );
    _FREE_ ( nets[i].pins );
  }
  _FREE_ ( nets );

#ifdef TRACE
  fprintf (stderr, "== compute_partition () ...\n");
#endif

  // Call PaToH
  compute_partition ( &h, n_parts, p );
  dest_hypergraph ( &h );
}

void build_explicit_block(
  const struct bcsr_t *A,
  index_t *part_rows,
  index_t part_size,
  struct set *workspace,
  level_t k,
  struct akx_explicit_block *this_block)
{
  index_t i, j;

  // Workspace for (block) column permutation
  index_t *__restrict__ perm;
  _ALLOC_ ((void**) &perm, A->nb * sizeof(index_t) );

  this_block->k = k;

  _ALLOC_ ((void**) &this_block->A_part, 1 * sizeof (struct bcsr_t));
  _ALLOC_ ((void**) &this_block->schedule, k * sizeof(index_t) );

  struct level_net n;
  build_net(A, &n, k, part_size, part_rows, workspace);

  // Copy and permute bcsr_t matrix into this_block->A_part
  // Copy schedule into some structure inside akx_thead_block
  // Write driver for do_akx (pthread call)

  // Explicitly partition A in block rows according to the permutation defined by the level_net
  struct bcsr_t *A_part = this_block->A_part;
  A_part->mb = n.levels[k] - n.levels[0];
  A_part->nb = n.levels[k+1] - n.levels[0];
  A_part->b_m = A->b_m;
  A_part->b_n = A->b_n;
  A_part->b_transpose = A->b_transpose;
  //    fprintf (stderr, "%d %d %d %d\n", this_block->A_part->mb,
  //	this_block->A_part->nb,
  //	this_block->A_part->b_m,
  //	this_block->A_part->b_n);
  _ALLOC_ ((void**) &A_part->browptr, (A_part->mb + 1) * sizeof (index_t) );

  this_block->V_size = A_part->nb * A_part->b_n;
  _ALLOC_ ((void**) &this_block->V, (k+1)*this_block->V_size * sizeof (value_t));

  // Count nnz and identify permutation
  A_part->nnzb = 0;
  for (i = n.levels[0]; i < n.levels[k]; ++i)
  {
    // TODO: unroll
    index_t browidx = n.pins[i];
    perm[browidx] = i; // TODO this will screw up if levels[0] != 0
    A_part->nnzb += A->browptr[browidx + 1] - A->browptr[browidx];
  }

  // TODO: if block_size is 1x1, consider padding the end of each brow to get
  // aligned SIMD for all values.
  //       if block_size is 2x2, 4x4, 8x8, don't worry (if value_t is float or double, we're aligned!)
  // if block size is 3x3, 5x5, 7x7, etc, maybe consider a more complicated type of padding
  // Also consider reblocking.

  _ALLOC_ ((void**) &A_part->bcolidx, A_part->nnzb * sizeof (index_t) );
  _ALLOC_ ((void**) &A_part->bvalues, A_part->nnzb * A_part->b_m * A_part->b_n * sizeof (value_t) );

  for (i = n.levels[k]; i < n.levels[k+1]; ++i)
  {
    // TODO: unroll
    index_t browidx = n.pins[i];
    perm[browidx] = i;
  }

  // Relabel bcolidx and permute values accordingly
  // TODO: consider sparse->dense->permute->sparse
  nnz_t cur_nnzb = 0;
  nnz_t cur_brow_begin = 0;
  level_t l;
  for (l = 0; l < k; ++l)
  {
    for (i = n.levels[l]; i < n.levels[l+1]; ++i)
    {
      // TODO: unroll
      index_t browidx = n.pins[i];
      A_part->browptr[ i ] = cur_brow_begin;

      for (j = A->browptr[browidx]; j < A->browptr[browidx + 1]; ++j)
      {
        // Use insertion sort to apply the symmetric permutation to the columns
        index_t tmp = cur_nnzb - 1;
        while ( tmp >= cur_brow_begin && A_part->bcolidx[ tmp ] > perm [ A->bcolidx[j]] )
        {
          A_part->bcolidx [tmp + 1] = A_part->bcolidx[ tmp ];

          // NOTE: unaligned copy that could easily be aligned for 2x2, 4x4, 8x8 block sizes
          _COPY_UAL_ ((void*) &A_part->bvalues[ tmp * A_part->b_m * A_part->b_n ],
            (void*) &A_part->bvalues[(tmp+1) * A_part->b_m * A_part->b_n ],
            A_part->b_m * A_part->b_n * sizeof(value_t));
          --tmp;
        }
        A_part->bcolidx [tmp + 1] = perm [ A->bcolidx[j]];
        _COPY_UAL_ ( (void*) &A->bvalues[j * A->b_m * A->b_n ],
            (void*) &A_part->bvalues[(tmp+1)* A_part->b_m * A_part->b_n ],
            A_part->b_m * A_part->b_n * sizeof (value_t) );
        ++cur_nnzb;
      }
      cur_brow_begin = cur_nnzb;
    }
    this_block->schedule[k - l - 1] = i * A->b_m;
  }
  A_part->browptr[i] = cur_nnzb;
  A_part->nnzb = cur_nnzb;
  //print_sp_matrix (this_block->A_part, 0);
  this_block->size = n.levels[k+1] - n.levels[0];
  this_block->perm = n.pins;

  _FREE_ (n.levels);
  this_block->implicit_blocks = 0;

  _FREE_ (perm);
}

struct akx_thread_block *make_thread_blocks(
    const struct bcsr_t *__restrict__ A,
    const struct partition_data *__restrict__ p,
    level_t k)
{
#ifdef TRACE
  fprintf (stderr, "== make_thread_blocks () ...\n");
#endif

  struct akx_thread_block *__restrict__ t_blocks;
  _ALLOC_ ((void**) &t_blocks, p->n_parts * sizeof (struct akx_thread_block) );

  // Workspace for transitive closure
  struct set workspace;
  workspace.capacity = A->nb;
  _ALLOC_ ((void**) &workspace.elements, A->nb * sizeof (index_t));
  _ALLOC_ ((void**) &workspace.flags,    A->nb * sizeof (flag_t));
  memset(workspace.flags, 0, A->nb * sizeof(flag_t));

  index_t i, j;
  part_id_t pp;
  for (pp = 0; pp < p->n_parts; ++pp)
  {
#ifdef TRACE
    fprintf (stderr, " = building thread block %d of %d ...\n", pp, p->n_parts);
#endif

    struct akx_thread_block * this_block = &t_blocks[pp];
    build_explicit_block(
      A,
      &p->part_to_row[p->ptr[pp]],
      p->ptr[pp + 1] - p->ptr[pp],
      &workspace,
      k,
      &this_block->orig_eb);

    this_block->explicit_blocks = 1;
    this_block->eb = &this_block->orig_eb;
  }

  _FREE_ ( workspace.elements );
  _FREE_ ( workspace.flags );
  return t_blocks;
}

void make_explicit_blocks(
    struct set *workspace,
    const struct akx_explicit_block *__restrict__ old_block,
    struct akx_explicit_block *__restrict__ new_blocks,
    int part_alg,
    int n_parts)
{
  level_t k = old_block->k;

  // Partition the thread block into cache blocks
  index_t b_m = old_block->A_part->b_m;
  index_t rows = (old_block->schedule[k-1] + b_m - 1) / b_m; // Non-ghost entries only
  struct partition_data cbp;
  if (part_alg)
    partition_matrix (old_block->A_part, rows, k, n_parts, &cbp);
  else
    partition_matrix_naive (rows, n_parts, &cbp);

  part_id_t pp;
  for (pp = 0; pp < n_parts; ++pp)
  {
    struct akx_explicit_block *eb = &new_blocks[pp];
#ifdef TRACE
    fprintf (stderr, " = building cache block %d of %d ...\n", pp, n_parts);
#endif
    build_explicit_block(
      old_block->A_part,             // matrix to take a partition of
      &cbp.part_to_row[cbp.ptr[pp]], // partition rows array
      cbp.ptr[pp + 1] - cbp.ptr[pp], // partition rows array size
      workspace,
      k,
      eb);

    // Fix permutation to contain indices for full matrix, not this thread block
    index_t i;
    for (i = 0; i < eb->size; i++)
    {
      eb->perm[i] = old_block->perm[eb->perm[i]];
    }
  }

  dest_partition_data (&cbp);
}

void make_implicit_blocks (
    struct set *workspace,
    struct akx_explicit_block *__restrict__ this_block,
    int cblock_part_alg,
    part_id_t cblocks_per_thread,
    int stanza)
{
  index_t i, j;
  level_t l;

  level_t k = this_block->k;

  assert(this_block->A_part->b_m == this_block->A_part->b_n);

  this_block->implicit_blocks = cblocks_per_thread;
  this_block->implicit_stanza = stanza;

  // Partition the thread block into cache blocks
  index_t b_m = this_block->A_part->b_m;
  index_t rows = (this_block->schedule[k-1] + b_m - 1) / b_m; // Non-ghost entries only
  struct partition_data cbp;
  if (cblock_part_alg)
    partition_matrix (this_block->A_part, rows, k, cblocks_per_thread, &cbp);
  else
    partition_matrix_naive (rows, cblocks_per_thread, &cbp);

  // Count number of computations done in this thread block,
  // and make room for worst-case computation sequence array
  i = 0;
  for (l = 0; l < k; l++)
    i += (this_block->schedule[l] + b_m - 1) / b_m;
  _ALLOC_ ((void **)&this_block->level_start, (cblocks_per_thread * k + 1) * sizeof(index_t));
  _ALLOC_ ((void **)&this_block->computation_seq, i*2 * sizeof(index_t));

  i = 0;
  part_id_t block;
  level_t *computed_level;
  _ALLOC_ ((void **)&computed_level, this_block->A_part->mb * sizeof(level_t));
  memset(computed_level, 0, this_block->A_part->mb * sizeof(level_t));

  // Make a copy of the thread block (structure only) with outside dependencies removed.
  // Register tiling causes the set of dependencies to grow faster at each level, and
  // we don't want this to result in going outside the thread block
  struct bcsr_t A_temp;
  A_temp.mb = this_block->A_part->mb;
  A_temp.nb = this_block->A_part->mb;
  A_temp.b_m = this_block->A_part->b_m;
  A_temp.b_n = this_block->A_part->b_m;
  A_temp.b_transpose = this_block->A_part->b_transpose;
  _ALLOC_((void **)&A_temp.browptr, sizeof(index_t) * (A_temp.mb + 1));
  _ALLOC_((void **)&A_temp.bcolidx, sizeof(nnz_t) * this_block->A_part->nnzb);
  A_temp.bvalues = NULL;
  nnz_t n = 0;
  for (i = 0; i < A_temp.mb; i++)
  {
    nnz_t j;
    A_temp.browptr[i] = n;
    for (j = this_block->A_part->browptr[i]; j != this_block->A_part->browptr[i+1]; j++)
      if (this_block->A_part->bcolidx[j] < A_temp.nb)
        A_temp.bcolidx[n++] = this_block->A_part->bcolidx[j];
  }
  A_temp.browptr[i] = n;
  A_temp.nnzb = n;

  i = 0;
  for (block = 0; block < cblocks_per_thread; block++)
  {
    // Compute dependencies of cache block
    struct level_net cbn;
    build_net(&A_temp, &cbn, k, 
              cbp.ptr[block + 1] - cbp.ptr[block], &cbp.part_to_row[cbp.ptr[block]],
              workspace);

    for (l = 0; l < k; l++)
    {
      // It is still not necessary to compute values outside the thread block,
      // even though register tiling can make it look like they're needed
      index_t limit = (this_block->schedule[l] + b_m - 1) / b_m;

      this_block->level_start[block*k+l] = i;
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
              this_block->computation_seq[i++] = localindex;
              this_block->computation_seq[i++] = localindex;
            }
            this_block->computation_seq[i-1]++;
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
            this_block->computation_seq[i++] = localindex;
            computed_level[localindex]++;
          }
        }
      }
    }
    _FREE_ (cbn.levels);
    _FREE_ (cbn.pins);
  }

  _FREE_ (A_temp.browptr);
  _FREE_ (A_temp.bcolidx);

  this_block->level_start[cblocks_per_thread*k] = i;
  _FREE_ (computed_level);
  dest_partition_data (&cbp);
}

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

void destroy_implicit_blocks (struct akx_explicit_block *__restrict__ block)
{
  if (block->implicit_blocks)
  {
    _FREE_ (block->level_start);
    _FREE_ (block->computation_seq);
    block->implicit_blocks = 0;
  }
}

void destroy_explicit_block(struct akx_explicit_block *__restrict__ block)
{
  destroy_implicit_blocks(block);
  _FREE_ (block->A_part->browptr);
  _FREE_ (block->A_part->bcolidx);
  _FREE_ (block->A_part->bvalues);
  _FREE_ (block->A_part);
  _FREE_ (block->V);
  _FREE_ (block->schedule);
  _FREE_ (block->perm);
}

void destroy_explicit_blocks (struct akx_thread_block *__restrict__ t_block)
{
  if (t_block->eb != &t_block->orig_eb)
  {
    part_id_t p;
    for (p = 0; p < t_block->explicit_blocks; p++)
    {
      destroy_explicit_block(&t_block->eb[p]);
    }
    _FREE_ (t_block->eb);
    t_block->explicit_blocks = 1;
    t_block->eb = &t_block->orig_eb;
  }
}

void destroy_thread_blocks (struct akx_thread_block *__restrict__ t_blocks, part_id_t n_blocks)
{
  part_id_t p;
  for (p = 0; p < n_blocks; ++p)
  {
    destroy_explicit_blocks(&t_blocks[p]);
    destroy_explicit_block(&t_blocks[p].orig_eb);
  }
  _FREE_ (t_blocks);
}

void dest_partition_data ( struct partition_data * p )
{
  _FREE_ (p->row_to_part);
  _FREE_ (p->row_to_part_row);
  _FREE_ (p->part_to_row);
  _FREE_ (p->ptr);
}

static PyObject *AkxObjectC_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *AkxObjectC_orig_tilesize(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_num_threadblocks(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_num_blocks(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_threadblocks(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_shape(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_nnzb(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_schedule(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_nnzb_computed(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_tilesize(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_tilecount(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_tile(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_block_split(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_explicitblocks(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_implicitblocks(AkxObjectC *self, PyObject *args);
static PyObject *AkxObjectC_powers(AkxObjectC *self, PyObject *args);
static void AkxObjectC_dealloc(AkxObjectC *self);

static PyMethodDef AkxObjectC_methods[] = {
	{ "num_threadblocks", (PyCFunction)AkxObjectC_num_threadblocks, METH_NOARGS },
	{ "num_blocks", (PyCFunction)AkxObjectC_num_blocks, METH_VARARGS },
	{ "orig_tilesize", (PyCFunction)AkxObjectC_orig_tilesize, METH_VARARGS },
	{ "threadblocks", (PyCFunction)AkxObjectC_threadblocks, METH_VARARGS },
	{ "block_shape", (PyCFunction)AkxObjectC_block_shape, METH_VARARGS },
	{ "block_nnzb", (PyCFunction)AkxObjectC_block_nnzb, METH_VARARGS },
	{ "block_schedule", (PyCFunction)AkxObjectC_block_schedule, METH_VARARGS },
	{ "block_nnzb_computed", (PyCFunction)AkxObjectC_block_nnzb_computed, METH_VARARGS },
	{ "block_tilesize", (PyCFunction)AkxObjectC_block_tilesize, METH_VARARGS },
	{ "block_tilecount", (PyCFunction)AkxObjectC_block_tilecount, METH_VARARGS },
	{ "block_tile", (PyCFunction)AkxObjectC_block_tile, METH_VARARGS },
	{ "block_split", (PyCFunction)AkxObjectC_block_split, METH_VARARGS },
	{ "explicitblocks", (PyCFunction)AkxObjectC_explicitblocks, METH_NOARGS },
	{ "implicitblocks", (PyCFunction)AkxObjectC_implicitblocks, METH_VARARGS },
	{ "powers", (PyCFunction)AkxObjectC_powers, METH_VARARGS },
	{ NULL, NULL, 0, NULL }
};

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
	AkxObjectC_methods,         /*tp_methods*/
	0,                          /*tp_members*/
	0,                          /*tp_getset*/
	0,                          /*tp_base*/
	0,                          /*tp_dict*/
	0,                          /*tp_descr_get*/
	0,                          /*tp_descr_set*/
	0,                          /*tp_dictoffset*/
	0,                          /*tp_init*/
	0,                          /*tp_alloc*/
	AkxObjectC_new,             /*tp_new*/
};

static PyMethodDef methods[] = {
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
init_akx_static(void)
{
  PyObject *module = Py_InitModule("_akx_static", methods);
  if (!module)
    return;

  if (PyType_Ready(&AkxObjectC_Type) < 0)
    return;

  Py_INCREF(&AkxObjectC_Type);
  PyModule_AddObject(module, "AkxObjectC", (PyObject *)&AkxObjectC_Type);

  import_array();
}

static PyObject *
AkxObjectC_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
  AkxObjectC *self;
  PyArrayObject *indptr, *indices, *data;

  if (!PyArg_ParseTuple(args, "O!O!O!",
      &PyArray_Type, &indptr, &PyArray_Type, &indices, &PyArray_Type, &data))
  {
    return NULL;
  }
  kwds = kwds; // unused variable

  self = PyObject_New(AkxObjectC, subtype);
  if (!self)
    return NULL;

  self->indptr = indptr;
  self->indices = indices;
  self->data = data;
  Py_INCREF(indptr);
  Py_INCREF(indices);
  Py_INCREF(data);

  /*******************************************\
   * COPY CSR MATRIX TO LOCAL DATA STRUCTURE *
   \*******************************************/
#ifdef TRACE
  fprintf (stderr, "== A allocs and copies ...\n");
#endif
  self->A.mb = self->A.nb = indptr->dimensions[0] - 1;
  self->A.b_m = data->nd > 2 ? data->dimensions[1] : 1;
  self->A.b_n = data->nd > 2 ? data->dimensions[2] : 1;
  self->A.b_transpose = 0;
  self->A.nnzb = data->dimensions[0];
  self->A.browptr = (index_t *)indptr->data;
  self->A.bcolidx = (index_t *)indices->data;
  self->A.bvalues = (value_t *)data->data;
  //print_sp_matrix (&self->A, 0);

  self->p.n_parts = 0;

  self->powers_func = NULL;

  return (PyObject *)self;
}

static PyObject *
AkxObjectC_orig_tilesize(AkxObjectC *self, PyObject *args)
{
  return Py_BuildValue("ii", self->A.b_m, self->A.b_n);
}

static PyObject *
AkxObjectC_num_threadblocks(AkxObjectC *self, PyObject *args)
{
  return PyInt_FromLong(self->p.n_parts);
}

static PyObject *
AkxObjectC_num_blocks(AkxObjectC *self, PyObject *args)
{
  int tbno;
  if (!PyArg_ParseTuple(args, "i", &tbno))
    return NULL;

  if (tbno < 0 || tbno >= self->p.n_parts)
  {
    PyErr_SetString(PyExc_IndexError, "block index out of range");
    return NULL;  
  }
  return PyInt_FromLong(self->tb[tbno].explicit_blocks);
}

static PyObject *
AkxObjectC_threadblocks(AkxObjectC *self, PyObject *args)
{
  if (self->powers_func) { Py_DECREF(self->powers_func); self->powers_func = NULL; }

  if (self->p.n_parts)
  {
    destroy_thread_blocks ( self->tb, self->p.n_parts );
    dest_partition_data ( &self->p );
    self->p.n_parts = 0;
  }

  if (PyTuple_GET_SIZE(args) != 0)
  {
    int k, part_alg, n_parts;
    if (!PyArg_ParseTuple(args, "iii", &k, &part_alg, &n_parts))
      return NULL;

    if (!part_alg)
      partition_matrix_naive(self->A.mb, n_parts, &self->p);
    else
      partition_matrix(&self->A, self->A.mb, k, n_parts, &self->p);

    self->tb = make_thread_blocks(&self->A, &self->p, k);
    // TODO Compute statistics
  }

  Py_RETURN_NONE;
}

static struct akx_explicit_block *get_block(AkxObjectC *self, int tbno, int ebno)
{
  if (tbno < 0 || tbno >= self->p.n_parts)
  {
    PyErr_SetString(PyExc_IndexError, "block index out of range");
    return NULL;  
  }
  struct akx_thread_block *tb = &self->tb[tbno];
  if (ebno < 0 || ebno >= tb->explicit_blocks)
  {
    PyErr_SetString(PyExc_IndexError, "block index out of range");
    return NULL;  
  }
  return &tb->eb[ebno];
}

static PyObject *
AkxObjectC_block_shape(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno;
  struct akx_explicit_block *block;
  if (!PyArg_ParseTuple(args, "ii", &tbno, &ebno))
    return NULL;

  block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  struct bcsr_t *A = block->A_part;
  return Py_BuildValue("ii", A->mb * A->b_m, A->nb * A->b_n);
}

static PyObject *
AkxObjectC_block_nnzb(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno;
  struct akx_explicit_block *block;
  if (!PyArg_ParseTuple(args, "ii", &tbno, &ebno))
    return NULL;

  block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  return PyInt_FromLong(block->A_part->nnzb);
}

static PyObject *
AkxObjectC_block_schedule(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno;
  struct akx_explicit_block *block;
  if (!PyArg_ParseTuple(args, "ii", &tbno, &ebno))
    return NULL;

  block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  // TODO: this assumes that index_t is same size as NPY_INT
  npy_intp size = block->k;
  PyObject *obj = PyArray_SimpleNewFromData(1, &size, NPY_INT, block->schedule);
  PyArray_FLAGS(obj) &= ~NPY_WRITEABLE;
  return obj;
}

static PyObject *
AkxObjectC_block_nnzb_computed(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno;
  if (!PyArg_ParseTuple(args, "ii", &tbno, &ebno))
    return NULL;

  struct akx_explicit_block *block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  long nnz = 0;
  level_t level;
  for (level = 0; level < block->k; level++)
    nnz += block->A_part->browptr[(block->schedule[level] + block->A_part->b_m - 1) / block->A_part->b_m];
  return PyInt_FromLong(nnz);
}

static PyObject *
AkxObjectC_block_tilesize(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno;
  struct akx_explicit_block *block;
  if (!PyArg_ParseTuple(args, "ii", &tbno, &ebno))
    return NULL;

  block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  struct bcsr_t *A = block->A_part;
  return Py_BuildValue("iii", A->b_m, A->b_n, A->b_transpose);
}

static PyObject *
AkxObjectC_block_tilecount(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno, b_m, b_n, samples;
  struct akx_explicit_block *block;
  if (!PyArg_ParseTuple(args, "iiiii", &tbno, &ebno, &b_m, &b_n, &samples))
    return NULL;

  block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  struct bcsr_t *A = block->A_part;

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

static PyObject *
AkxObjectC_block_tile(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno, b_m, b_n, b_transpose;
  struct akx_explicit_block *block;
  struct bcsr_t *A;

  if (!PyArg_ParseTuple(args, "iiiii", &tbno, &ebno, &b_m, &b_n, &b_transpose))
    return NULL;

  block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  if (block->implicit_blocks)
  {
    PyErr_SetString(PyExc_IndexError, "block is already partitioned into cache blocks");
    return NULL;
  }
  if (b_m > MAX_TILE_HEIGHT)
  {
    PyErr_SetString(PyExc_ValueError, "tile size too large");
    return NULL;
  }
  A = block->A_part;

  if (A->b_m != 1 || A->b_n != 1)
  {
    PyErr_SetString(PyExc_ValueError, "block is already tiled");
    return NULL;
  }

  index_t top, bottom, left, right;
  index_t curptr[MAX_TILE_HEIGHT];
  nnz_t nnz = 0;

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

  struct bcsr_t *Anew;
  _ALLOC_ ((void **)&Anew, sizeof(struct bcsr_t));
  Anew->mb = (A->mb + b_m - 1) / b_m;
  Anew->b_m = b_m;
  Anew->nb = (A->nb + b_n - 1) / b_n;
  Anew->b_n = b_n;
  Anew->b_transpose = b_transpose;
  Anew->nnzb = nnz;
  _ALLOC_ ((void **)&Anew->browptr, (Anew->mb + 1) * sizeof(index_t));
  _ALLOC_ ((void **)&Anew->bcolidx, Anew->nnzb * sizeof(index_t));
  _ALLOC_ ((void **)&Anew->bvalues, Anew->nnzb * (b_m * b_n) * sizeof(value_t));

  index_t *browptr = Anew->browptr;
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

      Anew->bcolidx[nnz] = next / b_n;
      left  = Anew->bcolidx[nnz] * b_n;
      right = left + b_n;

      tile = &Anew->bvalues[nnz * b_m * b_n];
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

  block->A_part = Anew;

  _FREE_ (block->V);
  index_t padded_height = Anew->mb * Anew->b_m;
  index_t padded_width = Anew->nb * Anew->b_n;
  block->V_size = (padded_height > padded_width ? padded_height : padded_width);
  _ALLOC_ ((void **)&block->V, (block->k+1) * block->V_size * sizeof (value_t));

  if (self->powers_func) { Py_DECREF(self->powers_func); self->powers_func = NULL; }

  Py_RETURN_NONE;
}

static PyObject *
AkxObjectC_block_split(AkxObjectC *self, PyObject *args)
{
  int tbno, ebno, part_alg, n_parts;
  if (!PyArg_ParseTuple(args, "iiii", &tbno, &ebno, &part_alg, &n_parts))
    return NULL;

  struct akx_explicit_block *block = get_block(self, tbno, ebno);
  if (!block)
    return NULL;

  if (n_parts < 2)
  {
    PyErr_SetString(PyExc_ValueError, "n_parts < 2");
    return NULL;
  }

  if (self->powers_func) { Py_DECREF(self->powers_func); self->powers_func = NULL; }

  // Workspace for transitive closure
  struct set workspace;
  workspace.capacity = block->A_part->nb;
  _ALLOC_ ((void**) &workspace.elements, workspace.capacity * sizeof (index_t));
  _ALLOC_ ((void**) &workspace.flags,    workspace.capacity * sizeof (flag_t));
  memset(workspace.flags, 0, workspace.capacity * sizeof(flag_t));

  struct akx_thread_block *tb = &self->tb[tbno];
  struct akx_explicit_block *new_eb;
  _ALLOC_ ((void**) &new_eb, (tb->explicit_blocks + n_parts - 1) * sizeof(struct akx_explicit_block));
  memcpy(new_eb, tb->eb, ebno * sizeof(struct akx_explicit_block));
  make_explicit_blocks(&workspace, block, &new_eb[ebno], part_alg, n_parts);
  memcpy(&new_eb[ebno + n_parts], &tb->eb[ebno + 1], (tb->explicit_blocks - ebno - 1) * sizeof(struct akx_explicit_block));

  if (tb->eb != &tb->orig_eb)
  {
    destroy_explicit_block(block);
    _FREE_ (tb->eb);
  }
  tb->explicit_blocks += n_parts - 1;
  tb->eb = new_eb;

  _FREE_ ( workspace.elements );
  _FREE_ ( workspace.flags );

  Py_RETURN_NONE;
}

static PyObject *
AkxObjectC_explicitblocks(AkxObjectC *self, PyObject *args)
{
  if (self->p.n_parts == 0)
  {
    PyErr_SetString(PyExc_ValueError, "thread blocks not yet created");
    return NULL;
  }

  if (self->powers_func) { Py_DECREF(self->powers_func); self->powers_func = NULL; }

  part_id_t pp;
  for (pp = 0; pp < self->p.n_parts; ++pp)
  {
    destroy_explicit_blocks(&self->tb[pp]);
  }

  Py_RETURN_NONE;
}

static PyObject *
AkxObjectC_implicitblocks(AkxObjectC *self, PyObject *args)
{
  if (self->p.n_parts == 0)
  {
    PyErr_SetString(PyExc_ValueError, "thread blocks not yet created");
    return NULL;
  }

  if (self->powers_func) { Py_DECREF(self->powers_func); self->powers_func = NULL; }

  part_id_t pp, pp2;
  for (pp = 0; pp < self->p.n_parts; ++pp)
  {
    struct akx_thread_block *tb = &self->tb[pp];
    for (pp2 = 0; pp2 < tb->explicit_blocks; ++pp2)
    {
      destroy_implicit_blocks(&tb->eb[pp2]);
    }
  }

  if (PyTuple_GET_SIZE(args) != 0)
  {
    int part_alg, n_parts, stanza;
    if (!PyArg_ParseTuple(args, "iii", &part_alg, &n_parts, &stanza))
      return NULL;

    // Workspace for transitive closure
    struct set workspace;
    workspace.capacity = 0;
    for (pp = 0; pp < self->p.n_parts; ++pp)
    {
      struct akx_thread_block *tb = &self->tb[pp];
      for (pp2 = 0; pp2 < tb->explicit_blocks; ++pp2)
      {
        struct akx_explicit_block *eb = &tb->eb[pp2];
        if (eb->A_part->nb > workspace.capacity)
          workspace.capacity = eb->A_part->nb;
      }
    }
    _ALLOC_ ((void**) &workspace.elements, workspace.capacity * sizeof (index_t));
    _ALLOC_ ((void**) &workspace.flags,    workspace.capacity * sizeof (flag_t));
    memset(workspace.flags, 0, workspace.capacity * sizeof (flag_t));

    for (pp = 0; pp < self->p.n_parts; ++pp)
    {
#ifdef TRACE
      fprintf (stderr, " = building cache blocks for thread block %d of %d ...\n", pp, self->p.n_parts);
#endif
      struct akx_thread_block *tb = &self->tb[pp];
      for (pp2 = 0; pp2 < tb->explicit_blocks; ++pp2)
      {
        make_implicit_blocks(&workspace, &tb->eb[pp2], part_alg, n_parts, stanza);
      }
    }

    _FREE_ ( workspace.elements );
    _FREE_ ( workspace.flags );
  }

  Py_RETURN_NONE;
}

static PyObject *
AkxObjectC_powers(AkxObjectC *self, PyObject *args)
{
  PyArrayObject *vecs, *coeffs = NULL;
  if (!PyArg_ParseTuple(args, "O!|O!", &PyArray_Type, &vecs, &PyArray_Type, &coeffs))
    return NULL;

  if (!self->powers_func) {
    PyObject *akx = PyImport_ImportModule("akx");
    if (akx) {
      PyObject *powers_codegen = PyObject_GetAttrString(akx, "_powers_codegen");
      if (powers_codegen) {
        if (coeffs)
          self->powers_func = PyObject_CallFunction(powers_codegen, "OOO", self, vecs, coeffs);
        else
          self->powers_func = PyObject_CallFunction(powers_codegen, "OO", self, vecs);
        Py_DECREF(powers_codegen);
      }
      Py_DECREF(akx);
    }
  }
  if (!self->powers_func)
    return NULL;

  if (coeffs)
    return PyObject_CallFunction(self->powers_func, "OOO", self, vecs, coeffs);
  else
    return PyObject_CallFunction(self->powers_func, "OO", self, vecs);
}

static void
AkxObjectC_dealloc(AkxObjectC *self)
{
  if (self->p.n_parts)
  {
    destroy_thread_blocks ( self->tb, self->p.n_parts );
    dest_partition_data ( &self->p );
  }

  Py_DECREF(self->indptr);
  Py_DECREF(self->indices);
  Py_DECREF(self->data);

  if (self->powers_func) { Py_DECREF(self->powers_func); self->powers_func = NULL; }
}
