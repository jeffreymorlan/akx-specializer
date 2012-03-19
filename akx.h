#ifndef AKX_H_
#define AKX_H_

// Typedefs to make changing types easier
typedef int level_t;     // Integers from 0 to k+1
typedef int part_id_t;   // Integers from 0 to n_parts + 1
typedef int count_t;     // Integers from 0 to some big number
typedef int index_t;     // Integers from 0 to n (or n+1?)
typedef double value_t;  // Floating-point values for matrix and vector elements
typedef int nnz_t;       // Integers from 0 to n^2
typedef int pin_t;       // Integers from 0 to n^2
typedef char flag_t;     // Integers from 0 to 1

#define P( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

// Always padded always square always square tiles
// TODO: find some way to keep track of padding
struct bcsr_t
{
	index_t mb;
	index_t nb;
	index_t b_m;
	index_t b_n;
	int b_transpose; // 0 = row-major, 1 = column-major
	nnz_t nnzb;
	union {
		index_t *__restrict__ browptr;
		uint16_t *__restrict__ browptr16;
	};
	union {
		index_t *__restrict__ bcolidx;
		uint16_t *__restrict__ bcolidx16;
	};
	value_t *__restrict__ bvalues;
};

// Preprocessing macros
void _COPY_AL_ ( const void * src, void * dst, size_t bytes)
{
  memcpy (dst, src, bytes);
}

void _COPY_UAL_ (const void * src, void *dst, size_t bytes)
{
  memcpy (dst, src, bytes);
}

void * _ALLOC_ (size_t bytes)
{
  return _mm_malloc (bytes, 16);
}

void _FREE_ ( void * ptr )
{
  _mm_free (ptr);
}

struct hypergraph
{
	index_t n_nets;
	pin_t n_pins;
	pin_t   *__restrict__ netptr;
	index_t *__restrict__ pins;
};

struct set
{
	index_t capacity;
	flag_t *__restrict__ flags;
	index_t *__restrict__ elements;
};

struct level_net
{
	index_t n_pins;
	level_t n_levels;

	index_t *__restrict__ pins;
	index_t *__restrict__ levels;
};

struct partition_data
{
	index_t *part_to_row;
	index_t *ptr;
};

typedef struct
{
	PyObject_HEAD
	level_t k;

	struct bcsr_t A_part;
	index_t V_size;
	value_t *__restrict__ V;
	index_t *__restrict__ schedule; // how many rows to process for each level
	                                // (number of rows, not number of row tiles)
	index_t perm_size;
	index_t *__restrict__ perm;

	flag_t symmetric_opt; // if 1, only upper triangle is stored
	part_id_t implicit_blocks; // 0 = no implicit cache blocking
	index_t *__restrict__ level_start;
	union {
		index_t *__restrict__ computation_seq;
		uint16_t *__restrict__ computation_seq16;
	};
	int implicit_stanza; // 1 if computation_seq is stanza encoded, 0 if not
	flag_t browptr_comp;
	flag_t bcolidx_comp;
	flag_t computation_seq_comp;
} AkxBlock;

struct akx_data
{
	level_t k;
	value_t *V_global;
	index_t V_global_m;
	struct akx_thread_block *__restrict__ thread_block;
	part_id_t nblocks;
	AkxBlock **blocks;
	level_t steps;
	value_t *__restrict__ coeffs;
};

typedef struct {
  PyObject_HEAD
  level_t k;
  index_t matrix_size;
  int nthreads;
  AkxBlock **blocks;
  int *thread_offset;
} AkxObjectC;

#endif
