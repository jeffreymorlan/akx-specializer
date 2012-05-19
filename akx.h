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

// Block-compressed-sparse-row matrix
struct bcsr_t {
	index_t mb;           // Height in tiles
	index_t nb;           // Width in tiles
	index_t b_m;          // Tile height
	index_t b_n;          // Tile width
	flag_t b_transpose;   // 0 = row-major tiles, 1 = column-major tiles
	flag_t browptr_comp;  // if 1, browptr is 16 bits per entry
	flag_t bcolidx_comp;  // if 1, bcolidx is 16 bits per entry
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
#define _ALLOC_(bytes) _mm_malloc(bytes, 16)
#define _FREE_(ptr) _mm_free(ptr)

struct set {
	index_t capacity;
	flag_t *__restrict__ flags;
	index_t *__restrict__ elements;
};

struct level_net {
	index_t n_pins;
	level_t n_levels;

	index_t *__restrict__ pins;
	index_t *__restrict__ levels;
};

struct partition_data {
	index_t *part_to_row;
	index_t *ptr;
};

typedef struct {
	PyObject_HEAD
	level_t k;

	struct bcsr_t A_part;
	flag_t symmetric_opt; // if 1, only upper triangle is stored
	index_t *__restrict__ schedule; // how many rows to process for each level
	                                // (number of rows, not number of row tiles)
	index_t perm_size;
	index_t *__restrict__ perm;
} AkxBlock;

typedef struct {
	PyObject_HEAD

	level_t k;
	index_t mb;
	part_id_t nblocks;
	index_t *__restrict__ level_start;
	index_t *__restrict__ computation_seq;
	flag_t stanza; // 1 if computation_seq is stanza encoded, 0 if not
} AkxImplicitSeq;

struct akx_task {
	AkxBlock *__restrict__ block;
	AkxImplicitSeq *__restrict__ imp; // NULL if none
	index_t V_size;
	value_t *__restrict__ V;
};

struct akx_data {
	level_t k;
	value_t *V_global;
	index_t V_global_m;
	part_id_t ntasks;
	struct akx_task *tasks;
	level_t steps;
	value_t *__restrict__ coeffs;
};

typedef struct {
	PyObject_HEAD
	level_t k;
	index_t matrix_size;
	int nthreads;
	struct akx_task *tasks;
	int *thread_offset;
} AkxObjectC;

#endif
