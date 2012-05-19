"""Matrix powers kernel specializer configuration."""

# Enable specialization
specialize = True

# Directory containing patoh.h and libpatoh.a
patoh_path = '/home/eecs/jmorlan/PaToH'

# Set to directory containing libiomp5.so to use Intel OpenMP rather than pthreads
# (using pthreads was observed to interact poorly with MKL performance-wise)
iomp5_path = None
#iomp5_path = '/home/eecs/jmorlan/mkl'

# If true, use MKL BLAS (must be in LD_LIBRARY_PATH)
use_mkl = False
# If true, use ACML BLAS (must be in LD_LIBRARY_PATH)
use_acml = False

# List of thread counts for tuner to try
thread_counts = [8]
