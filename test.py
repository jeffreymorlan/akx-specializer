#!/usr/bin/python2.5

import sys

import optparse
parser = optparse.OptionParser("usage: %prog [options] matrixfile")
parser.add_option("-o", "--output", help="output vectors to FILE", metavar="FILE")
parser.add_option("-s", "--sejits", help="enable specialization", action="store_true", default=False)
parser.add_option("-m",             help="total number of steps", type="int", default=0)
parser.add_option("-k",             help="number of steps per iteration", type="int", default=1)
parser.add_option("--tb-part",      help="thread block partitioning method (1 = hypergraph)", type="int", default=0)
parser.add_option("--tb-num",       help="number of threads", type="int", default=1)
parser.add_option("--tile-test",    help="show memory usage of different tile sizes", action="store_true", default=False)
parser.add_option("--tile-height",  help="tile size height", type="int", default=1)
parser.add_option("--tile-width",   help="tile size width", type="int", default=1)
parser.add_option("--tile-trans",   help="column-major tile ordering", type="int", default=0)
parser.add_option("--cb-part",      help="cache block partitioning method (1 = hypergraph)", type="int", default=0)
parser.add_option("--cb-num",       help="number of cache blocks (0 = no cache blocking)", type="int", default=0)
parser.add_option("--cb-exp",       help="explicit cache blocking", action="store_true", default=False)
parser.add_option("--cb-rle",       help="enable run length encoding", action="store_true", default=False)
options, args = parser.parse_args()

if len(args) != 1:
	parser.print_usage()
	exit()
filename, = args

print >>sys.stderr, "Importing modules...",
import numpy
import scipy.sparse
print >>sys.stderr, "done"

print >>sys.stderr, "Reading matrix...",
if filename.endswith('.bin'):
	import struct
	file = open(filename, "rb")
	rows, cols, nnz = struct.unpack('III', file.read(12))
	indptr  = numpy.fromfile(file, dtype=numpy.int32, count=rows+1)
	indices = numpy.fromfile(file, dtype=numpy.int32, count=nnz)
	data    = numpy.fromfile(file, dtype=numpy.double, count=nnz)
	file.close()
	matrix  = scipy.sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
else:
	import scipy.io.mmio
	matrix = scipy.io.mmio.mmread(filename).tocsr()
print >>sys.stderr, "done"

print >>sys.stderr, "Initializing akx...",
import akx
akx.specialize = options.sejits
akxobj = akx.AkxObject(matrix)
print >>sys.stderr, "done"

if options.tb_num:
	print >>sys.stderr, "Creating thread blocks...",
	akxobj.threadblocks(options.k, options.tb_part, options.tb_num)
	print >>sys.stderr, "done"

if options.tile_test:
	for i in xrange(options.tb_num):
		print "Block %d:" % i,
		mb, nb = akxobj.block_shape(i, 0)
		print "%dx%d" % (mb, nb)
		for b_m in xrange(1, 5):
			for b_n in xrange(1, 5):
				print "  %dx%d:" % (b_m, b_n),
				tiles = akxobj.block_tilecount(i, 0, b_m, b_n, 10000)
				bytes = (4 * (mb + (b_m - 1) / b_m) # browptr
				      + 4 * tiles                   # bcolidx
				      + 8 * tiles * b_m * b_n)      # bvalues
				print "%8d tiles, %9d bytes" % (tiles, bytes)
	exit(1)

if options.cb_num and options.cb_exp:
	print >>sys.stderr, "Creating cache blocks...",
	for tb in xrange(akxobj.num_threadblocks()):
		akxobj.block_split(tb, 0, options.cb_part, options.cb_num)
	print >>sys.stderr, "done"
if options.tile_height != 1 or options.tile_width != 1:
	print >>sys.stderr, "Tiling thread blocks...",
	for i in xrange(akxobj.num_threadblocks()):
		print >>sys.stderr, i,
		for j in xrange(akxobj.num_blocks(i)):
			akxobj.block_tile(i, j, options.tile_height, options.tile_width, options.tile_trans)
	print >>sys.stderr, "done"
if options.cb_num and not(options.cb_exp):
	print >>sys.stderr, "Creating cache blocks...",
	akxobj.implicitblocks(options.cb_part, options.cb_num, options.cb_rle)
	print >>sys.stderr, "done"

if options.m == 0:
	options.m = options.k
vecs = numpy.empty((1 + options.m, matrix.shape[0]))
vecs[0] = 1.0

num_flops = options.m * 2 * matrix.nnz
seconds, n_iterations = akx.benchmark(akxobj, vecs)
Mflops_s = 1.0e-6 * num_flops * n_iterations / seconds

print >>sys.stderr, ((
	"\n"+
	"**************************\n"+
	"*\n"+
	"* Matrix powers:\n"+
	"*\n" +
	"*  filename = %s\n"+
	"*       nnz = %d\n"+
	"*       dim = %d\n"+
	"*\n"+
	"*         m = %d\n"+
	"*   runtime = %g seconds\n"+
	"*             avg of %d its\n"+
	"* spmv-perf = %g Mflops/s\n"+
	"*\n"+
	"**************************\n") %
	(filename, matrix.nnz, matrix.shape[0],
	options.m, seconds, n_iterations, Mflops_s))

print >>sys.stderr, "Verifying..."
vecscheck = numpy.empty((1 + options.m, matrix.shape[0]))
vecscheck[0] = vecs[0]
for i in xrange(1, len(vecscheck)):
	vecscheck[i] = matrix * vecscheck[i-1]
print >>sys.stderr, "Maximum absolute error:", abs(vecs - vecscheck).max()
print >>sys.stderr, "Maximum relative error:", abs((vecs - vecscheck) / vecscheck).max()

if options.output:
	f = open(options.output, "w")
	print >>f, "Krylov vectors:"
	print >>f, "".join("x_%d\t" % n for n in xrange(options.m + 1))
	for element in vecs.transpose():
		print >>f, "".join("%.3g\t" % n for n in element)
