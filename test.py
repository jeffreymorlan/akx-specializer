#!/usr/bin/python
import sys
import optparse

parser = optparse.OptionParser("usage: %prog [options] matrixfile")
parser.add_option("-o", "--output", help="output vectors to FILE", metavar="FILE")
parser.add_option("-s", "--sejits", help="enable specialization", action="store_true", default=False)
parser.add_option("-m",             help="total number of steps", type="int", default=0)
parser.add_option("-k",             help="number of steps per iteration", type="int", default=1)
parser.add_option("-n",             help="newton basis coefficient", type="float", default=0.0)
parser.add_option("--tune",         help="use akx.tune (only --tb-num, --sym used)", action="store_true", default=False)
parser.add_option("--tb-part",      help="thread block hypergraph partitioning k value", type="int", default=0)
parser.add_option("--tb-num",       help="number of threads", type="int", default=1)
parser.add_option("--eb-part",      help="explicit block hypergraph partitioning k value", type="int", default=0)
parser.add_option("--eb-num",       help="number of explicit blocks (0 = no cache blocking)", type="int", default=0)
parser.add_option("--tile-test",    help="show memory usage of different tile sizes", action="store_true", default=False)
parser.add_option("--tile-height",  help="tile size height", type="int", default=1)
parser.add_option("--tile-width",   help="tile size width", type="int", default=1)
parser.add_option("--tile-trans",   help="column-major tile ordering", type="int", default=0)
parser.add_option("--sym",          help="symmetric optimization", action="store_true", default=False)
parser.add_option("--ib-part",      help="implicit block hypergraph partitioning k value", type="int", default=0)
parser.add_option("--ib-num",       help="number of implicit blocks (0 = no cache blocking)", type="int", default=0)
parser.add_option("--ib-stanza",    help="stanza encoding", action="store_true", default=False)
parser.add_option("--index-comp",   help="index compression", action="store_true", default=False)
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
import akxconfig
akxconfig.specialize = options.sejits
akxconfig.thread_counts = [options.tb_num]
import akx
print >>sys.stderr, "done"

if options.m == 0:
	options.m = options.k

if not options.sejits:
	akxobj = akx.AkxObjectPy(matrix)
elif options.tune:
	akxobj = akx.tune(matrix, options.k, options.sym, basis=(options.n != 0), show=sys.stderr)
else:
	tbpart = None
	if options.tb_part:
		print >>sys.stderr, "Partitioning...",
		tbpart, tbsizes, tbcut = akx.tb_partition(matrix.indptr, matrix.indices, matrix.data, options.tb_part, options.tb_num)
		print >>sys.stderr, "sizes =", tbsizes, "cut =", tbcut

	print >>sys.stderr, "Creating thread blocks...",
	tb = akx.threadblocks(matrix.indptr, matrix.indices, matrix.data, options.k, options.tb_num, tbpart)
	print >>sys.stderr, "done"

	if options.eb_num:
		print >>sys.stderr, "Creating cache blocks...",
		for i in xrange(options.tb_num):
			print >>sys.stderr, i,
			ebpart = None
			if options.eb_part:
				ebpart = tb[i].partition(options.eb_part, options.eb_num)[0]
			tb[i] = tb[i].split(options.eb_num, ebpart)
		print >>sys.stderr, "done"
	else:
		# Single block per thread
		tb = [[block] for block in tb]

	if options.tile_test:
		for i in tb:
			for b in i:
				mb, nb = b.shape()
				for b_m in xrange(1, 5):
					for b_n in xrange(1, 5):
						print "  %dx%d:" % (b_m, b_n),
						tiles = b.tilecount(b_m, b_n, 10000)
						bytes = (4 * (mb + (b_m - 1) / b_m) # browptr
									+ 4 * tiles                   # bcolidx
									+ 8 * tiles * b_m * b_n)      # bvalues
						print "%8d tiles, %9d bytes" % (tiles, bytes)
		exit()
	if options.tile_height != 1 or options.tile_width != 1:
		print >>sys.stderr, "Tiling blocks...",
		for i, t in enumerate(tb):
			for j in xrange(len(t)):
				print >>sys.stderr, "%d/%d" % (i, j),
				t[j] = t[j].tile(options.tile_height, options.tile_width, options.tile_trans)
		print >>sys.stderr, "done"

	nflops = sum(sum(b.flopcount() for b in t) for t in tb)

	if options.sym:
		print >>sys.stderr, "Symmetric opt..."
		for i, t in enumerate(tb):
			for j in xrange(len(t)):
				old = t[j].nnzb()
				t[j] = t[j].symm_opt()
				print >>sys.stderr, "Block (%d,%d) reduced from %d to %d nonzeros" % (i, j, old, t[j].nnzb())

	if options.ib_num:
		print >>sys.stderr, "Creating cache blocks...",
		for i, t in enumerate(tb):
			for j in xrange(len(t)):
				print >>sys.stderr, "%d/%d" % (i, j),
				ibpart = None
				if options.ib_part:
					ibpart = t[j].partition(options.ib_part, options.ib_num)[0]
				t[j] = (t[j], t[j].implicitblocks(options.ib_num, ibpart, options.ib_stanza))
		print >>sys.stderr, "done"

	if options.index_comp:
		print >>sys.stderr, "Index compression...",
		for i, t in enumerate(tb):
			for j in xrange(len(t)):
				print >>sys.stderr, "%d/%d" % (i, j),
				if type(t[j]) == tuple:
					t[j] = (t[j][0].index_comp(), t[j][1])
				else:
					t[j] = t[j].index_comp()
		print >>sys.stderr, "done"
	akxobj = akx.cgen(options.k, matrix.shape[0], tb, basis=(options.n != 0))

vecs = numpy.zeros((1 + options.m, matrix.shape[0]))
vecs[0] = 1.0
if options.n == 0:
	proc = lambda akxobj: akxobj.powers(vecs)
else:
	coeffs = numpy.empty(options.m)
	coeffs.fill(options.n)
	proc = lambda akxobj: akxobj.newton(vecs, coeffs)

useful_flops = options.m * 2 * matrix.nnz
seconds = akx.benchmark(akxobj, proc)
Mflops_s = 1.0e-6 * useful_flops / seconds
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
	"*   runtime = %g ms\n"+
	"* spmv-perf = %g Mflops/s\n"+
	"*\n"+
	"**************************\n") %
	(filename, matrix.nnz, matrix.shape[0], options.m, 1000 * seconds, Mflops_s))

print >>sys.stderr, "Verifying..."
vecscheck = numpy.empty((1 + options.m, matrix.shape[0]))
vecscheck[0] = vecs[0]
for i in xrange(1, len(vecscheck)):
	vecscheck[i] = matrix * vecscheck[i-1] - options.n * vecscheck[i-1]
print >>sys.stderr, "Maximum absolute error:", abs(vecs - vecscheck).max()
print >>sys.stderr, "Maximum relative error:", abs((vecs - vecscheck) / vecscheck).max()

if options.output:
	f = open(options.output, "w")
	print >>f, "Krylov vectors:"
	print >>f, "".join("x_%d\t" % n for n in xrange(options.m + 1))
	for element in vecs.transpose():
		print >>f, "".join("%.3g\t" % n for n in element)
