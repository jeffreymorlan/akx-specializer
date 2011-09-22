#!/usr/bin/python2.5

import sys
import time
import optparse

parser = optparse.OptionParser("usage: %prog [options] matrixfile")
parser.add_option("--scipy",        help="use scipy CG", action="store_true", default=False)
parser.add_option("-s", "--sejits", help="enable specialization", action="store_true", default=False)
parser.add_option("-m",             help="total number of steps", type="int", default=120)
parser.add_option("-k",             help="number of steps per iteration", type="int", default=1)
parser.add_option("--tb-part",      help="thread block partitioning method (1 = hypergraph)", type="int", default=0)
parser.add_option("--tb-num",       help="number of threads", type="int", default=1)
parser.add_option("--tile",         help="use tiling", action="store_true", default=False)
parser.add_option("--cb-part",      help="cache block partitioning method (1 = hypergraph)", type="int", default=0)
parser.add_option("--cb-num",       help="number of cache blocks (0 = no cache blocking)", type="int", default=0)
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

b = numpy.ones(matrix.shape[0])

if options.scipy:
	import scipy.linalg
	for i in xrange(5):
		cg_time = time.time()
		x, info = scipy.linalg.cg(matrix, b, maxiter=options.m)
		cg_time = time.time() - cg_time
		print "time =", cg_time
	residual = (matrix * x) - b
	print "|r|^2 =", numpy.dot(residual, residual)
else:
	import cacg
	print >>sys.stderr, "Initializing akx...",
	import akx

	akx.specialize = options.sejits
	akxobj = akx.AkxObject(matrix)
	print >>sys.stderr, "done"

	if options.tb_num:
		print >>sys.stderr, "Creating thread blocks...",
		akxobj.threadblocks(options.k, options.tb_part, options.tb_num)
		print >>sys.stderr, "done"
	if options.tile:
		print >>sys.stderr, "Tiling thread blocks...",
		akx.tile(akxobj)
		print >>sys.stderr, "done"
	if options.cb_num:
		print >>sys.stderr, "Creating cache blocks...",
		akxobj.implicitblocks(options.cb_part, options.cb_num, options.cb_rle)
		print >>sys.stderr, "done"

	for i in xrange(5):
		cg_time = time.time()
		x = cacg.cg3_ca(akxobj, b, s=options.k, maxiter=options.m)
		cg_time = time.time() - cg_time
		print "time =", cg_time
	residual = (matrix * x) - b
	print "|r|^2 =", numpy.dot(residual, residual)
