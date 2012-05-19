#!/usr/bin/python

import sys
import time
import optparse

parser = optparse.OptionParser("usage: %prog [options] matrixfile")
parser.add_option("--scipy",        help="use scipy CG", action="store_true", default=False)
parser.add_option("-s", "--sejits", help="enable specialization", action="store_true", default=False)
parser.add_option("-m",             help="total number of steps", type="int", default=120)
parser.add_option("-k",             help="number of steps per iteration", type="int", default=1)
parser.add_option("--tb-num",       help="number of threads", type="int", default=0)
parser.add_option("-x",             help="use explicit cache blocking only", action="store_true", default=False)
parser.add_option("-i",             help="use implicit cache blocking only", action="store_true", default=False)

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
	if scipy.version.version == '0.6.0':
		matrix  = scipy.sparse.csr_matrix((data, indices, indptr), dims=(rows, cols))
	else:
		matrix  = scipy.sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
else:
	import scipy.io.mmio
	matrix = scipy.io.mmio.mmread(filename).tocsr()
print >>sys.stderr, "done"

b = numpy.ones(matrix.shape[0])

if options.scipy:
	if scipy.version.version == '0.6.0':
		import scipy.linalg as linalg
	else:
		import scipy.sparse.linalg as linalg
	for i in xrange(5):
		cg_time = time.time()
		x, info = linalg.cg(matrix, b, maxiter=options.m)
		cg_time = time.time() - cg_time
		print "time =", cg_time
	for i in xrange(5):
		cg_time = time.time()
		for j in xrange(options.m):
			dummy = matrix * x
		cg_time = time.time() - cg_time
		print "mul_time =", cg_time
else:
	print >>sys.stderr, "Initializing akx...",
	import akxconfig
	if options.tb_num:
		akxconfig.threadcounts = [options.tb_num]
	import akx
	print >>sys.stderr, "done"

	if not options.sejits:
		akxobj = akx.AkxObjectPy(matrix)
	else:
		tune_time = time.time()
		akxobj = akx.tune(matrix, options.k, True, show=sys.stdout, use_exp=not(options.i), use_imp=not(options.x))
		tune_time = time.time() - tune_time
		print "tuning time =", tune_time

	import cacg
	for i in xrange(5):
		cg_time = time.time()
		x = cacg.cg3_ca(akxobj, b, s=options.k, maxiter=options.m, tol=0, show_times=sys.stdout)
		cg_time = time.time() - cg_time
		print "time =", cg_time
residual = (matrix * x) - b
print "|r|^2 =", numpy.dot(residual, residual)
print "gradient =", .5 * numpy.dot(x, residual - b)
