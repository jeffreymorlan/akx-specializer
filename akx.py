patoh_path = '/home/eecs/jmorlan/PaToH'
iomp5_path = None #'/home/eecs/jmorlan/mkl'
use_mkl = False

if use_mkl:
	from ctypes import CDLL, RTLD_GLOBAL, c_int, c_double, POINTER
	omp = CDLL("libiomp5.so", mode=RTLD_GLOBAL)
	mkl = CDLL("libmkl_rt.so") 
else:
	mkl = None

class AkxObjectPy(object):
	def __init__(self, matrix):
		self.matrix = matrix

	def powers(self, vecs, coeffs=None):
		for i in xrange(1, len(vecs)):
			vecs[i] = self.matrix * vecs[i-1]
			if coeffs:
				vecs[i] += vecs[i-1] * coeffs[i-1]

def benchmark(akxobj, vecs):
	import time
	seconds = -1.0
	n_iterations = 1
	akxobj.powers(vecs)   # Make sure code is compiled
	while seconds < 0.5:
		start = time.time()
		for i in xrange(n_iterations):
			akxobj.powers(vecs)
		end = time.time()
		seconds = end - start
		n_iterations *= 2
	return seconds, n_iterations / 2

def tile(block):
	m, n = block.shape()
	print "%dx%d" % (m, n),
	# Choose size that minimizes size of block in memory
	sizes = []
	for b_m in xrange(1, 9):
		b_n = b_m
		#print "  %dx%d:" % (b_m, b_n),
		tiles = block.tilecount(b_m, b_n, 10000)
		bytes = (4 * ((m + b_m - 1) / b_m + 1) # browptr
					+  4 * tiles                     # bcolidx
					+  8 * tiles * b_m * b_n)        # bvalues
		#print bytes
		sizes.append((bytes, b_m, b_n))
	bytes, b_m, b_n = min(sizes)
	if b_m != 1 or b_n != 1:
		print "Tiling to %dx%d" % (b_m, b_n)
		return block.tile(b_m, b_n, 0)
	return block

def tune(matrix, k, vecs):
	_akx = _make_module()

	results = []
	for nthreads in 4, 8:
		for usepatoh in 0, 1:

			tbpart = None
			if usepatoh:
				tbpart, tbsizes, tbcut = _akx.tb_partition(matrix.indptr, matrix.indices, matrix.data, k, nthreads)
			tb = _akx.threadblocks(matrix.indptr, matrix.indices, matrix.data, k, nthreads, tbpart)

			for tiling in 0, 1:
				akxobj = _powers_cgen(k, matrix.shape[0], [[b] for b in tb])
				seconds, n_iterations = benchmark(akxobj, vecs)
				print "%2d/%d | %d | ---,- | %g" % (nthreads, usepatoh, tiling, seconds / n_iterations)
				results.append((seconds / n_iterations, nthreads, usepatoh, tiling, 0, 0))

				if k != 1:
					for nblocks in 1, 2, 4, 8, 16, 32, 64, 128, 256:
						for stanza in 0, 1:
							for i in xrange(len(tb)):
								tb[i].implicitblocks(nblocks, None, stanza)
							akxobj = _powers_cgen(k, matrix.shape[0], [[b] for b in tb])
							seconds, n_iterations = benchmark(akxobj, vecs)
							print "%2d/%d | %d | %3d,%d | %g" % (nthreads, usepatoh, tiling, nblocks, stanza, seconds / n_iterations)
							results.append((seconds / n_iterations, nthreads, usepatoh, tiling, nblocks, stanza))

				if tiling == 0:
					for i in xrange(len(tb)):
						tb[i].implicitblocks()
						tb[i] = tile(tb[i])

	best, nthreads, usepatoh, tiling, nblocks, stanza = min(results)
	tbpart = None
	if usepatoh:
		tbpart, tbsizes, tbcut = _akx.tb_partition(matrix.indptr, matrix.indices, matrix.data, k, nthreads)
	tb = _akx.threadblocks(matrix.indptr, matrix.indices, matrix.data, k, nthreads, tbpart)
	if tiling:
		tb = map(tile, tb)
	if nblocks:
		for block in tb:
			block.implicitblocks(nblocks, None, stanza)
	akxobj = _powers_cgen(k, matrix.shape[0], [[b] for b in tb])
	return akxobj

def gram_matrix(vecs):
	import numpy
	vcount, vsize = vecs.shape
	if vecs.strides[1] != 8:
		raise ValueError('not contiguous')
	if mkl:
		gram = numpy.zeros((vcount, vcount))
		mkl.cblas_dsyrk(
			c_int(102), c_int(121), c_int(112), c_int(vcount), c_int(vsize), c_double(1),
			vecs.ctypes.data_as(POINTER(c_double)), c_int(vecs.strides[0] / 8),
			c_double(0), gram.ctypes.data_as(POINTER(c_double)), c_int(vcount))
		for i in xrange(vcount - 1):
			gram[i][i+1:] = gram.transpose()[i][i+1:]
		return gram
	else:
		return numpy.dot(vecs, vecs.transpose())

def combine_vecs(invecs, d, outvecs):
	import numpy
	incount, vsize = invecs.shape
	outcount, vsize2 = outvecs.shape
	outcount2, incount2 = d.shape
	if incount != incount2 or outcount != outcount2 or vsize != vsize2:
		raise ValueError('size mismatch')
	if not(d.strides[1] == invecs.strides[1] == outvecs.strides[1] == 8):
		raise ValueError('not contiguous')
	if mkl:
		mkl.cblas_dgemm(c_int(102), c_int(111), c_int(111), c_int(vsize), c_int(outcount), c_int(incount), c_double(1),
			invecs.ctypes.data_as(POINTER(c_double)), c_int(invecs.strides[0] / 8),
			d.ctypes.data_as(POINTER(c_double)), c_int(d.strides[0] / 8),
			c_double(0), outvecs.ctypes.data_as(POINTER(c_double)), c_int(outvecs.strides[0] / 8))
	else:
		outvecs[:] = numpy.dot(d, invecs)

_akx = None
toolchain = None
template_powers = None

def _make_module():
	global _akx
	if _akx:
		return _akx

	global asp, codepy
	import asp.codegen.templating.template
	import codepy.jit
	import codepy.toolchain
	import os

	global toolchain, template_powers

	thisdir = os.path.dirname(__file__)

	toolchain = codepy.toolchain.guess_toolchain()
	toolchain.cc = "gcc"
	toolchain.cflags = ["-O3", "-march=core2", "-msse3", "-fPIC"]
	toolchain.include_dirs.append(thisdir)
	toolchain.add_library('PaToH',
		[patoh_path],
		[patoh_path],
		['patoh']);
	if iomp5_path:
		toolchain.cflags.append('-fopenmp')
		toolchain.add_library('Intel OpenMP',
			[],
			[iomp5_path],
			['iomp5']);

	_akx = codepy.jit.extension_from_string(toolchain, '_akx_static',
		open(os.path.join(thisdir, 'akx-static.c')).read(),
		source_name='akx-static.c', debug=True)

	template_powers = asp.codegen.templating.template.Template(
		filename=os.path.join(thisdir, 'akx-powers.tpl'))

	return _akx

def _powers_cgen(k, matrix_size, blocks, usecoeffs=False):
	variants = set()
	for tb in blocks:
		for block in tb:
			# TODO: make sure type == AkxBlock
			variants.add(block.variant())
	#print variants

	# TODO: validate, make sure k, matrix size match

	module = codepy.jit.extension_from_string(toolchain, '_akx_powers',
		template_powers.render(variants=variants, usecoeffs=usecoeffs),
		source_name='akx-powers.c', debug=False)
	return module.AkxObjectC(k, matrix_size, blocks)
