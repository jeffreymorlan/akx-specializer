specialize = False
patoh_path = '/home/eecs/jmorlan/PaToH'
iomp5_path = None #'/home/eecs/jmorlan/mkl'
use_mkl = False

if use_mkl:
	from ctypes import CDLL, RTLD_GLOBAL, c_int, c_double, POINTER
	omp = CDLL("libiomp5.so", mode=RTLD_GLOBAL)
	mkl = CDLL("libmkl_rt.so") 
else:
	mkl = None

class AkxObject(object):
	def __new__(cls, matrix):
		if matrix.getformat() == 'csr':
			pass
		elif matrix.getformat() == 'bsr':
			b_m, b_n = matrix.blocksize
			if b_m != b_n:
				raise ValueError("matrix has non-square tiles")
		else:
			raise TypeError("matrix should be csr or bsr, but was %s" % type(matrix))

		# Redirect to C implementation if possible
		if specialize:
			module = _make_module()
			return module.AkxObjectC(matrix.indptr, matrix.indices, matrix.data)
		return object.__new__(cls)

	def __init__(self, matrix):
		self.matrix = matrix

	def threadblocks(self, *params):
		pass

	def cacheblocks(self, *params):
		pass

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

def tile(akxobj):
	for i in xrange(akxobj.num_threadblocks()):
		for j in xrange(akxobj.num_blocks(i)):
			print "Block %d/%d:" % (i, j),
			m, n = akxobj.block_shape(i, j)
			print "%dx%d" % (m, n),
			# Choose size that minimizes size of block in memory
			sizes = []
			for b_m in xrange(1, 9):
				b_n = b_m
				#print "  %dx%d:" % (b_m, b_n),
				tiles = akxobj.block_tilecount(i, j, b_m, b_n, 10000)
				bytes = (4 * ((m + b_m - 1) / b_m + 1) # browptr
							+  4 * tiles                     # bcolidx
							+  8 * tiles * b_m * b_n)        # bvalues
				#print bytes
				sizes.append((bytes, b_m, b_n))
			bytes, b_m, b_n = min(sizes)
			if b_m != 1 or b_n != 1:
				print "Tiling to %dx%d" % (b_m, b_n)
				akxobj.block_tile(i, j, b_m, b_n, 0)

def tune(akxobj, k, vecs):
	results = []

	for nthreads in 4, 8:
		for usepatoh in 0, 1:
			akxobj.threadblocks(k, usepatoh, nthreads)
			for tiling in 0, 1:
				seconds, n_iterations = benchmark(akxobj, vecs)
				print "%2d/%d | %d | ---,- | %g" % (nthreads, usepatoh, tiling, seconds / n_iterations)
				results.append((seconds / n_iterations, nthreads, usepatoh, tiling, 0, 0))

				if k != 1:
					for nblocks in 1, 2, 4, 8, 16, 32, 64, 128, 256:
						for stanza in 0, 1:
							akxobj.implicitblocks(0, nblocks, stanza)
							seconds, n_iterations = benchmark(akxobj, vecs)
							print "%2d/%d | %d | %3d,%d | %g" % (nthreads, usepatoh, tiling, nblocks, stanza, seconds / n_iterations)
							results.append((seconds / n_iterations, nthreads, usepatoh, tiling, nblocks, stanza))

				if tiling == 0:
					akxobj.implicitblocks()
					tile(akxobj)

	best, nthreads, usepatoh, tiling, nblocks, stanza = min(results)
	akxobj.threadblocks(k, usepatoh, nthreads)
	if tiling:
		tile(akxobj)
	if nblocks:
		akxobj.implicitblocks(0, nblocks, stanza)

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

toolchain = None
static_code = None
template_powers = None

def _make_module(**args):
	global asp, codepy, os
	import asp.codegen.templating.template
	import codepy.jit
	import codepy.toolchain
	import os

	#print "compiling, params:", args

	global static_code, toolchain
	if not static_code:
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

		static_code = open(os.path.join(thisdir, 'akx-static.c')).read()

	return codepy.jit.extension_from_string(toolchain, '_akx_static',
		static_code, source_name='akx-static.c', debug=True)

def _powers_codegen(akxobj, *args):
	#print "_powers_codegen called"

	orig_b_m, orig_b_n = akxobj.orig_tilesize()

	variants = set()
	for tb in xrange(akxobj.num_threadblocks()):
		for eb in xrange(akxobj.num_blocks(tb)):
			variants.add(akxobj.block_variant(tb, eb))
			#print "TB %2d: %s %s" % (tb, akxobj.block_shape(tb, eb), akxobj.threadblock_tilesize(tb, eb))

	global template_powers
	if not template_powers:
		thisdir = os.path.dirname(__file__)
		template_powers = asp.codegen.templating.template.Template(
			filename=os.path.join(thisdir, 'akx-powers.tpl'))

	module = codepy.jit.extension_from_string(toolchain, '_akx_powers',
		template_powers.render(orig_b_m=orig_b_m, orig_b_n=orig_b_n, variants=variants,
		                       usecoeffs=(len(args) > 1)),
		source_name='akx-powers.c', debug=False)
	return module.powers
