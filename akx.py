"""Matrix powers kernel specializer."""

import akxconfig
from ctypes import CDLL, RTLD_GLOBAL, c_char, c_int, c_double, POINTER
import numpy
import time

__all__ = [
	"AkxObjectPy",
	"tb_partition", "threadblocks", "AkxBlock", "AkxImplicitSeq",
	"benchmark", "tile", "partition", "cgen",
	"tune",
	"gram_matrix", "combine_vecs"
]

if akxconfig.use_mkl:
	omp = CDLL("libiomp5.so", mode=RTLD_GLOBAL)
	mkl = CDLL("libmkl_rt.so") 
elif akxconfig.use_acml:
	acml = CDLL("libacml_mp.so")

if akxconfig.specialize:
	import asp.codegen.templating.template
	import codepy.jit
	import codepy.toolchain
	import os

	thisdir = os.path.dirname(__file__)

	toolchain = codepy.toolchain.guess_toolchain()
	toolchain.cc = "gcc"
	toolchain.cflags = ["-O3", "-march=core2", "-msse3", "-fPIC"]
	toolchain.include_dirs.append(thisdir or '.')
	toolchain.add_library('PaToH',
		[akxconfig.patoh_path],
		[akxconfig.patoh_path],
		['patoh']);
	if akxconfig.iomp5_path:
		toolchain.cflags.append('-fopenmp')
		toolchain.add_library('Intel OpenMP',
			[],
			[akxconfig.iomp5_path],
			['iomp5']);

	# Load the static C code and import its symbols
	_akx = codepy.jit.extension_from_string(toolchain, '_akx_static',
		open(os.path.join(thisdir, 'akx-static.c')).read(),
		source_name='akx-static.c')
	tb_partition = _akx.tb_partition
	threadblocks = _akx.threadblocks
	AkxBlock = _akx.AkxBlock
	AkxImplicitSeq = _akx.AkxImplicitSeq

	template_powers = asp.codegen.templating.template.Template(
		filename=os.path.join(thisdir, 'akx-powers.tpl'))

class AkxObjectPy(object):
	"""Naive pure-Python implemention of matrix powers kernel."""
	def __init__(self, matrix):
		self.matrix = matrix

	def powers(self, vecs):
		"""Monomial basis: x_{i+1} = Ax_i"""
		for i in xrange(1, len(vecs)):
			vecs[i] = self.matrix * vecs[i-1]

	def newton(self, vecs, coeffs):
		"""Newton basis: x_{i+1} = Ax_i - \lambda_i*x_i"""
		for i in xrange(1, len(vecs)):
			vecs[i] = self.matrix * vecs[i-1] - coeffs[i-1] * vecs[i-1]

def benchmark(akxobj, proc):
	"""Benchmark an AkxObject; return average time per invocation in seconds."""
	seconds = -1.0
	n_iterations = 1
	proc(akxobj)
	while seconds < 0.5:
		start = time.time()
		for i in xrange(n_iterations):
			proc(akxobj)
		end = time.time()
		seconds = end - start
		n_iterations *= 2
	return seconds / (n_iterations / 2)

def tile(block, samples=10000):
	"""Tile a block, choosing the tile size that minimizes memory footprint."""
	m, n = block.shape()
	sizes = []
	for b_m in xrange(1, 9):
		b_n = b_m
		tiles = block.tilecount(b_m, b_n, samples)
		bytes = (4 * ((m + b_m - 1) / b_m + 1) # browptr
					+  4 * tiles                     # bcolidx
					+  8 * tiles * b_m * b_n)        # bvalues
		sizes.append((bytes, b_m, b_n))
	bytes, b_m, b_n = min(sizes)
	if b_m != 1 or b_n != 1:
		return block.tile(b_m, b_n, 0)
	return block

def partition(matrix, k, nthreads, filename=None):
	"""Compute a partitioning of a matrix into thread blocks, optionally caching in a file."""
	if nthreads == 1:
		return numpy.zeros(matrix.shape[0], dtype=numpy.int32)

	if filename is None:
		return tb_partition(matrix.indptr, matrix.indices, matrix.data, k, nthreads)[0]

	try:
		f = open('%s_%d_%d' % (filename, k, nthreads), 'rb')
		return numpy.fromfile(f, dtype=numpy.int32, count=matrix.shape[0])
	except IOError:
		tbpart = tb_partition(matrix.indptr, matrix.indices, matrix.data, k, nthreads)[0]
		open('%s_%d_%d' % (filename, k, nthreads), 'wb').write(tbpart)
		return tbpart

def cgen(k, matrix_size, blocks, basis=0):
	"""Create an AkxObjectC, generating necessary code."""
	variants = set()
	for tb in blocks:
		for block in tb:
			if type(block) == tuple:
				if type(block[0]) != AkxBlock or type(block[1]) != AkxImplicitSeq:
					raise TypeError('Must be AkxBlock or (AkxBlock,AkxImplicitSeq) pair')
				variants.add(block[0].variant())
			else:
				if type(block) != AkxBlock:
					raise TypeError('Must be AkxBlock or (AkxBlock,AkxImplicitSeq) pair')
				variants.add(block.variant())

	module = codepy.jit.extension_from_string(toolchain, '_akx_powers',
		template_powers.render(variants=variants, basis=basis),
		source_name='akx-powers.c')
	return module.AkxObjectC(k, matrix_size, blocks)

def tune(matrix, k, symmetric, basis=0, filename=None, show=None, use_exp=True, use_imp=True):
	"""Create an efficient AkxObject by auto-tuning."""
	if not akxconfig.specialize:
		return AkxObjectPy(matrix)

	vecs = numpy.ones((1 + k, matrix.shape[0]))
	if basis == 0:
		proc = lambda akxobj: akxobj.powers(vecs)
	elif basis == 1:
		coeffs = numpy.ones(k)
		proc = lambda akxobj: akxobj.newton(vecs, coeffs)
	else:
		raise ValueError('unknown basis type')

	best = (float('Inf'), None)
	for nthreads in akxconfig.thread_counts:
		tbpart = partition(matrix, 1, nthreads, filename)

		if use_exp and k > 1:
			tb = threadblocks(matrix.indptr, matrix.indices, matrix.data, k, nthreads, tbpart)
			tb_flops = sum(b.flopcount() for b in tb)
			flops = tb_flops
			tb = [[b] for b in tb]

			breaking = False
			for maxsize in 4000000, 2000000, 1000000, 500000, 250000:
				for i in xrange(nthreads):
					pending = tb[i]
					tb[i] = []
					while pending:
						block = pending.pop()
						size = 4*(block.shape()[0]+1) + 12*block.nnzb()
						if size < maxsize:
							tb[i].append(block)
						else:
							cbpart = block.partition(1, 2)[0]
							pending.extend(block.split(2, cbpart))
							flops -= block.flopcount()
							flops += (pending[-2].flopcount() + pending[-1].flopcount())
							if flops > 2*tb_flops:
								# redundant work exceeds useful work, probably time to give up
								breaking = True
								break
					if breaking:
						break
				if breaking:
					break

				if symmetric:
					tb2 = [[tile(b).symm_opt().index_comp() for b in t] for t in tb]
				else:
					tb2 = [[tile(b).index_comp() for b in t] for t in tb]
				akxobj = cgen(k, matrix.shape[0], tb2, basis)
				seconds = benchmark(akxobj, proc)
				if show:
					print >>show, "%2d | X-%7d (%9d) | %g" % (nthreads, maxsize, flops, seconds)
				best = min(best, (seconds, akxobj))

		if use_imp or k == 1:
			tb = threadblocks(matrix.indptr, matrix.indices, matrix.data, k, nthreads, tbpart)
			tb = map(tile, tb)
			for symm in [0, 1][:symmetric+1]:
				if symm:
					tb = [b.symm_opt() for b in tb]

				akxobj = cgen(k, matrix.shape[0], [[b.index_comp()] for b in tb], basis)
				seconds = benchmark(akxobj, proc)
				if show:
					print >>show, "%2d | %d | ----- | %g" % (nthreads, symm, seconds)
				best = min(best, (seconds, akxobj))

				if k != 1:
					for nblocks in 2, 4, 8, 16, 32, 64, 128, 256:
						ib = [b.implicitblocks(nblocks, None, True) for b in tb]
						akxobj = cgen(k, matrix.shape[0], [[(b.index_comp(), i)] for b, i in zip(tb, ib)], basis)
						seconds = benchmark(akxobj, proc)
						if show:
							print >>show, "%2d | %d | I-%3d | %g" % (nthreads, symm, nblocks, seconds)
						best = min(best, (seconds, akxobj))

	return best[1]

def gram_matrix(vecs):
	"""Compute the dot product of each pair of vectors: G = V^T * V."""
	vcount, vsize = vecs.shape
	if vecs.strides[1] != 8:
		raise ValueError('not contiguous')
	if akxconfig.use_mkl:
		gram = numpy.zeros((vcount, vcount))
		mkl.cblas_dsyrk(
			c_int(102), c_int(121), c_int(112), c_int(vcount), c_int(vsize), c_double(1),
			vecs.ctypes.data_as(POINTER(c_double)), c_int(vecs.strides[0] / 8),
			c_double(0), gram.ctypes.data_as(POINTER(c_double)), c_int(vcount))
		for i in xrange(vcount - 1):
			gram[i][i+1:] = gram.transpose()[i][i+1:]
		return gram
	elif akxconfig.use_acml:
		gram = numpy.zeros((vcount, vcount))
		acml.dsyrk(
			c_char('U'), c_char('T'), c_int(vcount), c_int(vsize), c_double(1),
			vecs.ctypes.data_as(POINTER(c_double)), c_int(vecs.strides[0] / 8),
			c_double(0), gram.ctypes.data_as(POINTER(c_double)), c_int(vcount))
		for i in xrange(vcount - 1):
			gram[i][i+1:] = gram.transpose()[i][i+1:]
		return gram
	else:
		return numpy.dot(vecs, vecs.transpose())

def combine_vecs(invecs, d, outvecs):
	"""Linearly combine set of input vectors to produce set of output vectors."""
	incount, vsize = invecs.shape
	outcount, vsize2 = outvecs.shape
	outcount2, incount2 = d.shape
	if incount != incount2 or outcount != outcount2 or vsize != vsize2:
		raise ValueError('size mismatch')
	if not(d.strides[1] == invecs.strides[1] == outvecs.strides[1] == 8):
		raise ValueError('not contiguous')
	if akxconfig.use_mkl:
		mkl.cblas_dgemm(c_int(102), c_int(111), c_int(111), c_int(vsize), c_int(outcount), c_int(incount), c_double(1),
			invecs.ctypes.data_as(POINTER(c_double)), c_int(invecs.strides[0] / 8),
			d.ctypes.data_as(POINTER(c_double)), c_int(d.strides[0] / 8),
			c_double(0), outvecs.ctypes.data_as(POINTER(c_double)), c_int(outvecs.strides[0] / 8))
	elif akxconfig.use_acml:
		acml.dgemm(c_char('N'), c_char('N'), c_int(vsize), c_int(outcount), c_int(incount), c_double(1),
			invecs.ctypes.data_as(POINTER(c_double)), c_int(invecs.strides[0] / 8),
			d.ctypes.data_as(POINTER(c_double)), c_int(d.strides[0] / 8),
			c_double(0), outvecs.ctypes.data_as(POINTER(c_double)), c_int(outvecs.strides[0] / 8))
	else:
		outvecs[:] = numpy.dot(d, invecs)
