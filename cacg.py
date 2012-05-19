import akx
import numpy
import time

def cg3_ca(A, b, tol=1e-5, callback=None, s=1, maxiter=1000, show_times=None):
	N = b.shape[0]

	xrV       = numpy.empty((2*s + 3, N))
	gamma     = numpy.empty(s + 1)
	rho       = numpy.empty(s + 1)
	old_xrV   = numpy.zeros((2*s + 3, N))
	old_gamma = numpy.ones(s + 1)
	old_rho   = numpy.ones(s + 1)
	mu        = None

	V = numpy.empty((2, N))
	V[0] = old_xrV[1]
	A.powers(V[:2])
	old_xrV[s+2] = b - V[1]

	powers_time = 0
	dots_time = 0
	d_time = 0
	r_time = 0

	for k in xrange(maxiter / s):
		gamma[0] = old_gamma[s]
		rho[0] = old_rho[s]

		start_time = time.time()
		A.powers(old_xrV[s+2:])
		powers_time += time.time() - start_time

		start_time = time.time()
		G = akx.gram_matrix(old_xrV[2:])
		dots_time += time.time() - start_time

		start_time = time.time()
		# d[j,i]: formula for (A^i*r_j) in terms of old_xrV entries
		d = numpy.zeros((2 * s + 3, s + 1, 2 * s + 1))
		for j in xrange(s):
			d[j+1,0][j] = 1.0    # old_r[j+1] = 1.0 * old_r[j+1]
		for i in xrange(s+1):
			d[s+1,i][s+i] = 1.0  # V[i] = 1.0 * V[i]
		for j in xrange(2, s+1):
			for i in xrange(1, j):
				d[j,i] = d[j-1,i-1] * (1 - old_rho[j]) / (old_rho[j] * old_gamma[j]) \
				       + d[j  ,i-1] / old_gamma[j] \
				       - d[j+1,i-1] / (old_rho[j] * old_gamma[j])

		for j in xrange(1, s+1):
			prev_mu = mu
			#mu = numpy.dot(r[j], r[j])
			mu = numpy.dot(d[s+j,0], numpy.dot(G, d[s+j,0]))
			if mu < tol:
				return old_xrV[1]
			#nu = numpy.dot(w, r[j])
			nu = numpy.dot(d[s+j,1], numpy.dot(G, d[s+j,0]))
			gamma[j] = mu / nu
			if k == 0 and j == 1:
				rho[j] = 1.0
			else:
				rho[j] = 1 / (1.0 - (gamma[j] / gamma[j-1]) * (mu / prev_mu) / rho[j-1])

			for i in xrange(0, s+1-j):
				d[s+j+1,i] = rho[j] * (d[s+j,i] - gamma[j] * d[s+j,i+1]) + (1 - rho[j]) * d[s+j-1,i]
		d2 = numpy.zeros((s+3, 2*s+3))
		for j in xrange(s+1):
			d2[2+j][2:] = d[s+1+j,0]
		d2[0][0] = 1.0
		d2[1][1] = 1.0
		for j in xrange(1, s+1):
			d2[0], d2[1] = d2[1], (rho[j] * (d2[1] + gamma[j] * d2[j+1]) + (1 - rho[j]) * d2[0])
		d_time += time.time() - start_time

		start_time = time.time()
		akx.combine_vecs(old_xrV, d2, xrV[:s+3])
		r_time += time.time() - start_time

		if callback is not None:
			callback(xrV[1])

		old_xrV,   xrV   = xrV,   old_xrV
		old_gamma, gamma = gamma, old_gamma
		old_rho,   rho   = rho,   old_rho

	if show_times:
		print >>show_times, "Powers:%.6f Dots:%.6f D:%.6f R:%.6f" % (powers_time, dots_time, d_time, r_time)
	return old_xrV[1]
