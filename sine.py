import kalman
from numpy import *
from scipy.stats import norm

Ts = 0.2
process_sigmasq = 1e-2
measurement_sigmasq = 0.5

filt = kalman.kalman(
	x=array([[0.0, 1.0, 1.0]]).T,
	P=array([
		[2.0, 0.0, 0.0],
		[0.0, 2.0, 0.0],
		[0.0, 0.0, 2.0]]),
)

class SineProcess:
	def F(self, state):
		x = state[0][0]
		omega = state[2][0]
		return array([
			[1.0, Ts, 0.0],
			[-(omega**2) * Ts, 1.0, -2 * omega * x * Ts],
			[0.0, 0.0, 1.0]])

	def step(self, state):
		substeps = 100
		stepsize = Ts / substeps
		x, xdot, omega = state[0][0], state[1][0], state[2][0]
		coeff = -(omega ** 2)
		for _ in xrange(substeps):
			xdotdot = coeff * x
			xdot = xdot + stepsize * xdotdot
			x = x + stepsize * xdot
		return array([[x, xdot, omega]]).T

	def Q(self, state):
		deriv = state[2][0] * state[0][0] * Ts
		return process_sigmasq * Ts * array([
			[0.0, 0.0, 0.0],
			[0.0, 4.0/3.0 * deriv ** 2, -deriv],
			[0.0, -deriv, 1.0]])

process = SineProcess()

measurement = kalman.observation_model(
	H=[[1.0, 0.0, 0.0]],
	R=[[measurement_sigmasq]]
)

true_amplitude = 1
true_rv = norm(0, 0.3)
true_theta = 0
for t in xrange(int(1000 / Ts) + 1):
	true = true_amplitude * math.sin(true_theta)
	noisy = true + true_rv.rvs()
	filt.update(measurement, noisy)

	print t * Ts, true, noisy, filt.x[0][0], filt.x[1][0], filt.x[2][0]

	true_theta = true_theta + Ts * (1 + 0.5 * sin(t * Ts * math.pi / 100))
	filt.predict(process)
