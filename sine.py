import kalman
import matplotlib.pyplot as plt
from numpy import *
from scipy.integrate import odeint
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
	def __init__(self):
		self.stepsused = []

	def F(self, state):
		x = state[0][0]
		omega = state[2][0]
		return array([
			[1.0, Ts, 0.0],
			[-(omega**2) * Ts, 1.0, -2 * omega * x * Ts],
			[0.0, 0.0, 1.0]])

	def step(self, state):
		x, xdot, omega = state[0][0], state[1][0], state[2][0]
		coeff = -(omega ** 2)
		def func(y, t):
			return [coeff * y[1], y[0]]
		results, info = odeint(func, [xdot, x], [0, Ts], full_output=True)
		self.stepsused.append(info['nst'][0])
		xdot, x = results[1]
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

results = []

true_amplitude = 1
true_rv = norm(0, 0.3)
true_theta = 0
for t in xrange(int(400 / Ts) + 1):
	true_omega = 0.75 + 0.5 * sin(t * Ts * math.pi / 100)
	true_theta = true_theta + Ts * true_omega
	filt.predict(process)

	true = true_amplitude * math.sin(true_theta)
	noisy = true + true_rv.rvs()
	filt.update(measurement, noisy)

	results.append([t * Ts, true, noisy, filt.x[0][0], filt.x[1][0], true_omega, filt.x[2][0]])

print "integration steps: min=%s mean=%s max=%s" % (min(process.stepsused), sum(process.stepsused) / len(process.stepsused), max(process.stepsused))
print "estimated final covariance:"
print filt.P

results = vstack(results)

plt.subplot(211)
plt.title('Sine wave')
plt.plot(results[:,0], results[:,2], label='Measurement')
plt.plot(results[:,0], results[:,3] - results[:,1], label='Residual')
plt.legend()

plt.subplot(212)
plt.title('Frequency')
plt.plot(results[:,0], results[:,5], label='True')
plt.plot(results[:,0], results[:,6], label='Estimate')
plt.legend()

plt.show()
