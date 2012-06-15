import kalman
import matplotlib.pyplot as plt
from numpy import *
from scipy.integrate import odeint
from scipy.stats import norm

Ts = 0.2
process_sigmasq = 1e-2
measurement_sigmasq = 0.5

filt = kalman.kalman(
	x=array([[0.0, 0.0, 0.0, 0.0, 1.0]]).T,
	P=2 * eye(5),
)

class SineProcess:
	def __init__(self):
		self.stepsused = []

	def F(self, state):
		x, _, y, _, omega = ravel(state)
		vd = -(omega**2) * Ts
		vw = -2 * omega * Ts
		return array([
			[1.0,  Ts, 0.0, 0.0, 0.0   ],
			[ vd, 1.0, 0.0, 0.0, vw * x],
			[0.0, 0.0, 1.0,  Ts, 0.0   ],
			[0.0, 0.0,  vd, 1.0, vw * y],
			[0.0, 0.0, 0.0, 0.0, 1.0   ]])

	def integrate(self, coeff, *init):
		def func(y, t):
			return [coeff * y[1], y[0]]
		results, info = odeint(func, init, [0, Ts], full_output=True)
		self.stepsused.append(info['nst'][0])
		return results[1]

	def step(self, state):
		x, xdot, y, ydot, omega = ravel(state)
		coeff = -(omega ** 2)
		xdot, x = self.integrate(coeff, xdot, x)
		ydot, y = self.integrate(coeff, ydot, y)
		return array([[x, xdot, y, ydot, omega]]).T

	def Q(self, state):
		x, _, y, _, omega = ravel(state)
		vw = -2 * omega * Ts
		vwx = vw * x
		vwy = vw * y
		return process_sigmasq * Ts * array([
			[0.0,       0.0, 0.0,       0.0,   0.0],
			[0.0, vwx*vwx/3, 0.0, vwx*vwy/3, vwx/2],
			[0.0,       0.0, 0.0,       0.0,   0.0],
			[0.0, vwy*vwx/3, 0.0, vwy*vwy/3, vwy/2],
			[0.0,     vwx/2, 0.0,     vwy/2,   1.0],
		])

process = SineProcess()

measurement = kalman.observation_model(
	H=[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
	R=measurement_sigmasq * eye(2),
)

results = []

true_amplitude = 1
true_rv = norm(0, 0.3)
true_theta = 0
for t in xrange(int(400 / Ts) + 1):
	true_omega = 0.75 + 0.5 * sin(t * Ts * math.pi / 100)
	true_theta = true_theta + Ts * true_omega
	filt.predict(process)

	true = true_amplitude * array([math.cos(true_theta), math.sin(true_theta)])
	noisy = true + true_rv.rvs(size=2)
	filt.update(measurement, *noisy)

	results.append(hstack(([t * Ts, true_omega], true, noisy, ravel(filt.x))))

print "integration steps: min=%s mean=%s max=%s" % (min(process.stepsused), sum(process.stepsused) / len(process.stepsused), max(process.stepsused))
print "estimated final covariance:"
print filt.P

results = vstack(results)

plt.subplot(411)
plt.title('Real part')
plt.plot(results[:,0], results[:,4], label='Measurement')
plt.plot(results[:,0], results[:,6] - results[:,2], label='Residual')
plt.legend()

plt.subplot(412)
plt.title('Imaginary part')
plt.plot(results[:,0], results[:,5], label='Measurement')
plt.plot(results[:,0], results[:,8] - results[:,3], label='Residual')
plt.legend()

plt.subplot(413)
plt.title('Frequency')
plt.plot(results[:,0], results[:,1], label='True')
plt.plot(results[:,0], results[:,10], label='Estimate')
plt.legend()

plt.subplot(414)
plt.title('Amplitude')
plt.plot(results[:,0], true_amplitude * ones_like(results[:,0]), label='True')
plt.plot(results[:,0], sqrt(results[:,6] ** 2 + results[:,8] ** 2), label='Estimate')
plt.legend()

plt.show()
