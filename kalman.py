from numpy import *
from numpy.linalg import *

class kalman:
	def __init__(self, x, P, F, Q, H, R):
		self.x = x
		self.P = P
		self.F = F
		self.Q = Q
		self.H = H
		self.R = R

	def predict(self):
		self.x = dot(self.F, self.x)
		self.P = dot(self.F, dot(self.P, self.F.transpose())) + self.Q

	def update(self, z):
		y = z - dot(self.H, self.x)
		S = dot(self.H, dot(self.P, self.H.transpose())) + self.R
		K = dot(dot(self.P, self.H.transpose()), inv(S))
		self.x = self.x + dot(K, y)
		self.P = dot(eye(K.shape[0]) - dot(K, self.H), self.P)

class linearmotion(kalman):
	def __init__(self, dt, process_sigmasq, measurement_sigmasq):
		G = array([dt*dt*dt*dt/4, dt*dt*dt*dt/4, dt*dt*dt*dt/4, dt*dt/2, dt*dt/2, dt*dt/2, dt, dt, dt])
		kalman.__init__(self,
			x=zeros((9,1)),
			P=eye(9),
			F=array([
				[1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0,  dt, 0.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 1.0,  dt, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, 0.0],
				[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt],
				[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
			Q=dot(G.transpose(), G) * process_sigmasq,
			H=array([
				[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
			R=array([
				[measurement_sigmasq, 0.0, 0.0],
				[0.0, measurement_sigmasq, 0.0],
				[0.0, 0.0, measurement_sigmasq]])
		)

	def measure(self, *z):
		self.predict()
		self.update(array([z]).transpose())
		print self.x
