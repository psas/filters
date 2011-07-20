from numpy import *
from numpy.linalg import *

class observation_model:
	def __init__(self, H, R):
		self.H = array(H)
		self.R = array(R)

class kalman:
	def __init__(self, x, P, F, Q):
		self.x = x
		self.P = P
		self.F = F
		self.Q = Q

	def predict(self):
		self.x = dot(self.F, self.x)
		self.P = dot(self.F, dot(self.P, self.F.transpose())) + self.Q

	def update(self, obs, *z):
		y = array([z]).transpose() - dot(obs.H, self.x)
		S = dot(obs.H, dot(self.P, obs.H.transpose())) + obs.R
		K = dot(dot(self.P, obs.H.transpose()), inv(S))
		self.x = self.x + dot(K, y)
		self.P = dot(eye(K.shape[0]) - dot(K, obs.H), self.P)
