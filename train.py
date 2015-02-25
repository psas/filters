import kalman
import matplotlib.pyplot as plt
from numpy import *
from scipy.stats import norm

dt = 0.1
accel_sigma = 2.0
measurement_sigma = 1.0

process = kalman.process_model(
    F=array([
        [1, dt],
        [0, 1],
    ]),
    Q=array([
        [(dt**4)/4, (dt**3)/2],
        [(dt**3)/2, dt**2],
    ]) * (accel_sigma ** 2)
)

measure = kalman.observation_model(
    H=[[1.0, 0.0]],
    R=[[measurement_sigma ** 2]],
)

filt = kalman.kalman(
    x=zeros((2,1)),
    P=zeros((2,2)),
)

accel_rv = norm(0, accel_sigma)
measurement_rv = norm(0, measurement_sigma)

accelerations = (accel_rv.rvs() for _ in xrange(1000))
true_positions = []
true_velocities = []
estimated_positions = []
estimated_velocities = []

true_position = 0.0
true_velocity = 0.0
for acceleration in accelerations:
    filt.predict(process)
    measurement = true_velocity + measurement_rv.rvs()
    filt.update(measure, measurement)

    true_position += true_velocity * dt + acceleration * (dt ** 2) / 2
    true_velocity += acceleration * dt

    true_positions.append(true_position)
    true_velocities.append(true_velocity)
    estimated_positions.append(filt.x[0])
    estimated_velocities.append(filt.x[1])

plt.title('Velocity')
plt.plot(estimated_velocities, label='Estimate')
plt.plot(true_velocities, label='True')
plt.legend()
plt.show()
