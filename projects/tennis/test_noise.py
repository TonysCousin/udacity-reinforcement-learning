# demonstrating the noise distribution
#
# theta default = 0.15. Increasing has effect of narrowing range of output
# (theta=0.4 limits to ~0.93, theta=0.8 limits to ~0.75)

import numpy as np
from ou_noise import OUNoise

NUM = 10000

def main():
    n = OUNoise(size=2, seed=44969, mu=0.0, theta=0.8, sigma=0.2)
    n.reset()
    print("initialized")

    sum = np.array([0.0, 0.0])
    min_val = np.array([999.0, 999.0])
    max_val = np.array([-999.0, -999.0])
    for i in range(NUM):
        sample = n.sample()
        #print("{:7.4f}  {:7.4f}".format(sample[0], sample[1]))
        sum += sample
        min_val = np.minimum(min_val, sample)
        max_val = np.maximum(max_val, sample)

    print("min = {:7.4f}  {:7.4f}".format(min_val[0], min_val[1]))
    print("max = {:7.4f}  {:7.4f}".format(max_val[0], max_val[1]))
    print("avg = {:7.4f}  {:7.4f}".format(sum[0]/NUM, sum[1]/NUM))


if __name__ == "__main__":
    main()
