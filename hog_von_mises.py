#%%
import pystan

import matplotlib.pyplot as plt
import numpy
import os
import itertools
import math
import scipy.integrate
import mahotas
import collections
import skimage.measure, skimage.io, skimage.feature, skimage.util, skimage.filters
import seaborn
import random

#
#,,
#             '/home/bbales2/microhog/rafting_rotated_2d/2na/9/signalx.png',
#             '/home/bbales2/microhog/rafting_rotated_2d/ah/9/signalx.png''/home/bbales2/web/hog/static/images/renen5strain02.png',
#             '/home/bbales2/web/hog/static/images/renen5strain22.png'
#
ims2 = []
for path in ['/home/bbales2/web/hog/static/images/molybdenum0.png',
             '/home/bbales2/web/hog/static/images/molybdenum1.png']:
    im = skimage.io.imread(path, as_grey = True).astype('float')

    im = skimage.transform.rescale(im, 0.25)

    im -= im.mean()
    im /= im.std()

    stats = []
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if i == 0:
                dy = im[i + 1, j] - im[i, j]
            elif i == im.shape[0] - 1:
                dy = im[i, j] - im[i - 1, j]
            else:
                dy = (im[i + 1, j] - im[i - 1, j]) / 2.0

            if j == 0:
                dx = im[i, j + 1] - im[i, j]
            elif j == im.shape[1] - 1:
                dx = im[i, j] - im[i, j - 1]
            else:
                dx = (im[i, j + 1] - im[i, j - 1]) / 2.0

            angle = (numpy.arctan2(dy, dx) + numpy.pi)# / (2.0 * numpy.pi)
            mag = numpy.sqrt(dy**2 + dx**2)

            stats.append((angle, mag))

    stats = numpy.array(stats)

    plt.imshow(im)
    plt.show()

    1/0

    hog = microstructure.features.hog2(im, bins = 20, stride = 1, sigma = 1.0)

#%%
stats[:, 0] = stats[:, 0] - stats[:, 0].min()
#%%
idxs = range(len(stats))

random.shuffle(idxs)

seaborn.distplot(stats[idxs[:10000], 0])
plt.show()
seaborn.distplot(stats[idxs[:10000], 1])
plt.show()

idxs = numpy.argsort(stats[:, 1])[-2000:]

seaborn.distplot(stats[idxs, 0])
plt.show()
#%%

model_code = """
data {
  int<lower=1> K; //Number of Von Mises distributions to fit
  int<lower=1> N;
  real<lower=0.0, upper=2.0 * pi()> y[N];
}

parameters {
  real<lower=0.0, upper=2.0 * pi()> mu[K];
  simplex[K + 1] theta;
  real<lower=0.0> kappa[K];
}

model {
  real ps[K + 1];

  for (k in 1:K) {
    kappa[k] ~ normal(5.0, 10.0);
  }

  for (n in 1:N) {
    for (k in 1:K) {
      ps[k] <- log(theta[k]) + von_mises_log(y[n], mu[k], kappa[k]);
    }

    ps[K + 1] <- log(theta[K + 1]) + uniform_log(y[n], 0.0, 2.0 * pi());

    increment_log_prob(log_sum_exp(ps));
  }
}

generated quantities {
  real out;

  {
    int k;

    k <- categorical_rng(theta);

    if (k <= K)
    {
      out <- von_mises_rng(mu[k], kappa[k]);
    }
    else
    {
      out <- uniform_rng(0.0, 2.0 * pi());
    }
  }
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%
N = 100
idxs2 = range(len(idxs))

random.shuffle(idxs2)
seaborn.distplot(stats[idxs[idxs2[:N]], 0], bins = 20)
plt.show()

samples = stats[idxs[idxs2[:N]], 0]
#%%

fit = sm.sampling(data = {
    'K' : 4,
    'N' : N,
    'y' : samples
})
#%%
print fit

plt.hist(fit.extract()['out'][2000:] % (2.0 * numpy.pi), normed = True, alpha = 0.5)
plt.hist(samples, normed = True, alpha = 0.5)
plt.show()
