# ABCsampler
![](https://img.shields.io/badge/Python-3-blue)

Simple model class and non-vectorized samplers for Approximate Bayesian Statistics (ABC) for use with slow simulator functions
where sampling speed doesn't really matter.

## How to use

### Model construction
```
from abcsamplers import ABCModel

def simulator(prior_sample):
    return list(prior_sample.values())


def summary(simulator_out):
    return np.sum(simulator_out)


def distance(s1, observed):
    return np.abs(observed - s1)


groups = ["Prior 1", "Prior 2", "Prior 3", "Prior 4"]
priors = {group:scipy.stats.norm(0, 1) for group in groups}

model = abcsamplers.ABCModel(priors, simulator, summary, distance
```

### Rejection sampling
```
from abcsamplers import REJ

observed = [4]
rej = abcsamplers.REJ(model, observed)
rej.sample(1000, threshold=0.1)
```

### Sequential Monte Carlo sampling
```
from abcsamplers import SMC

observed = [4]
smc = abcsamplers.SMC(model, observed)
thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
smc.sample(1000, thresholds)
```

## Features

- [x] Models wih simple hierarchy like `priors` -> `simulator` -> `summary` -> `distance`
- [ ] Graph-based model construction allowing for complex prior relationships, multiple summary statistics and distance functions


## Dependencies
`pip install tqdm scipy`
