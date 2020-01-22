from abc import ABC, abstractmethod
from typing import Dict, Iterable
import tqdm
import scipy.stats


class ABCModel:
    def __init__(
            self,
            priors:Dict[str, object],
            simulator:callable, 
            summary:callable,
            distance:callable,
        ):
        self.priors = [priors]
        self.simulator = simulator
        self.summary = summary
        self.distance = distance


class Sampler(ABC):
    def __init__(self, model:ABCModel, observed:Iterable):
        self.model = model
        self.observed = observed

        self.acceptance_rates = []
        self.outputs = []
        

    @abstractmethod
    def sample(self):
        pass

    @staticmethod  # ? should this be a non-static method?
    def draw(priors:Dict[str, object]) -> Dict[str, float]:
        # ? replace try/except with checking for prior __module__ ? or __class__?
        try:
            return {
                group:prior.rvs() for group, prior in priors.items()
            }
        except:
            return {
                group:prior.resample(1)[0, 0] for group, prior in priors.items()
            }

    @staticmethod
    def append_samples(sample, storage):
        for k, v in sample.items():
            storage[k].append(v)


class REJ(Sampler):
    def __init__(self, model:ABCModel, observed:Iterable):
        super().__init__(model, observed)

    def sample(
            self, 
            n_samples:int, 
            threshold:float,
            iter_max:int=None
        ):
        n_iter, n_accepted = 0, 0
        accepted_samples = {
                name:[] for name in self.model.priors[0].keys()
            }

        pbar = tqdm.tqdm(total=n_samples)
        while n_accepted < n_samples:
            # sample from current priors
            samples = self.draw(self.model.priors[0])
            # simulate
            sim = self.model.simulator(samples)
            # summary statistics
            s = self.model.summary(sim)
            # distance
            d = self.model.distance(s, self.observed)
            # acceptance criteria
            if d < threshold:
                n_accepted += 1
                self.append_samples(samples, accepted_samples)
                pbar.update(1)
            n_iter += 1
        
        # store accepted samples for epoch
        self.outputs.append(accepted_samples)
        # store acceptance rate for epoch
        self.acceptance_rates.append(n_accepted / n_iter)


class SMC(Sampler):
    def __init__(self, model:ABCModel, observed:Iterable, kde:callable=None):
        super().__init__(model, observed)
        self.kde = scipy.stats.kde.gaussian_kde if not kde else kde
        self.thresholds = []

    def sample(
            self, 
            n_samples:int, 
            thresholds:Iterable[float],
            iter_max:int=None
        ):
        for n_epoch, threshold in enumerate(thresholds):
            
            print(f"Beginning epoch: {n_epoch + 1}")
            # initialize epoch
            n_iter, n_accepted = 0, 0
            accepted_samples = {
                name:[] for name in self.model.priors[n_epoch].keys()
            }

            pbar = tqdm.tqdm(total=n_samples)
            while n_accepted < n_samples:
                # sample from current priors
                samples = self.draw(self.model.priors[n_epoch])
                # simulate
                sim = self.model.simulator(samples)
                # summary statistics
                s = self.model.summary(sim)
                # distance
                d = self.model.distance(s, self.observed)
                # acceptance criteria
                if d < threshold:
                    n_accepted += 1
                    self.append_samples(samples, accepted_samples)
                    pbar.update(1)
                n_iter += 1
            del pbar
            
            # store accepted samples for epoch
            self.outputs.append(accepted_samples)
            # store acceptance rate for epoch
            self.acceptance_rates.append(n_accepted / n_iter)

            # generate new priors from accepted samples
            smc_priors = {}
            for name, values in accepted_samples.items():
                kde = self.kde(values)
                smc_priors[name]= kde
            self.model.priors.append(smc_priors)
            self.thresholds.append(threshold)
