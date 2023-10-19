#%%
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%%
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import collections

#%%
class ProbDist:
    def __init__(self, dim):
        self.dim = dim
        if self.dim == 1:
            pass
        else:
            raise NotImplementedError

#%%
class UnivPD(ProbDist):
    def __init__(self, dim = 1):
        super().__init__(dim)

class MultivPD(ProbDist):
    def __init__(self, dim):
        super().__init__(dim)

#a = UnivPD()
#b = MultivPD(dim = 2) #same as univProbDist, will raise Error if dimensionality != 1

#%%
class DUPD(UnivPD):
    def __init__(self, sample_space_end, sample_space_start = 0):
        super().__init__()
        self.sample_space = range(sample_space_start, sample_space_end + 1)
        self.pmf = {}
        for event in self.sample_space:
            self.pmf[event] = self.calculateProbability(event)
        self.experiments = 0
        self.cdf_keys = list(self.sample_space)
        self.cdf_values = list(np.cumsum(list(self.pmf.values())))
        self.cdf = {self.cdf_keys[i]: self.cdf_values[i] for i in range(len(self.cdf_keys))}


    def calculateProbability(self, k):
        if k in self.sample_space and len(self.sample_space) > 0:
            return 1/len(self.sample_space)
        else:
            return 0

    def probabilityOfSampleSpace(self): #sollte zu 1 aufsummieren
        sum = 0
        for event in self.sample_space:
            sum+= self.calculateProbability(event)
            return sum

    def runEvent(self, variance_factor = 0.01): #pseudorandom-sampling: generiert random event aus sample space entsprechend zugeh.prob
        observation = random.random()
        accumulate = 0
        for event, probability in self.pmf.items():
            accumulate += probability
            if accumulate >= observation:
                accumulate = 0
                return event
        return self.runEvent(variance_factor)

    def simulateExperiment(self): #wrapper um runEvent
        value = self.runEvent()
        self.experiments += 1
        if self.experiments == len(self.sample_space):
            self.experiments = 0
        return value

    def plot(self, order_labels = None):
        labels = []
        probabilities = []
        for event, probability in self.pmf.items():
            labels.append(event)
            probabilities.append(probability)

        if order_labels is not None:
            labels.sort(key = order_labels)
            probabilities = []
            for l in labels:
                probabilities.append(self.pmf[l])
        x = np.arange(len(labels))
        width = .35
        fig, ax = plt.subplots()
        rects = ax.bar(x, probabilities, width, label = "Probability")

        ax.set_ylabel("Probabilities")
        ax.set_title("Probability distribution")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax = self.autolabel(rects, ax)
        fig.tight_layout()
        plt.ylim(0, max(probabilities)*1.2)
        plt.show()

    def autolabel(self, rects, ax): #plot-helper-func1
        """
        Text label with exact prob above each bar
        """
        for rect in rects:
            height = self.truncate(rect.get_height(), 3)
            ax.annotate(str(height),
                        xy = (rect.get_x() + rect.get_width() / 2, height),
                        xytext = (0, 3),
                        textcoords = "offset points",
                        ha = "center", va = "bottom")
        return ax

    def truncate(self, number, digits): #plot-helper-func2
        stepper = 10.0 ** digits
        return np.math.trunc(stepper * number) / stepper

    def expectedValue(self):
        self.expected_value = 0
        for event, probability in self.pmf.items():
            if isinstance(event, int):
                self.expected_value += event*probability
            else:
                return self.pmf
        return self.expected_value

    def variance(self):
        self.Variance = 0
        for event, probability in self.pmf.items():
            if isinstance(event, int):
                self.Variance += ((event - self.expectedValue())**2)*probability
            else:
                raise ValueError
        return self.Variance

#%%
#a = UnivPD()
#test output
#a = DUPD(sample_space_end=6)
#a.cdf
#a.plot()
#a.expectedValue()
#a.variance()


#%%
# test pseudorandom-sampling
#a.simulateExperiment()
#events = []
#for i in range(10000000):
#    event = a.simulateExperiment()
#    events.append(event)

#b = collections.Counter(events).values()
#c = list(b)
#d = np.divide(c, sum(c))
#print(d)

#%%
class BinomialDistribution(DUPD):
    """
    bla
    """
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.q = 1-p
        super().__init__(sample_space_end = self.n)

    def calculateProbability(self, k):
        exp = self.p**k * self.q**(self.n-k)
        return self.getCombinations(self.n, k)*exp

    def getCombinations(self, n, k):
        return np.math.factorial(n)/(np.math.factorial(k) * np.math.factorial(n-k))

    def expectedValue(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * self.q

#%%
#test
#a = BinomialDistribution(n = 6, p = .4)
#a.cdf
#a.variance()
#a.plot()
#print(a.pmf)

#%%
class NegBinomialDistribution(DUPD):
    """
    bla
    """
    def __init__(self, r, p):
        self.r = r
        self.p = p
        self.q = 1-p
        super().__init__(sample_space_start= 1, sample_space_end = int(round(self.expectedValue(),0))*3) #negBin hat alle ganzen Zahlen als sample space daher 3*E(x) als hinreichend groß

    def calculateProbability(self, n):
        exp = self.p**self.r * self.q**(n-self.r)
        return self.getCombinations(n, self.r)*exp

    def getCombinations(self, n, r):
        return np.math.comb(n-1, r-1)

    def expectedValue(self):
        return self.r / self.p

    def variance(self):
        return (self.r * self.q) / self.p**2

#%%
#b = NegBinomialDistribution(p = .02, r = 20)
#b.plot()



