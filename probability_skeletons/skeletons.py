#%%
# Required libraries
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import collections
%matplotlib inline



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
#b = MultivPD(dim = 1) #same as univProbDist, will raise Error if dimensionality != 1

#%%
class DUPD(UnivPD):
    def __init__(self, sample_space_end, sample_space_start = 0):
        super().__init__()
        self.sample_space = range(sample_space_start, sample_space_end + 1)
        self.distribution = {}
        for event in self.sample_space:
            self.distribution[event] = self.calculateProbability(event)
        self.experiments = 0
        print(self.distribution)


    def calculateProbability(self, y):
        if y in self.sample_space and len(self.sample_space) > 0:
            return 1/len(self.sample_space) #Laplace-Wahrscheinlichkeit als a priori
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
        for event, probability in self.distribution.items():
            accumulate += probability
            if accumulate >= observation:
                accumulate = 0
                return event
        return self.runEvent(variance_factor)

    def simulateExperiment(self, variance_factor = 0.01): #wrapper um runEvent
        value = self.runEvent()
        self.experiments += 1
        if self.experiments == len(self.sample_space):
            self.experiments = 0
            self.generator = None
        return value

    def plot(self, order_labels = None):
        labels = []
        probabilities = []
        for event, probability in self.distribution.items():
            labels.append(event)
            probabilities.append(probability)

        if order_labels is not None:
            labels.sort(key = order_labels)
            probabilities = []
            for l in labels:
                probabilities.append(self.distribution[l])
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
        return math.trunc(stepper * number) / stepper

    def expectedValue(self):
        self.expected_value = 0
        for event, probability in self.distribution.items():
            if isinstance(event, int):
                self.expected_value += event*probability
            else:
                return self.distribution
        return self.expected_value

    def variance(self):
        self.Variance = 0
        for event, probability in self.distribution.items():
            if isinstance(event, int):
                self.Variance += ((event - self.expectedValue())**2)*probability
            else:
                raise ValueError
        return self.Variance

#%%
#test output
a = DUPD(sample_space_end=6)
a.plot()
a.expectedValue()
a.variance()

#%%
# test pseudorandom-sampling
a.simulateExperiment()
events = []
for i in range(10000000):
    event = a.simulateExperiment()
    events.append(event)

b = collections.Counter(events).values()
c = list(b)
d = np.divide(c, sum(c))
print(d)
