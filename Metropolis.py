
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class Metropolis: 
    def __init__(self, logTarget, initialState):
        self.step_size = 1.0  #static value
        self.logTarget = logTarget
        self.state = initialState
        self.samples = []
    
    def __accept(self, proposal):
        difference = self.logTarget(proposal) - self.logTarget(self.state)
        #calculate the acceptance ratio
        acceptance_ratio = np.exp(min(1.0, difference)) 
        #generate a random number between 0 and 1
        random_number = np.random.random()
        #accept the proposed state
        if acceptance_ratio > random_number: 
            self.state = proposal
            return True
        else:
            return False

    def adapt(self, blockLengths):
        for i in blockLengths:
            proposal = np.random.normal(self.state, self.step_size, size=i)
            acceptance_ratio = np.exp(self.logTarget(proposal) - self.logTarget(self.state))
            accepted = proposal[np.random.rand(i) < acceptance_ratio]
            num_accepted = len(accepted)
            num_proposals = i
            acceptance_ratio = num_accepted / num_proposals
            self.step_size *= 1.1 if acceptance_ratio > 0.5 else 0.9
        return self

    def sample(self, nSamples):
        for i in range(nSamples):
            #propose a new state using a normal distribution
            new_state =np.random.normal(loc=self.state,scale=1.0)
            #run the new_state through the accept method to decide whether to accept
            self.__accept(new_state)
            #add the current state to the list of samples
            self.samples.append(self.state)
        return self
           
    def summary(self):
        samples = self.samples
        n = len(samples)
        mean = sum(samples) / n
        var = sum((x - mean) ** 2 for x in samples) / (n - 1)
        std = var ** 0.5
        sorted_samples = sorted(samples)
        c025 = sorted_samples[int(n * 0.025)]
        c975 = sorted_samples[int(n * 0.975)]
        return {'mean': mean, 'std': std, 'c025': c025, 'c975': c975}

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def hitRate(self):
        return(self.hits/(self.hits + self.misses))
    
    def falseAlarmrate(self):
        return(self.falseAlarms/(self.falseAlarms+ self.correctRejections))
        
    def d_prime(self):
        return (norm.ppf(self.hitRate()) - norm.ppf(self.falseAlarmrate()))

    def criterion(self):
        return -0.5 * (norm.ppf(self.hitRate()) + norm.ppf(self.falseAlarmrate()))
   #Add Overloading Operators 
    def __add__(self,other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, 
        self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)
    
    def __mul__(self,scalar):
        return SignalDetection( self.hits * scalar, self.misses * scalar, self.falseAlarms * scalar,
        self.correctRejections * scalar)
        
    def nLogLikelihood(self, hitRate, falseAlarmrate):
        return -((self.hits * np.log(hitRate)) + (self.misses * np.log(1-hitRate)) + (self.falseAlarms * np.log(falseAlarmrate)) + (self.correctRejections * np.log(1-falseAlarmrate)))
    
    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        hitRates = norm.cdf((dprime - 2 * np.array(criteriaList)) / 2)
        falseAlarmRates = norm.cdf((- dprime - 2 * np.array(criteriaList)) / 2)
        sdtList = [SignalDetection(np.random.binomial(signalCount, hitRates[i]), np.random.binomial(signalCount, 1 - hitRates[i]), 
        np.random.binomial(noiseCount, falseAlarmRates[i]), np.random.binomial(noiseCount, 1 - falseAlarmRates[i])) for i in range(len(criteriaList))]
        return sdtList
    
    @staticmethod
    def plot_roc(sdtList):
        fig, ax = plt.subplots()
        ax.set(xlim=[0, 1], ylim=[0, 1], xlabel="false alarm rate", ylabel="hit Rate",
           title="ROC curve")
        ax.plot(np.linspace(0, 2, 80), np.linspace(0, 2, 80), '--', color='black')
        for sdt in sdtList:
            ax.plot(sdt.falseAlarmrate(), sdt.hitRate(), 'o', color='black')
            ax.grid()
            
    @staticmethod
    def rocCurve(falseAlarmrate, a):
        hitRate = norm.ppf(falseAlarmrate)
        return norm.cdf(a + hitRate)

    @staticmethod
    def fit_roc(sdtList):
        SignalDetection.plot_roc(sdtList)
        res = minimize_scalar(SignalDetection.rocLoss, method='bounded', bounds=[0, 1], args=(sdtList,))
        a_hat = res.x
        t = np.arange(0,1,0.01)
        loss = SignalDetection.rocCurve(t, a_hat)
        plt.plot(t, loss, '-', color='r')
        return a_hat

    @staticmethod
    def rocLoss(a, sdtList):
        return sum([sdt.nLogLikelihood(sdt.rocCurve(sdt.falseAlarmrate(), a), sdt.falseAlarmrate()) for sdt in sdtList])
    
import scipy.stats
import unittest

class TestSignalDetection(unittest.TestCase):
    def test_simulate(self):
        # Test with a single criterion value
        dPrime       = 1.5
        criteriaList = [0]
        signalCount  = 1000
        noiseCount   = 1000

        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]

        self.assertEqual(sdt.hits             , sdtList[0].hits)
        self.assertEqual(sdt.misses           , sdtList[0].misses)
        self.assertEqual(sdt.falseAlarms      , sdtList[0].falseAlarms)
        self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)

        # Test with multiple criterion values
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual    (sdt.hits              ,  signalCount)
            self.assertLessEqual    (sdt.misses            ,  signalCount)
            self.assertLessEqual    (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual    (sdt.correctRejections ,  noiseCount)

    def test_nLogLikelihood(self):
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                           5 * np.log(1-hit_rate) +
                           3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)

    def test_rocLoss(self):
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884206555698
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)

    def test_integration(self):
        dPrime = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()

if __name__ == '__main__':
    unittest.main() # for jupyter

def fit_roc_bayesian(sdtList):

    # Define the log-likelihood function to optimize
    def loglik(a):
        return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

    # Create a Metropolis sampler object and adapt it to the target distribution
    sampler = Metropolis(logTarget = loglik, initialState = 0)
    sampler = sampler.adapt(blockLengths = [2000]*3)

    # Sample from the target distribution
    sampler = sampler.sample(nSamples = 4000)

    # Compute the summary statistics of the samples
    result  = sampler.summary()

    # Print the estimated value of the parameter a and its credible interval
    print(f"Estimated a: {result['mean']} ({result['c025']}, {result['c975']})")

    # Create a mosaic plot with four subplots
    fig, axes = plt.subplot_mosaic(
        [["ROC curve", "ROC curve", "traceplot"],
         ["ROC curve", "ROC curve", "histogram"]],
        constrained_layout = True
    )

    # Plot the ROC curve of the SDT data
    plt.sca(axes["ROC curve"])
    SignalDetection.plot_roc(sdtList = sdtList)

    # Compute the ROC curve for the estimated value of a and plot it
    xaxis = np.arange(start = 0.00,
                      stop  = 1.00,
                      step  = 0.01)

    plt.plot(xaxis, SignalDetection.rocCurve(xaxis, result['mean']), 'r-')

    # Shade the area between the lower and upper bounds of the credible interval
    plt.fill_between(x  = xaxis,
                     y1 = SignalDetection.rocCurve(xaxis, result['c025']),
                     y2 = SignalDetection.rocCurve(xaxis, result['c975']),
                     facecolor = 'r',
                     alpha     = 0.1)

    # Plot the trace of the sampler
    plt.sca(axes["traceplot"])
    plt.plot(sampler.samples)
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.title('Trace plot')

    # Plot the histogram of the samples
    plt.sca(axes["histogram"])
    plt.hist(sampler.samples,
             bins    = 51,
             density = True)
    plt.xlabel('a')
    plt.ylabel('density')
    plt.title('Histogram')

    # Show the plot
    plt.show()

# Define the number of SDT trials and generate a simulated dataset
sdtList = SignalDetection.simulate(dprime       = 1,
                                   criteriaList = [-1, 0, 1],
                                   signalCount  = 40,
                                   noiseCount   = 40)

# Fit the ROC curve to the simulated dataset
fit_roc_bayesian(sdtList)


    



