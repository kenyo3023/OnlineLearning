import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

class BetaBinomial():
    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b
        self.iter = 0
        self.thetas = np.linspace(0.01, 0.99, 100)
    
    def nCr(self, n, k):
        '''calculate combination'''
        f = math.factorial
        return f(n) / (f(k) * f(n-k))

    def likelihood(self, n, k, theta):
        '''pdf of binomial distribution'''
        return self.nCr(n, k) * (theta)**k * (1-theta)**(n-k)
    
    def betadistribution(self, a, b):
        N = a + b
        pdf_beta = [self.nCr(N, a) * (theta)**(a-1) * (1-theta)**(b-1) for theta in self.thetas]
        return pdf_beta
    
    def plot(self, prior, likelihood, posterior, ylim):
        sns.set_style("darkgrid")
        plt.figure(figsize=(20,5))
        titles = ['prior', 'likelihood', 'posterior']
        for i, p in enumerate([prior, likelihood, posterior]):
            plt.subplot(130+(i+1))
            sns.lineplot(self.thetas, p)
            plt.xlim(0, 1)
            plt.ylim(0, ylim)
            plt.title(titles[i])
            plt.xlabel('mu')
        plt.show()
    
    def update(self, case):
        self.iter += 1
        self.case_str = case
        self.case = [int(c) for c in case]
        self.n = len(self.case)
        self.k = sum(self.case)
        print('case %d: %s'%(self.iter, self.case_str))
        
        theta = self.k / self.n
        self.ll = self.likelihood(self.n, self.k, theta)
        if showdistribution:
            self.lldistribution = [self.likelihood(self.n, self.k, theta) for theta in self.thetas]
        print('Binomial Likelihood: %.17f'%self.ll)

        print('Beta prior: a = %d b = %d'%(self.a, self.b))
        prior_pdf = self.betadistribution(self.a, self.b)
        self.ylim = math.ceil(max(prior_pdf))
        self.a += self.k
        self.b += self.n - self.k
        print('Beta posterior: a = %d b = %d'%(self.a, self.b))
        posterior_pdf = self.betadistribution(self.a, self.b)

        if showdistribution:
            self.plot(prior_pdf, self.lldistribution, posterior_pdf, self.ylim)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--a', help='alpha', default=0, type=int)
    parser.add_argument('--b', help='beta', default=0, type=int)
    parser.add_argument('--dataset', help='the path of the dataset', default='dataset/testfile.txt')
    parser.add_argument('--showdistribution', help='whether to show the prior, likelihood, posterior distribution', 
                        action="store_true")
    args = parser.parse_args()
    
    a = args.a
    b = args.b
    dataset = args.dataset
    showdistribution = args.showdistribution

    with open(dataset) as f:
        testfile = f.read()
    testfile = testfile.split('\n')

    BB = BetaBinomial(a=a, b=b)
    for test in testfile:
        BB.update(test)
        print('\n')