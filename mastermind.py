import numpy as np
import matplotlib.pyplot as plt
import random

"""
Colors:
000 --> 0 --> red
001 --> 1 --> yellow
010 --> 2 --> xxx
011 --> 3 --> blue
100 --> 4 --> xxx
101 --> 5 --> orange
110 --> 6 --> xxx
111 --> 7 --> xxx
"""

class BinaryString():
    def __init__(self, n=4, value=None) -> None:
        #binary string is a combination of numbers in a given order 
        if value is not None:
            self.value = value
        else:
            self.value = ''.join([str(random.randint(0, 1)) for _ in range(n*3)])

        self.flip = {'1':'0', '0':'1'}
    
    def __repr__(self) -> str:
        return self.value
    
    def __len__(self) -> int:
        return len(self.value)

    def mutate(self):
        mutated_s = ''
        for el in self.value:
            if random.random() < 1/len(self.value):
                new_el = self.flip[el]
            else:
                new_el = el
            mutated_s += new_el

        mutated_bs = BinaryString(value=mutated_s)
        return mutated_bs

    def crossover(self, bs):
        assert len(bs) == len(self)
        splitpoint = random.randint(0, len(bs)-1)
        os1 = BinaryString(value=bs.value[:splitpoint] + self.value[splitpoint:])
        os2 = BinaryString(value=self.value[:splitpoint] + bs.value[splitpoint:])
        return os1, os2

class Population:
    def __init__(self, N=50, n=4, values=None) -> None:
        self.N = N
        if values is not None:
            self.values = values
        else:
            self.values = np.array([BinaryString(n) for _ in range(N)])
        self.curr = 0
    
    def __str__(self):
        """ Represent the class instance as a string. """
        return f"Population object with size {self.N}."
    
    def __len__(self):
        return len(self.values)

    def __iter__(self):
        """ Return the iterator object (implicitly called before loops). """
        return self
    
    def __next__(self):
        """Return the next BinaryString in the sequence.
        
        Implicitly called at each loop increment. Raises a StopIteration 
        exception when there are no more samples to return (queue is empty),
        which is implicitly captured by looping constructs to stop iterating.
        """
        try:
            next_candidate = self.values[self.curr]
            self.curr += 1
        except IndexError:
            self.curr = 0
            raise StopIteration
        return next_candidate
    
    def extend(self, N):
        """Extend population through crossover and mutation
        operators."""    
        #crossover operator
        children = []    
        for bs in self.values:
            other_bs = self.values[random.randint(0, len(self)-1)]
            os1, os2 = bs.crossover(other_bs)
            children.append(os1)
            children.append(os2)

        mutations = []
        while len(children) + len(self) + len(mutations) < N:
            rand_child = random.choice(children)
            mutations.append(rand_child.mutate())
        
        mutations = np.array(mutations)
        children = np.array(children)
        self.values = np.hstack((self.values, mutations, children)).ravel()

        return self

class MasterMind:
    """MasterMind game class. Game master has a secret 
    combination of N colors; the goal is to find this 
    combination through evolutionary algorithms. Very 
    similar to popular game Wordle in that we have the 
    following information for each guess: 
    
    - p1: Number of pieces with the right color and the 
    correct position. 
    
    - p2: Number of pieces with the right color but the 
    wrong position.

    Genotype: binary string of fixed size (Nx3 bits). 

    Cross-over operator swaps portions of binary string 
    (sections of three). 

    Mutation operator randomly flips the bits in binary string.
    """
    def __init__(self, n=4, alpha=0.5, N=50, elite_frac=0.10) -> None:
        self.n = n #number of colors in game-master's deck
        self.N = N #population number
        self.alpha = alpha #relative importance of p2 wrt p1
        self.n_elite = int(elite_frac * N)

        self.gm = BinaryString(n) #secret combination
        self.gm_phenotype = [self.gm.value[3*i:3*(i+1)] for i in range(self.n)]
        self.population = Population(N, n) #guess
        self.mean_fitnesses = []

    @property
    def p1(self):
        """Number of pieces with the right color and the 
        correct position."""
        p1 = np.zeros(self.N)
        for i, candidate in enumerate(self.population):
            candidate_phenotype = [candidate.value[3*i:3*(i+1)] for i in range(self.n)]
            for cand_color, true_color in zip(candidate_phenotype, self.gm_phenotype):
                if cand_color == true_color:
                    p1[i] += 1 
        return p1
    
    @property
    def p2(self):
        """Number of pieces with the right color but the 
        wrong position."""
        p2 = np.zeros(self.N)
        for i, candidate in enumerate(self.population):
            candidate_phenotype = [candidate.value[3*i:3*(i+1)] for i in range(self.n)]
            for cand_color, true_color in zip(candidate_phenotype, self.gm_phenotype):
                if cand_color != true_color and cand_color in self.gm_phenotype:
                    p2[i] += 1 
        return p2

    @property
    def fitness(self):
        """Fitness function; maximum = N."""
        return self.p1 + self.alpha*self.p2

    def perform_selection(self):
        elite_indices = self.fitness.argsort()[-self.n_elite:]
        elite_bs = Population(values=self.population.values[elite_indices])
        self.population = elite_bs.extend(self.N)
        
    def solve(self):
        converged = False
        i = 0
        while not converged:
            self.mean_fitnesses.append(np.mean(self.fitness))
            self.perform_selection()

            if np.mean(self.fitness) > 0.98*self.n:
                converged = True

            if i > 10000:
                break
            i += 1
        
        guess = self.population.values[self.fitness.argsort()[-1]]
        return guess

    
if __name__ == '__main__':
    mm = MasterMind(n=20)
    guess = mm.solve()
    print('guess = ', guess)
    print('true  = ', mm.gm.value)

    plt.figure()
    plt.plot(mm.mean_fitnesses)
    plt.show()