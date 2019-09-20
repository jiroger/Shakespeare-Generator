import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class Shakespeare:
    def __init__(self):
        path_to_shakespeare = "shakespeare_input.txt"
        with open(path_to_shakespeare, "r") as f:
            self.shakespeare_text = f.read()

    @staticmethod
    def unzip(pairs):
        """
        Splits list of pairs (tuples) into separate lists.

        Example: pairs = [("a", 1), ("b", 2)] --> ["a", "b"] and [1, 2]
        """
        return tuple(zip(*pairs))
    

    def normalize(self, counter):
        """ 
        Convert counter to a list of (letter, frequency) pairs, sorted in descending order of frequency.

        Parameters
        -----------
        counter: A Counter-instance

        Returns
        -------
        A list of tuples - (letter, frequency) pairs. 

        """
        total = sum(counter.values())
        return [(char, cnt/total) for char, cnt in counter.most_common()]
    
    
    def train_lm(self, n):
        """ 
        Train character-based n-gram language model.
        
        Given a sequence of n-1 characters, model will learn what the probability
        distribution is for the n-th character in the sequence.

        Tildas ("~") are used for padding the history when necessary, so that it's 
        possible to estimate the probability of a seeing a character when there 
        aren't (n - 1) previous characters of history available.
           
        Parameters
        -----------
        text: str 
            A string (doesn't need to be lowercased).
        n: int
            The length of n-gram to analyze.
        
        Returns
        -------
        A dict that maps histories (strings of length (n-1)) to lists of (char, prob) 
        pairs, where prob is the probability of char appearing after 
        that specific history. 

        """
        raw_lm = defaultdict(Counter)
        history = "~" * (n - 1)

        # count number of times characters appear following different histories
        for x in self.shakespeare_text:
            raw_lm[history][x] += 1
            history = history[1:] + x

        # create final dictionary by normalizing
        lm = { history : self.normalize(counter) for history, counter in raw_lm.items() }

        return lm

    def generate_letter(self, lm, history):
        """ 
        Randomly picks letter according to probability distribution associated with 
        the specified history.

        Note: returns dummy character "~" if history not found in model.

        Parameters
        ----------
        lm: Dict[str, Tuple[str, float]] 
            The n-gram language model. I.e. the dictionary: history -> (char, freq)

        history: str
            A string of length (n-1) to use as context/history for generating 
            the next character.

        Returns
        -------
        str
            The predicted character. '~' if history is not in language model.
        """
        if not history in lm:
            return "~"
        letters, probs = Shakespeare.unzip(lm[history])
        
        i = np.random.choice(letters, p=probs)
        return i

    def generate_text(self, lm, n, nletters=100):      
        """ 
        Randomly generates nletters of text with n-gram language model lm.
    
        Parameters
        ----------
        lm: Dict[str, Tuple[str, float]] 
            The n-gram language model. I.e. the dictionary: history -> (char, freq)
        n: int
            Order of n-gram model.
        nletters: int
            Number of letters to randomly generate.
        
        Returns
        -------
        str
            Model-generated text.
        """
        history = "~" * (n - 1)
        text = []
        for i in range(nletters):
            c = self.generate_letter(lm, history)
            text.append(c)
            history = history[1:] + c
        return "".join(text)    

model = Shakespeare()
lm3 = model.train_lm(3)
print(model.generate_text(lm3, 3, 500))