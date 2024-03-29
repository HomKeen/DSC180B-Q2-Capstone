'''
General purpose functions, for use wherever we need them.
'''

import time

class Timer:
    def __init__(self):
        self.start = 0
        self.reset()

    def reset(self):
        self.start = time.time()
    
    def get(self, reset=False):
        elapsed = time.time() - self.start
        if reset:
            self.reset()
        
        return elapsed