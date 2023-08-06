from blist import sortedlist
import numpy as np
import math

from osm.data_streams.budget_manager.abstract_budget_manager import AbstractBudgetManager

#minimalist value to be added to values to avoid same value sorting
#epsilon = 1 / (2 ** 20)

class IncrementalPercentileFilter(AbstractBudgetManager):
    def __init__(self, budget_threshold, window_size):
        """
        handles inline budgeting of streams by sorting merit values and deciding acquisition 
        on their rank in the list
        ignores costs

        usable only for uniform acquisition costs
        :param budget_threshold:The relative amount of budget that can be spent
        in terms of queried / called; ignores budget and budget_spent
        :param window_size:The size of the list in which merit values are ranked
        """
        super().__init__(budget_threshold=budget_threshold)

        self.counter = 0
        self.window_size = window_size
        self.values_list = [None] * window_size
        self.values = sortedlist()

    def _acquire(self, value, cost = 1):
        """
        adds a value to its list and returns whether to acquire the value or not
        :param value: a float value akin to a merit or quality
        """
        if not isinstance(value, float):
            raise ValueError("value must be a float")

        self.counter = self.counter + 1
        #randomize to ensure no equal values
        #value += np.random.uniform() * epsilon
        i = (self.counter-1) % self.window_size

        #replace oldest value if window size reached
        if self.counter > self.window_size:
            oldest_val = self.values_list[i]
            self.values.remove(oldest_val)

        self.values_list[i] = value
        self.values.add(value)

        return self.values[math.floor(min(self.window_size, self.counter) * (1 - self.budget_threshold))] <= value

    def get_name(self):
        return "IPF"

class TrendCorrectedIncrementalPercentileFilter(AbstractBudgetManager):
    def __init__(self, budget_threshold, window_size):
        """
        handles inline budgeting of streams by sorting merit values and deciding acquisition 
        on their rank in the list
        transforms the incoming merit according to a curve fitting to eliminate the trend within the data
        ignores costs

        usable only for uniform acquisition costs
        :param budget_threshold:The relative amount of budget that can be spent
        in terms of queried / called; ignores budget and budget_spent
        :param window_size:The size of the list in which merit values are ranked
        """
        super().__init__(budget_threshold=budget_threshold)

        self.counter = 0
        self.window_size = window_size
        self.values_list = [None] * window_size
        self.sumX = 0; self.sumY = 0; self.sumXX = 0; self.sumXY = 0
        #self.poly = lambda x: 0
    
    def _acquire(self, value, cost = 1):
        """
        adds a value to its list and returns whether to acquire the value or not
        :param value: a float value akin to a merit or quality
        """
        if not isinstance(value, float):
            raise ValueError("value must be a float")
        
        self.counter += 1
        #randomize to ensure no equal values
        #value += np.random.uniform() * epsilon
        i = (self.counter - 1) % self.window_size
        
        old_value = self.values_list[i]
        self.values_list[i] = value
        self.sumX += self.counter
        self.sumY += value
        self.sumXX += self.counter * self.counter
        self.sumXY += value * self.counter #* value
        if self.counter > self.window_size:
            self.sumX -= self.counter - self.window_size
            self.sumY -= old_value
            self.sumXX -= (self.counter - self.window_size) ** 2
            self.sumXY -= old_value * (self.counter - self.window_size)
        
        poly = lambda x: 0
        n = min(self.counter, self.window_size)
        if n > 2:
            meanX = self.sumX / n
            meanY = self.sumY / n
            varXX = self.sumXX - self.sumX * meanX
            varXY = self.sumXY - self.sumX * meanY
            frac = varXY / varXX
            poly = lambda x: frac * (x - meanX) + meanY
        
        transformed_list = [self.values_list[k] - poly(self.counter - n + k) for k in range(n)]
        ordered_list = sorted(transformed_list)
        return ordered_list[math.floor(n * (1 - self.budget_threshold))] <= transformed_list[i]
        
    def get_name(self):
        return "TCIPF"