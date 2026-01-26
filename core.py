import pandas as pd
import numpy as np

from collections import defaultdict
import math
from typing import List, Literal


class Probability:
    def value(self, time:int=None):
        raise NotImplementedError()

    def __repr__(self): return str(self.value())
    def _binary_operation(self, other, op): return op(self.value(), other)
    def __add__(self, other): return self._binary_operation(other, lambda x, y: x + y)
    def __sub__(self, other): return self._binary_operation(other, lambda x, y: x - y)
    def __mul__(self, other): return self._binary_operation(other, lambda x, y: x * y)
    def __truediv__(self, other): return self._binary_operation(other, lambda x, y: x / y)
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self._binary_operation(other, lambda x, y: y - x)
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other): return self._binary_operation(other, lambda x, y: y / x)


class ProbabilityWithRange(Probability):
    def __init__(self, prob:float, dist_param_1:float=None, dist_param_2:float=None, 
                 distribution:Literal['uniform', 'beta', 'binomial', 'gamma', 'normal', 'lognormal']='uniform'):
        '''
        when dist_param_1 and dist_param_2 are both None, and distribution is set to its default value (uniform), 
        this object behaves identically to a constant probability.

        dist_param_1 and dist_param_2 = 
            `lower boundary` and `upper boundary` for `uniform` distribution, 
            `a` and `b` for `beta` distribution, 
            `n` and `p` for `binomial` distribution, 
            `shape` and `scale ` for `gamma` distribution, 
            `mean` and `standard deviation` for `normal` distribution, 
            `log mean` and `log standard deviation` for `lognormal` distribution, 
        '''
        self.distribution = distribution
        self.dist_param_1 = dist_param_1 if dist_param_1 is not None else prob
        self.dist_param_2 = dist_param_2 if dist_param_2 is not None else prob
        self.prob = prob

    def sample_value(self, random_state):
        rng = np.random.default_rng(random_state)
        self.prob = getattr(rng, self.distribution)(self.dist_param_1, self.dist_param_2)

    def value(self, time:int=None):
        return self.prob


class TimeVaringProbability(Probability):
    def __init__(self, probs:List[ProbabilityWithRange]):
        self.probs = probs
        self.time = 0
    
    def sample_value(self, random_state):
        for item in self.probs:
            item.sample_value(random_state)
        
    def value(self, time=None):
        if time is not None:
            self.time = time
        return self.probs[self.time].value()


class ComplementProbability(Probability):
    def __init__(self):
        self._other_probs = None

    def set_other_probabilities(self, other_probs):
        self._other_probs = [p for p in other_probs if p is not self]

    def value(self, time=None):
        return 1 - sum([p.value(time) for p in self._other_probs])

    def __repr__(self): 
        if self._other_probs is not None:
            return str(self.value())
        else:
            return str('Complement Probability')

class Node:
    def __init__(self, node_name:str, trans_prob:Probability=ProbabilityWithRange(1.0), **variables):
        self.node_name = node_name
        self.controller = None
        self.children = []

        if isinstance(trans_prob, (float, int)):
            self.trans_prob = ProbabilityWithRange(trans_prob)
        else:
            self.trans_prob = trans_prob

        self.variables = variables
    
    def reset(self):
        if hasattr(self, 'reset_memory'):
            self.reset_memory()
        
        for child in self.children:
            child.reset()

    def set_controller(self, controller:'MarkovController'):
        self.controller = controller
        for child in self.children:
            child.set_controller(controller)
    
    def add_child(self, child:'Node'):
        if not isinstance(child, Node):
            raise TypeError(f"Invalid type: {type(child)} for {child.node_name}. Must be a Node instance.")
        self.children.append(child)
        if self.controller:
            child.set_controller(self.controller)
    
    def lookup(self, node_name:str):
        if self.node_name == node_name:
            return self
        
        for child in self.children:
            found = child.lookup(node_name)
            if found:
                return found
        return None
    
    def init_prob(self, random_state=None):
        if random_state is not None:
            if not isinstance(self.trans_prob, ComplementProbability):
                self.trans_prob.sample_value(random_state)
            
        for child in self.children:
            child.init_prob(random_state)

        for child in self.children:
            if isinstance(child.trans_prob, ComplementProbability):
                probs = [child.trans_prob for child in self.children]
                child.trans_prob.set_other_probabilities(probs)
                break

    def verify(self):
        if not self.children:
            return

        total_prob = sum(child.trans_prob for child in self.children)
        
        if not math.isclose(total_prob, 1.0, rel_tol=1e-9):
            raise ValueError(f"Children of '{self.node_name}' sum to {total_prob}, expected 1.0")
            
        for child in self.children:
            child.verify()    

    def forward(self, time:int, input_prob:float, **input_variables):
        output_prob = input_prob * self.trans_prob.value(time=time)
        output_variables = {}

        for k in list(input_variables.keys()|self.variables.keys()):
            output_variables[k] = input_variables.get(k, 0) + self.variables.get(k, 0)

        for child in self.children:
            child.forward(time=time, input_prob=output_prob, **output_variables)
    
    def on_controller_message(self, data):
        raise NotImplementedError(f'message handler is not implemented.')

    def notify_controller(self, data):
        if self.controller:
            self.controller.handle_node_message(self, data)
        else:
            raise ValueError(f'no controller for {self.node_name}')

    def __str__(self) -> str:
        variables = ', '.join(f"{k}={v}" for k, v in self.variables.items())
        return f"{self.__class__.__name__}(name={self.node_name}, prob={self.trans_prob}, {variables})"

    def __repr__(self):
        return str(self)


class ChanceNode(Node):
    def __init__(self, node_name, trans_prob:Probability=ProbabilityWithRange(1.0), **variables):
        super().__init__(node_name, trans_prob, **variables)


class MarkovState(Node):
    def __init__(self, node_name, init_prob:Probability=ProbabilityWithRange(1.0), **variables):
        super().__init__(node_name, init_prob, **variables)
        self.initial_prob = []
    
    def reset_memory(self):
        self.initial_prob = []

    def start(self, time, cycle_start_prob):
        if time == 0:
            self.initial_prob.append(cycle_start_prob)
        
        self.initial_prob.append(0)
        
        self.notify_controller({'prob':self.initial_prob[time: time+2], **self.variables})
        self.trans_prob = ProbabilityWithRange(cycle_start_prob)
        self.forward(time=time)

    def on_controller_message(self, data):
        # controller send a cycle start event
        time = data['time']
        cycle_start_prob = data['cycle_start_prob']
        self.start(time, cycle_start_prob)

    def forward(self, time:int):
        for child in self.children:
            child.forward(time=time, input_prob=self.trans_prob, **{k:0 for k in self.variables.keys()})

class StateTransition(Node):
    def __init__(self, node_name, trans_prob:Probability=ProbabilityWithRange(1.0), dst_state:MarkovState=None, **variables):
        super().__init__(node_name, trans_prob, **variables)
        self.children = [dst_state]
        self.cumulative_probs = []
        self.cumulative_variables = []
    
    def reset(self):
        return self.reset_memory()

    def reset_memory(self):
        self.cumulative_probs = []
        self.cumulative_variables = []

    def forward(self, time, input_prob, **input_variables):
        if time == 0:
            self.cumulative_probs.append(0)
        
        output_prob = input_prob * self.trans_prob.value(time=time)
        self.cumulative_probs.append(output_prob)

        destination = self.children[0]
        output_variables = {}
        
        for k in list(input_variables.keys() | self.variables.keys()):
            output_variables[k] = input_variables.get(k, 0) + self.variables.get(k, 0) + destination.variables.get(k, 0)
        
        self.notify_controller({'dst':destination.node_name, 'prob': self.cumulative_probs[time: time+2], **output_variables})

    def verify(self):
        if len(self.children) == 1 and isinstance(self.children[0], MarkovState):
            return True
        raise ValueError(f"Transition '{self.node_name}' must have exactly one MarkovState child.")
    
    def lookup(self, node_name):
        return self if self.node_name == node_name else None

    def init_prob(self, random_state=None):
        if random_state is not None:
            if not isinstance(self.trans_prob, ComplementProbability):
                self.trans_prob.sample_value(random_state)
    
    def set_controller(self, controller:'MarkovController'):
        self.controller = controller


class MarkovController(Node):
    def __init__(self, total_cycles=1, count_method:Literal['start', 'half', 'end']='half', discount_rate=0):
        super().__init__(node_name='controller')
        self.total_cycles = total_cycles
        self.count_method = count_method
        self.discount_rate = discount_rate

        self.model_time = 0
        self.total_probs = defaultdict(list)
        self.total_variables = defaultdict(list)
        self.cycle_probs = defaultdict(list)
        self.cycle_variables = defaultdict(list)
        self.next_cycle_start_prob = defaultdict(list)

    def run(self):
        self.reset()
        for state in self.children:
            self.next_cycle_start_prob[state.node_name] = state.trans_prob.value(0)

        for _ in range(self.total_cycles):
            self.start_one_cycle()
            self.end_one_cycle()

        return pd.DataFrame.from_dict(self.total_probs), pd.DataFrame.from_dict(self.total_variables)

    def add_child(self, child:Node):
        if not isinstance(child, MarkovState):
            raise TypeError(f"Invalid type: {type(child)}. Must be a MarkovState instance.")
        child.set_controller(self)
        self.children.append(child)

    def start_one_cycle(self):
        next_cycle_start_prob = self.next_cycle_start_prob.copy()
        self.next_cycle_start_prob.clear()
        for node_name, cycle_start_prob in next_cycle_start_prob.items():
            self.send_to_node(node_name=node_name, data={'time':self.model_time, 'cycle_start_prob':cycle_start_prob})
        
    def end_one_cycle(self):
        for key in self.next_cycle_start_prob.keys():
            self.next_cycle_start_prob[key] = np.sum(self.next_cycle_start_prob[key])

        for key in self.cycle_probs.keys():
            self.total_probs[key].append(np.sum(self.cycle_probs[key]))
        
        for key in self.cycle_variables.keys():
            self.total_variables[key].append(np.sum(self.cycle_variables[key]))
        
        self.cycle_probs.clear()
        self.cycle_variables.clear()

        self.model_time += 1
    
    def reset_memory(self):
        self.model_time = 0

        self.total_probs.clear()
        self.total_variables.clear()
        self.cycle_probs.clear()
        self.cycle_variables.clear()
        self.next_cycle_start_prob.clear()

    def send_to_node(self, node_name, data):
        target = self.lookup(node_name)
        if target:
            target.on_controller_message(data)
        else:
            raise ValueError(f'invalid node: {node_name}')
        
    def handle_node_message(self, node, data:dict):
        # receive data from a markov state or a state transition object
        if isinstance(node, MarkovState):
            prob_data = data.pop('prob')
            node_name = node.node_name
        elif isinstance(node, StateTransition):
            destination_name = data.pop('dst')
            prob_data = data.pop('prob')
            node_name = destination_name

            self.next_cycle_start_prob[destination_name].append(prob_data[1])
        else:
            return
        
        prob_methods = {'start': prob_data[0], 'end': prob_data[1], 'half': np.mean(prob_data)}
        prob = prob_methods[self.count_method]
        
        self.cycle_probs[node_name].append(prob)
        
        discount_factor = (1 + self.discount_rate) ** self.model_time
        for key, value in data.items():
            discounted_value = (value * prob) / discount_factor
            self.cycle_variables[key].append(discounted_value)
