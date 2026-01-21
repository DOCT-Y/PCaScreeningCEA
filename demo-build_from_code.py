import pandas as pd

from core import (ProbabilityWithRange, TimeVaringProbability, ComplementProbability, 
                  MarkovController, MarkovState, ChanceNode, StateTransition)


# some parameters
total_cycles = 20
count_method = 'half'
discount_rate = 0.02

background_mortality = TimeVaringProbability([
    ProbabilityWithRange(
        0.0001 + time * 0.0001, 
        0.0001 + time * 0.0001 - 0.0001, 
        0.0001 + time * 0.0001 + 0.0001, 
        'uniform'
     ) 
    for time in range(total_cycles)
])

# define markov states, chance nodes, state transitions
state_a = MarkovState(node_name='state_a', init_prob=1, cost=50, utility=1)
state_b = MarkovState(node_name='state_b', init_prob=0, cost=100, utility=0.7)
state_c = MarkovState(node_name='state_c', init_prob=0, cost=0, utility=0)
controller = MarkovController(total_cycles=total_cycles, count_method=count_method, discount_rate=discount_rate)

k1 = ChanceNode(node_name='k1', trans_prob=ComplementProbability(), cost=20, utility=-0.1)
k2 = ChanceNode(node_name='k1', trans_prob=ComplementProbability(), cost=30, utility=-0.2)

a_to_a = StateTransition(node_name='a_to_a', trans_prob=ComplementProbability(), dst_state=state_a, cost=1)
a_to_b = StateTransition(node_name='a_to_b', trans_prob=0.2, dst_state=state_b, cost=2, utility=-0.1)
a_to_c = StateTransition(node_name='a_to_c', trans_prob=background_mortality, dst_state=state_c, cost=2, utility=-0.1)

b_to_c = StateTransition(node_name='b_to_c', trans_prob=background_mortality, dst_state=state_c, cost=0, utility=0)
k1_to_c = StateTransition(node_name='k1_to_c', trans_prob=0.01, dst_state=state_c, cost=0, utility=0)
k2_to_a = StateTransition(node_name='k2_to_a', trans_prob=0.3, dst_state=state_a, cost=6, utility=-0.2)
k2_to_b = StateTransition(node_name='k2_to_b', trans_prob=ComplementProbability(), dst_state=state_b, cost=7, utility=-0.1)

c_to_c = StateTransition(node_name='c_to_c', trans_prob=1, dst_state=state_c, cost=0, utility=0)

# define parent-child relationships
state_a.add_child(a_to_a)
state_a.add_child(a_to_b)
state_a.add_child(a_to_c)

k2.add_child(k2_to_a)
k2.add_child(k2_to_b)
k1.add_child(k2)
k1.add_child(k1_to_c)
state_b.add_child(k1)
state_b.add_child(b_to_c)

state_c.add_child(c_to_c)

controller.add_child(state_a)
controller.add_child(state_b)
controller.add_child(state_c)

# initialize probabilities and verify model
controller.init_prob()
controller.verify()

# run model
prob_df, variable_df = controller.run()

print(prob_df)
print(variable_df)
print(variable_df.sum(axis=0))

'''
     state_a   state_b   state_c
0   0.899950  0.100000  0.000050
1   0.749524  0.249276  0.001200
2   0.673450  0.322608  0.003942
3   0.634306  0.358178  0.007515
4   0.613491  0.374967  0.011542
5   0.601760  0.382407  0.015833
6   0.594518  0.385188  0.020294
7   0.589484  0.385639  0.024878
8   0.585523  0.384917  0.029560
9   0.582074  0.383599  0.034327
10  0.578858  0.381970  0.039173
11  0.575735  0.380172  0.044093
12  0.572638  0.378277  0.049085
13  0.569534  0.376320  0.054146
14  0.566407  0.374318  0.059275
15  0.563248  0.372282  0.064470
16  0.560054  0.370215  0.069730
17  0.556825  0.368121  0.075054
18  0.553560  0.366001  0.080439
19  0.550259  0.363856  0.085885

         cost   utility
0   55.597550  0.959945
1   67.760844  0.846416
2   77.731424  0.747721
3   81.518997  0.691599
4   82.315430  0.656381
5   81.675745  0.631504
6   80.357265  0.611820
7   78.728608  0.594818
8   76.969019  0.579275
9   75.165492  0.564592
10  73.360090  0.550473
11  71.572995  0.536775
12  69.813736  0.523425
13  68.086659  0.510383
14  66.393587  0.497631
15  64.735118  0.485155
16  63.111254  0.472948
17  61.521710  0.461003
18  59.966064  0.449316
19  58.443828  0.437882

cost       1414.825414
utility      11.809063
'''
