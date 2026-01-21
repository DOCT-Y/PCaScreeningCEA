rm(list = ls())

library(heemod)
library(data.table)


get_background_mortality <- function(model_time) {
    return(0 + 0.0001 * model_time)
}


CYCLES <- 20
state_names <- c("a","a_to_a","b_to_a","b","a_to_b","b_to_b", "c", "b_k1_to_c")
INIT   <- c(1, rep(0, length(state_names)-1))

count_method <- c("beginning", "life-table", "end")[2]

# define parameters
param <- define_parameters(
    # State costs
    cost_a=50, 
    cost_b=100, 
    cost_c=0, 
    
    # Transition costs
    cost_a_to_a=50+1, 
    cost_a_to_b=100+2, 
    cost_b_to_k1_to_c=0+20, 
    cost_b_to_k1_to_k2_to_a=50+6+30+20, 
    cost_b_to_k1_to_k2_to_b=100+7+30+20, 
    
    # State utilities
    utility_a=1, 
    utility_b=0.7, 
    utility_c=0, 
    
    # Transition utilities
    utility_a_to_a=1+0, 
    utility_a_to_b=0.7+(-0.1), 
    utility_b_to_k1_to_c=0+(-0.1), 
    utility_b_to_k1_to_k2_to_a=1+(-0.2)+(-0.2)+(-0.1), 
    utility_b_to_k1_to_k2_to_b=0.7+(-0.1)+(-0.2)+(-0.1), 
    
    # Time-varying probability
    mortality = get_background_mortality(model_time),
    
    # Transition probabilities
    p_a_to_a=1-0.2-mortality, 
    p_a_to_b=0.2, 
    p_a_to_c=mortality, 
    
    p_b_to_c=mortality, 
    p_b_to_k1_to_c=(1-mortality)*0.01, 
    p_b_to_k1_to_k2_to_a=(1-mortality)*(1-0.01)*0.3, 
    p_b_to_k1_to_k2_to_b=(1-mortality)*(1-0.01)*(1-0.3), 
    
    p_c_to_c=1, 
    
    discount_rate=0.02
)


# define transition matrix
mat_strans <- define_transition(
    state_names = state_names,
    0, p_a_to_a, 0, 0, p_a_to_b, 0, p_a_to_c, 0, 
    0, p_a_to_a, 0, 0, p_a_to_b, 0, p_a_to_c, 0, 
    0, p_a_to_a, 0, 0, p_a_to_b, 0, p_a_to_c, 0, 
    0, 0, p_b_to_k1_to_k2_to_a, 0, 0, p_b_to_k1_to_k2_to_b, p_b_to_c, p_b_to_k1_to_c, 
    0, 0, p_b_to_k1_to_k2_to_a, 0, 0, p_b_to_k1_to_k2_to_b, p_b_to_c, p_b_to_k1_to_c, 
    0, 0, p_b_to_k1_to_k2_to_a, 0, 0, p_b_to_k1_to_k2_to_b, p_b_to_c, p_b_to_k1_to_c, 
    0, 0, 0, 0, 0, 0, p_c_to_c, 0, 
    0, 0, 0, 0, 0, 0, p_c_to_c, 0, 
)

# define markov states
state_a <- define_state(cost=discount(cost_a, r=discount_rate), qaly=discount(utility_a, r=discount_rate))
state_b <- define_state(cost=discount(cost_b, r=discount_rate), qaly=discount(utility_b, r=discount_rate))
state_c <- define_state(cost=discount(cost_c, r=discount_rate), qaly=discount(utility_c, r=discount_rate))

state_a_to_a <- define_state(cost=discount(cost_a_to_a, r=discount_rate), qaly=discount(utility_a_to_a, r=discount_rate))
state_a_to_b <- define_state(cost=discount(cost_a_to_b, r=discount_rate), qaly=discount(utility_a_to_b, r=discount_rate))

state_b_to_k1_to_c <- define_state(cost=discount(cost_b_to_k1_to_c, r=discount_rate), qaly=discount(utility_b_to_k1_to_c, r=discount_rate))
state_b_to_k1_to_k2_to_a <- define_state(cost=discount(cost_b_to_k1_to_k2_to_a, r=discount_rate), qaly=discount(utility_b_to_k1_to_k2_to_a, r=discount_rate))
state_b_to_k1_to_k2_to_b <- define_state(cost=discount(cost_b_to_k1_to_k2_to_b, r=discount_rate), qaly=discount(utility_b_to_k1_to_k2_to_b, r=discount_rate))


# make up the markov model
strategy <- define_strategy(
    transition=mat_strans,
    a=state_a,
    b=state_b,
    c=state_c,
    a_to_a=state_a_to_a,
    b_to_a=state_b_to_k1_to_k2_to_a,
    a_to_b=state_a_to_b, 
    b_to_b=state_b_to_k1_to_k2_to_b, 
    b_k1_to_c=state_b_to_k1_to_c
)

# run model
res <- run_model(
    strategy_1 = strategy, parameters=param, init=INIT, cycles=CYCLES,
    method = count_method,
    cost=cost, effect=qaly
)

prob_dt <- dcast(as.data.table(get_counts(res)), model_time ~ state_names, value.var = "count", fun.aggregate = sum)
values_dt <- dcast(as.data.table(get_values(res)), model_time ~ value_names, value.var = "value", fun.aggregate = sum)

print(prob_dt)
print(values_dt)
print(res$run_model)