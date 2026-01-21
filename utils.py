import pandas as pd

from core import (ProbabilityWithRange, TimeVaringProbability, ComplementProbability, 
                  MarkovController, MarkovState, ChanceNode, StateTransition)


def parse_value_from_str(value_str: str):
    result = []

    rows = value_str.split(';')
    for row in rows:
        parsed_row = []
        values = row.split(',')

        for value in values:
            value = value.strip()
            if not value:
                continue

            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value

            parsed_row.append(parsed_value)

        result.append(parsed_row)

    if len(result) == 1:
        return result[0]
    return result


def parse_parameters(param_df):
    parameters = {}

    for _, row in param_df.iterrows():
        parameter_name, parameter_type, value_str = row[['parameter_name', 'parameter_type', 'value']]
        parameter_type = parameter_type.lower()
        if parameter_type == 'complement probability':
            parameters[parameter_name] = ComplementProbability()
        elif parameter_type == 'constant probability':
            parameters[parameter_name] = ProbabilityWithRange(*parse_value_from_str(value_str))
        elif parameter_type == 'probability with range':
            parameters[parameter_name] = ProbabilityWithRange(*parse_value_from_str(value_str))
    
    for _, row in param_df.iterrows():
        parameter_name, parameter_type, value_str = row[['parameter_name', 'parameter_type', 'value']]
        if parameter_type == 'time-varing probability':
            parameter_values = []
            parsed_values = parse_value_from_str(value_str)
            for parsed_value in parsed_values:
                if isinstance(parsed_value[0], str) and len(parsed_value) == 1:
                    if parsed_value[0] in parameters:
                        parameter_values.append(parameters[parsed_value[0]])
                    else:
                        raise KeyError(f'invalid parameter value: {parsed_value[0]}')
                else:
                    parameter_values.append(ProbabilityWithRange(*parsed_value))
            parameters[parameter_name] = TimeVaringProbability(parameter_values)
    return parameters


def make_node(node_name:str, parent:str, transition_prob:float=1.0, **varaibles):
    if parent == '__start__':
        return MarkovState(node_name=node_name, init_prob=transition_prob, **varaibles)
    else:
        return ChanceNode(node_name=node_name, trans_prob=transition_prob, **varaibles)
    

def parse_model(node_df:pd.DataFrame, transition_df:pd.DataFrame, parameters:dict, **controller_params):
    controller = MarkovController(**controller_params)

    nodes = {}
    
    for _, row in node_df.iterrows():
        node_name = row.pop('node_name')
        parent = row.pop('parent')
        transition_prob = row.pop('transition_probability')

        variables = row.to_dict()

        try:
            transition_prob = float(transition_prob)
        except Exception:
            transition_prob = parameters.get(transition_prob)

        for k in variables.keys():
            try:
                variables[k] = float(variables[k])
            except Exception:
                variables[k] = parameters.get(variables[k])

        nodes[node_name] = make_node(node_name=node_name, parent=parent, transition_prob=transition_prob, **variables)

    for _, row in node_df.iterrows():
        name, parent = row['node_name'], row['parent']
        node_obj = nodes[name]
        
        if parent == '__start__':
            controller.add_child(node_obj)
        elif parent in nodes:
            nodes[parent].add_child(node_obj)

    for _, row in transition_df.iterrows():
        node_name = row.pop('node_name')
        parent = row.pop('parent')
        dst_state = row.pop('dst_state')
        transition_prob = row.pop('transition_probability')

        variables = row.to_dict()
        
        try:
            transition_prob = float(transition_prob)
        except Exception:
            transition_prob = parameters.get(transition_prob)

        for k in variables.keys():
            try:
                variables[k] = float(variables[k])
            except Exception:
                variables[k] = parameters.get(variables[k])

        transition_node = StateTransition(node_name=node_name, trans_prob=transition_prob, dst_state=nodes[dst_state], **variables)

        if parent in nodes:
            nodes[parent].add_child(transition_node)

    controller.init_prob()
    controller.verify()

    return controller
