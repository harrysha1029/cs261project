def social_cost(agents, alt, metric):
    """Computes the social cost of
    some point with respect to a list of
    agents and a distance metric

    Args:
        agents ([Agent]): A list of agents
        alt ([type]): Some point
        metric (Function): The metric to be used in the calculation

    Returns:
        (float) The social cost
    """
    costs = [metric(alt, a) for a in agents]
    return sum(costs)


def distortion(agents, alt, opt, metric):
    """Returns the distortion of some point
    alt with respect to agents and the some optimal
    solution

    Args:
        agents ([Agent]): A list of agents
        alt ([type]): Some point in the metric space
        opt ([type]): The optimal solution
        metric (Function): The metric to use

    Returns:
        float: The distorion of alt w.r.t. agents, opt and metric
    """
    return social_cost(agents, alt, metric) / social_cost(agents, opt, metric)
