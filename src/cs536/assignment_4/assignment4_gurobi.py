import gurobipy as gp
from gurobipy import GRB


def optimize_topology_for_traffic(T, d=4, time_limit=None, mip_gap=None, verbose=True):
    """
    Solve the joint topology + multicommodity-flow design problem.

    Args:
        T: 8x8 traffic matrix as a list of lists, numpy array, or similar.
           T[i][i] should be 0.
        d: required in-degree and out-degree for every node (default 4).
        time_limit: optional solver time limit in seconds.
        mip_gap: optional relative MIP gap.
        verbose: if False, suppress Gurobi solver output.

    Returns:
        model, lambda_value, selected_arcs, routed_flow
        where selected_arcs is a list of (i, j) pairs and routed_flow is a dict
        keyed by ((s,t), (i,j)).
    """
    n = len(T)
    if n != 8:
        raise ValueError("This assignment fixes n = 8.")
    if any(len(row) != n for row in T):
        raise ValueError("T must be an 8x8 matrix.")
    if any(T[i][i] != 0 for i in range(n)):
        raise ValueError("Diagonal entries must be 0.")

    nodes = range(n)
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    comms = [(s, t) for s in nodes for t in nodes if s != t]

    m = gp.Model("best_topology_for_given_traffic")
    if not verbose:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap

    # x[i,j] = 1 if directed edge i -> j is included in the topology.
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")

    # f[s,t,i,j] = amount of commodity (s,t) routed on arc i -> j.
    f = m.addVars(comms, arcs, lb=0.0, vtype=GRB.CONTINUOUS, name="f")

    # Concurrent-flow scaling factor.
    lam = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="lambda")

    # Objective: maximize lambda.
    m.setObjective(lam, GRB.MAXIMIZE)

    # Every node must have exactly d outgoing and d incoming edges.
    m.addConstrs((gp.quicksum(x[i, j] for j in nodes if j != i) == d for i in nodes),
                 name="outdeg")
    m.addConstrs((gp.quicksum(x[i, j] for i in nodes if i != j) == d for j in nodes),
                 name="indeg")

    # Capacity constraint: a selected arc has capacity 1, otherwise 0.
    m.addConstrs(
        (gp.quicksum(f[s, t, i, j] for (s, t) in comms) <= x[i, j] for (i, j) in arcs),
        name="cap",
    )

    # Flow conservation for each commodity.
    for (s, t) in comms:
        demand = float(T[s][t])
        for v in nodes:
            outflow = gp.quicksum(f[s, t, v, j] for j in nodes if j != v)
            inflow = gp.quicksum(f[s, t, i, v] for i in nodes if i != v)
            if v == s:
                m.addConstr(outflow - inflow == lam * demand,
                            name=f"flow_src_{s}_{t}_{v}")
            elif v == t:
                m.addConstr(outflow - inflow == -lam * demand,
                            name=f"flow_dst_{s}_{t}_{v}")
            else:
                m.addConstr(outflow - inflow == 0.0,
                            name=f"flow_mid_{s}_{t}_{v}")

    m.optimize()

    if m.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Model did not solve successfully. Status = {m.Status}")

    lambda_value = lam.X
    selected_arcs = [(i, j) for (i, j) in arcs if x[i, j].X > 0.5]
    routed_flow = {
        ((s, t), (i, j)): f[s, t, i, j].X
        for (s, t) in comms
        for (i, j) in arcs
        if f[s, t, i, j].X > 1e-9
    }

    return m, lambda_value, selected_arcs, routed_flow


if __name__ == "__main__":
    # Example hose-model traffic matrix (every row and column sum <= 4).
    T = [
        [0, 2, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 2, 0, 0, 1, 0],
        [0, 0, 2, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 2, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 2, 0],
    ]

    _, lam, arcs, flow = optimize_topology_for_traffic(T, d=4, verbose=True)
    print(f"Optimal lambda: {lam:.6f}")
    print("Selected arcs:")
    for a in arcs:
        print(a)
    print(f"Number of positive flow variables: {len(flow)}")
