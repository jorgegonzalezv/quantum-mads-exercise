import numpy as np
from scipy.optimize import LinearConstraint, Bounds
from scipy.optimize import milp


# nodes
N_DEPOSIT_STATIONS = 3  # C
N_STORAGES = 2  # D
N_TERMINALS = 3  # T
N_CHARGING_STATIONS = 2  # S

# initial conditions
TERMINAL_MAX_CAPACITY = 5
STORAGE_MAX_CAPACITY = 5
TERMINAL_INITIAL_CONTAINERS = 1
STORAGE_INITIAL_CONTAINERS = 1
TERMINAL_TWO_DEMAND = 2
CHARGING_STATIONS_DEMAND = np.array([1, 1])
DEPOSIT_STATIONS_PRODUCTION = np.array([1, 1, 1])

# big M
M = 9999

# Respect order: C1, C2, C3, D1, D2, T1, T2, T3, S1, S2
COST_STORAGE_VECTOR = np.array([22, 28, 30, 20, 25])
COST_TRANSPORT_MATRIX = np.array(
    [
        [M, M, M, 12, 10, 14, 15, 12, M, M],
        [M, M, M, 19, 12, 15, 17, 12, M, M],
        [M, M, M, 16, 18, 20, 19, 17, M, M],
        [M, M, M, 0, 18, 18, 16, 17, 11, 17],
        [M, M, M, 16, 0, 14, 17, 18, 12, 14],
        [M, M, M, 16, 15, 0, 15, 18, 14, 15],
        [M, M, M, 14, 17, 15, 0, 12, 18, 17],
        [M, M, M, 13, 16, 17, 16, 0, 12, 20],
    ]
)

# cost function/vector: f = c @ x
cost_transport_vector = COST_TRANSPORT_MATRIX.flatten()
cost = np.concatenate([cost_transport_vector, COST_STORAGE_VECTOR])
idxs = (
    np.array(range(len(cost_transport_vector)))
    .reshape(COST_TRANSPORT_MATRIX.shape)
    .astype(int)
)

# auxiliar index helper groups, hardcoded for layout purposes
from_C_idxs = idxs[:3]
from_D_idxs = idxs[3:5]
from_T_idxs = idxs[5:]
from_C_to_S_idxs = idxs[:3, -2:]
to_C_idxs = idxs[:, :3].T
to_D_idxs = idxs[:, 3:5].T
to_T_idxs = idxs[:, 5:8].T
to_S_idxs = idxs[:, 8:].T
N_D_idxs = np.cumsum(np.ones(2)) + idxs[-1, -1]
N_T_idxs = np.cumsum(np.ones(3)) + N_D_idxs[-1]
N_D_idxs = N_D_idxs.astype(int)
N_T_idxs = N_T_idxs.astype(int)

# constraints
n_contraints = (
    (N_STORAGES + N_TERMINALS)
    + (N_STORAGES + N_TERMINALS)
    + N_CHARGING_STATIONS
    + N_DEPOSIT_STATIONS
    + (N_STORAGES + N_TERMINALS)
    + 1
)

constraint_idx = 0
constraints_matrix = np.zeros((n_contraints, len(cost)))
bound_lower = -np.inf * np.ones(n_contraints)
bound_upper = np.inf * np.ones(n_contraints)

# 1. conservation of mass
for storage_idx in range(N_STORAGES):
    constraints_matrix[constraint_idx, from_C_idxs[storage_idx]] = 1
    constraints_matrix[
        constraint_idx, to_D_idxs[storage_idx]
    ] -= 1  # for loopback purposes
    constraints_matrix[constraint_idx, N_D_idxs[storage_idx]] = 1
    bound_lower[constraint_idx] = STORAGE_INITIAL_CONTAINERS
    bound_upper[constraint_idx] = STORAGE_INITIAL_CONTAINERS
    constraint_idx += 1

for T_idxs_idx in range(N_TERMINALS):
    constraints_matrix[constraint_idx, from_T_idxs[T_idxs_idx]] = 1
    constraints_matrix[constraint_idx, to_T_idxs[T_idxs_idx]] -= 1
    constraints_matrix[constraint_idx, N_T_idxs[T_idxs_idx]] = 1
    bound_lower[constraint_idx] = TERMINAL_INITIAL_CONTAINERS
    bound_upper[constraint_idx] = TERMINAL_INITIAL_CONTAINERS

    if T_idxs_idx + 1 == 2:  # T_idxs 2 does not have containers
        bound_lower[constraint_idx] = 0
        bound_upper[constraint_idx] = 0

    constraint_idx += 1

# 2. storage capacity
for storage_idx in range(N_STORAGES):
    constraints_matrix[constraint_idx, N_D_idxs[storage_idx]] = 1
    bound_lower[constraint_idx] = 0
    bound_upper[constraint_idx] = STORAGE_MAX_CAPACITY
    constraint_idx += 1

for T_idxs_idx in range(N_TERMINALS):
    constraints_matrix[constraint_idx, N_T_idxs[T_idxs_idx]] = 1
    bound_lower[constraint_idx] = 0
    bound_upper[constraint_idx] = TERMINAL_MAX_CAPACITY
    if T_idxs_idx + 1 == 2:
        bound_lower[constraint_idx] = TERMINAL_TWO_DEMAND

    constraint_idx += 1

# 3. demand satisfaction
for charging_station_idx in range(N_CHARGING_STATIONS):
    constraints_matrix[constraint_idx, to_S_idxs[charging_station_idx]] = 1
    bound_lower[constraint_idx] = CHARGING_STATIONS_DEMAND[
        charging_station_idx
    ]
    bound_upper[constraint_idx] = CHARGING_STATIONS_DEMAND[charging_station_idx]
    constraint_idx += 1

# 4. production is not stored
for deposit_station_idx in range(N_DEPOSIT_STATIONS):
    constraints_matrix[constraint_idx, from_C_idxs[deposit_station_idx]] = 1
    bound_lower[constraint_idx] = DEPOSIT_STATIONS_PRODUCTION[deposit_station_idx]
    bound_upper[constraint_idx] = DEPOSIT_STATIONS_PRODUCTION[deposit_station_idx]
    constraint_idx += 1

# 5. TODO por aqui INITIAL CAPACITY (poner nombres explicitos)
for storage_idx in range(N_STORAGES):
    constraints_matrix[constraint_idx, from_D_idxs[storage_idx]] = 1
    bound_lower[constraint_idx] = STORAGE_INITIAL_CONTAINERS
    bound_upper[constraint_idx] = STORAGE_INITIAL_CONTAINERS
    constraint_idx += 1

for T_idxs_idx in range(N_TERMINALS):
    constraints_matrix[constraint_idx, from_T_idxs[T_idxs_idx]] = 1
    bound_lower[constraint_idx] = TERMINAL_INITIAL_CONTAINERS
    bound_upper[constraint_idx] = TERMINAL_INITIAL_CONTAINERS

    if T_idxs_idx + 1 == 2:  # T_idxs 2 does not have containers
        bound_lower[constraint_idx] = 0
        bound_upper[constraint_idx] = 0

    constraint_idx += 1

# 6. movements not allowed
constraints_matrix[
    constraint_idx,
    np.concatenate([from_C_to_S_idxs.flatten(), to_C_idxs.flatten()]),
] = 1
bound_lower[constraint_idx] = 0  # equality constraint
bound_upper[constraint_idx] = 0
constraint_idx += 1

# optimize!
integrality = np.ones_like(cost)  # all variables are integer variables
constraints = LinearConstraint(constraints_matrix, bound_lower, bound_upper)
bounds = Bounds(lb=0, ub=np.inf)
res = milp(
    c=cost,
    constraints=constraints,
    integrality=integrality,
    bounds=bounds,
)

print(f"Total cost: {res.fun}")
print(f"Transport matrix:")
print("C1,C2,C3,D1,D2,T1,T2,T3,S1,S2")
print(res.x[:-5].reshape(COST_TRANSPORT_MATRIX.shape).astype(int))
print("Containers at storage and terminal nodes:")
print(res.x[-5:].astype(int))
