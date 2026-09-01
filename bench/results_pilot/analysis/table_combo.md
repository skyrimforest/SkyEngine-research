# 基准汇总 (按求解器组合)

| combo | makespan(mean/min) | overhead | agv_busy | agv_loaded | empty_pick | blocking | n |
|---|---|---|---|---|---|---|---|
| cpsat+eecbs+couplingH | 844/299 | 15.2 | 0.9 | 0.4 | 13.7 | 2.0 | 16 |
| cpsat+eecbs+nearest | 1018/304 | 14.4 | 0.8 | 0.4 | 13.6 | 2.5 | 17 |
| greedy+astar+couplingH | 1531/483 | 30.6 | 0.6 | 0.2 | 29.3 | 25.5 | 18 |
| greedy+astar+nearest | 1661/459 | 37.4 | 0.6 | 0.2 | 24.8 | 21.7 | 18 |
| greedy+astar+random | 1368/400 | 29.6 | 0.7 | 0.3 | 26.2 | 17.9 | 18 |