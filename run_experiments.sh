# Set 1: Baseline Replication (3 graphs)

# Vary beta (0.5, 1, 1.5, 2) with fixed gamma=0.5
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=0.5 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=1.5 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=2.0 gamma=0.5 learning-rate-schedule='constant'" --app .

# Vary gamma (0.2, 0.4, 0.6, 0.8) with fixed beta=1.0
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=1.0 gamma=0.2 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=1.0 gamma=0.4 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=1.0 gamma=0.6 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=200 learning-rate=0.0001 beta=1.0 gamma=0.8 learning-rate-schedule='constant'" --app .

# Set 2: Federation Parameter Effects (6 graphs)
# Vary number of nodes (10, 50, 100, 500)
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=50 --run-config="num-nodes=50 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=500 --run-config="num-nodes=500 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .

# Vary aggregation frequency (1, 5, 10, 20 epochs between rounds)
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=1 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=5 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=10 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .

# Vary number of rounds frequency (5, 20, 50, 100 rounds)
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=2 local-epochs=10 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=20 local-epochs=10 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=50 local-epochs=10 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=10 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .

# Set 3: Local vs Global Evolution (4 graphs)
# These runs capture both local and global evolution
#flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
# Repeat with different node counts to get divergence data
#flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
#flower-simulation --num-supernodes=50 --run-config="num-nodes=50 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
#flower-simulation --num-supernodes=500 --run-config="num-nodes=500 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .

# Set 4: Mitigation Effectiveness (9 graphs)
# 3x3 matrix varying learning rate schedules and node counts
# Learning rate schedules: constant, decay, feature-specific
# Node counts: 10, 100, 500

# Constant learning rate
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .
flower-simulation --num-supernodes=500 --run-config="num-nodes=500 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .

# Decay learning rate
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='decay'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='decay'" --app .
flower-simulation --num-supernodes=500 --run-config="num-nodes=500 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='decay'" --app .

# Feature-specific learning rate
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='feature_specific'" --app .
flower-simulation --num-supernodes=100 --run-config="num-nodes=100 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='feature_specific'" --app .
flower-simulation --num-supernodes=500 --run-config="num-nodes=500 num-server-rounds=100 local-epochs=20 learning-rate=0.0001 beta=1.0 gamma=0.5 learning-rate-schedule='feature_specific'" --app .
