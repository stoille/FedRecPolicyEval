# Set 1: Baseline Replication (3 graphs)

# Vary beta (0.5, 1, 1.5, 2) with fixed gamma=0.5
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=0.5 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=1.5 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=2.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Vary gamma (0.2, 0.4, 0.6, 0.8) with fixed beta=1.0
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=1.0 gamma=0.2 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=1.0 gamma=0.4 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=1.0 gamma=0.6 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=1 --run-config="num-nodes=1 num-server-rounds=1 local-epochs=75 learning-rate=0.01 beta=1.0 gamma=0.8 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Set 2: Federation Parameter Effects (6 graphs)
# Vary number of nodes (5, 10, 20, 30)
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=20 --run-config="num-nodes=20 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=30 --run-config="num-nodes=30 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Vary aggregation frequency (5, 10, 20, 30 epochs between rounds)
#flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=10 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=20 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=30 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Vary number of rounds (2, 10, 20, 30 rounds)
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=2 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=20 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=30 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Set 3: Local vs Global Evolution (4 graphs)
# These runs capture both local and global evolution
# Repeat with different node counts to get divergence data
# Reused from Set 2
# Set 4: Mitigation Effectiveness (9 graphs)
# 3x3 matrix varying learning rate schedules and node counts
# Learning rate schedules: constant, decay, feature-specific
# Node counts: 5, 10, 15

# Constant learning rate
#flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
#flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=15 --run-config="num-nodes=15 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='constant'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Decay learning rate
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='decay'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='decay'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=15 --run-config="num-nodes=15 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='decay'" --app .;afplay /System/Library/Sounds/Ping.aiff

# Feature-specific learning rate
flower-simulation --num-supernodes=5 --run-config="num-nodes=5 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='feature_specific'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=10 --run-config="num-nodes=10 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='feature_specific'" --app .;afplay /System/Library/Sounds/Ping.aiff
flower-simulation --num-supernodes=15 --run-config="num-nodes=15 num-server-rounds=10 local-epochs=5 learning-rate=0.01 beta=1.0 gamma=0.5 learning-rate-schedule='feature_specific'" --app .;afplay /System/Library/Sounds/Ping.aiff
