from kuramoto import KuramotoModel

# N = 100 K = 0.1

model = KuramotoModel(N=100, K=0.1)
model.run_simulation()
model.animate()

# N = 100 K = 2

model = KuramotoModel(N=100, K=2)
model.run_simulation()
model.animate()

# N = 100 K = 10

model = KuramotoModel(N=100, K=5)
model.run_simulation()
model.animate()