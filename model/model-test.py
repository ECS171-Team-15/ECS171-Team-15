from model import create_model

model = create_model(0.3, [1000, 50, 20], 100000)
model.summary()
