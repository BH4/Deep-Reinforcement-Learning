from Pong import Pong

env = Pong(10, 5, "Random", simpleOutput=True, paddleSize=2)

print(env.draw_state())
print(env.observe())
print(env.act(0))
print(env.act(0))
print(env.act(0))
print(env.act(0))
print(env.act(0))
print(env.act(0))
print(env.act(0))
print(env.act(0))
print(env.draw_state())
print(env.observe())
