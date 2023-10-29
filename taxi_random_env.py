import gym

# Criação do ambiente
env = gym.make("Taxi-v3", render_mode="human")
env.reset()
env.render()


total_epochs, total_rewards = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, reward = 0, 0
    
    done = False
    
    while not done:
        action = env.action_space.sample()
        result = env.step(action)

        state, reward, done, _, info = env.step(action)
        
        epochs += 1

    total_rewards += reward
    total_epochs += epochs

print(f"Resultados após {episodes} episódios:")
print(f"Média de etapas por episódio: {total_epochs / episodes}")
print(f"Média de recompensas por episódio: {total_rewards / episodes}")
