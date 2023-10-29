import gym
import numpy as np

# Criação do ambiente
env = gym.make("Taxi-v3")

# Tabela Q de dimensões [número de estados, número de ações]
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Parâmetros do algoritmo
alpha = 0.1
gamma = 0.6
epsilon = 0.2

total_epochs, total_rewards = 0, 0
episodes = 10000

# Treinamento do agente
for _ in range(episodes):
    state = env.reset()
    epochs, reward = 0, 0
    done = False
    
    while not done:
        if isinstance(state, tuple):
            state = state[0]

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore ação
        else:
            action = np.argmax(Q[state])  # Escolhe a ação com maior Q

        next_state, reward, done, _, _ = env.step(action)
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]

        old_value = Q[state, action]
        next_max = np.max(Q[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q[state, action] = new_value

        state = next_state
        epochs += 1

    total_rewards += reward
    total_epochs += epochs

print(f"Resultados após {episodes} episódios:")
print(f"Média de etapas por episódio: {total_epochs / episodes}")
print(f"Média de recompensas por episódio: {total_rewards / episodes}")


env_visual = gym.make("Taxi-v3", render_mode="human")

# Visualização do comportamento do agente após o treinamento
num_episodes_for_visualization = 3
print("\nVisualizando o comportamento do agente após treinamento:")
for _ in range(num_episodes_for_visualization):
    state = env_visual.reset()
    if isinstance(state, tuple):
        state = state[0]
    done = False
    print("####### NOVO EPISÓDIO #########")
    
    while not done:
        action = np.argmax(Q[state])
        next_state, _, done, _, _ = env_visual.step(action)
        env_visual.render()
        state = next_state
        if isinstance(state, tuple):
            state = state[0]
