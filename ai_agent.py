import gym
import numpy as np

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras.models import load_model


import random

class AiAgent:
    def __init__(self, state_size, action_size, explorationRate, explorationRate_min):
        self.learning_rate = 1e-3
        self.explorationRate = explorationRate #percentage of random acting
        self.explorationRate_min = explorationRate_min
        self.explorationRate_decay = 0.995
        self.gamma = 0.95
        self.memory = deque(maxlen=100000)
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()



    def _build_model(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def train_model(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            target_f = self.model.predict(state)
            #print(reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f[0][action] = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.explorationRate > self.explorationRate_min:
            self.explorationRate *= self.explorationRate_decay


    def act(self, state):
        #print(state)
        if random.uniform(0,1) <= self.explorationRate:
            return random.randrange(0, self.action_size)

        predictions = self.model.predict(state)
        return np.argmax(predictions[0])


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def loadModel(self,modelPath):
        self.model = load_model(modelPath)
        #self.model.load_weights(modelPath)
        print("Loaded model")

    def saveModel(self,modelPath):
        self.model.save(modelPath, overwrite=True)
        #self.model.save_weights(modelPath)



if __name__ == "__main__":
    modelPath = './model.h5'
    episodes = 1000
    scores = []
    maxSteps = 500
    explorationRate = 0.9
    explorationRate_min = 0.1
    batchSize = 32


    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = AiAgent(state_size, action_size, explorationRate, explorationRate_min)

    try:
        #agent.loadModel(modelPath)
        print(" ")
    except (ValueError, RuntimeError, TypeError, NameError, OSError):
        pass

    ###play
    for e in range(1,episodes+1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        for time in range(maxSteps):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                #print("[Episode {}] Score: {}".format(e, time))
                scores.append(time+1)
                break

        agent.train_model(batchSize)

        if e % 100 == 0 and e > 0:
            print("[Episode {}] Average score of last 100: {}".format(e ,sum(scores[-100:]) / 100))
            print("\t Exploration rate: {}".format(agent.explorationRate))
            print("\t Worst score of last 100: {}".format(min(scores[-100:])))
            print("\t Best score of last 100: {}".format(max(scores[-100:])))

    #agent.saveModel(modelPath)
    print("Average score: {}".format(sum(scores)/len(scores)))
    print("Highscore: {}".format(max(scores)))
