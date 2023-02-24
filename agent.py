import numpy as np

from action_value_table import ActionValueTable

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3  # agents actions
GAMMA = 0.99
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON


    def update(self, state, action, reward, next_state, done):
        '''
        This function will update the values stored in self.value using Q-learning. 

        HINT: Use self.values.get_value and self.values.set_value
        HINT: Remember to add a special case to handle the terminal state

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze
            
            action : (int) the action taken at state

            reward : float

        returns:
            action : (int) a epsilon-greedy action for state 
        '''
        if done:
            newValue = self.values.get_value(state, action) + ((STEP_SIZE * reward) - self.values.get_value(state,action))
            self.values.set_value(state,action,newValue)
        else:
            nextAction = self.get_greedy_action(next_state)
            newValue = self.values.get_value(state, action) + (self.step_size * (reward + (GAMMA * self.values.get_value(next_state, nextAction)) - self.values.get_value(state, action)))
            self.values.set_value(state,action,newValue)




    def get_action(self, state):
        '''
        This function returns an action from self.actions given a state. 

        Implement this function using an epsilon-greedy policy. 

        HINT: use np.random.rand() to generate random numbers
        HINT: If more than one action has maximum value, treat them all as the greedy action. In other words,
        if there are b greedy actions, each should have probability epsilon/b + |A|, where |A| is 
        the number of actions in this state.

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze

        returns:
            action : (int) a epsilon-greedy action for state 

        '''

        n = np.random.rand()
        if n < EPSILON:
            action = np.random.choice(range(4))
        else:
            actionValues = []
            actionIndices = []
            maxValues = []
            for i in self.actions:
                actionValues.append(self.values.get_value(state, i))
                actionIndices.append(i)
            for i in range(len(actionValues)):
                if actionValues[i] == np.amax(actionValues):
                    maxValues.append(actionIndices[i])

            action = np.random.choice(maxValues)

        return action


    def get_greedy_action(self, state):
        '''
        This function returns an action from self.actions given a state. 

        Implement this function using a greedy policy, i.e. return the action with the highest value
        HINT: If more than more than one action has maximum value, uniformly randomize amongst them

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze

        returns:
            action : (int) a greedy action for state 
        '''

        actionValues = []
        actionIndices = []
        maxValues = []
        for i in self.actions:
            actionValues.append(self.values.get_value(state, i))
            actionIndices.append(i)
        for i in range(len(actionValues)):
            if actionValues[i] == np.amax(actionValues):
                maxValues.append(actionIndices[i])

        action = np.random.choice(maxValues)
        return action
