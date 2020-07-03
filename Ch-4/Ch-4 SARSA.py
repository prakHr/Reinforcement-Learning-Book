import gym
import numpy as np

def eps_greedy(Q,s,eps=0.1):
    if eps>np.random.uniform(0,1):
        #Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        #Choose the index of max action among Q[s] 
        return greedy(Q,s)

def greedy(Q,s):
    #returns the index of maximum Q value in state s
    return np.argmax(Q[s])

def run_episodes(env,Q,num_episodes=100,to_print=False):
    total_reward=[]
    state=env.reset()
    
    for _ in range(num_episodes):
        done=False
        game_reward=0
        
        while not done:
            #select a greedy actino
            next_state,reward,done,_=env.step(greedy(Q,state))

            state=next_state
            game_reward+=reward
            if done:
                state=env.reset()
                total_reward.append(game_reward)
                
    if to_print:
        print('Mean score: %.3f of %i games!'%(np.mean(total_reward),num_episodes))
    
    return np.mean(total_reward)

def Q_learning(env,\
                        lr=0.01,\
                        num_episodes=10000,\
                        eps=0.3,\
                        gamma=0.95,\
                        eps_decay=0.00005):

    nA=env.action_space.n
    nS=env.observation_space.n
    Q=np.zeros((nS,nA))
    
    games_reward=[]
    test_rewards=[]

    for ep in range(num_episodes):
        state=env.reset()
        done=False
        total_reward=0
        if eps>0.01:
            eps=eps-eps_decay
        while not done:
            action = eps_greedy(Q,state,eps)
            next_state,reward,done,_=env.step(action)
            Q[state][action]=Q[state][action]+lr*(reward+gamma*np.max(Q[next_state])-Q[state][action])
            state=next_state
            total_reward+=reward
            if done:
                games_reward.append(total_reward)
        # Test the policy every 300 episodes and print the results
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            
    return Q
        
def SARSA_state_action_reward_next_state_next_action(env,\
                                                     lr=0.01,\
                                                     num_episodes=10000,\
                                                     eps=0.3,\
                                                     gamma=0.95,\
                                                     eps_decay=0.00005):
    nS=env.observation_space.n
    nA=env.action_space.n
    
    test_rewards=[]
    Q=np.zeros((nS,nA))

    games_reward=[]

    for ep in range(num_episodes):
        #resets the environment after every episode
        state=env.reset()
        #boolean variable set to True when env is in terminal state
        done=False
        total_reward=0
        if eps>0.01:
            eps=eps-eps_decay
        action = eps_greedy(Q,state,eps)
        while not done:
            next_state,reward,done,_=env.step(action)
            next_action=eps_greedy(Q,next_state,eps)
            Q[state][action]=Q[state][action]+lr*(reward+gamma*Q[next_state][next_action]-Q[state][action])
            state=next_state
            action=next_action
            total_reward+=reward
            if done:
                games_reward.append(total_reward)
        
        # Test the policy every 300 episodes and print the results
        if(ep % 300) == 0:
            test_reward=run_episodes(env,Q,1000)
            print("Episode: {:5d} Eps: {:2.4f} Rew: {:2.4f}".format(ep,eps,test_reward))
            test_rewards.append(test_reward)
    return Q

if __name__=='__main__':
    env=gym.make('Taxi-v3')
    env.reset()
    Q=Q_learning(env,\
                                                       lr=0.1,\
                                                       num_episodes=10000,\
                                                     eps=0.4,\
                                                     gamma=0.95,\
                                                     eps_decay=0.001)
    '''
    Q=SARSA_state_action_reward_next_state_next_action(env,\
                                                       lr=0.1,\
                                                       num_episodes=10000,\
                                                     eps=0.4,\
                                                     gamma=0.95,\
                                                     eps_decay=0.001)
    '''
    





















            
            
