import numpy as np
import gym

def eval_state_action(V,s,a,gamma=0.99):
    return np.sum([p*(reward+gamma*V[next_s]) for p,next_s,reward,_ in env.P[s][a]])

def value_iteration(eps=0.0001):
    V=np.zeros(nS)
    it=0
    while True:
        delta=0
        #update the value of each state
        for s in range(nS):
            old_v=V[s]
            V[s]=np.max([eval_state_action(V,s,a) for a in range(nA)])
            delta=max(delta,np.abs(old_v-V[s]))
        if delta<eps:break
        else:
            print('Iteration: ',it,' delta: ',np.round(delta,5))
        it+=1

    return V

def run_episodes(env,V,num_games=100):
    total_reward=0
    state=env.reset()
    for _ in range(num_games):
        done=False
        #env.render()
        while not done:
            #choose the best action using the value function
            action=np.argmax([eval_state_action(V,state,a) for a in range(nA)])
            next_state,reward,done,_=env.step(action)
            state=next_state
            total_reward+=reward
            if done:#Done means reached the goal then we have to restart
                #env.render()
                state=env.reset()
    
    print('Won %i of %i games!'%(total_reward, num_games))

            
if __name__=="__main__":
    env=gym.make('FrozenLake-v0')
    env=env.unwrapped
    nA=env.action_space.n
    nS=env.observation_space.n
    V=value_iteration(eps=0.0001)
    run_episodes(env,V,100)
    print(V.reshape((4,4)))
