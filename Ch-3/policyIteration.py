import numpy as np
import gym
'''
conda create -n reinforce python=3.6 anaconda
conda activate reinforce
conda install -c conda-forge tensorflow=1.14,gym,pybox2d,opencv
pip install roboschool(does not work in windows)
python <script_name>.py
conda deactivate
'''
"""
cd "C:\Users\HP\Desktop\RL\Chapter-3\policyIteration.py"
conda activate reinforce
python policyIteration.py

"""

def eval_state_action(V,s,a,gamma=0.99):
    #Formula applied here
    return np.sum([p*(rew+gamma*V[next_s]) for p,next_s,rew,_ in env.P[s][a]])

def policy_evaluation(V,policy,eps=0.0001):
    while True:
        delta=0
        for s in range(nS):
            old_v=V[s]
            V[s]=eval_state_action(V,s,policy[s])
            delta=max(delta,np.abs(old_v-V[s]))
        #Update value function until it reaches a steady state
        if eps>delta:
            break

def policy_improvement(V,policy):
    policy_stable=True
    for s in range(nS):
        old_a=policy[s]
         # update the policy with the action that bring to the highest state value
        policy[s]=np.argmax([eval_state_action(V,s,a) for a in range(nA)])
        if old_a!=policy[s]:
            policy_stable=False
    return policy_stable

def run_episodes(env,policy,num_games=100):
    total_reward=0
    state=env.reset()
    for _ in range(num_games):
        done=False
        #env.render()
        while not done:
             # select the action accordingly to the policy
            next_state,reward,done,_=env.step(policy[state])
            state=next_state
            total_reward+=reward 
            if done:
                state=env.reset()
            
    print('Won %i of %i games!'%(total_reward,num_games))
    
if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    nA = env.action_space.n
    nS = env.observation_space.n
    V = np.zeros(nS)
    policy = np.zeros(nS)

    policy_stable=False
    it=0
    while not policy_stable:
        policy_evaluation(V,policy)
        policy_stable=policy_improvement(V,policy)
        it+=1
    print('Converged after %i policy iterations'%(it))
    run_episodes(env,policy)
    print(V.reshape((4,4)))
    print(policy.reshape((4,4)))
    
