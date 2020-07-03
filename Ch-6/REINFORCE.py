import gym
import tensorflow as tf
import numpy as np
from datetime import datetime
import time

def mlp(x,hidden_layers,output_size,activation=tf.nn.relu,last_activation=None):
    for l in hidden_layers:
        x=tf.layers.dense(x,units=l,activation=activation)
    return tf.layers.dense(x,units=output_size,activation=last_activation)

def softmax_entropy(logits):
    return tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)

def discounted_rewards(rews,gamma):
    rtg=np.zeros_like(rews,dtype=np.float32)
    rtg[-1]=rews[-1]
    for i in reversed(range(len(rews)-1)):
        rtg[i]=rews[i]+gamma*rtg[i+1]
    return rtg

class Buffer():
    def __init__(self,gamma=0.99):
        self.gamma=gamma
        self.obs=[]
        self.act=[]
        self.ret=[]

    def store(self,temp_traj):
        if len(temp_traj)>0:
            self.obs.extend(temp_traj[:,0])
            rtg=discounted_rewards(temp_traj[:,1],self.gamma)
            self.ret.extend(rtg)
            self.act.extend(temp_traj[:,2])
            

    def get_batch(self):
        b_ret=self.ret
        return self.obs,self.act,b_ret

    def __len__(self):
        assert (len(self.obs)==len(self.act)==len(self.ret))
        return len(self.obs)
    
def REINFORCE(env_name,hidden_sizes=[32],lr=5e-3,num_epochs=50,gamma=0.99,steps_per_epoch=100):
    tf.reset_default_graph()

    
    env=gym.make(env_name)
    
    obs_dim=env.observation_space.shape
    act_dim=env.action_space.n

    obs_ph=tf.placeholder(shape=(None,obs_dim[0]),dtype=tf.float32,name='obs')
    act_ph=tf.placeholder(shape=(None,),dtype=tf.int32,name='act')
    ret_ph=tf.placeholder(shape=(None,),dtype=tf.float32,name='ret')

    p_logits=mlp(obs_ph,hidden_sizes,act_dim,activation=tf.tanh)
    
    ##############REINFORCE Update##############
    act_multn=tf.squeeze(tf.random.multinomial(p_logits,1))
    actions_mask=tf.one_hot(act_ph,depth=act_dim)

    p_log=tf.reduce_sum(actions_mask*tf.nn.log_softmax(p_logits),axis=1)

    # entropy useful to study the algorithms
    entropy = -tf.reduce_mean(softmax_entropy(p_logits))
    p_loss=-tf.reduce_mean(p_log*ret_ph)
    
    p_opt=tf.train.AdamOptimizer(lr).minimize(p_loss)
     # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)
    # Set scalars and hisograms for TensorBoard
    tf.summary.scalar('p_loss', p_loss, collections=['train'])
    tf.summary.scalar('entropy', entropy, collections=['train'])
    tf.summary.histogram('p_soft', tf.nn.softmax(p_logits), collections=['train'])
    tf.summary.histogram('p_log', p_log, collections=['train'])
    tf.summary.histogram('act_multn', act_multn, collections=['train'])
    tf.summary.histogram('p_logits', p_logits, collections=['train'])
    tf.summary.histogram('ret_ph', ret_ph, collections=['train'])
    train_summary = tf.summary.merge_all('train')

    tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    pre_scalar_summary = tf.summary.merge_all('pre_train')

    hyp_str = '-steps_{}-aclr_{}'.format(steps_per_epoch, lr)
    file_writer = tf.summary.FileWriter('log_dir/{}/REINFORCE_{}_{}'.format(env_name, clock_time, hyp_str), tf.get_default_graph())
    
    #############start a session, reset the global variables of the computational graph,and initialize some further variables
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    step_count=0
    train_rewards=[]
    train_ep_len=[]
    timer=time.time()
    ####Then, we create the two inner cycles that will interact with the environment to gather
    ######experience and optimize the policy, and print a few statistics
    for ep in range(num_epochs):
        obs=env.reset()
        buffer=Buffer(gamma)
        env_buf=[]
        ep_rews=[]
        while len(buffer)<steps_per_epoch:
            #run the policy
            act=sess.run(act_multn,feed_dict={obs_ph:[obs]})
            #take a step in the env
            obs2,rew,done,_=env.step(np.squeeze(act))

            env_buf.append([obs.copy(),rew,act])
            obs=obs2.copy()
            step_count+=1
            ep_rews.append(rew)

            if done:
                #add the full trajectory to the env
                buffer.store(np.array(env_buf))
                env_buf=[]
                train_rewards.append(np.sum(ep_rews))
                train_ep_len.append(len(ep_rews))
                obs=env.reset()
                ep_rews=[]

        # collect the episodes' information
        obs_batch,act_batch,ret_batch=buffer.get_batch()
        # run pre_scalar_summary before the optimization phase
        epochs_summary = sess.run(pre_scalar_summary, feed_dict={obs_ph:obs_batch, act_ph:act_batch, ret_ph:ret_batch})
        file_writer.add_summary(epochs_summary, step_count)

        #Policy optimization
        sess.run(p_opt,feed_dict={obs_ph:obs_batch,act_ph:act_batch,ret_ph:ret_batch})
        # run train_summary to save the summary after the optimization
        train_summary_run = sess.run(train_summary, feed_dict={obs_ph:obs_batch, act_ph:act_batch, ret_ph:ret_batch})
        file_writer.add_summary(train_summary_run, step_count)

        #Print some statistics
        if ep % 10==0:
            print('Ep:%d MnRew:%.2f MxRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d -- Time:%d' % (ep, np.mean(train_rewards), np.max(train_rewards), np.mean(train_ep_len), len(buffer), step_count,time.time()-timer))
            summary = tf.Summary()
            summary.value.add(tag='supplementary/len', simple_value=np.mean(train_ep_len))
            summary.value.add(tag='supplementary/train_rew', simple_value=np.mean(train_rewards))
            file_writer.add_summary(summary, step_count)
            file_writer.flush()

            timer = time.time()
            train_rewards=[]
            train_ep_len=[]
    env.close()
    file_writer.close()

if __name__=="__main__":
    REINFORCE('LunarLander-v2', hidden_sizes=[64], lr=8e-3, gamma=0.99,num_epochs=1000, steps_per_epoch=1000)



























            
    




























        
    
    
    
    
    
