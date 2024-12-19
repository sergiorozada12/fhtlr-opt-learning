import numpy as np
from itertools import product
from src.utils import ReplayBuffer, Discretizer

class BackwardPropagation:
    def __init__(self,H,nS,nA,R,P) -> None:
        self.R = R
        self.P = P

        self.H = H

        self.Q = np.zeros(
                    np.concatenate([[H], [nS], [nA]])
                )
    
    def run(self) -> np.ndarray:
        R_sa = np.sum(np.multiply(self.R, self.P),axis=0)
        for h in range(1,self.H+1):
            if h == 1:
                self.Q[self.H-h,:,:] = R_sa
            else:
                self.Q[self.H-h,:,:] = R_sa + np.sum((self.P*np.max(self.Q[self.H-(h-1),:,:], axis=1)[:,np.newaxis,np.newaxis]),axis=0)

    def q_reshape(self, discretizer):
        Q_reshape = np.zeros(
            np.concatenate([[self.H], discretizer.bucket_states, discretizer.bucket_actions])
        )
        for h in range(self.H):
            for nA, a in enumerate(product(*[ np.arange(0,da) for da in discretizer.bucket_actions])):
                Q_reshape [h,...,a] = np.transpose(self.Q[h,:,nA].reshape(*discretizer.bucket_states))
        return Q_reshape

class FrontPolicyImprovement:
    def __init__(self,H,nS,nA,R,P) -> None:

        self.R_sa = np.sum(np.multiply(R, P),axis=0)
        self.P_sa_s = np.transpose(P, (1, 2, 0))

        self.H = H
        self.nS = nS
        self.nA = nA

        self.Q = np.zeros(
                    np.concatenate([[H], [nS], [nA]])
                )
    
    def front_policy_eval(self,Pi):
        V = np.zeros((self.H,self.nS))
        for _ in range(100):
            for h in range(self.H):
                P_s_s = np.einsum('ijk, ij -> ik', self.P_sa_s, Pi[h,:])
                R_s = np.einsum('ij, ij -> i', self.R_sa, Pi[h,:])
                V[h,:] = R_s
                if h < self.H-1 :
                    V[h,:] += P_s_s @ V[h+1,:]
            
        Q_aux = np.roll((self.R_sa[:,:,np.newaxis] + self.P_sa_s @ np.transpose(V)), shift=-1, axis=2)
        Q_aux[:,:,self.H-1] = self.R_sa
        self.Q = np.transpose(Q_aux,(2,0,1))
        return V, self.Q
    
    def policy_improvement(self,Q):
        Pi = np.zeros((self.H,self.nS, self.nA))
        for h in range(self.H):
            for s in range(self.nS):
                a = np.argmax(Q[h, s, :])
                Pi[h,s, a] = 1
        return Pi

    def run(self) -> np.ndarray:
        V_old = np.zeros((self.H,self.nS))
        Q_old = np.zeros((self.H,self.nS, self.nA))
        Pi = np.random.rand(self.H, self.nS, self.nA)
        Pi /= Pi.sum(axis=1, keepdims=True)
        error_plicy = []
        for _ in range(100):
            V_new, Q_new = self.front_policy_eval(Pi)
            Pi = self.policy_improvement(Q_new)

            #print(V)
            #assert np.all(V >= V_old)
            #assert np.all(Q >= Q_old)
            
            error_plicy.append(np.linalg.norm(V_new - V_old))
            V_old = V_new
            Q_old = Q_new
        
        return Q_old, error_plicy
    
    def q_reshape(self, discretizer):
        Q_reshape = np.zeros(
            np.concatenate([[self.H], discretizer.bucket_states, discretizer.bucket_actions])
        )
        for h in range(self.H):
            for nA, a in enumerate(product(*[ np.arange(0,da) for da in discretizer.bucket_actions])):
                Q_reshape [h,...,a] = np.transpose(self.Q[h,:,nA].reshape(*discretizer.bucket_states))
        return Q_reshape

class BackPolicyImprovement:
    def __init__(self,H,nS,nA,R,P) -> None:

        self.R_sa = np.sum(np.multiply(R, P),axis=0)
        self.P_sa_s = np.transpose(P, (1, 2, 0))

        self.H = H
        self.nS = nS
        self.nA = nA

        self.Q = np.zeros(
                    np.concatenate([[H], [nS], [nA]])
                )
    
    def back_policy_eval(self,Pi):
        V = np.zeros((self.H,self.nS))
        #Calculo de V(s)
        for h in range(1,self.H+1):
            P_s_s = np.einsum('ijk, ij -> ik', self.P_sa_s, Pi[self.H-h,:])
            R_s = np.einsum('ij, ij -> i', self.R_sa, Pi[self.H-h,:])
            if h == 1:
                V[self.H-h,:] = R_s
            else:
                V[self.H-h,:] = R_s + P_s_s @ V[self.H-(h-1),:]
        
        for h in range(self.H):
            if h == self.H-1:
                self.Q[h,:,:] = self.R_sa
            else:
                self.Q[h,:,:] = self.R_sa +self.P_sa_s @ V[h+1,:]  
        return V, self.Q

    def policy_improvement(self,Q):
        Pi = np.zeros((self.H,self.nS, self.nA))
        for h in range(self.H):
            for s in range(self.nS):
                a = np.argmax(Q[h, s, :])
                Pi[h,s, a] = 1
        return Pi

    def run(self) -> np.ndarray:
        V_old = np.zeros((self.H,self.nS))
        Q_old = np.zeros((self.H,self.nS, self.nA))
        Pi = np.random.rand(self.H, self.nS, self.nA)
        Pi /= Pi.sum(axis=1, keepdims=True)
        error_plicy = []
        for _ in range(100):
            V_new, Q_new = self.back_policy_eval(Pi)
            Pi = self.policy_improvement(Q_new)

            #print(V)
            #assert np.all(V >= V_old)
            #assert np.all(Q >= Q_old)
            
            error_plicy.append(np.linalg.norm(V_new - V_old))
            V_old = V_new
            Q_old = Q_new
        
        return Q_old, error_plicy
    
    def q_reshape(self, discretizer):
        Q_reshape = np.zeros(
            np.concatenate([[self.H], discretizer.bucket_states, discretizer.bucket_actions])
        )
        for h in range(self.H):
            for nA, a in enumerate(product(*[ np.arange(0,da) for da in discretizer.bucket_actions])):
                Q_reshape [h,...,a] = np.transpose(self.Q[h,:,nA].reshape(*discretizer.bucket_states))
        return Q_reshape


class ClassicDP:
    
    def __init__(self,nS,nA,R,P,gamma = 0.9) -> None:

        self.R_sa = np.sum(np.multiply(R, P),axis=0)
        self.P_sa_s = np.transpose(P, (1, 2, 0))

        self.nS = nS
        self.nA = nA

        self.Q = np.zeros(
                    np.concatenate([[nS], [nA]])
                )
        self.gamma = gamma
    
    def policy_evaluation(self,Pi):
        P_s_s = np.einsum('ijk, ij -> ik', self.P_sa_s, Pi)
        R_s = np.einsum('ij, ij -> i', self.R_sa, Pi)
        
        V = np.zeros(self.nS)
        for _ in range(100):
            V = R_s + self.gamma * P_s_s @ V
            
        Q = self.R_sa + self.gamma * self.P_sa_s @ V
        return V, Q

    def policy_improvement(self,Q):
        Pi = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            a = np.argmax(Q[s, :])
            Pi[s, a] = 1
        return Pi

    def run(self):
        V_old = np.zeros(self.nS)
        Q_old = np.zeros((self.nS, self.nA))
        Pi = np.random.rand(self.nS, self.nA)
        Pi /= Pi.sum(axis=1, keepdims=True)
        error_plicy = []
        for _ in range(30):
            V, Q = self.policy_evaluation(Pi)
            Pi = self.policy_improvement(Q)

            #print(V)
            #assert np.all(V >= V_old)
            #assert np.all(Q >= Q_old)
            
            error_plicy.append(np.linalg.norm(V - V_old))
            V_old = V
            Q_old = Q
        self.Q = Q
        return Q
    
    def run_value_iter(self):
        V_old = np.zeros(self.nS)
        error_value = []
        gamma = 1
        for _ in range(60):
            V = np.max(self.R_sa + gamma * self.P_sa_s @ V_old, axis=1)
            assert np.all(V >= V_old)
            error_value.append(np.linalg.norm(V - V_old))
            V_old = V
        V_value = V
        self.Q = self.R_sa + gamma * self.P_sa_s @ V
        self.V = V 
        return self.Q
    