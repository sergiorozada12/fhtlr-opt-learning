import numpy as np
from tensorly.tenalg import khatri_rao
import torch
import tensorly as tl


def calculate_Ds1(S1,S2, A, H, ENV, Pi):
    P_sa_s = np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.nS*ENV.nA,ENV.nS)
    I_s1 = torch.eye(S1.shape[0])
    Ds1_h = []
    for h in range(ENV.H):
        if h == ENV.H -1:
            Y =  np.kron(I_s1,khatri_rao([S2, H[h].reshape(1,-1), A]))
            Ds1_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(I_s1,khatri_rao([S2, H[h+1].reshape(1,-1), A])))
            Y =  np.kron(I_s1,khatri_rao([S2, H[h].reshape(1,-1), A]))
            Ds1_h.append(X-Y)

    Ds1 = np.vstack(Ds1_h)

    return Ds1

def calculate_Ds2(S1,S2, A, H, ENV, Pi):

    P_s2s1a_s2s1 = np.transpose((np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.W,ENV.W,ENV.nA,ENV.W,ENV.W)),(1,0,2,4,3)).reshape(ENV.nS*ENV.nA,ENV.nS)
    Pi_t = np.transpose(Pi.reshape(ENV.H,ENV.W,ENV.W,ENV.nA),(0,2,1,3)).reshape(ENV.H,ENV.nS,ENV.nA)
    I_s2 = torch.eye(S2.shape[0])
    Ds2_h = []
    for h in range(ENV.H):
        if h == ENV.H -1:
            Y =  np.kron(I_s2,khatri_rao([S1, H[h].reshape(1,-1), A]))
            Ds2_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi_t[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_s2s1a_s2s1[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(I_s2,khatri_rao([S1, H[h+1].reshape(1,-1), A])))
            Y =  np.kron(I_s2,khatri_rao([S1, H[h].reshape(1,-1), A]))
            Ds2_h.append(X-Y)

    Ds2 = np.vstack(Ds2_h)
    return Ds2

def calculate_Da(S1,S2, A, H, ENV, Pi):
    P_sa_s = np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.nS*ENV.nA,ENV.nS)
    a = A.reshape(-1)
    I_a = torch.eye(A.shape[0])
    Da_h = []
    for h in range(ENV.H):
        if h == ENV.H-1:
            Y =  np.kron(khatri_rao([H[h].reshape(1,-1), S1,S2]),I_a)
            Da_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(khatri_rao([H[h+1].reshape(1,-1), S1,S2]),I_a))
            Y =  np.kron(khatri_rao([H[h].reshape(1,-1), S1,S2]),I_a)
            Da_h.append(X-Y)

    Da = np.vstack(Da_h)

    return Da

def calculate_Dh(S1,S2, A, H, ENV, Pi,k):
    P_sa_s = np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.nS*ENV.nA,ENV.nS)
    h = H.reshape(-1)
    I_h = torch.eye(A.shape[0])
    Dh_h = []

    M_h_h = []
    for h in range(ENV.H):
        Mh = np.zeros((k, ENV.H*k))
        Mh[:, h*k:(h+1)*k] = np.eye(k) 
        M_h_h.append(Mh)

    for h in range(ENV.H):
        if h == ENV.H-1:
            Y = np.matmul(khatri_rao([S1,S2, A]),M_h_h[h])
            Dh_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(np.matmul(P_pi_h,khatri_rao([S1,S2, A])),M_h_h[h+1])
            Y = np.matmul(khatri_rao([S1,S2, A]),M_h_h[h])
            Dh_h.append(X-Y)

    Dh = np.vstack(Dh_h)
    return Dh

def normalize_cp_factors2(factors):
    """
    Normaliza los factores de una descomposición CP sin cambiar la norma total del tensor reconstruido.
    
    Parámetros:
        weights: array de tamaño (R,), contiene los valores escalares (lambda_r).
        factors: lista de matrices [A, B, C], cada una de tamaño (I, R), (J, R), (K, R).
        
    Retorna:
        weights_normalized: Pesos ajustados.
        factors_normalized: Factores normalizados.
    """
    # Copia de los factores para evitar modificar los originales
    factors_normalized = [np.copy(factor) for factor in factors]

   
    norms = [np.linalg.norm(factor) for factor in factors]
    geometric_mean = np.prod(norms)**(1/len(norms)) # Media geométrica de las normas

    # Ajustar los factores para que su norma sea la misma
    for i, factor in enumerate(factors_normalized):
        factor /= norms[i]  # Normalizar cada factor
        factor *= geometric_mean  # Escalar por la media geométrica

    # Ajustar el peso asociado al componente
    weights = np.prod(norms) / geometric_mean**len(factors)

    return weights, factors_normalized

def run_test_episode(env, Q, H):
    G = 0
    s, _ = env.reset()
    for h in range(H):
        s1, s2 = s
        dist = Q.forward(np.array([h, s2, s1]))
        a = [torch.argmax(dist)]
        s, r, d, _, _ = env.step(a)
        G += r

        if d:
            break
    return G

class bcd:

    def __init__(self,Q,Pi,discretizer,env,k,Q_opt) -> None:

        self.Q = Q
        self.Pi = Pi
        self.discretizer = discretizer
        self.ENV = env
        self.k = k
        self.Q_opt = Q_opt

    def run(self,num_iter):
        r = np.sum(np.multiply(self.ENV.R, self.ENV.P),axis=0).reshape(-1)
        r_t = np.transpose(np.sum(np.multiply(self.ENV.R, self.ENV.P),axis=0).reshape(self.ENV.W,self.ENV.W,-1),(1,0,2)).reshape(-1)
        r_gorro = np.tile(r, (self.ENV.H, 1)).reshape(-1)
        r_gorro_t = np.tile(r_t, (self.ENV.H, 1)).reshape(-1)

        fo_values = []
        convs = []
        errors = []
        
        for i in range(num_iter):
            H = self.Q.factors[0].detach().numpy()
            A = self.Q.factors[1+len(self.discretizer.bucket_states):][0].detach().numpy()
            S1 = self.Q.factors[1:len(self.Q.factors)-len(self.discretizer.bucket_actions)][0].detach().numpy()
            S2 = self.Q.factors[1:len(self.Q.factors)-len(self.discretizer.bucket_actions)][1].detach().numpy()
            
            
            S1_i = np.matmul(np.linalg.pinv(calculate_Ds1(S1,S2, A, H, self.ENV, self.Pi)), -r_gorro).reshape(self.ENV.W,self.k)
            
            S2_i = np.matmul(np.linalg.pinv(calculate_Ds2(S1_i,S2, A, H, self.ENV, self.Pi)), -r_gorro_t).reshape(self.ENV.W,self.k)
            
            A_i = np.matmul(np.linalg.pinv(calculate_Da(S1_i,S2_i, A, H, self.ENV, self.Pi)), -r_gorro).reshape(self.k,self.ENV.nA).T
            
            H_i = np.matmul(np.linalg.pinv(calculate_Dh(S1_i,S2, A_i, H, self.ENV, self.Pi, self.k)), -r_gorro).reshape(self.ENV.H,self.k)

            _, [S1_i,S2_i,A_i,H_i] = normalize_cp_factors2([S1_i,S2_i,A_i,H_i])

            fo_values.append(np.linalg.norm(np.tile(r, (self.ENV.H, 1)).reshape(-1) + np.matmul(calculate_Da(S1,S2, A, H, self.ENV, self.Pi),(A.T).reshape(-1)),ord = 2)**2)
            convs.append(np.linalg.norm(S1-S1_i, ord = "fro")+np.linalg.norm(S2-S2, ord = "fro") + np.linalg.norm(A-A_i, ord = "fro") + np.linalg.norm(H-H_i, ord = "fro"))
            errors.append(np.linalg.norm( self.Q_opt - tl.cp_to_tensor(([1]*self.k,[factor.detach().numpy() for factor in self.Q.factors]))))
            
            if fo_values[-1] < 10e-9:
                break
            
            new_values = [torch.tensor(H_i),torch.tensor(S1_i),torch.tensor(S2_i),torch.tensor(A_i)]
            with torch.no_grad():  # Desactivamos el cálculo de gradientes para la actualización directa
                for param, new_value in zip(self.Q.factors, new_values):
                    param.copy_(new_value)  # Copiar los nuevos valores al parámetro existente

        return fo_values,errors,convs,self.Q
    
    def bcd_policy_improvement(self,num_iter_policy,num_iter_bcd):

        fo_list = []
        errors_list = []
        conv_list = []
        return_mean = []
        return_std = []

        for i in range(num_iter_policy):
            Q_tensor = tl.cp_to_tensor(([1]*self.k,[factor.detach().numpy() for factor in self.Q.factors]))

            Pi = np.zeros((self.ENV.H, self.ENV.W, self.ENV.W, self.ENV.nA))
            for h in range(self.ENV.H):
                for s1 in range(self.ENV.W):
                    for s2 in range(self.ENV.W):
                        #a = np.argmax(Q.forward(np.array([h, s])).detach().numpy())
                        a = np.argmax(Q_tensor[h,s1,s2,:])
                        Pi[h,s1,s2, a] = 1

            self.Pi = Pi.reshape(self.ENV.H, self.ENV.nS, self.ENV.nA)

            fo_values,errors,convs, self.Q = self.run(num_iter_bcd)

            list = [run_test_episode(self.ENV,self.Q,self.ENV.H) for i in range(1000)]
            return_mean.append(np.mean(list))
            return_std.append(np.std(list))
            fo_list.extend(fo_values)
            errors_list.extend(errors)
            conv_list.extend(convs)
        
            if errors_list[-1] < 10^-3:
                break
        
        return fo_list,errors_list,conv_list,return_mean,return_std, self.Q
    

class bcgd:

    def __init__(self,Q,Pi,discretizer,env,k,Q_opt,alpha, normalize = 0,decay =0.99999) -> None:

        self.Q = Q
        self.Pi = Pi
        self.discretizer = discretizer
        self.ENV = env
        self.k = k
        self.Q_opt = Q_opt
        self.alpha = alpha
        self.normalize = normalize
        self.decay = decay

    def run(self,bcgd):
        r = np.sum(np.multiply(self.ENV.R, self.ENV.P),axis=0).reshape(-1)
        r_t = np.transpose(np.sum(np.multiply(self.ENV.R, self.ENV.P),axis=0).reshape(self.ENV.W,self.ENV.W,-1),(1,0,2)).reshape(-1)
        r_gorro = np.tile(r, (self.ENV.H, 1)).reshape(-1)
        r_gorro_t = np.tile(r_t, (self.ENV.H, 1)).reshape(-1)

        fo_values = []
        convs = []
        errors = []

        for i in range(bcgd):
            H = self.Q.factors[0].detach().numpy()
            A = self.Q.factors[1+len(self.discretizer.bucket_states):][0].detach().numpy()
            S1 = self.Q.factors[1:len(self.Q.factors)-len(self.discretizer.bucket_actions)][0].detach().numpy()
            S2 = self.Q.factors[1:len(self.Q.factors)-len(self.discretizer.bucket_actions)][1].detach().numpy()
            
            #Update S1
            Ds1_n = calculate_Ds1(S1,S2, A, H, self.ENV, self.Pi)
            S1_grad = np.matmul(np.transpose(Ds1_n),(np.matmul(Ds1_n, S1.reshape(-1)) + r_gorro)).reshape(self.ENV.W,self.k)
            S1_i = (S1 - self.alpha *S1_grad)

            #Update S2
            Ds2_n = calculate_Ds2(S1_i,S2, A, H, self.ENV, self.Pi)
            S2_grad = np.matmul(np.transpose(Ds2_n),(np.matmul(Ds2_n, S2.reshape(-1)) + r_gorro_t)).reshape(self.ENV.W,self.k)
            S2_i = (S2 - self.alpha *S2_grad)
            #Update A
            Da_n = calculate_Da(S1_i,S2_i, A, H, self.ENV, self.Pi)
            A_grad = np.matmul(np.transpose(Da_n),(np.matmul(Da_n, (A.T).reshape(-1)) + r_gorro)).reshape(self.k,self.ENV.nA).T
            A_i = (A - self.alpha *A_grad)

            #Update H
            Dh_n = calculate_Dh(S1_i,S2_i, A_i, H, self.ENV, self.Pi, self.k)
            H_grad = np.matmul(np.transpose(Dh_n),(np.matmul(Dh_n, H.reshape(-1)) + r_gorro)).reshape(self.ENV.H,self.k)
            H_i = (H - self.alpha *H_grad)
            
            _, [S1_i,S2_i,A_i,H_i] = normalize_cp_factors2([S1_i,S2_i,A_i,H_i])

            fo_values.append(np.linalg.norm(np.tile(r, (self.ENV.H, 1)).reshape(-1) + np.matmul(calculate_Da(S1,S2, A, H, self.ENV, self.Pi),(A.T).reshape(-1)),ord=2)**2)
            convs.append(np.linalg.norm(S1-S1_i) + np.linalg.norm(S2-S2_i) + np.linalg.norm(A-A_i) + np.linalg.norm(H-H_i))
            errors.append(np.linalg.norm( self.Q_opt - tl.cp_to_tensor(([1]*self.k,[factor.detach().numpy() for factor in self.Q.factors]))))
            self.alpha = self.alpha*self.decay

            if errors[-1] < 10^-6:
                break
            
            new_values = [torch.tensor(H),torch.tensor(S1_i),torch.tensor(S2_i),torch.tensor(A)]
            with torch.no_grad():  # Desactivamos el cálculo de gradientes para la actualización directa
                for param, new_value in zip(self.Q.factors, new_values):
                    param.copy_(new_value)  # Copiar los nuevos valores al parámetro existente
        
        return fo_values,errors,convs,self.Q
    
    def bcgd_policy_improvement(self,num_iter_policy,num_iter_bcgd):

        fo_list = []
        errors_list = []
        conv_list = []
        return_mean = []
        return_std = []

        for i in range(num_iter_policy):
            Q_tensor = tl.cp_to_tensor(([1]*self.k,[factor.detach().numpy() for factor in self.Q.factors]))

            Pi = np.zeros((self.ENV.H, self.ENV.W, self.ENV.W, self.ENV.nA))
            for h in range(self.ENV.H):
                for s1 in range(self.ENV.W):
                    for s2 in range(self.ENV.W):
                        #a = np.argmax(Q.forward(np.array([h, s])).detach().numpy())
                        a = np.argmax(Q_tensor[h,s1,s2,:])
                        Pi[h,s1,s2, a] = 1

            self.Pi = Pi.reshape(self.ENV.H, self.ENV.nS, self.ENV.nA)

            list = [run_test_episode(self.ENV,self.Q,self.ENV.H) for i in range(1000)]
            return_mean.append(np.mean(list))
            return_std.append(np.std(list))

            fo_values,errors,convs, self.Q = self.run(num_iter_bcgd)
            fo_list.extend(fo_values)
            errors_list.extend(errors)
            conv_list.extend(convs)
        
            if errors_list[-1] < 10^-3:
                break
        
        return fo_list,errors_list,conv_list,return_mean,return_std, self.Q