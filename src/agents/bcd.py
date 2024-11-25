import numpy as np
from tensorly.tenalg import khatri_rao
import torch
import tensorly as tl


def calculate_Ds(S, A, H, ENV, Pi):
    P_sa_s = np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.nS*ENV.nA,ENV.nS)
    s = S.reshape(-1)
    I_s = torch.eye(S.shape[0])
    Ds_h = []
    for h in range(ENV.H):
        if h == ENV.H -1:
            Y =  np.kron(I_s,khatri_rao([H[h].reshape(1,-1), A]))
            Ds_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(I_s,khatri_rao([H[h+1].reshape(1,-1), A])))
            Y =  np.kron(I_s,khatri_rao([H[h].reshape(1,-1), A]))
            Ds_h.append(X-Y)

    Ds = np.vstack(Ds_h)

    return Ds

def calculate_Da(S, A, H, ENV, Pi):
    P_sa_s = np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.nS*ENV.nA,ENV.nS)
    a = A.reshape(-1)
    I_a = torch.eye(A.shape[0])
    Da_h = []
    for h in range(ENV.H):
        if h == ENV.H-1:
            Y =  np.kron(khatri_rao([H[h].reshape(1,-1), S]),I_a)
            Da_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(khatri_rao([H[h+1].reshape(1,-1), S]),I_a))
            Y =  np.kron(khatri_rao([H[h].reshape(1,-1), S]),I_a)
            Da_h.append(X-Y)

    Da = np.vstack(Da_h)

    return Da

def calculate_Dh(S, A, H, ENV, Pi,k):
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
            Y = np.matmul(khatri_rao([S, A]),M_h_h[h])
            Dh_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h+1,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(np.matmul(P_pi_h,khatri_rao([S, A])),M_h_h[h+1])
            Y = np.matmul(khatri_rao([S, A]),M_h_h[h])
            Dh_h.append(X-Y)

    Dh = np.vstack(Dh_h)
    return Dh

class bcd:

    def __init__(self,Q,Pi,discretizer,env,num_iter,k,Q_opt) -> None:

        self.Q = Q
        self.Pi = Pi
        self.discretizer = discretizer
        self.ENV = env
        self.num_iter = num_iter
        self.k = k
        self.Q_opt = Q_opt

    def run(self):
        r = np.sum(np.multiply(self.ENV.R, self.ENV.P),axis=0).reshape(-1)
        r_gorro = np.tile(r, (self.ENV.H, 1)).reshape(-1)

        fo_values = []
        convs = []
        errors = []

        for i in range(self.num_iter):
            H = self.Q.factors[0].detach().numpy()
            A = self.Q.factors[1+len(self.discretizer.bucket_states):][0].detach().numpy()
            S = self.Q.factors[1:len(self.Q.factors)-len(self.discretizer.bucket_actions)][0].detach().numpy()
            
            fo_values.append(np.linalg.norm(np.tile(r, (self.ENV.H, 1)).reshape(-1) + np.matmul(calculate_Da(S, A, H, self.ENV, self.Pi),(A.T).reshape(-1)),ord=2)**2)
            
            S_i = np.matmul(np.linalg.pinv(calculate_Ds(S, A, H, self.ENV, self.Pi)), -r_gorro).reshape(self.ENV.nS,self.k)
            A_i = np.matmul(np.linalg.pinv(calculate_Da(S_i, A, H, self.ENV, self.Pi)), -r_gorro).reshape(self.k,self.ENV.nA).T
            H_i = np.matmul(np.linalg.pinv(calculate_Dh(S_i, A_i, H, self.ENV, self.Pi, self.k)), -r_gorro).reshape(self.ENV.H,self.k)
            
            convs.append(np.linalg.norm(S-S_i) + np.linalg.norm(A-A_i) + np.linalg.norm(H-H_i))
            errors.append(np.linalg.norm( self.Q_opt - tl.cp_to_tensor(([1]*self.k,[factor.detach().numpy() for factor in self.Q.factors]))))

            if errors[-1] < 0.1:
                break
            
            new_values = [torch.tensor(H_i),torch.tensor(S_i),torch.tensor(A_i)]
            with torch.no_grad():  # Desactivamos el cálculo de gradientes para la actualización directa
                for param, new_value in zip(self.Q.factors, new_values):
                    param.copy_(new_value)  # Copiar los nuevos valores al parámetro existente

        return fo_values,errors,convs,self.Q
    

def normalize_cp_factors1(weights, factors):
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
    R = weights.shape[0]  # Número de componentes

    # Normalizar cada componente r
    for r in range(R):
        norms = [np.linalg.norm(factor[:, r]) for factor in factors]  # Normas de A[:, r], B[:, r], C[:, r]
        geometric_mean = np.cbrt(np.prod(norms))  # Media geométrica de las normas

        # Ajustar los factores para que su norma sea la misma
        for i, factor in enumerate(factors_normalized):
            factor[:, r] /= norms[i]  # Normalizar cada factor
            factor[:, r] *= geometric_mean  # Escalar por la media geométrica

        # Ajustar el peso asociado al componente
        weights[r] *= np.prod(norms) / geometric_mean**len(factors)

    return weights, factors_normalized

def normalize_cp_factors2(weights, factors):
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

   
    norms = [np.linalg.norm(factor) for factor in factors]  # Normas de A[:, r], B[:, r], C[:, r]
    geometric_mean = np.cbrt(np.prod(norms))  # Media geométrica de las normas

    # Ajustar los factores para que su norma sea la misma
    for i, factor in enumerate(factors_normalized):
        factor /= norms[i]  # Normalizar cada factor
        factor *= geometric_mean  # Escalar por la media geométrica

    # Ajustar el peso asociado al componente
    weights *= np.prod(norms) / geometric_mean**len(factors)

    return weights, factors_normalized

class bcgd:

    def __init__(self,Q,Pi,discretizer,env,num_iter,k,Q_opt,alpha, normalize = 0) -> None:

        self.Q = Q
        self.Pi = Pi
        self.discretizer = discretizer
        self.ENV = env
        self.num_iter = num_iter
        self.k = k
        self.Q_opt = Q_opt
        self.alpha = alpha
        self.normalize = normalize

    def run(self):
        r = np.sum(np.multiply(self.ENV.R, self.ENV.P),axis=0).reshape(-1)
        r_gorro = np.tile(r, (self.ENV.H, 1)).reshape(-1)

        fo_values = []
        convs = []
        errors = []

        for i in range(self.num_iter):
            H = self.Q.factors[0].detach().numpy()
            A = self.Q.factors[1+len(self.discretizer.bucket_states):][0].detach().numpy()
            S = self.Q.factors[1:len(self.Q.factors)-len(self.discretizer.bucket_actions)][0].detach().numpy()
            fo_values.append(np.linalg.norm(np.tile(r, (self.ENV.H, 1)).reshape(-1) + np.matmul(calculate_Ds(S, A, H, self.ENV, self.Pi),(S).reshape(-1)),ord=2)**2)
            
            #Update S
            Ds_n = calculate_Ds(S, A, H, self.ENV, self.Pi)
            S_grad = np.matmul(np.transpose(Ds_n),(np.matmul(Ds_n, S.reshape(-1)) + r_gorro)).reshape(self.ENV.nS,self.k)
            S_i = (S - self.alpha *S_grad)
            
            #Update A
            Da_n = calculate_Da(S_i, A, H, self.ENV, self.Pi)
            A_grad = np.matmul(np.transpose(Da_n),(np.matmul(Da_n, (A.T).reshape(-1)) + r_gorro)).reshape(self.k,self.ENV.nA).T
            A_i = (A - self.alpha *A_grad)
            
            #Update H
            Dh_n = calculate_Dh(S_i, A_i, H, self.ENV, self.Pi, self.k)
            H_grad = np.matmul(np.transpose(Dh_n),(np.matmul(Dh_n, H.reshape(-1)) + r_gorro)).reshape(self.ENV.H,self.k)
            H_i = (H - self.alpha *H_grad)
            
            if self.normalize == 1:
                _, [H_i,S_i,A_i] = normalize_cp_factors1(np.array([np.linalg.norm(H_i),np.linalg.norm(S_i),np.linalg.norm(A_i)]), [H_i,S_i,A_i])
            elif self.normalize == 1:
                 _, [H_i,S_i,A_i] = normalize_cp_factors2(np.array([np.linalg.norm(H_i),np.linalg.norm(S_i),np.linalg.norm(A_i)]), [H_i,S_i,A_i])

            convs.append(np.linalg.norm(S-S_i) + np.linalg.norm(A-A_i) + np.linalg.norm(H-H_i))
            errors.append(np.linalg.norm( self.Q_opt - tl.cp_to_tensor(([1]*self.k,[factor.detach().numpy() for factor in self.Q.factors]))))

            if errors[-1] < 10^-6:
                break
            
            new_values = [torch.tensor(H),torch.tensor(S_i),torch.tensor(A)]
            with torch.no_grad():  # Desactivamos el cálculo de gradientes para la actualización directa
                for param, new_value in zip(self.Q.factors, new_values):
                    param.copy_(new_value)  # Copiar los nuevos valores al parámetro existente
        
        return fo_values,errors,convs,self.Q