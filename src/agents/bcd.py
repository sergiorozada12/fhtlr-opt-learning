import numpy as np
from tensorly.tenalg import khatri_rao
import torch


def calculate_Ds(S, A, H, ENV, Pi):
    P_sa_s = np.transpose(ENV.P, (1, 2, 0)).reshape(ENV.nS*ENV.nA,ENV.nS)
    s = S.reshape(-1)
    I_s = torch.eye(S.shape[0])
    Ds_h = []
    for h in range(ENV.H):
        if h == ENV.H -1:
            Y =  np.kron(I_s,khatri_rao([H[h].detach().numpy().reshape(1,-1), A.detach().numpy()]))
            Ds_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(I_s,khatri_rao([H[h+1].detach().numpy().reshape(1,-1), A.detach().numpy()])))
            Y =  np.kron(I_s,khatri_rao([H[h].detach().numpy().reshape(1,-1), A.detach().numpy()]))
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
            Y =  np.kron(I_a,khatri_rao([H[h].detach().numpy().reshape(1,-1), S.detach().numpy()]))
            Da_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(P_pi_h,np.kron(I_a,khatri_rao([H[h+1].detach().numpy().reshape(1,-1), S.detach().numpy()])))
            Y =  np.kron(I_a,khatri_rao([H[h].detach().numpy().reshape(1,-1), S.detach().numpy()]))
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
            Y = np.matmul(khatri_rao([A.detach().numpy(), S.detach().numpy()]),M_h_h[h])
            Dh_h.append(-Y)
        else:
            P_pi_h = np.zeros((ENV.nS*ENV.nA, ENV.nS*ENV.nA))
            Pi_h = Pi[h,:]
            for nSnA in range(ENV.nS*ENV.nA):
                P_pi_h[nSnA,:] = (Pi_h * P_sa_s[nSnA,:].reshape(-1, 1)).reshape(-1)
            X = np.matmul(np.matmul(P_pi_h,khatri_rao([A.detach().numpy(), S.detach().numpy()])),M_h_h[h+1])
            Y = np.matmul(khatri_rao([A.detach().numpy(), S.detach().numpy()]),M_h_h[h])
            Dh_h.append(X-Y)

    Dh = np.vstack(Dh_h)
    return Dh
