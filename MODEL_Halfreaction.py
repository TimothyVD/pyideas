#MODEL_Halfreaction
import numpy as np

def system(ODES,t,Parameters):
    k1 = Parameters['k1']
    k1m = Parameters['k1m']
    k2 = Parameters['k2']
    k2m = Parameters['k2m']
    k3 = Parameters['k3']
    k3m = Parameters['k3m']
    k4 = Parameters['k4']
    k4m = Parameters['k4m']

    EP = ODES[0]
    En = ODES[1]
    Es = ODES[2]
    EsQ = ODES[3]
    PP = ODES[4]
    PQ = ODES[5]
    SA = ODES[6]
    SB = ODES[7]

    dEP = k4*En*PP - k4m*EP
    dEn = k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ
    dEs = - k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ
    dEsQ = k3*Es*PQ - k3m*EsQ
    dPP = k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP
    dPQ = k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ
    dSA = - k1*En*SA + k1m*Es*PP
    dSB = - k2*Es*SB + k2m*En*PQ
    return [dEP, dEn, dEs, dEsQ, dPP, dPQ, dSA, dSB]


def system_with_sens(ODES,t,Parameters):
    k1 = Parameters['k1']
    k1m = Parameters['k1m']
    k2 = Parameters['k2']
    k2m = Parameters['k2m']
    k3 = Parameters['k3']
    k3m = Parameters['k3m']
    k4 = Parameters['k4']
    k4m = Parameters['k4m']

    EP = ODES[0]
    En = ODES[1]
    Es = ODES[2]
    EsQ = ODES[3]
    PP = ODES[4]
    PQ = ODES[5]
    SA = ODES[6]
    SB = ODES[7]

    dEP = k4*En*PP - k4m*EP
    dEn = k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ
    dEs = - k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ
    dEsQ = k3*Es*PQ - k3m*EsQ
    dPP = k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP
    dPQ = k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ
    dSA = - k1*En*SA + k1m*Es*PP
    dSB = - k2*Es*SB + k2m*En*PQ

    #Sensitivities

    state_len = len(ODES)/(len(Parameters)+1)
    dxdtheta = np.array(ODES[state_len:].reshape(state_len,len(Parameters)))

    dfdtheta = np.array([[0, 0, 0, 0, 0, 0, En*PP, -EP],
       [-En*SA, Es*PP, Es*SB, -En*PQ, 0, 0, EP - En*PP, 0],
       [En*SA, -Es*PP, -Es*SB, En*PQ, -Es + EsQ, 0, 0, 0],
       [0, 0, 0, 0, Es*PQ, -EsQ, 0, 0],
       [En*SA, -Es*PP, 0, 0, 0, 0, -En*PP, EP],
       [0, 0, En*SB, -En*PQ, -Es*PQ, EsQ, 0, 0],
       [-En*SA, Es*PP, 0, 0, 0, 0, 0, 0],
       [0, 0, -Es*SB, En*PQ, 0, 0, 0, 0]], dtype=object)

    dfdx = np.array([[-k4m, PP*k4, 0, 0, En*k4, 0, 0, 0],
       [k4, -PP*k4 - PQ*k2m - SA*k1, PP*k1m + SB*k2, 0, -En*k4 + Es*k1m,
        -En*k2m, -En*k1, Es*k2],
       [0, PQ*k2m + SA*k1, -PP*k1m - SB*k2 - k3, k3, -Es*k1m, En*k2m,
        En*k1, -Es*k2],
       [0, 0, PQ*k3, -k3m, 0, Es*k3, 0, 0],
       [k4m, -PP*k4 + SA*k1, -PP*k1m, 0, -En*k4 - Es*k1m, 0, En*k1, 0],
       [0, -PQ*k2m + SB*k2, -PQ*k3, k3m, 0, -En*k2m - Es*k3, 0, En*k2],
       [0, -SA*k1, PP*k1m, 0, Es*k1m, 0, -En*k1, 0],
       [0, PQ*k2m, -SB*k2, 0, 0, En*k2m, 0, -Es*k2]], dtype=object)

    dxdtheta = dfdtheta + np.dot(dfdx,dxdtheta)
    return [dEP, dEn, dEs, dEsQ, dPP, dPQ, dSA, dSB]+ list(dxdtheta.reshape(-1,))