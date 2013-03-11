#MODEL_Halfreaction

#System definition

def system(States,t,Parameters):
    k1 = Parameters['k1']
    k1m = Parameters['k1m']
    k2 = Parameters['k2']
    k2m = Parameters['k2m']
    k3 = Parameters['k3']
    k3m = Parameters['k3m']
    k4 = Parameters['k4']
    k4m = Parameters['k4m']

    A = States[0]
    B = States[1]
    E = States[2]
    EP = States[3]
    Es = States[4]
    EsQ = States[5]
    P = States[6]
    Q = States[7]

    dA = - k1*E*A + k1m*Es*P
    dB = - k2*Es*B + k2m*E*Q
    dE = k1m*Es*P + k4*EP + k2*Es*B - k1*E*A - k4*E*P - k2m*E*Q
    dEP = k4*E*P - k4m*EP
    dEs = - k1m*Es*P + k3*EsQ - k2*Es*B + k1*E*A - k3*Es + k2m*E*Q
    dEsQ = k3*Es*Q - k3m*EsQ
    dP = k1*E*A - k1m*Es*P - k4*E*P + k4m*EP
    dQ = k2*E*B - k2m*E*Q - k3*Es*Q + k3m*EsQ
    return [dA, dB, dE, dEP, dEs, dEsQ, dP, dQ]

#Sensitivities

def sensitivities(States,Parameters):

    A = States[0]
    B = States[1]
    E = States[2]
    EP = States[3]
    Es = States[4]
    EsQ = States[5]
    P = States[6]
    Q = States[7]

    k1 = Parameters['k1']
    k1m = Parameters['k1m']
    k2 = Parameters['k2']
    k2m = Parameters['k2m']
    k3 = Parameters['k3']
    k3m = Parameters['k3m']
    k4 = Parameters['k4']
    k4m = Parameters['k4m']

    dAdk1 = -A*E*k1/(A + 1.0e-6)
    dAdk1m = Es*P*k1m/(A + 1.0e-6)
    dAdk2 = 0
    dAdk2m = 0
    dAdk3 = 0
    dAdk3m = 0
    dAdk4 = 0
    dAdk4m = 0
    dBdk1 = 0
    dBdk1m = 0
    dBdk2 = -B*Es*k2/(B + 1.0e-6)
    dBdk2m = E*Q*k2m/(B + 1.0e-6)
    dBdk3 = 0
    dBdk3m = 0
    dBdk4 = 0
    dBdk4m = 0
    dEdk1 = -A*E*k1/(E + 1.0e-6)
    dEdk1m = Es*P*k1m/(E + 1.0e-6)
    dEdk2 = B*Es*k2/(E + 1.0e-6)
    dEdk2m = -E*Q*k2m/(E + 1.0e-6)
    dEdk3 = 0
    dEdk3m = 0
    dEdk4 = k4*(-E*P + EP)/(E + 1.0e-6)
    dEdk4m = 0
    dEPdk1 = 0
    dEPdk1m = 0
    dEPdk2 = 0
    dEPdk2m = 0
    dEPdk3 = 0
    dEPdk3m = 0
    dEPdk4 = E*P*k4/(EP + 1.0e-6)
    dEPdk4m = -EP*k4m/(EP + 1.0e-6)
    dEsdk1 = A*E*k1/(Es + 1.0e-6)
    dEsdk1m = -Es*P*k1m/(Es + 1.0e-6)
    dEsdk2 = -B*Es*k2/(Es + 1.0e-6)
    dEsdk2m = E*Q*k2m/(Es + 1.0e-6)
    dEsdk3 = k3*(-Es + EsQ)/(Es + 1.0e-6)
    dEsdk3m = 0
    dEsdk4 = 0
    dEsdk4m = 0
    dEsQdk1 = 0
    dEsQdk1m = 0
    dEsQdk2 = 0
    dEsQdk2m = 0
    dEsQdk3 = Es*Q*k3/(EsQ + 1.0e-6)
    dEsQdk3m = -EsQ*k3m/(EsQ + 1.0e-6)
    dEsQdk4 = 0
    dEsQdk4m = 0
    dPdk1 = A*E*k1/(P + 1.0e-6)
    dPdk1m = -Es*P*k1m/(P + 1.0e-6)
    dPdk2 = 0
    dPdk2m = 0
    dPdk3 = 0
    dPdk3m = 0
    dPdk4 = -E*P*k4/(P + 1.0e-6)
    dPdk4m = EP*k4m/(P + 1.0e-6)
    dQdk1 = 0
    dQdk1m = 0
    dQdk2 = B*E*k2/(Q + 1.0e-6)
    dQdk2m = -E*Q*k2m/(Q + 1.0e-6)
    dQdk3 = -Es*Q*k3/(Q + 1.0e-6)
    dQdk3m = EsQ*k3m/(Q + 1.0e-6)
    dQdk4 = 0
    dQdk4m = 0
    return [dAdk1, dAdk1m, dAdk2, dAdk2m, dAdk3, dAdk3m, dAdk4, dAdk4m, dBdk1, dBdk1m, dBdk2, dBdk2m, dBdk3, dBdk3m, dBdk4, dBdk4m, dEdk1, dEdk1m, dEdk2, dEdk2m, dEdk3, dEdk3m, dEdk4, dEdk4m, dEPdk1, dEPdk1m, dEPdk2, dEPdk2m, dEPdk3, dEPdk3m, dEPdk4, dEPdk4m, dEsdk1, dEsdk1m, dEsdk2, dEsdk2m, dEsdk3, dEsdk3m, dEsdk4, dEsdk4m, dEsQdk1, dEsQdk1m, dEsQdk2, dEsQdk2m, dEsQdk3, dEsQdk3m, dEsQdk4, dEsQdk4m, dPdk1, dPdk1m, dPdk2, dPdk2m, dPdk3, dPdk3m, dPdk4, dPdk4m, dQdk1, dQdk1m, dQdk2, dQdk2m, dQdk3, dQdk3m, dQdk4, dQdk4m]