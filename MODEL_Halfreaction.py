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

    EP = States[0]
    En = States[1]
    Es = States[2]
    EsQ = States[3]
    PP = States[4]
    PQ = States[5]
    SA = States[6]
    SB = States[7]

    dEP = k4*En*PP - k4m*EP
    dEn = k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ
    dEs = - k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ
    dEsQ = k3*Es*PQ - k3m*EsQ
    dPP = k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP
    dPQ = k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ
    dSA = - k1*En*SA + k1m*Es*PP
    dSB = - k2*Es*SB + k2m*En*PQ
    return [dEP, dEn, dEs, dEsQ, dPP, dPQ, dSA, dSB]

#Sensitivities

def sensitivities(States,Parameters):

    EP = States[0]
    En = States[1]
    Es = States[2]
    EsQ = States[3]
    PP = States[4]
    PQ = States[5]
    SA = States[6]
    SB = States[7]

    k1 = Parameters['k1']
    k1m = Parameters['k1m']
    k2 = Parameters['k2']
    k2m = Parameters['k2m']
    k3 = Parameters['k3']
    k3m = Parameters['k3m']
    k4 = Parameters['k4']
    k4m = Parameters['k4m']

    dEPdk1 = 