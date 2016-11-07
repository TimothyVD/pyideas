import numpy as np
from pyideas import (Model, DirectLocalSensitivity, NumericalLocalSensitivity)


def LSA_comparison():
    # Plain ping-pong bi-bi kinetics
    ping_pong = ('(Vf*Vr*(MBA*Pyr-Ace*Ala/Keq))/(Vr*Km*Pyr + Vr*Kp*MBA +'
                 'Vr*Pyr*MBA + Vf*Kac*Ala/Keq + Vf*Kal*Ace/Keq +'
                 'Vf*Ace*Ala/Keq + Vr*Km*Pyr*Ala/Kialp +'
                 'Vf*Kal*MBA*Ace/(Keq*Kimp))')

    # Replace reactants of Shin & Kim by current reactants
    ping_pong = ping_pong.replace('MBA', 'IPA')   # IPA  = isopropylamine
    ping_pong = ping_pong.replace('Pyr', 'BA')    # BA   = benzylacetone
    ping_pong = ping_pong.replace('Ace', 'ACE')   # ACE  = acetone
    ping_pong = ping_pong.replace('Ala', 'MPPA')  # MPPA =

    # Define system of interest
    system = {'v': ping_pong,
              'Keq': '((Vf/Vr)**2)*(Kac*Kal)/(Km*Kp)',
              'dE': '-decay*E',
              'dIPA': '-v*E',
              'dBA': '-v*E',
              'dACE': 'v*E',
              'dMPPA': 'v*E'}

    # Define model parameters
    parameters = {  # Forward reaction parameters calibrated with data ULUND
                  'Vf': 2.47e-2,        # umol/U/min
                  'Km': 143.18,         # mM
                  'Kp': 3.52,           # mM
                  # Backward reaction parameters calibrated with data UGENT
                  'Vr': 1.996e-2,       # umol/U/min
                  'Kac': 207.64,        # mM
                  'Kal': 2.301,         # mM
                  # Remaining parameters to calibrate (first guess)
                  'Kialp': 1.39889,     # mM
                  'Kimp': 2.9295,       # mM
                  'decay': 0.00302837}  # 1/min

    # Make model instance
    M1 = Model('PPBB', system, parameters)

    res_time = 0.01/1.66e-4/2.

    # Set independent (time) range
    M1.independent = {'t': np.linspace(0, res_time, 1000)}
    # Set initial conditions
    M1.initial_conditions = {'IPA': 65., 'BA': 5., 'ACE': 0.,
                             'MPPA': 0., 'E': 0.68}

    # Initialize model => Generate underlying functions
    M1.initialize_model()

    # DirectLocalSensitivity instance
    M1sens_dir = DirectLocalSensitivity(M1,
                                        parameters=['Kp', 'Vf', 'Km', 'Kal'])

    # NumericalLocalSensitivity instance
    M1sens_num = NumericalLocalSensitivity(M1,
                                           parameters=['Kp', 'Vf',
                                                       'Km', 'Kal'])
    M1sens_num.perturbation = 1e-5

    # Calculate direct sensitivity
    dir_sens = M1sens_dir.get_sensitivity(method='PRS')
    # Calculate numerical sensivity
    num_sens = M1sens_num.get_sensitivity(method='PRS')

    return dir_sens, num_sens

if __name__ == "__main__":
    dir_sens, num_sens = LSA_comparison()
