from __future__ import division

# Some interesting python packages
import sys
import os

# import numpy  #matlab in python
import sympy
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #plotten in python
from matplotlib import colors
from biointense import *   #de package laden
from collections import OrderedDict

#Read output files of the respiro-experiment
mycols = ["Time","Untitled", "pH(V)","DO(V)","NOx(V)","pH","pH Filt","DO","DO Filt","NOx","NOx Filt","OUR","OUR Filt","Comment"]
file_path = os.path.join(os.getcwd(), 'data','respirometry.txt')
respirodata = pd.read_table(file_path, decimal=',', skiprows=24, names=mycols ) 

#The untitled column is not of interest; last argument (the 1) is to drop columns and not rows!
respirodata = respirodata.drop(["Untitled","pH(V)","DO(V)", "NOx(V)","pH","NOx","NOx Filt","OUR","OUR Filt"],1)

#Adjust data in case of bad calibration
LowReach = 0.28
respirodata["DO"] = respirodata["DO"]-LowReach

#plot data
respirodata["DO"].plot(figsize=(12,4))

#Find where first spike was added
Add=0

#Find array values of beginning and end
spike1 = np.where(pd.notnull(respirodata["Comment"])==True)[0][Add]
NumSample = 6
NumSpikes = 6
EndSpike6 = np.where(pd.notnull(respirodata["Comment"])==True)[0][NumSpikes*(NumSample+1)-3]
                                                           


# In[17]:

#Select useful part of DO-profile and set relative times
respirodataTimeAdapt = respirodata.ix[spike1:EndSpike6]
respirodataTimeAdapt["NewTime"] = (respirodataTimeAdapt.index.values-respirodataTimeAdapt.index.values[0])/60/60/24
respirodataWithTime = respirodataTimeAdapt
respirodataTimeAdapt = respirodataTimeAdapt.set_index("NewTime")
respirodataTimeAdapt["DO"].plot(figsize=(14,6))


##### Calculation of average DO at setpoints

# In[18]:

#Find new array values of spikes;
#The next 6 or 7 Comments are sampling points
Start = 0
NumSample = 6

#first sp
StartSpike1 = Add
spike1 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike1]
print spike1
EndSpike1 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike1+NumSample-1]
print EndSpike1

#second sp
StartSpike2 = StartSpike1+NumSample#+1
spike2 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike2]
print spike2
EndSpike2 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike2+NumSample]
print EndSpike2

#third sp
StartSpike3 = StartSpike2+NumSample+1
spike3 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike3]
print spike3
EndSpike3 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike3+NumSample]
print EndSpike3

#fourth sp
StartSpike4 = StartSpike3+NumSample+1
spike4 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike4]
print spike4
EndSpike4 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike4+NumSample]
print EndSpike4

#fifth sp
StartSpike5 = StartSpike4+NumSample+1
spike5 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike5]
print spike5
EndSpike5 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike5+NumSample]
print EndSpike5

#sixth sp
StartSpike6 = StartSpike5+NumSample+1
spike6 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike6]
print spike6
EndSpike6 = np.where(pd.notnull(respirodataTimeAdapt["Comment"])==True)[0][StartSpike6+NumSample-2]
print EndSpike6

respirodataTimeAdapt.iloc[int(EndSpike6)]

#Select parts of DO-profile
respiroprofile1 = respirodataTimeAdapt.iloc[int(spike1):int(EndSpike1)]
respiroprofile2 = respirodataTimeAdapt.iloc[int(spike2):int(EndSpike2)]
respiroprofile3 = respirodataTimeAdapt.iloc[int(spike3):int(EndSpike3)]
respiroprofile4 = respirodataTimeAdapt.iloc[int(spike4):int(EndSpike4)]
respiroprofile5 = respirodataTimeAdapt.iloc[int(spike5):int(EndSpike5)]
respiroprofile6 = respirodataTimeAdapt.iloc[int(spike6):int(EndSpike6)]

#check plots if necessary
respiroprofile1["DO"].plot()
respiroprofile2["DO"].plot()
respiroprofile3["DO"].plot()
respiroprofile4["DO"].plot()
respiroprofile5["DO"].plot()
respiroprofile6["DO"].plot()
#plt.show()

#calculate average DO's
Mean1=np.mean(respiroprofile1["DO"])
Std1=np.std(respiroprofile1["DO"])
print Mean1
print Std1
Mean2=np.mean(respiroprofile2["DO"])
Std2=np.std(respiroprofile2["DO"])
print Mean2
print Std2
Mean3=np.mean(respiroprofile3["DO"])
Std3=np.std(respiroprofile3["DO"])
print Mean3
print Std3
Mean4=np.mean(respiroprofile4["DO"])
Std4=np.std(respiroprofile4["DO"])
print Mean4
print Std4
Mean5=np.mean(respiroprofile5["DO"])
Std5=np.std(respiroprofile5["DO"])
print Mean5
print Std5
Mean6=np.mean(respiroprofile6["DO"])
Std6=np.std(respiroprofile6["DO"])
print Mean6
print Std6

DurationSpike = 3.5*1e-5 #3 seconds expressed in days
SpikeTime1 = respirodataWithTime["NewTime"].iloc[int(spike1)]
SpikeTime1End = SpikeTime1 + DurationSpike
SpikeTime2 = respirodataWithTime["NewTime"].iloc[int(spike2)]
SpikeTime2End = SpikeTime2 + DurationSpike
SpikeTime3 = respirodataWithTime["NewTime"].iloc[int(spike3)]
SpikeTime3End = SpikeTime3 + DurationSpike
SpikeTime4 = respirodataWithTime["NewTime"].iloc[int(spike4)]
SpikeTime4End = SpikeTime4 + DurationSpike
SpikeTime5 = respirodataWithTime["NewTime"].iloc[int(spike5)]
SpikeTime5End = SpikeTime5 + DurationSpike
SpikeTime6 = respirodataWithTime["NewTime"].iloc[int(spike6)]
SpikeTime6End = SpikeTime6 + DurationSpike
End = respirodataWithTime["NewTime"].iloc[int(EndSpike6)]
print SpikeTime1
print SpikeTime1End
print SpikeTime2
print SpikeTime2End
print SpikeTime3
print SpikeTime4
print SpikeTime5
print SpikeTime6

#Read own datafile with concentrations of NH3, NO2 and NO3
mycols2 = ["Time2","DO2","NH3","NO2","NO3"]
file_path2 = os.path.join(os.getcwd(), 'data','respirometry2.txt')
Ndata = pd.read_table(file_path2, decimal=',',names=mycols2)

#We set the time column as current index
Ndata = Ndata.set_index("Time2")

#Plot data
plt.figure()
Ndata["NH3"].plot(style='ro')
Ndata["NO2"].plot(style='bo')
plt.show()

Ndata["NO3"].plot(style='co')
plt.show()

#Een naam voor het model waarmee je werkt
Modelname = 'KoModel'

#specific conditions of test and the parameters dependent on them
#from Henze 2008, accepted in ASM models
Temp=20
theta_s_AOB=1.123
DOsp=4

# KsAOB=0.12 #Manser2005
# KsNOB=0.3 #Manser2005
# --> as numbers in model! Are assumed and don't have to be estimated

#De parameters van het model. Als geen parameter, maar constante, dan gewoon in system zetten met getalwaarde
#muMaxXAOB en muMaxXNOB in mgN/L/sec, want data staat ook in sec!!
Parameters = {'RmaxXAOB':400,'RmaxXNOB':400,
          'KoAOB':0.9,'KoNOB':0.3}
Algebraic = {'DO':'DO','NH3spike':'NH3spike','NO2spike':'NO2spike','NH3':'NH3', 'NO2':'NO2', 'NO3':'NO3'}

#De systeem-vergelijkingen:
System = {'dNH3':'-RmaxXAOB*(DO/(DO+KoAOB))*(NH3/(0.12+NH3))+NH3spike',
          'dNO2':'RmaxXAOB*(DO/(DO+KoAOB))*(NH3/(0.12+NH3))-RmaxXNOB*(DO/(DO+KoNOB))*(NO2/(0.3+NO2))+NO2spike',
          'dNO3':'RmaxXNOB*(DO/(DO+KoNOB))*(NO2/(0.3+NO2))'
          }
#units:
#   RmaxXAOB and RmaxXNOB: [mgN/L/d]
#   Rmax = muMax/Y [mgN/gVSS/d]
#            muMax [d^-1]
#            Y_AOB = 3.42 gVSS/mgNH4-N
#            Y_NOB = 1.14 gVSS/mgNO2-N
#   X [gVSS/L]
#   KoAOB and KoNOB: [mgO2/L]
#   NH3 and NO2 and NO3: [mgN/L]
#   dNH3 and dNO2 and dNO3: [mgN/L/d]

#'Ntot':'NH3+NO2+NO3',

#We maken nu het model aan en zetten het 
M1 = DAErunner(ODE=System, Parameters=Parameters, Algebraic=Algebraic,
               Modelname=Modelname, external_par=['DO','NH3spike','NO2spike'],
               print_on=False)


#### Het model initialiseren met de juiste tijd en randvwn

# In[33]:

#Enkele zaken die we moeten instellen om verder aan de slag te gaan:
M1.set_measured_states(['NH3','NO2','NO3'])
InitNH3 = Ndata["NH3"].ix[0]
InitNO2 = Ndata["NO2"].ix[0]
InitNO3 = Ndata["NO3"].ix[0]
M1.set_initial_conditions({'NH3': InitNH3, 'NO2': InitNO2, 'NO3': InitNO3})
#Neem voorlopig 35 minuten (voor 1ste DO niveau)
#35 min is 0.024 d
M1.set_xdata({'start':0,'end':End,'nsteps':1000})     

#Definition of step functions
#Oxygen concentration
arrayO2 = [[SpikeTime1,Mean1],[SpikeTime2,Mean2],[SpikeTime3,Mean3],[SpikeTime4,Mean4],[SpikeTime5,Mean5],[SpikeTime6,Mean6]]

#NH4-spikes
VolumeReactor = 2 #L
ConcSpikeSolNH3 = 20000 #mgN/L
VolumeSpikeSolNH3 = 0.001 #L
#Aimed concentration in reactor: 10 mgN/L
#TimeNH3 = 3.5*1e-5 #timesteps --> +-sec, has to be in days!
spikeLevelNH3 = 20000
#*(Ndata["NH3"].ix[NumSample]-Ndata["NH3"].ix[NumSample+1])*1000 #ConcSpikeSolNH3*VolumeSpikeSolNH3/DurationSpike/VolumeReactor
spikeLevelNH3Half = 10000
arrayNH3 = [[SpikeTime1,0],#[SpikeTime1,spikeLevelNH3],[SpikeTime1End*5,0],
            [SpikeTime2,spikeLevelNH3],[SpikeTime2End,0],
            [SpikeTime3,spikeLevelNH3],[SpikeTime3End,0],
            [SpikeTime4,spikeLevelNH3],[SpikeTime4End,0],
            [SpikeTime5,spikeLevelNH3Half],[SpikeTime5End,0],
            [SpikeTime6,spikeLevelNH3Half],[SpikeTime6End,0],
            ]
#print spikeLevelNH3

ConcSpikeSolNO2 = 20000 #mgN/L
VolumeSpikeSolNO2 = 0.0005 #L
#Aimed concentration in reactor: 5 mgN/L
#TimeNO2 = 3.5*1e-5 #timesteps
spikeLevelNO2 = 0#10000 #10000 #ConcSpikeSolNO2*VolumeSpikeSolNO2/DurationSpike/VolumeReactor
spikeLevelNO2Half = 0#5000 #5000
arrayNO2 = [[SpikeTime1,0],#[SpikeTime1,spikeLevelNO2],[SpikeTime1End,0],
            [SpikeTime2,spikeLevelNO2],[SpikeTime2End,0],
            [SpikeTime3,spikeLevelNO2],[SpikeTime3End,0],
            [SpikeTime4,spikeLevelNO2],[SpikeTime4End,0],
            [SpikeTime5,spikeLevelNO2Half],[SpikeTime5End,0],
            [SpikeTime6,spikeLevelNO2Half],[SpikeTime6End,0],
            ]

step = M1.makeStepFunction({'DO':np.array(arrayO2),'NH3spike':np.array(arrayNH3),'NO2spike':np.array(arrayNO2)}, accuracy = 1e-3)
M1.addExternalFunction(step)


# In[35]:

M1.Initial_Conditions


# In[36]:

M1.ode_procedure = 'odespy'
M1.ode_integrator = 'Heun'


#### Het model doorrekenen

# In[37]:

M1.ode_solver_options = {'rtol':0.0, 'atol':1e-6}#{'hmax':1e-6}
output = M1.solve_ode(plotit=True)


# In[38]:

M1.solve_algebraic()


# In[39]:

M1.algeb_solved.plot(subplots=True,figsize=(12,8))


# In[40]:

#Deze output heeft een pandas.dataframe vorm en kan nu voor allerlei anlayse of plotting gebruikt worden 
output.describe()
output.plot(figsize(16,8))  #check met de [tab] naar de opties om anders te plotten


#### Lokale gevoeligheidsanalyse

# In[41]:

#numeriek op basis van modelruns met perturbatie
temp1 = M1.numeric_local_sensitivity()


# Gevoeligheid NH3

# In[42]:

M1.numerical_sensitivity['NH3'].plot(subplots=True, figsize=(8,8))  
#verander de NH3 eens naar NO2 of NO3 om ander sensitiviteiten te zien


# In[43]:

#Dit kan ook analystisch om te checken en zou zelfde moeten zijn
M1.analytic_local_sensitivity()
M1.analytical_sensitivity['NH3'].plot(subplots=True, figsize=(8,8))


# Gevoeligheid NO2

# In[44]:

M1.numerical_sensitivity['NO2'].plot(subplots=True, figsize=(8,8))


# Gevoeligheid NO3

# In[45]:

M1.numerical_sensitivity['NO3'].plot(subplots=True, figsize=(8,8))  


#### Metingen in package zetten

# We kunnen dit profiel gebruiken om ons model te fitten. Daarvoor maken we deze eerst aan als ode_measurement. Er zijn verschillende methoden om deze op te nemen, maar op basis van bovenstaande pandas dataframes, we hebben een compatibel systeem:

# In[46]:

#Ndata was defined earlier and is now used to define Ndataprofile, containing the measurements to compare with the data
Ndataprofile = ode_measurements(Ndata[["NH3","NO2","NO3"]], print_on=False)


# Hierbij krijgt elke meting een gelijk gewicht bij de WSSE berekening, dit kan geupdate worden. Bv als verschillende variabelen of als we verschillende profielen tegelijk willen fitten. Hier werken we even door met DO en alle gewichten gelijk.

# In[47]:

Ndataprofile.add_measured_errors({'NH3':0.1,'NO2':0.24,'NO3':0.66}, method = 'absolute')
Ndataprofile.Meas_Errors


# In[48]:

#Possibility to check values
Ndataprofile.Data_dict["NH3"].head()
Ndataprofile.Data_dict["NO2"].tail()
Ndataprofile.Data_dict["NO3"].tail()


# De tijdstippen waarop we een meting hebben is belangrijk, want daar willen we ook een output van het model van:

# In[49]:

Ndataprofile.get_measured_xdata()


### Parameterschatting

# In[50]:

backup_xdata = M1._xdata


# Nu we een model hebben dat we willen testen en een set van metingen hebben gekozen, willen we deze nu fitten om de parameters te vinden:

#### Object aanmaken voor fit-procedure

# In[51]:file_path = os.path.join(os.getcwd(), 'data','respirometry.txt')

Fitprofile1 = ode_optimizer(M1, Ndataprofile, print_on=False)


# In[66]:

Fitprofile1._model._xdata = np.sort(np.append(backup_xdata, Ndataprofile.get_measured_xdata()[1:]))


# In[68]:

Fitprofile1.ModMeas.head()


# In[69]:

Fitprofile1.get_all_parameters()


#### Manuele kalibratie

# In[70]:

#adapting the initial conditions of the model towards the O example
M1.set_initial_conditions({'NH3':Ndata["NH3"].ix[0], 'NO2': Ndata["NO2"].ix[0], 'NO3': Ndata["NO3"].ix[0]})


# In[71]:

#do manual calibration
Fitprofile1.set_fitting_parameters({'KoAOB':0.82593226,'KoNOB':0.4075632,'RmaxXAOB':394.35987034357737,'RmaxXNOB':320.66655535456482})
#({'KoAOB': 1.4214515312372935, 'KoNOB': 0.028782494246204958, 
#                                    'RmaxXAOB': 524.55983016322068, 'RmaxXNOB': 302.61900525750229})
#({'KoAOB':0.82593226,'KoNOB':0.5075632,'RmaxXAOB':394.35987034357737,'RmaxXNOB':350.66655535456482})
#[('KoAOB', 1.4214515312372935), ('KoNOB', 0.028782494246204958), ('RmaxXAOB', 524.55983016322068), ('RmaxXNOB', 302.61900525750229)]
#{'muMaxXAOB':0.00462962,'muMaxXNOB':0.00462962,
 #         'KoAOB':0.2,'KoNOB':0.8,
  #        'KsAOB':0.5,'KsNOB':0.5}
Fitprofile1._model.solve_ode(plotit=False)#, xlabel("Time [d]"))#, ylabel("Concentration [mgN/L]")))
Fitprofile1.plot_ModMeas()


# In[72]:

#Fitprofile1._solve_for_opt()
#Fitprofile1._solve_for_visual()
#Fitprofile1.plot_ModMeas()
Fitprofile1._get_fitting_parameters()


# In[73]:

Fitprofile1.get_WSSE()
print Fitprofile1.WSSE


#### Lokale optimalisatie

# De lokale optimalisatie gebruikt de scipy.optimize.minimize()-functionaliteiten en alle opties kunnen ook meegegeven worden als extra argumenten; <br>
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

# In[74]:


#input_dict = OrderedDict(sorted({'KoAOB':0.8,'KoNOB':0.5,'RmaxXAOB':400,'RmaxXNOB':400}.items(), key=lambda t: t[0]))


# In[80]:

res = Fitprofile1.local_parameter_optimize(method = 'Nelder-Mead',
                                           options={'maxiter':200})


# De bekomen parameterwaarden:

# In[62]:

#figsize(12,8)
#Fitprofile1.plot_ModMeas()


# In[76]:

Fitprofile1.optimize_info


# In[76]:




# In[77]:

Fitprofile1.get_all_parameters()


# Een een vergelijking van de gemodelleerde en gemeten waarden in een scatterplot

# In[65]:

fig1,ax1 = plt.subplots(figsize=(6,6))
Fitprofile1.plot_spread_diagram('NH3', ax=ax1, )


# 

#### Betrouwbaarheidsintervallen

# In[78]:

FIM_info = ode_FIM(Fitprofile1,sensmethod='numerical')
FIM_info.get_newFIM()
FIM_info.get_parameter_confidence(alpha=0.95)
#FIM_info.get_correlations()
