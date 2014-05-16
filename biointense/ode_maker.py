"""
Created on Mon Mar 25 12:04:03 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator by Tvandaele
"""

from biointense import DAErunner

class odemaker(object):
    '''
    Class to generate an ODE system based on a symbolic diagram.
    
    Parameters
    ------------
    System : OrderedDict
        Ordered dict with the keys as the derivative of a state (written as 
        'd'+State), the values of the dictionary is the ODE system written 
        as a string     
    
    Examples
    ----------
    >>> input_list=[['SA + En <=> EnA','k1','k2'],
                    ['EnA ==> En + PP','kcat']]
    >>> MM = odemaker(input_list, Modelname = 'Michaelis_Menten')
    >>> MM.system
    >>> MM.system_list
    >>> MM.parameters
    '''
    
    def __init__(self, string_list, Modelname = 'MyModel',*args,**kwargs):
        '''

        '''
        try:
            self._print_on = kwargs.get('print_on')
        except:
            self._print_on = True
            
        self.string_list = string_list   
        
        self.modelname = Modelname
        
        self.odeMake()
        
      
    def odeMake(self):
        '''
        Generate an ODE system based on a symbolic diagram.

        '''
        if self._print_on:
            print('Running odemaker to convert symbolic input to ODE system...')
        system = {}
        parameters = {}
        system_list = []
        for m,i in enumerate(self.string_list):
            i[0] = i[0].replace(" ","")
            if len(i) == 3 and '<=>' in i[0]:
                arrow_split = i[0].split('<=>')
                beforelist = arrow_split[0].split('+')
                afterlist = arrow_split[1].split('+')
                for n,j in enumerate(beforelist):
                    if j not in system_list:
                        system_list.append(j)
                    try:
                        system['d' + j] += ' - ' + i[1] + '*' + arrow_split[0].replace("+","*") + ' + ' + i[2] + '*' + arrow_split[1].replace("+","*")
                    except:
                        system['d' + j] = '-'+ i[1] + '*' + arrow_split[0].replace("+","*") + ' + ' + i[2] + '*' + arrow_split[1].replace("+","*")
                for n,j in enumerate(afterlist):
                    if j not in system_list:
                        system_list.append(j)
                    try:
                        system['d' + j] += ' + ' + i[1] + '*' + arrow_split[0].replace("+","*") + ' - ' + i[2] + '*' + arrow_split[1].replace("+","*")
                    except:
                        system['d' + j] = i[1] + '*' + arrow_split[0].replace("+","*") + ' - ' + i[2] + '*' + arrow_split[1].replace("+","*")
            elif len(i) == 2 and '==>' in i[0]:
                arrow_split = i[0].split('==>')
                beforelist = arrow_split[0].split('+')
                afterlist = arrow_split[1].split('+')
                for n,j in enumerate(beforelist):
                    if j not in system_list:
                        system_list.append(j)
                    try:
                        system['d' + j] += ' - ' + i[1] + '*' + arrow_split[0].replace("+","*")
                    except:
                        system['d' + j] = '-'+ i[1] + '*' + arrow_split[0].replace("+","*")
                for n,j in enumerate(afterlist):
                    if j not in system_list:
                        system_list.append(j)
                    try:
                        system['d' + j] += ' + ' + i[1] + '*' + arrow_split[0].replace("+","*")
                    except:
                        system['d' + j] = i[1] + '*' + arrow_split[0].replace("+","*")

            elif len(i) == 2 and '<==' in i[0]:
                arrow_split = i[0].split('<==')
                beforelist = arrow_split[0].split('+')
                afterlist = arrow_split[1].split('+')
                for n,j in enumerate(beforelist):
                    if j not in system_list:
                        system_list.append(j)
                    try:
                        system['d' + j] += ' + ' + i[1] + '*' + arrow_split[0].replace("+","*")
                    except:
                        system['d' + j] = i[1] + '*' + arrow_split[0].replace("+","*")
                for n,j in enumerate(afterlist):
                    if j not in system_list:
                        system_list.append(j)
                    try:
                        system['d' + j] += ' - ' + i[1] + '*' + arrow_split[0].replace("+","*")
                    except:
                        system['d' + j] = '-' + i[1] + '*' + arrow_split[0].replace("+","*")
            else:
                raise Exception('The input {0} cannot be converted properly, please change your input!'.format(i[0]))
    
            for n,j in enumerate(i):
                if n>0:
                    try:
                        parameters[j]
                        raise Exception('The parameter {0} has been defined more than once, please change the input!'.format(j))
                    except KeyError:
                        parameters[j] = 1
                
        self.system = system
        self.parameters = parameters
        self.system_list = system_list
        
        if self._print_on:
            print('...Done!')
        
    def passToOdeGenerator(self):
        return DAErunner(ODE = self.system, Parameters = self.parameters, Modelname = self.modelname)