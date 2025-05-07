# -*- coding: utf-8 -*-
"""

utilility functions

"""


'''
function for writing information in the data dictionary to file
'''
def write2file(filename, datadict):
    
    f = open(filename, 'w')
    for key in datadict:
        f.write(key + ': ' + str(datadict[key]) + '\n')
    f.close()