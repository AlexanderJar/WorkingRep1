#Converting tem files to inv files
#python xxx_txt ramp
#ramp time in microseconds

#importing packages
import numpy as np
import sys

#Reading information from command line (file name without extension and ramp time in Microseconds)
print(sys.argv)
INFILE = sys.argv[1]
ramp = float(sys.argv[2])

#loading data
print('Reading Data ...')
#just loading times, voltages and errors
#voltages are already normalized to current strength
data = np.loadtxt(INFILE+'.tem',skiprows=8,usecols=(1,2,3))

#reading from header information about loop sizes
f=open(INFILE+'.tem','r')
lines = f.readlines()
loop_info=lines[4]
xl = loop_info.split()
f.close()

print('Finished!')

#Information
TxLen = float(xl[2]) #Transmitter edge in m
RxLen = float(xl[5]) #Receiver edge in m

#output file name
OUTFILE = INFILE+'.inv'

print('Writing Output File ...')
#writing .inv file
fp=open(OUTFILE,'w') #output file

fp.write(' TEM\n')
fp.write(' %s\n'%INFILE)
#Receiver Loop position, central loop configuration, Transmitter Loop size
fp.write('%20.14f %20.14f %20.14f %20.14f\n'%(0,0,TxLen, TxLen))
fp.write('1\n')
#Ramp Time in second
fp.write('%20.8e\n'%(ramp*10**(-6)))
fp.write('%12d\n'%len(data))

#Coloumn 1: time in s
#Coloumn 2: Voltage normalized to Current and Receiver Loop Area in V/A/m^2
#Coloum 3: Relative Error of Voltage 
for ii in range(0,len(data)):
	if (data[ii,1] == 0):#if measured voltage is zero
		fp.write(' %16.8e %16.8e %16.8f\n'%(data[ii,0]*10**-6,data[ii,1]/TxLen**2,0))
	else:
		if (data[ii,1]/TxLen**2,0<0.1): 
			fp.write(' %16.8e %16.8e %16.8e\n'%(data[ii,0]*10**-6,data[ii,1]/TxLen**2,np.abs(data[ii,2]/data[ii,1])))
		else:
			fp.write(' %16.8e %16.8e %16.8f\n'%(data[ii,0]*10**-6,data[ii,1]/TxLen**2,np.abs(data[ii,2]/data[ii,1])))
fp.close()
print('Finished!')