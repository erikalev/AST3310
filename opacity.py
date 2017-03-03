import sys
import numpy as np

#collecting variables T and rho from the command line call from c++
T = float(sys.argv[1])          # fetching the T-value from the command line
rho = float(sys.argv[2])/1000.0 # fetching the rho-value from the command line and converting it to cgs
R = rho/((T/1e6)**3)  # as defined by Appendix D
R = np.log10(R)

infile = open("/home/erik/Appendix_C/opacity.txt", "r")

# the "closest-", "min", "max" and "second_closest"-variables are variables I use to locate the correct opacity-values from opacity.txt. 

closest_R_distance = 1e6                # initial value for the closest distance between my R-value and the R-value in opacity.txt
min_R_value = -8.0                      # initial value of the closest R_value in opacity.txt from the left 
min_R_index = 1                         # initial index of the left R_value

second_closest_R_distance = 1e6         # initial value for the second closest distance between my R-value and the R-value in opacity.txt
max_R_value = -8.0                      # initial value of the closest R_value in opacity.txt from the right
max_R_index = 1                         # initial index of the right R_value

closest_log10T_distance = 1e6           # initial value for the closest distance between my log10T-value and the log10T-value in opacity.txt
closest_log10T_value = 3.75             # initial value of the log10T-value in opacity.txt closest to one sent in by my code 
closest_log10T_index = 2                # initial value of the closest_log10T_value index

second_closest_log10T_distance = 1e6    # initial value for the second closest distance between my log10T-value and the log10T-value in opacity.txt
second_closest_log10T_value = 3.75      # initial value of the log10T-value in opacity.txt second closest to one sent in by my code
second_closest_log10T_index = 2         # initial value of the second_closest_log10T_value index


T = np.log10(T)                         # changing the T_value sent in by my code to log10(T)

#test-values
#T = 3.795
#R = -5.95

# the opacity-values I need to do a 2D-interpolation of
K1 = 0                                  # upper left opacity value
K2 = 0                                  # upper right opacity value
K3 = 0                                  # lower left opacity value
K4 = 0                                  # lower right opacity value

i = -1                                  # this is just a counter 

for line in infile:
    i+=1    
    words = line.split()
    if i == 0:
        #R-values are located in the first row. This first if-test finds the 2 R-values closest to the one sent in and saves their values and index 
        for j in range(1, len(words)):
            if abs(R - float(words[j])) < closest_R_distance: 
                # A least squares method to locate all values closer than the previous value
                # If true then values are updated
                closest_R_distance = second_closest_R_distance 
                min_R_value = max_R_value
                min_R_index = max_R_index

                max_R_index = j  
                second_closest_R_distance = abs(R - float(words[j]))
                max_R_value = float(words[j])
            
            else:
        
                if abs(R - float(words[j])) < closest_R_distance:
                    min_R_value = float(words[j])
                    min_R_index = max_R_index
                    max_R_index = j

    elif i == 1:
        # no values in this row
        pass 

    else:
        if i==2:
            # setting initial values to the K-variables 
            K1 = float(words[0])
            K2 = float(words[0])
            K3 = float(words[0])
            K4 = float(words[0])

        if abs(T - float(words[0])) < closest_log10T_distance: 
            # an other least squares methof for the log10T-values
            # If true the values are updated
            # K1-K4 values are also updated
            second_closest_log10T_distance = closest_log10T_distance
            second_closest_log10T_value = closest_log10T_value
            closest_log10T_value = float(words[0])
            closest_log10T_distance = abs(T - float(words[0]))
            K1 = K3
            K2 = K4
            K3 = float(words[min_R_index])
            K4 = float(words[max_R_index])

        elif abs(T - float(words[0])) < second_closest_log10T_distance:
            # A least squares method for checking if the second closest log10T-value on the right hand side is 
            # closer than the one on the left hand side. If so the values are updated and opacity.txt is closed  
            K1 = K3
            K2 = K4
            K3 = float(words[min_R_index])
            K4 = float(words[max_R_index])
            closest_log10T_value = second_closest_log10T_value
            second_closest_log10T_value = float(words[0])
            infile.close()
            break
        else:
            infile.close()
            break

N = 101                   # number of interpolation points

R_array = np.linspace(min_R_value, max_R_value, N)  # array of R-values
R_index = 0                                         # variable for closest value index
least_distance = 1e6                                # least squares variable for distance
for i in range(len(R_array)):
    if abs(R - R_array[i]) < least_distance:
        # finds the val ue in 
        least_distance = abs(R - R_array[i])
        R_index = i

if closest_log10T_value < second_closest_log10T_value:
    # test to check that the log10T values are in the correct order
    T_array = np.linspace(closest_log10T_value, second_closest_log10T_value, N)     # array of log10T-values
    T_index = 0
    least_distance = 1e6
    for i in range(len(R_array)):
        if abs(T - T_array[i]) < least_distance:
            least_distance = abs(T - T_array[i])
            T_index = i
else:
    # changing order
    T_array = np.linspace(second_closest_log10T_value, closest_log10T_value, N)
    T_index = 0
    least_distance = 1e6
    for i in range(len(R_array)):
        if abs(T - T_array[i]) < least_distance:
            least_distance = abs(T - T_array[i])
            T_index = i

K1K3 = np.linspace(K1, K3, N)       # the first column vector in my [[K1, K2],[K3, K4]] matrix
K2K4 = np.linspace(K2, K4, N)       # the last row vector in my [[K1, K2],[K3, K4]] matrix   

matrix = np.zeros((N, N))           # creating the [[K1, K2],[K3, K4]] matrix

for i in range(N):  
    matrix[i,:] = np.linspace(K1K3[i], K2K4[i], N) # 2D interpolation 


outfile = open("/home/erik/Appendix_C/opacity_result.txt", "w") # file to dumt the opacity-value in
outfile.write(str(matrix[T_index, R_index]))                    # locating the best interpolation approximation and dumping it to a file
outfile.close()                                                 # done
