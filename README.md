# Homogenous_regions
It uses self organizing maps for delineating homogenous regions based on precipitation data.
Homogenous_regions_code.m matlab code provides the user to give the output homogenous regions map based on IMD daily gridded precipitation data for the period of 1951-2020.
The repository consists of different functions which have been used in the Homogenous_regions.m code.
appent.m is a function used to compute the apportionment entropy of rainfall.
cent.m is used to compute the marginal entropy of rainfall.
centri.m is used to identify the centroid of the rainfall.
daviesbouldin.m is used to compute the values of davies bouldin index for different gridsizes of the homogenous regions.
Input_data.mat file consists of the precipitation data for the period of 1951-2020.
clustered_data.mat consists of the homogenous regions data.

The functions and the code has to be in the same folder to avoid the errors due to path.
Initially, The Input data has to be loaded and the run the entire code, the output homogenous regions will be generated as the output.





