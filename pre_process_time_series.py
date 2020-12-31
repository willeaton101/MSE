import numpy as np

# Script contains functions for both coarse-grained averaging and moving-averaging of time series.
# Moving average is more favourable for small-size time-series as coarse-averaging will rapidly decimate the time-series.

#_____________________________________________________________________________________________________________________________________________
def coarse_grain_avg(u, tau, time=[]):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function produces a coarse-grain averaged time series of user-inputted time-series with scale tau
    # In the case that u = u(t), user may want accompanying t time-series to be calculated
    # INPUTS:
    #    u [1D array]             - time-series to be coarse-grained
    #    tau [int]                - coarse-grain scale
    #    time (opt) [int, array]  - default set to 'No' indicates time array doesnt need to be calculated; otherwise 1D time array/time-series
    # OUTPUTS:
    #    cgu [1D array]           - coarse-grained time-series
    #    cg_time (opt) [1D array] - accompanying time array for time-series
    # =====================================================================================================================================

    # Get size of time-series:
    u_length = len(u)

    # Calculate N/tau - length of coarse-grained time-series
    N_tau = int(u_length/tau)

    # Initialise empty coarse-grain array:
    cgu = np.zeros(N_tau)
    avg_t_temp = [] # Will hold the first and last cg-averaged time values to produce the cg_time array

    # Coarse grain the time-series:
    for j in range(N_tau):
        cgu[j] = (1/tau)*np.sum(u[j*tau : (j+1)*tau])


        # Calculates the first and last corase-grain time values from the array which can then be extrapolated for the rest of the array
        if j==0 or j==N_tau-1:
            if len(time) != len(u):
                # Not used if time = 0 but still calculated
                avg_t_temp.append((1/tau)*np.sum(time[j*tau : (j+1)*tau]))

    # Calculate new accompanying 'time' array if defined by user (not = 0):
    if len(time) != len(u):
        # Interpolate the rest of the cg_time array values
        cg_time =  np.linspace(avg_t_temp[0], avg_t_temp[1], num=N_tau)
        return cg_time, cgu
    else:
        return cgu





def moving_avg(f, time=[], half_window=1000, convert_t='NO'):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function produces a moving-average time series of user-inputted time-series with window-size 2*spacing+1
    # In the case that u = u(t), user may want accompanying t time-series to be calculated
    # INPUTS:
    #    f [1D array]             - number of seconds
    #    time (opt) [int]         - coarse-grain scale
    #    time (opt) [int, array]  - default set to 'No' indicates time array doesnt need to be calculated; otherwise 1D time array/time-series
    # OUTPUTS:
    #    cgu [1D array]           - coarse-grained time-series
    #    cg_time (opt) [1D array] - accompanying time array for time-series
    # =====================================================================================================================================


    # Initialise mov_avg array:
    avg = np.zeros(int(len(f) - 2 * half_window))

    # Moving-average calculations cuts off the first and last number of elements equal to the value of 'half_window'
    for ts_index in range(half_window, int(len(f) - half_window)):
        avg[ts_index - half_window]= 1 / (half_window * 2) * np.sum(f[ts_index - half_window:ts_index + half_window])

    # If convering a time series eg u(t), may wish to slice time array to same size:
    if convert_t.upper() == 'YES':
        try:
            t_avg = time[half_window:int(len(f) - half_window)]
            return t_avg, avg
        except time == []:
            print('Error: No time array inputted')

    else:
        return avg
