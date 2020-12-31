import numpy as np
import scipy.stats
import timeit

def conv_to_hms(seconds):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function converts a number of seconds to format hh:mm:ss for printing time updates to user
    # INPUTS:
    #    seconds [int, float]  - number of seconds
    # OUTPUTS:
    #    hms_string [str]      - string containing hh:mm:ss
    # =====================================================================================================================================

    hours = int(seconds//3600)
    minutes = int((seconds - hours*3600)//60)
    seconds = int(seconds - hours*3600 - minutes*60)

    # Format string
    hms_string = f'{hours}:{minutes}:{seconds}'
    return hms_string



def truth_to_sampEn(T, N_min_m):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function converts truth table (T tensor) and converts to a SampEn(m) value
    # INPUTS:
    #    T [3D array]             - T tensor dimensions [2 x (N-m) x (N-m)]
    #    N_min_m [int]            - Value of (N-m) for calculation
    # OUTPUTS:
    #    SampEn [float]           - SampEn(m) value
    # =====================================================================================================================================


    # Initialise a_i, b_i vectors
    a_i = np.zeros(N_min_m)
    b_i = np.zeros(N_min_m)


    # Each element in a_i, b_i is the sum of one column in T
    # Normalise the a_i and b_i values - essentially a probability for that x_i that it matches an x_j within the template vector set
    for column in range(N_min_m):
        b_i[column] = np.sum(T[0,:, column])
        a_i[column] = np.sum(T[1,:, column])

    # Calculate A_m, B_m by summing all of the normalised a_i, b_i (probabilities) and dividing by the number summed (avg):
    A_m = np.sum(a_i) / (N_min_m*(N_min_m-1))
    B_m = np.sum(b_i) / (N_min_m*(N_min_m-1))

    # Calculate Sample Entropy value
    SampEn = -np.log(A_m / B_m)

    return SampEn



def gen_template_sets(m, u):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Creates sets of template vectors for different scales (m) from time-series u
    # Set of template vectors, each of size m x 1 stored as an m x N-m+1 array
    # Each of these N-m+1 arrays are stored in a list called templates such that templates[0] has the m x N-m+1 array for m[0]
    # INPUTS:
    #    m [1D array]             - Array holding all m values requiring templates
    #    u [1D array]             - Time-series from which templates are formed
    # OUTPUTS:
    #    templates [list]         - List of 2D arrays holding templates for each m
    # =====================================================================================================================================

    # Get lengths and initialise template list
    length_u = len(u)
    length_m = len(m)
    templates = []

    # Generate an m x N-m array for each value of m and append it to the template array
    # Loop for each m value
    for m_index in range(length_m):

        # Get N-m+1 value for initialising 2D array
        N__m = int(length_u - m[m_index]+1)

        # Initialise m x N-m+1 array:
        m_vectors = np.zeros((m[m_index], N__m))

        # Map time series into templates array
        for m_val in range(N__m):
            m_vectors[:, m_val] = u[m_val:m_val + m[m_index]]

        # Append the m x N-m array for m into the templates array:
        templates.append(m_vectors)

    # User update message
    print("Templates generated")
    return templates



def run_lowest_m(N_value, m_value, r_value, templates, p_time=1000, return_time='YES'):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function calculating matches for x[m[0]] and x[m[1]] using method which forms Truth (T) tensors
    # INPUTS:
    #    N_value [int]               - Length of time-series
    #    m_value [int]               - m[0] value (ie. the lowest scale being tested)
    #    r_value [float]             - Tolerance/threshold value
    #    templates [list]            - Templates list - function utilises templates[0] and templates[1] only
    #    p_time (opt) [int]          - set number of matches to be completed before printing update to user - default = 1000
    # OUTPUTS:
    #    indices [list]              - List of 1 x 2 arrays holding indices (i,j) of vectors/elements in T tensor where matches occurred
    #    T [3D array]                - 2 x N-m x N-m array holding matches (1s) and non-matches (0s) between i,j vectors
    #                                  [0 x N-m x N-m] is matrix for m[0] and [0 x N-m x N-m] is matrix for m[1]
    #    single_duration (opt) [int] - Number of seconds taken to run this function - used for timing runs. Can be switched off by using return_time = 'NO'
    # =====================================================================================================================================

    # Get start time:
    if return_time.upper() == 'YES':
        start_single_time = timeit.default_timer()

    # Calculate N-m
    N_minus_m = N_value-m_value

    # Initialise T tensor:
    T = np.zeros((2, N_minus_m, N_minus_m))

    # Initialise list holding matching indices
    indices = []

    # Loop to test matches
    for i in range(N_minus_m):

        # ______________________________________________________________________
        # Print progress update message
        if i % p_time == 0 and i > 0:
            # Time taken so far:
            so_far = timeit.default_timer() - start_single_time

            # Estimate end time:
            remaining = (so_far / i) * (N_minus_m - i)

            print(f'i = {i}/{N_minus_m}')
            print(f'Current run-time: {conv_to_hms(so_far)}')
            print(f'Estimated time remaining: {conv_to_hms(remaining)}')
        # ______________________________________________________________________

        # Designate x_i vector from x[m=0] set:
        x_i = templates[0][:, i]

        # Loop to compare all x_j with above x_i and ensure self-matches (ie i==j) arent counted by starting at i+1
        for j in range(i+1, N_minus_m):
               # Designate x_j vector from x[m=0] set:
                x_j = templates[0][:, j]

                # Calculate resultant (distance) vector and check if distance within threshold r
                if np.amax(np.abs(x_i - x_j)) < r_value:

                    # If within threshold, set element to 1 in T[0] matrix
                    T[0,j,i] = 1

                    # In the case that the m vectors match, we then test the m+1 vectors:
                    # Generate the x_i and x_j from template[1] (m+1) set
                    x_i_plus_1 = templates[1][:, i]
                    x_j_plus_1 = templates[1][:, j]

                    # If within threshold, set element to 1 in T[1] matrix
                    if np.amax(np.abs(x_i_plus_1 - x_j_plus_1)) < r_value:
                        T[1,j,i] = 1

                        # Add index to array holding matched indices
                        indices.append([j,i])


    # Mirror both the m and m+1 truth matrices along their diagnoals by adding their transpose:
    T[0,:,:] += np.matrix.transpose(T[0,:,:])
    T[1,:,:] += np.matrix.transpose(T[1,:,:])

    # Return time taken to user
    if return_time.upper() == 'YES':
        single_end_time = timeit.default_timer()
        single_duration = single_end_time - start_single_time

    return indices, T, single_duration


def calculate_T(r_val, T, template, index_array):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function calculating matches for x[m]] and x[m+1]] using method which forms Truth (T) tensors, for m[>1]
    # INPUTS:
    #    r_value [float]             - Tolerance/threshold value
    #    T [3D array]                - T tensor holding matched indices
    #    template [2D array]         - 2D array holding templates for m+1 only
    #    index_array [list]          - List of 1 x 2 arrays holding indices (i,j) of vectors/elements in T tensor where matches occurred
    # OUTPUTS:
    #    new_ind [list]              - Updated version of index_array where any indices no longer matching have been removed
    #    T [3D array]                - Updated T tensor
    # =====================================================================================================================================

    # Get number of indices which require testing (length of index_array list)
    len_index = len(index_array)
    # Initialise list for indices
    new_ind = []
    # Loop through indices in list
    for h in range(len_index):
        # Get i, j from list
        i = index_array[h][1]
        j = index_array[h][0]

        # Designate x_i vector:
        x_i_plus_1 = template[:, i]
        x_j_plus_1 = template[:, j]

        # Test match
        if np.amax(np.abs(x_i_plus_1 - x_j_plus_1)) < r_val:
           # Update T[1] matrix if match occurs
           T[1,j,i] = 1
           # Add matched indices (i,j) to new array
           new_ind.append(index_array[h])

    # Mirror both the m and m+1 truth matrices along their diagnoals:
    T[1,:,:] += np.matrix.transpose(T[1,:,:])

    return new_ind, T




def remove_end_indices(array, T):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Function removing any indices held in the index array which will no longer be valid when the T matrices are sliced
    # INPUTS:
    #    T [3D array]                - T tensor holding matched indices
    #    array [list]                - List of 1 x 2 arrays holding indices (i,j) of vectors/elements in T tensor where matches occurred
    # OUTPUTS:
    #    new_array [list]            - Updated version of index_array where indices that wont be valid, after slicing, have been removed
    # =====================================================================================================================================

    # Get value of last column/row in T (ie the row/column to be sliced):
    val = T.shape[2]-1
    # Initialise updated array
    new_array = []

    # Loop for each index in the list
    for ind in array:
        # As long as both indices are below the value then add to new array
        # (for some reason removing values that violate the opposite condition doesnt seem to work)
        if ind[0] < val and ind[1] < val:
            new_array.append(ind)


    return new_array

def update_arrays(index_array, T):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Updates the index array and T tensor prior to the next round of SampEn calculations.
    # INPUTS:
    #    T [3D array]                - T tensor holding matched indices
    #    index_array [list]          - List of 1 x 2 arrays holding indices (i,j) of vectors/elements in T tensor where matches occurred
    # OUTPUTS:
    #    new_index [list]            - Updated version of index_array where indices that wont be valid, after slicing, have been removed
    #    T [3D array]                - Updated T tensor
    # =====================================================================================================================================

    # Need to remove any indices which are about to be sliced out:
    new_index = remove_end_indices(index_array, T)

    # We now take the T[1,:,:] (our m+1 from previous SampEn calculation) and make it T[0,:,:] as it is now m
    # The last column and row must be sliced to reflect the increasing value of m and therefore reduction in dimension of the set of test
    # vectors (N-m)
    T = T[:, :-1, :-1] # Slice last row and column
    T[0, :, :] = T[1, :, :] # Set new T[m] = old T[m+1]
    T[1, :, :] = 0 # initialise new T[m+1] to be calculated using calculate_T() function

    return new_index, T




# __________________________________________________________________________________________________________-
def calculate_MSE(u, m, r=0, p_time=1000, m_1=1, return_duration='YES', print_update='YES'):
    # =====================================================================================================================================
    # DESCRIPTION:
    # Front-end function for calculating the MSE of a time-series.
    # INPUTS:
    #    u [1D array]                - Time-series for which MSE will be calculated
    #    m [int]                     - Maximum scale to be investigated. All scales from 1 to m will be investigated
    #    r (opt) [float]             - Threshold/tolerance value for matches - if not entered program defaults to r = std. dev. of time-series
    #    p_time (opt) [int]          - Number of iterations in first calculation to be completed before printing update. Default = 1000
    #    m_1 (opt) [int]             - Minimum scale (m) investigated. Default = 1
    #    return_duration (opt) [str] - If 'YES' will return an array of the calculation time for each SampEn(m), in seconds. Default = 'YES'
    #    print_update (opt) [str]    - If 'YES' will print update after each m is calculated. Default = 'YES
    # OUTPUTS:
    #    MSE [1D array]              - Array of SampEn(m) values for all m
    #    m_duration [1D array]       - Array holding calculation time of each SampEn(m)
    # =====================================================================================================================================

    # Save m vector as len_m
    len_m = m
    # Convert m into a 1D array holding all m values to calculate
    m = np.arange(m_1,m+1)

    # Convert u elements to z-scores
    u = scipy.stats.zscore(u)

    # By default r is given as the standard dev of the time series unless otherwise specified - this will be 1 given conversion to z-scores
    if r == 0:
        r = np.std(u)

    # Get length of time-series
    N = len(u)

    # Generate empty MSE and m_duration array same lengths as the number of scales to test:
    MSE = np.zeros(len_m-1)
    m_duration = np.zeros(len_m-1)

    # Preamble complete - begin calculation of MSE's:

    # Generate all the 2D template arrays required as a list in the same order as m:
    templates = gen_template_sets(m,u)

    # First we conduct the matching process on the minimum m value
    index_array, truth, m_duration[0]= run_lowest_m(N_value=N, m_value=m[0], r_value=r, templates=templates, p_time=p_time)

    # Calculate and add the SampEn(m[0]) to the MSE array:
    MSE[0] = truth_to_sampEn(T=truth, N_min_m=N-m[0])

    # Begin timer
    start_loop_timer = timeit.default_timer()

    # Now calculating the SampEns with higher m:
    for m_loop_val in m[1:-1]:

        # Begin separate time for calculating this specific loop
        loop_time = timeit.default_timer()

        # First update the T tensor and index arrays
        index_array, truth = update_arrays(index_array, truth)

        # Calculate T - only have to test the matches which are non-zero in the truth matrix
        index_array, truth = calculate_T(r_val=r, T=truth, template=templates[m_loop_val], index_array=index_array)

        # Calculate and add the SampEn(m) to the MSE array:
        # Note this indexing may cause issues if user chooses m[0] > 1...need to update/investigate
        MSE[m_loop_val-1] = truth_to_sampEn(T=truth, N_min_m=N - m[m_loop_val])

        # Get current time:
        current_time = timeit.default_timer()
        # Time taken so far overall:
        current_duration = current_time - start_loop_timer
        m_duration[m_loop_val-1] = current_time - loop_time

        # Print update to user
        if print_update.upper() == 'YES':
            # Estimate end time and formatting user output:
            remaining = (current_duration/m_loop_val) * (len_m-1 - m_loop_val)
            print('  ')
            print('_______________________________________________________________')
            print(f'm = {m_loop_val}/{m[-1]-1}')
            print(f'Current run-time: {conv_to_hms(current_duration)}')
            print(f'Estimated time remaining: {conv_to_hms(remaining)}')
            print('_______________________________________________________________')


    if return_duration.upper()=='YES':
        return MSE, m_duration
    else:
        return MSE



