import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

def generate_initial_point(low, high, n, infected_idx, infection, default_val=1):
    """ generaete initial point

    Parameters: low (int/float): lower boundary
                high (int/float): upper boundary
                n (int): total points
                infected_idx (dict) - index of those asymptomatic or infected or recovered
                                      format -> {id/idx:status}
                default_val (int) - default value if not specified in infected_idx, optional, default is 1

    Returns: init_point (DataFrame) - a dataframe with 3 columns: x, y, status
    """
    # generate walk
    init_point = pd.DataFrame(np.random.uniform(low, high, n*2).reshape((n,2)), columns=['x','y'])
    status = [infected_idx.get(i, default_val) for i in range(n)]
    init_point['status'] = status

    infected_time = [1 if i in infection else np.nan for i in status]
    init_point['infected_time'] = infected_time

    return init_point

# help function, help compute_distance_check()
def check_list_val(in_list, compare, threshold, exclude=[]):
    """ check if any value in a list match a condition

    Parameters: in_list (list) - the input list
                compare (str) - the comparison condition, le:'<', leq:'<=', g:'>', geq:'>='
                threshold (int/float) - the threshold
                exclude (int/tuple/list/array-like) - exclude for a certain index

    Return: (boo) - True if any value matches, Flase o/w
    """
    my_list = np.delete(in_list, exclude)
    if compare is 'leq':
        check = my_list <= threshold
    elif compare is 'le':
        check = my_list < threshold
    elif compare is 'geq':
        check = my_list >= threshold
    elif compare is 'g':
        check = my_list > threshold
    else:
        raise(ValueError)

    return any(check)

# help function
def compute_distance_check(left, right, norm, threshold, compare, self_exclude=False):
    """ compute the distance pair between two matrix and check if a certain condition match

    Parameters: left (DataFrame) - main dataframe
                right (DataFrame) - comparison dataframe
                norm (int) - Minkowski norm type
                threshold (int/float) - the threshold
                compare (str) - the comparison condition, le:'<', leq:'<=', g:'>', geq:'>='
                self_exclude (boo) - for self checking, i.e. compare distance with all other value
                                     if true, exclude, self value, i.e. of course 0
                                     optional, default is False

    Returns: dist_matrix (DataFrame) - the distance matrix
                                       index denotes left index, column name denotes right index
             check_result (Series) - check result, index denotes the index for left
                                     True if any value matches, Flase o/w
    """
    dist_matrix = pd.DataFrame(distance_matrix(left, right, norm), index=left.index, columns=right.index)
    if self_exclude:
        check = [check_list_val(j, compare, threshold, i) for i,j in enumerate(dist_matrix.values)]
    else:
        check = [check_list_val(j, compare, threshold) for i,j in enumerate(dist_matrix.values)]

    check_result = pd.Series(check, index=left.index)
    return dist_matrix, check_result

# help function, help update_status_by_time()
def generate_period(in_df, status_duration):
    """ generate a pre-defined status period

    Parameters: in_df (DataFrame) - the population dataset
                status_duration (dict) - define random period in a list, dict format {status:[low,high]}
                                         eg, {2:[7,14]} -> status 2, random integer (days) between 7 to 14 days

    Returns: (DataFrame) - a dataframe which column denotes the random days, index denotes the id of the user
    """
    random_data = np.transpose([np.random.randint(status_duration[i][0], status_duration[i][1]+1, in_df.shape[0]) for i in list(status_duration)])
    df = pd.DataFrame(random_data, columns=list(status_duration))
    return df

def update_time_by_increment(in_df, increment):
    """ updated infected time by a specific increment value

    Parameters: my_df (DataFrame) - original dataframe with data
                increment (int) - increment value

    Returns: (DataFrame) - the updated dataframe
    """
    my_df = in_df.copy()
    my_df['infected_time'] += increment
    return my_df

def update_status_by_time(in_df, status_time, status_update_to):
    """ update status if the infected_time reach a pre-defined number
        once status updated, infected_time will be reset to 1

    Parameters: in_df (DataFrame) - original dataframe with data
                status_time (DataFrame) - index denotes the id, column denotes the status number, value is the duration
                                          generate by generate_period() function
                status_update_to (dict) - status update to what value,
                                          eg. {2:3, 3:5} -> status 2 update to 3, status 3 update to 5

    Returns: (DataFrame) - the updated data
    """
    merged_df = in_df.merge(right=status_time, left_index=True, right_index=True)
    subset = {i:merged_df[merged_df.status == i] for i in list(status_time)}

    for i in list(status_time):
        check = (subset[i]['infected_time'] - subset[i][i]) == 0
        merged_df.loc[check[check].index, 'status'] = status_update_to[i]
        merged_df.loc[check[check].index, 'infected_time'] = 1

    return merged_df[['x','y','status','infected_time']]

def update_status_by_location(in_df, norm, infection, non_im, threshold):
    """ update the status if the distance between a point and an infected point is less than a threshold

    Parameters: in_df (DataFrame) - original dataframe with data
                norm (int) - Minkowski norm type
                infection (array-like) - code denotes that is affected, eg [2,3]
                non_im_df (array-like) - code denotes people with no immunity, eg [1]
                threshold (int/float) - threshold distance

    Returns: return_df (DataFrame) - the updated status data
    """
    my_df = in_df.copy()
    good = my_df[my_df.status.isin(non_im)]
    bad = my_df[my_df.status.isin(infection)]

    dist_matrix, check = compute_distance_check(good[['x','y']], bad[['x','y']], norm, threshold, 'leq')

    my_df.loc[check[check].index, 'status'] = 2
    my_df.loc[check[check].index, 'infected_time'] = 1

    return my_df

def update_location_isolation(in_df, status, iso_x, iso_y, iso_inf_time):
    """ update location as isolation, i.e. move obs off the grid

    Parameters: in_df (DataFrame) - original dataframe with data
                status (tuple/list) - the status to isolate
                iso_x, iso_y (int/float) - the x, y coordinate of isolation location
                iso_inf_time (int) - the infection time to initiate isolation

    Returns: (DataFrame) - the updated data
    """
    my_df = in_df.copy()
    iso_group = in_df[(in_df.status.isin(status)) & (in_df.infected_time == iso_inf_time)]

    my_df.loc[iso_group.index, ['x','y']] = [iso_x, iso_y]

    return my_df

def update_location_return_isolation(in_df, status, return_time, low, high):
    """ return isolation points back to normal grid by generate random number within range

    Parameters: in_df (DataFrame) - original dataframe with data
                status (tuple/list) - the status to return from isolation
                return_time (int) - the infection time to return from isolation
                low (int/float): lower boundary
                high (int/float): upper boundary

    Returns: (DataFrame) - the updated data
    """
    my_df = in_df.copy()
    bad = in_df[(in_df.status.isin(status)) & (in_df.infected_time == return_time)]

    new_pts = np.random.uniform(low, high, bad.shape[0]*2).reshape((bad.shape[0], 2))
    my_df.loc[bad.index, ['x','y']] = new_pts
    return my_df

def update_location_random_walk(in_df, status, max_walk, norm, threshold):
    """ update location as random walk

    Parameters: in_df (DataFrame) - original dataframe with data
                status (tuple/list) - list of status to go for a walk
                max_walk (int/float) - the maximum walking distance
                norm (int) - Minkowski norm type
                threshold (int/float) - minimum sc distance

    Returns: (DataFrame) - the updated data
    """
    walk_group = in_df[in_df.status.isin(status)].copy()
    non_walk_group = in_df[~in_df.status.isin(status)]

    converge_counter = 0
    while True:
        converge_counter += 1
        walk_dist = np.random.uniform(-max_walk, max_walk, walk_group.shape[0]*2).reshape((walk_group.shape[0],2))
        walk_group[['x','y']] += walk_dist

        sd_dist, sd_result = a, b = compute_distance_check(walk_group[['x','y']], walk_group[['x','y']], norm, threshold, 'geq', True)

        if all(sd_result):
            break
        elif converge_counter == 700:
            raise ValueError('SD failed to converge, please reset boundary length and n.')
            break
        else:
            walk_group[['x','y']] -= walk_dist

    final_df = pd.concat([walk_group, non_walk_group]).sort_index()

    return final_df

# main class
class Simulation:
    def __init__(self, low, high, n, infected_idx, infection, non_im):
        self.low = low
        self.high = high
        self.n = n
        self.infected_idx = infected_idx
        self.infection = infection
        self.non_im = non_im

        self.initial_point = generate_initial_point(self.low, self.high, self.n, self.infected_idx, self.infection)

    def run(self, step, status_duration, status_update_to, \
            norm, infection_threshold, \
            iso_status, iso_x, iso_y, iso_inf_time, \
            re_iso_status, re_iso_time, \
            walk_status, max_walk, sd_threshold, \
            return_type='dict'):

        status_duration = status_duration
        infect_period = generate_period(self.initial_point, status_duration)

        record = dict()
        record[0] = self.initial_point

        original = self.initial_point
        for i in range(1,step+1):
            ## update status
            # increase time by 1 step
            new = update_time_by_increment(original, 1)
            # update status by infected time
            new = update_status_by_time(new, infect_period, status_update_to)
            # update status by location, i.e. calculate if a person is near an infected person
            new = update_status_by_location(new, norm, self.infection, self.non_im, infection_threshold)

            ## update location
            # isolation
            new = update_location_isolation(new, iso_status, iso_x, iso_y, iso_inf_time)
            # return from isolation
            new = update_location_return_isolation(new, re_iso_status, re_iso_time, self.low, self.high)
            # random walk guys
            new = update_location_random_walk(new, walk_status, max_walk, norm, sd_threshold)

            # append to the dictionary
            record[i] = new
            original = new


        if return_type is 'dict':
            return record

        elif return_type is 'df':
            df = self.dict_to_df(record)
            return df

    # data wrangling tool
    @staticmethod
    def dict_to_df(record):
        """ transform a dict from simulation records to a df

        Parameters: record (dict) - the dict from Simulation.run

        Returns: (DataFrame)
        """
        n = record[0].shape[0]
        frame = list()
        for i in list(record):
            record[i]['step'] = [i]*n
            frame.append(record[i][['x','y','status','step']])

        df = pd.concat(frame).reset_index(drop=True)
        return df

if __name__ == '__main__':
    # class parameters
    low = -350
    high = 350
    n = 50
    infected_idx = {4:2, 12:2, 8:2, 5:2, 25:2, 6:3, 11:3}
    infection = [2,3]
    non_im = [1]

    # individual experiment parameters
    step = 180
    status_duration = {2:[7,14], 3:[30,60]}
    status_update_to = {2:3, 3:4}
    norm = 2
    infection_threshold = 60
    iso_status = [3]
    iso_x = 1000
    iso_y = 1000
    iso_inf_time = 3
    re_iso_status = [4]
    re_iso_time = 2
    walk_status = [1,2,3,4]
    max_walk = 30
    sd_threshold = 20

    # start the basic set up
    my_simulation = Simulation(low, high, n, infected_idx, infection, non_im)

    # run the individual experiment
    # you can either specify the return type,
    # or in general keep a dict as the raw record
    # and use the class method self.dict_to_df() to transform the dict to df
    records = my_simulation.run(step, status_duration, status_update_to, \
                                norm, infection_threshold, \
                                iso_status, iso_x, iso_y, iso_inf_time, \
                                re_iso_status, re_iso_time, \
                                walk_status, max_walk, sd_threshold, 'dict')

    df = my_simulation.dict_to_df(records)

##
