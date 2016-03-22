#! /usr/bin/python

import sys
import os

import numpy as np
import pandas as pd
from pandas import Series
from datetime import datetime

import itertools
from collections import Counter
import operator
from itertools import tee, izip

import matplotlib.pyplot as pyplot
import matplotlib.ticker as tkr
from matplotlib import cm
import matplotlib.ticker as tkr
import seaborn as sns
sns.set_style('white')

import scipy
from scipy.optimize import minimize
from scipy.stats import binom

# change directory
os.chdir("/Users/Gabi/dev/Shareablee/RueLaLa/")
# read data
dat_rll_only = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/cleanData/dat_rll_only.csv")


'''
function to encode actiontypes:
1 - like
2 - comment
3 - login
4 - order
'''

def action_codes(x):
    if x == 'like':
        code = 1
    elif x =='comment':
        code = 2
    elif  x == 'Login':
        code = 3
    elif x == 'Order':
        code = 4
    return code

'''
function to return action type based on actioncode
'''
def map_codes(x):
    if x == 1:
        return 'like'
    elif x == 2:
        return 'comment'
    elif x == 3:
        return 'Login'
    else:
        return 'Order'

'''
function to pair up diff combinations of states
'''
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


# list of possible state pairs
coordinates = list(itertools.product([1,2,3,4], [1,2,3,4]))

def subset_data(segment):
    dat_sub = dat_rll_only.set_index([dat_rll_only.buyer_segment_cd])
    dat_sub = dat_sub.ix[[segment]]
    return dat_sub

def get_segment_states(segment):
    # temp_dat = dat_rll_only.set_index([dat_rll_only.buyer_segment_cd])
    # temp = temp_dat.ix[[segment]]

    # get data based on segment
    temp = subset_data(segment)

    temp = temp.set_index('time')[['email','actiontype','month','year','num_logins_orders','price','total_amt']]
    action_codes_seg = temp.actiontype.map(action_codes)
    temp['actiontype_codes'] = action_codes_seg.values
    temp = temp.sort_index(ascending = False)
    # quick look at data
    # print "data with encoded actions", temp.head()

    # reset index to email
    #email = tempC.groupby(['email',tempC.index.month])
    email = temp.groupby('email')
    # email: [state, state, state,...]
    user_states = email['actiontype_codes'].apply(lambda s: s.tolist()).reset_index()
    # filter by users with >1 action
    filter_rows = user_states.actiontype_codes.map(lambda x: len(x))
    filter_rows = filter_rows[filter_rows >1]
    # remove users who dont have at least 1 action
    user_states = user_states.ix[filter_rows.index]
    user_states = user_states.reset_index().drop('index', axis =1)

    # for each user update the counts for the diff states combinations
    user_pairs_list = list()
    all_user_pairs = list()

    # dictionary of user states - to update with counts by user
    states_dict = dict.fromkeys(coordinates,0)

    # fill in states dictionary for segment
    for e,i in enumerate(user_states.actiontype_codes):
        user_pairs = []
        user_pairs = pairwise(user_states.actiontype_codes[e])
        user_pairs_list = [p for p in user_pairs]
        user_counts = Counter(user_pairs_list)
        for k,v in user_counts.iteritems():
            states_dict[k]+= v

    # dataframe for states
    states_matrix = pd.DataFrame(columns=[1,2,3,4], index = [1,2,3,4], data=np.zeros((4,4)))

    # fill in states matrix for all users - ie count number of users (GROUP C) who went from state 1-1, 1-2,..
    # where states are:
    # 1 - like
    # 2 - comment
    # 3 - login
    # 4 - order

    keys = states_dict.keys()
    vals = states_dict.values()

    keys.sort()


    # fill in states matrix values
    states_matrix.ix[1][[1]] = states_dict[keys[0]]
    states_matrix.ix[1][[2]] = states_dict[keys[1]]
    states_matrix.ix[1][[3]] = states_dict[keys[2]]
    states_matrix.ix[1][[4]] = states_dict[keys[3]]
    states_matrix.ix[2][[1]] = states_dict[keys[4]]
    states_matrix.ix[2][[2]] = states_dict[keys[5]]
    states_matrix.ix[2][[3]] = states_dict[keys[6]]
    states_matrix.ix[2][[4]] = states_dict[keys[7]]
    states_matrix.ix[3][[1]] = states_dict[keys[8]]
    states_matrix.ix[3][[2]] = states_dict[keys[9]]
    states_matrix.ix[3][[3]] = states_dict[keys[10]]
    states_matrix.ix[3][[4]] = states_dict[keys[11]]
    states_matrix.ix[4][[1]] = states_dict[keys[12]]
    states_matrix.ix[4][[2]] = states_dict[keys[13]]
    states_matrix.ix[4][[3]] = states_dict[keys[14]]
    states_matrix.ix[4][[4]] = states_dict[keys[15]]

    # revert matrix column and index names back
    states_matrix.columns = states_matrix.columns.map(lambda x: map_codes(x))
    states_matrix.index = states_matrix.index.map(lambda x: map_codes(x))

    return states_matrix


'''
Function to build binomial likelihood
'''

def binomial_log_likelihood(x0):
    p = x0[0]
    ll = 0

    for row in data.values:
        i = np.array(row)
        x = np.sum(i)
        n = np.size(i) #12
        l = binom.pmf(x, n, p)
        ll += np.log(l)
    return -1 * ll




'''
function to build binomial pmf
'''
def beta_binomial_pmf(k, params):  # n, alpha, beta
    n, alpha, beta = params
    return scipy.misc.comb(n, k) * scipy.special.beta(k + alpha, n - k + beta) \
        / scipy.special.beta(alpha, beta)

'''
function to build beta binomial model
'''
def beta_binomial_log_likelihood(x0):
    alpha, beta = x0

    ll = 0

    for row in data.values:
        i = np.array(row)
        x = np.sum(i)
        n = np.size(i)
        l = beta_binomial_pmf(x, (n, alpha, beta))
        ll += np.log(l)

    return -1 * ll


'''
Function to convert counts to 0 or 1's for each user based on input segment
'''
def map_orders(segment):
    temp = subset_data(segment)
    orders = temp[temp.actiontype =='Order']
    orders = orders.set_index(pd.to_datetime(orders.time))

    orders_timeline = pd.crosstab(orders.email,  orders.index.month)
    orders_timeline = orders_timeline.reset_index()

    # convert counts to 0 or 1's for each user
    orders_binom = orders_timeline.set_index('email').applymap(lambda x: x>0).applymap(lambda x: 1 if x else 0)
    return orders_binom


'''
To run:
For example, segment C
python segment_transitions.py 'C'
'''

if __name__ == '__main__':
    print "*" * 45
    print "Begin processing."
    print "*" * 45
    input_segment = sys.argv[1]
    states_matrix = get_segment_states(input_segment)
    print "states_matrix:\n", states_matrix, '\n'

    # calculate transition matrix probabilities
    transition_matrix = states_matrix.apply(lambda row: (row / row.sum(axis=0)), axis=1)
    # format as % with 2 decimal places
    transition_matrix = transition_matrix.applymap(lambda x: "%.2f%%" % (100 * x))
    print "transition matrix of probabilities:\n" , transition_matrix

    print "*" * 45
    print "Begin modeling."
    print "*" * 45

    # set upper and lower bounds
    p_bounds = (0.000001,.999999)
    alpha_bounds  = (0.000001, None)
    beta_bounds = (0.000001, None)
    np.random.seed(1)

    # get data by segment & put into binomial form (0's and 1's)
    data = map_orders(input_segment)
    data = data.reset_index().drop('email',axis=1)

    # minimize likelihood function to get p hat binomial estimator
    binomial_res = minimize(binomial_log_likelihood, (.1,), bounds = (p_bounds,))

    # print '\n', "Binomial Model ... "
    print binomial_res, '\n'
    # get estimator of the model
    p_hat = binomial_res.x[0]

    # plot the distribution of an order at any month based on binomial model
    plot_n = 12
    bin_plot_x_range = range(0, plot_n + 1)
    import calendar

    bin_plot_y_range = [binom.pmf(x, plot_n, p_hat) for x in bin_plot_x_range]
    print bin_plot_y_range, "\n"
    print 'Binomial pmf with p_hat: %s' % p_hat, '\n'


    # plot probability distribution of an order within 12 months
    x_tick_labs = [calendar.month_abbr[i] for i in bin_plot_x_range]
    pyplot.plot(bin_plot_x_range, bin_plot_y_range)
    pyplot.xticks(bin_plot_x_range, x_tick_labs)
    pyplot.xlabel("Month")
    pyplot.title(str("Binomial distribution of an Order occuring in 12 months:\n") + str("%.2f%%" % (p_hat * 100)) )
    pyplot.show()

    # # minimize beta binomial model
    beta_binomial_res = minimize(beta_binomial_log_likelihood, (.00001, .00001),
                                 bounds = (alpha_bounds, beta_bounds))


    print '\n', "Beta Binomial Model ... "
    print beta_binomial_res

    alpha_hat = beta_binomial_res.x[0]
    beta_hat = beta_binomial_res.x[1]

    print '\n', 'Beta pmf with alpha_hat: %s, beta_hat: %s' % (alpha_hat, beta_hat), '\n'


    beta_x_range = np.linspace(0.0, 1.0, 100)
    beta_y_range = [scipy.stats.beta.pdf(x, alpha_hat, beta_hat) for x in beta_x_range]

    pyplot.plot(beta_x_range, beta_y_range)
    pyplot.title(str("Beta Binomial distribution of an Order occuring in 12 months with:\n" + "a hat " +  str(round(alpha_hat,2)) + "\nb hat " + str(round(beta_hat,2))))
    pyplot.show()

    print 'Beta Binomial pmf with E[p]: %s' % (alpha_hat / (alpha_hat + beta_hat),), '\n'



# get timeline info for each user - count number of orders by month for each user
# orders_c = tempC[tempC.actiontype =='Order']
# orders_c.index = pd.to_datetime(orders_c.index)
# orders_c_timeline = pd.crosstab(orders_c.email,  orders_c.index.month)
# orders_c_timeline = orders_c_timeline.reset_index()
# orders_c_timeline.head()









# # convert counts to 0 or 1's for each user
# order_c_binom = orders_c_timeline.set_index('email').applymap(lambda x: x>0).applymap(lambda x: 1 if x else 0)
# order_c_binom.head()






