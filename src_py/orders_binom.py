import os
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.ticker as tkr
from matplotlib import cm
import seaborn as sns
from pandas import Series
from datetime import datetime
import matplotlib.ticker as tkr
import itertools
from collections import Counter
import operator
from itertools import tee, izip

import scipy
from scipy.optimize import minimize
from scipy.stats import binom


os.chdir("/Users/Gabi/dev/Shareablee/RueLaLa/")


dat_rll_only = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/cleanData/dat_rll_only.csv")
print dat_rll_only.head()



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


#users_C = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["C"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["C"]].actiontype)
#pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["C"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["C"]].actiontype)
temp = dat_rll_only.set_index([dat_rll_only.buyer_segment_cd])
tempC = temp.ix[['C']]

tempC = tempC.set_index('time')[['email','actiontype','month','year','num_logins_orders','price','total_amt']]
c_actions_emails = pd.crosstab(tempC.email, tempC.actiontype)
c_actions_emails['fb'] =c_actions_emails.comment + c_actions_emails.like
# engage in FB, order and login
c_engagers_orders = c_actions_emails[(c_actions_emails.fb>0) & (c_actions_emails.Order>0) & (c_actions_emails.Login>0)]

# only fb engagers - no logins or orders
c_engagers_only = c_actions_emails[(c_actions_emails.fb>0) & (c_actions_emails.Order==0) & (c_actions_emails.Login==0)]

# fb and login - no orders - none
# login - no orders - none
# c_engage_login = c_actions_emails[(c_actions_emails.fb==0) & (c_actions_emails.Order==0) & (c_actions_emails.Login>0)]
# print c_engage_login.head()

# no fb - just orders
c_orders_only = c_actions_emails[(c_actions_emails.fb==0) & (c_actions_emails.Order>0)]


 # apply codes
action_codes_c = tempC.actiontype.map(action_codes)
tempC['actiontype_codes'] = action_codes_c.values
tempC = tempC.sort_index(ascending=False)
print "data with encoded actions", tempC.head()


# reset index to email
#email = tempC.groupby(['email',tempC.index.month])
email = tempC.groupby('email')
# email: [state, state, state,...]
user_states = email['actiontype_codes'].apply(lambda s: s.tolist()).reset_index()
# filter by users with >1 action
filter_rows = user_states.actiontype_codes.map(lambda x: len(x))
filter_rows = filter_rows[filter_rows >1]
# remove users who dont have at least 1 action
user_states = user_states.ix[filter_rows.index]
user_states = user_states.reset_index().drop('index', axis =1)




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
# dictionary of user states - to update with counts by user
states_dict = dict.fromkeys(coordinates,0)
states_dict


# for each user update the counts for the diff states combinations
user_pairs_list = list()
all_user_pairs = list()
for e,i in enumerate(user_states.actiontype_codes):
    user_pairs = []
    user_pairs = pairwise(user_states.actiontype_codes[e])
    user_pairs_list = [p for p in user_pairs]
    user_counts = Counter(user_pairs_list)
    for k,v in user_counts.iteritems():
        states_dict[k]+= v




# fill in states matrix for all users - ie count number of users (GROUP C) who went from state 1-1, 1-2,..
# where states are:
# 1 - like
# 2 - comment
# 3 - login
# 4 - order

keys = states_dict.keys()
vals = states_dict.values()

keys.sort()

# dataframe for states
states_matrix = pd.DataFrame(columns=[1,2,3,4], index = [1,2,3,4], data=np.zeros((4,4)))

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


print "states_matrix:\n", states_matrix


# calculate transition matrix probabilities
transition_matrix = states_matrix.apply(lambda row: row / row.sum(axis=0), axis=1)
transition_matrix

# get timeline info for each user - count number of orders by month for each user
orders_c = tempC[tempC.actiontype =='Order']
orders_c.index = pd.to_datetime(orders_c.index)
orders_c_timeline = pd.crosstab(orders_c.email,  orders_c.index.month)
orders_c_timeline = orders_c_timeline.reset_index()
orders_c_timeline.head()


# convert counts to 0 or 1's for each user
order_c_binom = orders_c_timeline.set_index('email').applymap(lambda x: x>0).applymap(lambda x: 1 if x else 0)
order_c_binom.head()



'''
Function to build binomial likelihood
'''
np.random.seed(1)
data = order_c_binom
data = data.reset_index().drop('email',axis=1)

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

p_bounds = (0.000001,.999999)
alpha_bounds  = (0.000001, None)
beta_bounds = (0.000001, None)


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

# minimize likelihood function to get p hat binomial estimator
binomial_res = minimize(binomial_log_likelihood, (.1,), bounds = (p_bounds,))

print '\n', "Binomial Model ... "
print binomial_res, '\n'
# get estimator of the model
p_hat = binomial_res.x[0]

plot_n = 12
bin_plot_x_range = range(0, plot_n + 1)
bin_plot_y_range = [binom.pmf(x, plot_n, p_hat) for x in bin_plot_x_range]
print bin_plot_y_range
print 'Binomial pmf with p_hat: %s' % p_hat, '\n'


# plot probability distribution of an order within 12 months
pyplot.plot(bin_plot_x_range, bin_plot_y_range)
pyplot.title(str("Binomial distribution of an Order occuring in 12 months:\n" + str(p_hat)))
pyplot.show()

# minimize beta binomial model
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
pyplot.title(str("Beta Binomial distribution of an Order occuring in 12 months:\n" + "with a hat, b hat" +  str(round(alpha_hat)) +  str(beta_hat)))
pyplot.show()

print 'Beta Binomial pmf with E[p]: %s' % (alpha_hat / (alpha_hat + beta_hat),), '\n'
