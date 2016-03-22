import os
#%matplotlib inline
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.ticker as tkr
from matplotlib import cm
from datetime import datetime
import seaborn as sns
sns.set_style('white')



os.chdir("/Users/Gabi/dev/Shareablee/RueLaLa/")



'''
script used to visualize trends by segment
track a few average users in segments and plot their activities
'''

# read data
dat_rll_only = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/cleanData/dat_rll_only.csv")
print dat_rll_only.head()

# make sure rll data has all segment and all Nones have been replaced
seg_list = sorted(dat_rll_only.buyer_segment_cd.unique().tolist())
print "List of segments\n" , seg_list

'''
Create subsets for each user segment
Find users exhibiting behavior similar to the avg in that category
'''

def subset_user_data(segment):
    user_seg = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[[segment]].email, dat_rll_only.set_index('buyer_segment_cd').ix[[segment]].actiontype)
    return user_seg


def get_avg_actions():
    # number of actions by segment
    actions_by_seg = pd.crosstab(dat_rll_only.actiontype, dat_rll_only.buyer_segment_cd)


    # emails and segments
    d = dat_rll_only[['email','buyer_segment_cd']].drop_duplicates()
    d.index.name = 'time'
    d = d.reset_index().drop('time',axis=1)
    email_grouped = d.groupby('buyer_segment_cd').count()
    # Number of users by Segment
    seg_num_users = pd.DataFrame([i for i in email_grouped.email], index = seg_list, columns = ['num_users'])


    # avg number of actions by segment
    actions_by_seg_2 = actions_by_seg.T # number of actions by segment
    avg_num_actions_df = pd.DataFrame(columns = seg_list) # df with avg number of actions per segment
    avg_num_actions = [np.round(actions_by_seg.T.ix[[i]].values/seg_num_users.ix[[i]].values.astype(float)) for i in seg_list]  # this creates arrays w avgs
    #avg_num_actions = [actions_by_seg_2.ix[[i]].values/seg_num_users_2.ix[[i]].values for i in seg_list]  # this creates arrays w avgs

    # put avg number of actions by segment in dataframe
    for e,i in enumerate(avg_num_actions):
        tempL = [item for sublist in i for item in sublist]
        colN = seg_list[e]
        avg_num_actions_df[colN] = tempL
    avg_num_actions_df.index = ['Login','Order','comment','like']

    return avg_num_actions_df




'''
function to order nmbers based on segment
after inspecting segments these are the number of orders an avg user in each segment has
'''
def get_avg_orders(segment):
    if segment == 'A':
        orders = [i for i in xrange(1,4)]
    elif segment == 'B':
        orders = [i for i in xrange(3,6)]
    elif segment == 'C':
        orders = [i for i in xrange(6,11)]
    elif segment == 'D':
        orders = [i for i in xrange(15,19)]
    elif segment == 'F':
        orders = [i for i in xrange(26,31)]
    else:
        orders == [0]
    return orders




# ### B ###
# print "\n avg B\n"
# print avg_num_actions_df[['B']]
# print "\n Sorted values closest to avg\n"
# print users_B[(users_B.Order>0) & (users_B.like>0)].sort_values(by = ['Order','like'], ascending=[False,False])


# # avg is 2 orders in segment B - pick ppl w 4 orders
# tracking_B = users_B[(users_B.Order.isin([5, 4, 3])) & ((users_B.like >0) | (users_B.comment>0))]
# print tracking_B


# ### C ###
# print "\n avg C\n"
# print avg_num_actions_df[['C']]
# print users_C[(users_C.Order>0) & (users_C.like>0)].sort_values(by = ['comment','Order'], ascending=[False,False]).head(50)

# # avg is 6 orders in segment A - pick ppl w 9 orders since they have more likes
# tracking_C = users_C[(users_C.Order.isin([10,9,8,7,6])) & (users_C.like >0)]
# print tracking_C



# ### D ###
# print "\n avg D\n"
# print avg_num_actions_df[['D']]

# print users_D[(users_D.Order>0) & (users_D.like>0)].sort_values(by = ['Order','like'], ascending=False).head(50)

# # avg is 11 orders in segment A - pick ppl w 19
# tracking_D = users_D[(users_D.Order.isin([18,17,15])) & (users_D.like>0)]
# tracking_D


# ### F ###
# print "\n avg F\n"
# print avg_num_actions_df[['F']]

# print users_F[(users_F.Order>0) & (users_F.like>0)].sort_values(by = ['like','Order'], ascending=False).head(45)

# # avg is 22 orders in segment A - pick ppl w 32 & 40 orders
# tracking_F = users_F[(users_F.Order.isin([30,26])) & (users_F.like>0)]
# tracking_F



'''Takes a dataframe and a subsequence of its columns,
   returns dataframe with seq as first columns if "front" is True,
   and seq as last columns if "front" is False.
'''
def set_column_sequence(dataframe, seq, front=True):
	cols = seq[:] # copy so we don't mutate seq
	for c in cols:
	    if c not in dataframe.columns:
	        cols.remove(c)
	    else:
	        for x in dataframe.columns:
	            if x not in cols:
	                if front: #we want "seq" to be in the front
	                    #so append current column to the end of the list
	                    cols.append(x)
	                else:
	                    #we want "seq" to be last, so insert this
	                    #column in the front of the new column list
	                    #"cols" we are building:
	                    cols.insert(0, x)
	            else:
	                pass
	return dataframe[cols]


'''
Individual plots for users tracked by segment
'''
# '''
# PLOTTING C
# '''
# print tracking_C
# #%matplotlib inline
'''
function to get
'''
def get_seg_actions(input_segment):
    avg_num_orders = get_avg_orders(input_segment)
    tracking_users = user_dat[(user_dat.Order.isin(avg_num_orders) & (user_dat.like>0))]
    seg_data = dat_rll_only[dat_rll_only.email.isin(tracking_users.reset_index().email)]
    seg_data = seg_data.set_index('time')
    seg_data.index = pd.to_datetime(seg_data.index)
    seg_actions = pd.crosstab([seg_data.index.date, seg_data.email], seg_data.actiontype)
    seg_emails = tracking_users.index

    seg_actions = set_column_sequence(seg_actions, ['email','like','comment','Login','Order'], front = False)
    seg_actions.index.name = 'time'
    seg_actions = seg_actions.reset_index(level=1)
    #seg_actions['FB'] = seg_actions.like + seg_actions.comment

    return [seg_actions, seg_emails]

def get_user_row(segment):
    if segment == 'A':
        user_row = 3
    elif segment =='B':
        user_row = 4
    elif segment =='C':
        user_row = 8
    elif segment =='D':
        user_row = 0
    elif segment =='F':
        user_row = 1
    return user_row


def plot_tracked():
    # row 0 in group C
    seg_actions = get_seg_actions(input_segment)[0]
    seg_emails = get_seg_actions(input_segment)[1]
    user_row = get_user_row(input_segment)
    if input_segment == 'D':
        nrows = 4
    else:
        nrows = 3
    fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=nrows, sharey=False)
    #seg_actions[seg_actions.email == seg_emails[user_row]][['Login','FB','Order']].plot(subplots = True,  figsize = (8,8), ax= axes)
    seg_actions[seg_actions.email == seg_emails[user_row]].plot(subplots = True,  figsize = (8,8), ax= axes)

    for ax in fig.get_axes():
        ax.set_xlabel("")
        ax.set_ylabel("")
    pyplot.tight_layout()
    pyplot.show(fig)



# c = dat_rll_only[dat_rll_only.email.isin(tracking_C.reset_index().email)]
# c = c.set_index('time')
# c.index = pd.to_datetime(c.index)
# c_actions = pd.crosstab([c.index.date, c.email], c.actiontype)
# #d_actions.reset_index(level=1)[['email']].drop_duplicates()
# c_emails = tracking_C.index
# #d_actions.reset_index().email
# #d_actions[d_actions.index.levels[1] == e[0]]
# c_Logins = c_actions.reset_index(level=1)[['email','Login']]
# c_Orders = c_actions.reset_index(level=1)[['email','Order']]
# #c_actions = c_actions.drop('Login', axis =1)#.plot(subplots= True)
# c_actions = set_column_sequence(c_actions, ['email','like','comment','Login','Order'], front = False)
# c_actions.index.name = 'time'
# c_actions = c_actions.reset_index(level=1)



# for e,i in enumerate(c_emails):
# #    fig, (ax1, ax2) = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=1 )
#     fig, ax1 = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=3)
#     c_actions[c_actions.email == c_emails[e]].plot(subplots = True, title = str("SEGMENT C - Email: " + i), ax=ax1, figsize = (10,8))
#     #c_Logins.plot(subplots = True, title = str("SEGMENT A - Email: " + i), ax= ax2, figsize = (10,8))


#     for ax in fig.get_axes():
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#         ax.yaxis.grid(False)
#         ax.xaxis.grid(False)

#     pyplot.tight_layout()
# pyplot.show(fig)



# # row 1
# fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
# c_actions[c_actions.email == c_emails[1]].plot(subplots = True, title = str("SEGMENT C - Email: " + c_emails[1]),
#                                                figsize = (10,8), ax= axes)
# import matplotlib.ticker as tkr
# for ax in fig.get_axes():
#     ax.set_xlabel("")
#     ax.set_ylabel("")


# # row 2
# fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
# c_actions[c_actions.email == c_emails[5]].plot(subplots = True, title = str("SEGMENT C - Email: " + c_emails[5]),
#                                                figsize = (10,8), ax= axes)

# for ax in fig.get_axes():
#     ax.set_xlabel("")
#     ax.set_ylabel("")


'''
To run:
For example, segment C
python track_avg_users.py.py 'C'
'''

if __name__ == '__main__':
    print "*" * 45
    print "Begin processing."
    print "*" * 45
    input_segment = sys.argv[1]


    '''
    visually inspect how avg number of orders by segment
    Track users for each segment based on users that order close to the avg of that subgroup
    ex: segment A: avg order number = 1, track users who ordered 1,2 or 3 times
    '''

    # subset data
    user_dat = subset_user_data(input_segment)
    # get average actions for all segment
    avg_num_actions = get_avg_actions()
    # filter by by segment
    avg_num_actions = avg_num_actions[[input_segment]]

    ### A ###
    print "\n avg number of actions for group \n", input_segment, "\n"
    print avg_num_actions


    print "\n sorted values closest to avg for segment \n", input_segment, "\n"
    print "\n", user_dat.sort_values(by = ['like','Order'], ascending=False).head()

    #avg is 1 orders in segment A - pick ppl w 2-3 orders
    avg_num_orders = get_avg_orders(input_segment)
    tracking_users = user_dat[(user_dat.Order.isin(avg_num_orders) & (user_dat.like>0))]
    print "Users being tracked\n", tracking_users

    plot_tracked()




# '''
# PLOTTING A
# '''
# print tracking_A
# #%matplotlib inline
# a = dat_rll_only[dat_rll_only.email.isin(tracking_A.reset_index().email)]
# a = a.set_index('time')
# a.index = pd.to_datetime(a.index)
# a_actions = pd.crosstab([a.index.date, a.email], a.actiontype)
# #d_actions.reset_index(level=1)[['email']].drop_duplicates()
# a_emails = tracking_A.index
# #d_actions.reset_index().email
# #d_actions[d_actions.index.levels[1] == e[0]]
# a_Logins = a_actions.reset_index(level=1)[['email','Login']]
# a_Orders = a_actions.reset_index(level=1)[['email','Order']]
# #a_actions = a_actions.drop('Login', axis =1)#.plot(subplots= True)
# a_actions = set_column_sequence(a_actions, ['email','like','comment','Login','Order'], front = False)
# a_actions.index.name = 'time'
# a_actions = a_actions.reset_index(level=1)



# for e,i in enumerate(a_emails):
# #    fig, (ax1, ax2) = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=1 )
#     fig, ax1 = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=3)
#     a_actions[a_actions.email == a_emails[e]].plot(subplots = True, title = str("SEGMENT A - Email: " + i), ax=ax1, figsize = (10,8))
#     #a_Logins.plot(subplots = True, title = str("SEGMENT A - Email: " + i), ax= ax2, figsize = (10,8))


#     for ax in fig.get_axes():
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#         ax.yaxis.grid(False)
#         ax.xaxis.grid(False)

#     pyplot.tight_layout()
# pyplot.show(fig)





# '''
# PLOTTING B
# '''
# print tracking_B
# #%matplotlib inline
# b = dat_rll_only[dat_rll_only.email.isin(tracking_B.reset_index().email)]
# b = b.set_index('time')
# b.index = pd.to_datetime(b.index)
# b_actions = pd.crosstab([b.index.date, b.email], b.actiontype)
# #d_actions.reset_index(level=1)[['email']].drop_duplicates()
# b_emails = tracking_B.index
# #d_actions.reset_index().email
# #d_actions[d_actions.index.levels[1] == e[0]]
# b_Logins = b_actions.reset_index(level=1)[['email','Login']]
# b_Orders = b_actions.reset_index(level=1)[['email','Order']]
# #b_actions = b_actions.drop('Login', axis =1)#.plot(subplots= True)
# b_actions = set_column_sequence(b_actions, ['email','like','comment','Login','Order'], front = False)
# b_actions.index.name = 'time'
# b_actions = b_actions.reset_index(level=1)



# for e,i in enumerate(b_emails):
# #    fig, (ax1, ax2) = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=1 )
#     fig, ax1 = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=3)
#     b_actions[b_actions.email == b_emails[e]].plot(subplots = True, title = str("SEGMENT B - Email: " + i), ax=ax1, figsize = (10,8))
#     #b_Logins.plot(subplots = True, title = str("SEGMENT A - Email: " + i), ax= ax2, figsize = (10,8))


#     for ax in fig.get_axes():
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#         ax.yaxis.grid(False)
#         ax.xaxis.grid(False)

#     pyplot.tight_layout()
# pyplot.show(fig)






