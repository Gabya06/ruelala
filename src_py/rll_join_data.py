# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.ticker as tkr
from matplotlib import cm
import seaborn as sns
from pandas import Series
from datetime import datetime

import itertools
from collections import Counter
import operatoror

# <codecell>

# read data
# posts of RLL customers (not necessarily on ruelala's pageid)
fb_rll_data = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/fb_rll_fixed.csv" ,
	dtype = {'fb_rll_temp.email': np.str, 'fb_rll_temp.actiontype':np.str,
	'fb_rll_temp.month': np.int64, 'fb_rll_temp.total_amt':np.int64,
	'fb_rll_temp.fb_appscopedid' : np.str})
# this only relating to posts for RUELALA - filtered out pageid
rll_fb_data = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/ruelala_fb_1192015.csv" ,
	dtype = {'fb_rll_temp.email': np.str, 'fb_rll_temp.actiontype':np.str,
	'fb_rll_temp.month': np.int64, 'fb_rll_temp.total_amt':np.int64,
	'fb_rll_temp.fb_appscopedid' : np.str})

buyer90_data = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/buyer90_filtered.csv" ,
	dtype = {'b.email': np.str, 'b.actiontype':np.str})
#	'b.month': np.int64, 'b.total_amt':np.int64})

print "FB All size\n", fb_rll_data.shape # 1000000 X 25
print "Buyer All size\n", buyer90_data.shape # 1000000 X 25
print "RLL FB size\n", rll_fb_data.shape #2808 X 25


# fix column names for fb data, buyer90 data and fb rll only data
fb_cols = fb_rll_data.columns
b_cols = buyer90_data.columns
rll_cols = rll_fb_data.columns

fb_cols = fb_cols.map(lambda x: x.replace("fb_rll_temp.",""))
rll_cols = rll_cols.map(lambda x: x.replace("rll_only_temp.",""))
b_cols = b_cols.map(lambda x: x.replace("b.",""))

fb_rll_data.columns = fb_cols
buyer90_data.columns = b_cols
rll_fb_data.columns = rll_cols

# <codecell>

print buyer90_data.columns
print fb_rll_data.columns
print rll_fb_data.columns

# <codecell>

f_dat = fb_rll_data[['email','actiontype','buyer_segment_cd','time','month','price','total_amt','num_logins_orders','user_type','fb_appscopedid',
             'spend_365','spend_lifetime','orders_lifetime','login_days_365','last_visit_spend','pref_city','pref_state','pageid','postymd']]
b_dat = buyer90_data[['email','actiontype','buyer_segment_cd','time','month','price','total_amt','num_logins_orders','user_type','fb_appscopedid',
             'spend_365','spend_lifetime','orders_lifetime','login_days_365','last_visit_spend','pref_city','pref_state','pageid','postymd']]
rll_dat = rll_fb_data[['email','actiontype','buyer_segment_cd','time','month','price','total_amt','num_logins_orders','user_type','fb_appscopedid',
             'spend_365','spend_lifetime','orders_lifetime','login_days_365','last_visit_spend','pref_city','pref_state','pageid','postymd']]


print "f dat size\n", f_dat.shape #(1000000 X 19)
print "b dat size\n", b_dat.shape #(1000000 X 19)
print "rll dat size\n", rll_dat.shape #(2808 X 19)

# <codecell>

b_dat.email.drop_duplicates().shape # 2007

# <codecell>

# remove 2013 data from buyer data
b_dat.time = pd.to_datetime(b_dat.time)
b_dat = b_dat.set_index('time')
b_dat = b_dat[b_dat.index.year != 2013]
b_dat = b_dat.reset_index()

# <codecell>

# concatenate dataframes - ignore 2013 buyer data
data_all = pd.DataFrame()
data_all = pd.concat([data_all, f_dat, b_dat])
# convert time column to datetime
data_all.time = pd.to_datetime(data_all.time)
print "combined fb all + buyer all \n", data_all.shape #(2000000, 19) wt 2013 and  (1692573,19) w.0 2013

# <codecell>

# reset index and use only dates
#data_all.time[0].map(lambda x: x.date())
data_all.time = data_all.time.map(lambda x: x.date())
data_all.time.head()

# <codecell>

# convert back to datetime - this  takes a longggg time
#data_all.time = data_all.time.map(lambda x: pd.Timestamp(x))
data_all.time = data_all.time.map(lambda x: pd.to_datetime(x))

# <codecell>

data_all = data_all.set_index(data_all.time)
data_all['year'] = data_all.index.map(lambda x:x.year)

# <codecell>

# filter out orders and logins since they dont have segment info (only likes and comments do)
users_2014 = data_all[(data_all.year == 2014) & (data_all.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
dedup_users_2014 = users_2014.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()#.to_dict()
dedup_users_2014[['email','buyer_segment_cd']].head()

# <codecell>

users_2015 = data_all[(data_all.year == 2015) & (data_all.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
dedup_users_2015 = users_2015.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()#.to_dict()

deduped_users_2014_2015 = pd.concat([dedup_users_2015, dedup_users_2014]).drop_duplicates()
deduped_users_2014_2015.head()

# <codecell>

# put all fb users in a dictionary
users_dict_2014_2015 = deduped_users_2014_2015.set_index('email').T.to_dict('records')

# <codecell>

# built dataframe for emails and buyer-segment-cd
# replace None buyer segments with segment inf
email_seg = data_all[['email','buyer_segment_cd']].reset_index(level=0)[['email','buyer_segment_cd']]
for row in email_seg.values:
    if row[1] =="None":
        row[1] = users_dict_2014_2015[0][row[0]]


# <codecell>

# data_all: fb users for all properties
# create new field called buyer_segment_cd_2 where None values are replaced by looked up segment info
data_all['buyer_segment_cd_2'] = [i for i in email_seg.buyer_segment_cd]
data_all[data_all.buyer_segment_cd=="None"][['buyer_segment_cd','buyer_segment_cd_2']].head(n=10)

# <codecell>

# check there s buyer segment in new column
data_all[data_all.buyer_segment_cd_2=="None"]

# <codecell>

# replace segment info now that it s filled in
# since data_all has additional content data this will have all segment info about rll users
data_all.buyer_segment_cd = data_all.buyer_segment_cd_2

# <codecell>

data_all[data_all.buyer_segment_cd == 'None']

# <codecell>

# drop column
data_all = data_all.drop('buyer_segment_cd_2', axis=1)

# <codecell>

data_all.columns

# <codecell>

# built dataframe for emails and buyer-segment-cd
# replace None buyer segments with segment info in Buyer_data since will also join this with RLL Fb only data
b_email_seg = b_dat[['email','buyer_segment_cd']]#.reset_index(level=0)[['email','buyer_segment_cd']]
for row in b_email_seg.values:
    if row[1] =="None":
        row[1] = users_dict_2014_2015[0][row[0]]

# <codecell>

# create new field called buyer_segment_cd_2 where None values are replaced by looked up segment info
b_dat['buyer_segment_cd_2'] = [i for i in b_email_seg.buyer_segment_cd]
b_dat[b_dat.buyer_segment_cd=="None"][['buyer_segment_cd','buyer_segment_cd_2']].head(n=10)

# <codecell>

# replace column with fixed buyer segment info
b_dat.buyer_segment_cd = b_dat.buyer_segment_cd_2

# <codecell>

# drop other col
b_dat[b_dat.buyer_segment_cd=="None"]
b_dat = b_dat.drop('buyer_segment_cd_2', axis =1)

# <codecell>

# dat_rll_only : fb users w RUELALA only + buyers -- 2014 - 2015 (get rid of 2013)
dat_rll_only = pd.DataFrame()
dat_rll_only = pd.concat([dat_rll_only, rll_dat, b_dat])

# convert time column to datetime date
dat_rll_only.time = pd.to_datetime(dat_rll_only.time)
dat_rll_only.time = dat_rll_only.time.map(lambda x: x.date())
dat_rll_only.time = dat_rll_only.time.map(lambda x: pd.to_datetime(x))
#dat_rll_only.time = dat_rll_only.time.map(lambda x: pd.Timestamp(x))
# # dat_rll_only.time = pd.to_datetime(dat_rll_only.time)
# # # set it as index to filter out 2013
dat_rll_only = dat_rll_only.set_index(dat_rll_only.time)
dat_rll_only = dat_rll_only[dat_rll_only.index.year != 2013]

# <codecell>

dat_rll_only = dat_rll_only.sort_index()
dat_rll_only = dat_rll_only.ix[datetime(2014,7,1):datetime(2015,6,30)]

# <codecell>

dat_rll_only.shape #475370 - 7/1/2014 - 6/30/2015

# <codecell>

# ******************************************************************** #
# MAKE SURE TO Filter out ruelala page_id***
rll_pageid = 22505733956
dat_rll_only[~dat_rll_only.pageid.isin(['None',rll_pageid])]

# <codecell>

dat_rll_only['year'] = dat_rll_only.index.year

# <codecell>

# make sure rll data has all segment and all Nones have been replaced
seg_list = sorted(dat_rll_only.buyer_segment_cd.unique().tolist())
print seg_list

# <codecell>

# dict of users
# {email:[login, order, comment, like]}
rll_users_counts = dat_rll_only.groupby(['email', 'actiontype'])[['actiontype']].count().unstack().fillna(0, axis = 1).T.to_dict('l')
len(rll_users_counts)#2831

# <codecell>

# dictionary of rll user only {email:segment}

rll_buyers_fb_users = dat_rll_only[(dat_rll_only.year.isin([2014,2015])) & (dat_rll_only.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
rll_deduped_users = rll_buyers_fb_users.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()
rll_users_dict = rll_deduped_users.set_index('email').T.to_dict('records')[0]
len(rll_users_dict)#860

# <codecell>

# number of actions by segment by year
actions_by_seg_year = pd.crosstab([dat_rll_only.index.year, dat_rll_only.actiontype], dat_rll_only.buyer_segment_cd)
actions_by_seg_year.index.levels[0].name = 'year'
print actions_by_seg_year
#actions_by_seg_year.reset_index().sort_values(by = 'actiontype')

pd.crosstab(dat_rll_only.buyer_segment_cd, dat_rll_only.actiontype)

# <codecell>

# number of actions by segment
actions_by_seg = pd.crosstab(dat_rll_only.actiontype, dat_rll_only.buyer_segment_cd)
actions_by_seg

# number of users in each segment - only rll users
seg_num_users_fb = pd.DataFrame([0 for i in seg_list], index = seg_list, columns = ['num_users'])
# loop thru dictionary values (email:segment)
for i in seg_list:
    seg_num_users_fb.ix[i] = sum([1 for s in rll_users_dict.values() if s == i])
print seg_num_users_fb.sum() # 860
print "\nNumber of FB users by Segment \n", seg_num_users_fb
print "\nNumber of actions by Segment \n" ,actions_by_seg

# <codecell>

d = dat_rll_only[['email','buyer_segment_cd']].drop_duplicates()
d = d.reset_index().drop('time',axis=1)
d
email_grouped = d.groupby('buyer_segment_cd').count()
seg_num_users = pd.DataFrame([i for i in email_grouped.email], index = seg_list, columns = ['num_users'])

print seg_num_users.sum() # 2831
print "Number of users by Segment\n", seg_num_users

# <codecell>

orders = dat_rll_only[dat_rll_only.actiontype =='Order']
orders.groupby([orders.index.year, orders.index.month]).count()[['actiontype']]

# <codecell>

likes_comm = dat_rll_only[dat_rll_only.actiontype.isin(['like','comment'])]
likes_comm_stats = likes_comm.groupby([likes_comm.index.year, likes_comm.index.month]).count()[['actiontype']]
total_fb = likes_comm_stats.sum().values.astype(float)[0]
print total_fb
likes_comm_stats['percent_total'] = likes_comm_stats.actiontype.map(lambda x: x/total_fb*100)
print likes_comm_stats
likes_comm_stats.ix[2014].ix[8:12].percent_total.sum()

# <codecell>

logins = dat_rll_only[dat_rll_only.actiontype.isin(['Login'])]
logins.groupby([logins.index.year, logins.index.month]).count()[['actiontype']]

# <codecell>


# <codecell>

# avg number of actions by segment

actions_by_seg_2 = actions_by_seg.T # number of actions by segment
seg_num_users_2 = seg_num_users.copy() #number of users in segment

avg_num_actions_df = pd.DataFrame(columns = seg_list) # df with avg number of actions per segment
avg_num_actions = [np.round(actions_by_seg_2.ix[[i]].values/seg_num_users.ix[[i]].values.astype(float)) for i in seg_list]  # this creates arrays w avgs
#avg_num_actions = [actions_by_seg_2.ix[[i]].values/seg_num_users_2.ix[[i]].values for i in seg_list]  # this creates arrays w avgs
print avg_num_actions

# <codecell>

# put avg number of actions by segment in dataframe
for e,i in enumerate(avg_num_actions):
    tempL = [item for sublist in i for item in sublist]
    print e , i
    colN = seg_list[e]
    avg_num_actions_df[colN] = tempL
avg_num_actions_df.index = ['Login','Order','comment','like']
avg_num_actions_df

# <codecell>


# get all users accross different groups - find users exhibiting behavior similar to the avg in that category
# emails in A
users_A = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["A"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["A"]].actiontype)
# emails in B
users_B = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["B"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["B"]].actiontype)
# emails in C
users_C = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["C"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["C"]].actiontype)

# emails in D
users_D = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["D"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["D"]].actiontype)

# emails in F
users_F = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["F"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["F"]].actiontype)

# emails in G - nothing there
users_G = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["G"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["G"]].actiontype)

# emails in L
users_L = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["L"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["L"]].actiontype)

# emails in N
users_N = pd.crosstab(dat_rll_only.set_index('buyer_segment_cd').ix[["N"]].email, dat_rll_only.set_index('buyer_segment_cd').ix[["N"]].actiontype)

# <codecell>

print "\n avg\n"
print avg_num_actions_df[['A']]

print "\n avg A\n"
print users_A.sort_values(by = ['like','Order'], ascending=False).head(35)

#avg is 14 orders in segment A - pick ppl w 2-3 orders
tracking_A = users_A[(users_A.Order.isin([3,2,1]) & (users_A.like>0))]
tracking_A

# <codecell>

print "\n avg B\n"
print avg_num_actions_df[['B']]
print "\n Sorted values closest to avg\n"
print users_B[(users_B.Order>0) & (users_B.like>0)].sort_values(by = ['Order','like'], ascending=[False,False])


# avg is 23 orders in segment B - pick ppl w 4 orders
tracking_B = users_B[(users_B.Order.isin([5, 4, 3])) & ((users_B.like >0) | (users_B.comment>0))]
print tracking_B

# <codecell>

print "\n avg C\n"
print avg_num_actions_df[['C']]
print users_C[(users_C.Order>0) & (users_C.like>0)].sort_values(by = ['comment','Order'], ascending=[False,False]).head(50)

# avg is 39 orders in segment A - pick ppl w 9 orders since they have more likes
tracking_C = users_C[(users_C.Order.isin([10,9,8,7,6])) & (users_C.like >0)]
print tracking_C

# <codecell>

print "\n avg D\n"
print avg_num_actions_df[['D']]

print users_D[(users_D.Order>0) & (users_D.like>0)].sort_values(by = ['Order','like'], ascending=False).head(50)

# avg is 17 orders in segment A - pick ppl w 19
tracking_D = users_D[(users_D.Order.isin([18,17,15])) & (users_D.like>0)]
tracking_D

# <codecell>

print "\n avg F\n"
print avg_num_actions_df[['F']]

print users_F[(users_F.Order>0) & (users_F.like>0)].sort_values(by = ['like','Order'], ascending=False).head(45)

# avg is 32 orders in segment A - pick ppl w 32 & 40 orders
tracking_F = users_F[(users_F.Order.isin([30,26])) & (users_F.like>0)]
tracking_F

# <codecell>

print "\n avg G\n"
print avg_num_actions_df[['G']]

print users_G[(users_G.Order>0)].sort_values(by = ['Order'], ascending=False).head(45)

# no likes/comments

# <codecell>

print "\n avg N\n"
print avg_num_actions_df[['N']]

print users_N.sort_values(by = 'like', ascending=False).head(15)

# 0 orders/logins

# <codecell>

def set_column_sequence(dataframe, seq, front=True):
    '''Takes a dataframe and a subsequence of its columns,
       returns dataframe with seq as first columns if "front" is True,
       and seq as last columns if "front" is False.
    '''
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

# <codecell>

print tracking_A
#%matplotlib inline
a = dat_rll_only[dat_rll_only.email.isin(tracking_A.reset_index().email)]#.groupby(dat_rll_only.index.month)[['email','actiontype']].count()
a_actions = pd.crosstab([a.index.date, a.email], a.actiontype)
#d_actions.reset_index(level=1)[['email']].drop_duplicates()
a_emails = tracking_A.index
#d_actions.reset_index().email
#d_actions[d_actions.index.levels[1] == e[0]]
a_Logins = a_actions.reset_index(level=1)[['email','Login']]
a_Orders = a_actions.reset_index(level=1)[['email','Order']]
#a_actions = a_actions.drop('Login', axis =1)#.plot(subplots= True)
a_actions = set_column_sequence(a_actions, ['email','like','comment','Login','Order'], front = False)
a_actions = a_actions.reset_index(level=1)


for e,i in enumerate(a_emails):
#    fig, (ax1, ax2) = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=1 )
    fig, ax1 = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=3)
    a_actions[a_actions.email == a_emails[e]].plot(subplots = True, title = str("SEGMENT A - Email: " + i), ax=ax1, figsize = (10,8))
    #a_Logins.plot(subplots = True, title = str("SEGMENT A - Email: " + i), ax= ax2, figsize = (10,8))


    for ax in fig.get_axes():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

    pyplot.tight_layout()

# <codecell>

# 1st person in tracking A
fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=3, sharey=False)
a_actions[a_actions.email == a_emails[2]].plot(subplots = True, title = str("SEGMENT A - Email: " + a_emails[2]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")
#pyplot.get_yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')));
#pyplot.tight_layout()

# <codecell>

# 2nd person in tracking A
fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=3, sharey=False)
a_actions[a_actions.email == a_emails[3]].plot(subplots = True, title = str("SEGMENT A - Email: " + a_emails[3]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")
#pyplot.get_yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')));
#pyplot.tight_layout()

# <codecell>

print tracking_B

b = dat_rll_only[dat_rll_only.email.isin(tracking_B.reset_index().email)]#.groupby(dat_rll_only.index.month)[['email','actiontype']].count()
b_actions = pd.crosstab([b.index.date, b.email], b.actiontype)
#d_actions.reset_index(level=1)[['email']].drop_duplicates()
b_emails = tracking_B.index
#d_actions.reset_index().email
#d_actions[d_actions.index.levels[1] == e[0]]
b_Logins = b_actions.reset_index(level=1)[['email','Login']]
#b_actions = b_actions.drop('Login', axis =1)#.plot(subplots= True)

b_actions = set_column_sequence(b_actions, ['email','like','comment','Order'], front = False)

b_actions = b_actions.reset_index(level=1)

for e,i in enumerate(b_emails):
    fig, (ax1,ax2) = pyplot.subplots(figsize = (12,8), sharex=True, ncols=1, nrows=2 )
    b_actions[b_actions.email == b_emails[e]].plot(subplots = True, title = str("SEGMENT B - Email: " + i), ax = ax1)
    b_Logins.plot(subplots = True, title = str("SEGMENT B - Email: " + i), ax=ax2)
    for ax in fig.get_axes():
        ax.set_xlabel("")
        ax.set_ylabel("")

    pyplot.tight_layout()

# <codecell>

fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
b_actions[b_actions.email == b_emails[0]].plot(subplots = True, title = str("SEGMENT B - Email: " + b_emails[0]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
b_actions[b_actions.email == b_emails[1]].plot(subplots = True, title = str("SEGMENT B - Email: " + b_emails[1]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")


# <codecell>

fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
b_actions[b_actions.email == b_emails[1]].plot(subplots = True, title = str("SEGMENT B - Email: " + b_emails[1]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
b_actions[b_actions.email == b_emails[3]].plot(subplots = True, title = str("SEGMENT B - Email: " + b_emails[2]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

print tracking_C

c = dat_rll_only[dat_rll_only.email.isin(tracking_C.reset_index().email)]#.groupby(dat_rll_only.index.month)[['email','actiontype']].count()
c_actions = pd.crosstab([c.index.date, c.email], c.actiontype)
#d_actions.reset_index(level=1)[['email']].drop_duplicates()
c_emails = tracking_C.index
#d_actions.reset_index().email
#d_actions[d_actions.index.levels[1] == e[0]]
c_Logins = c_actions.reset_index(level=1)[['email','Login']]
#b_actions = b_actions.drop('Login', axis =1)#.plot(subplots= True)

c_actions = set_column_sequence(c_actions, ['email','like','comment','Order'], front = False)

c_actions = c_actions.reset_index(level=1)

for e,i in enumerate(b_emails):
    fig, (ax1,ax2) = pyplot.subplots(figsize = (12,8), sharex=True, ncols=1, nrows=2 )
    c_actions[c_actions.email == c_emails[e]].plot(subplots = True, title = str("SEGMENT C - Email: " + i), ax = ax1)
    c_Logins.plot(subplots = True, title = str("SEGMENT C - Email: " + i), ax=ax2)
    for ax in fig.get_axes():
        ax.set_xlabel("")
        ax.set_ylabel("")

    pyplot.tight_layout()

# <codecell>

dat_b = dat_rll_only_2[dat_rll_only_2.buyer_segment_cd=='B']
b_1 = pd.crosstab(dat_b.index.date, dat_b.actiontype)
b_1.index.name = 'time'
b_p = b_1.apply(lambda x: x.cumsum())
b_p[['like','Order']].plot(subplots = True)

# <codecell>

# row 0 in group C
fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
c_actions[c_actions.email == c_emails[0]].plot(subplots = True, title = str("SEGMENT C - Email: " + c_emails[0]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

# row 1
fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
c_actions[c_actions.email == c_emails[1]].plot(subplots = True, title = str("SEGMENT C - Email: " + c_emails[1]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

# row 2
fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=2, sharey=False)
c_actions[c_actions.email == c_emails[5]].plot(subplots = True, title = str("SEGMENT C - Email: " + c_emails[5]),
                                               figsize = (10,8), ax= axes)
import matplotlib.ticker as tkr
for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

print tracking_D

d = dat_rll_only[dat_rll_only.email.isin(tracking_D.reset_index().email)]#.groupby(dat_rll_only.index.month)[['email','actiontype']].count()
d_actions = pd.crosstab([d.index.date, d.email], d.actiontype)
#d_actions.reset_index(level=1)[['email']].drop_duplicates()
d_emails = tracking_D.index
#d_actions.reset_index().email
#d_actions[d_actions.index.levels[1] == e[0]]
d_Logins = d_actions.reset_index(level=1)[['email','Login']]
d_actions = set_column_sequence(d_actions, ['email','like','comment','Order'], front = False)

#d_actions = d_actions.drop('Login', axis =1)#.plot(subplots= True)
d_actions = d_actions.reset_index(level=1)

for e,i in enumerate(d_emails):
    fig, (ax1, ax2) = pyplot.subplots(figsize = (12,8), sharex=True, ncols=1, nrows=2 )
    d_actions[d_actions.email == d_emails[e]].plot(subplots = True, title = str("SEGMENT D - Email: " + i), ax = ax1)
    #d_Logins.plot(subplots = True, title = str("SEGMENT D - Email: " + i), ax=ax2)
    pyplot.tight_layout()
    for ax in fig.get_axes():
        ax.set_xlabel("")
        ax.set_ylabel("")


# <codecell>

fig, axes = pyplot.subplots(figsize = (10,8), sharex=True, ncols=1, nrows=4, sharey=False)
d_actions[d_actions.email == d_emails[2]].plot(subplots = True, title = str("SEGMENT D - Email: " + d_emails[3]),
                                               figsize = (10,8), ax= axes)

for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")

# <codecell>

# Look at actions (fb comments + likes) throughout 2014-2015
actions = dat_rll_only.set_index('time')[['actiontype']]
actions.head()

# <codecell>

action_counts = dat_rll_only[['actiontype','num_logins_orders']]

# <codecell>

action_counts_res = pd.crosstab([action_counts.index.year, action_counts.index.month],action_counts.actiontype)
print action_counts_res

total_fb_actions = action_counts_res.comment.sum() + action_counts_res.like.sum()
action_counts_res.comment.add(action_counts_res.like).map(lambda x: x/total_fb_actions.astype(np.float)*100)

print "total", total_fb_actions
print action_counts.groupby([action_counts.index.year, action_counts.index.month, 'actiontype']).agg('count').reset_index(level=2).head()

# <codecell>


# <codecell>

# create dataframes for each year with actiontypes and #orders_logins
data_2014 = dat_rll_only[dat_rll_only.year == 2014][['actiontype','num_logins_orders']]
data_2015 = dat_rll_only[dat_rll_only.year == 2015][['actiontype','num_logins_orders']]

# <codecell>

# group by date & actiontype for each year and count number of rows for each actiontype --> counting number of users here
# reset index level 1 (action type) so have count by day of each action type per day
action_counts_2014 = data_2014.groupby([data_2014.index, 'actiontype']).agg('count')
action_counts_2014 = action_counts_2014.reset_index(level=1)

action_counts_2015 = data_2015.groupby([data_2015.index, 'actiontype']).agg('count')
action_counts_2015 = action_counts_2015.reset_index(level=1)


# <codecell>

action_counts_2014.head()

# <codecell>

# look at data for all orders
data_2014[data_2014.actiontype=='Order'].head()

# <codecell>

# verify counts are accurate in df_2013
val_counts_2014 = pd.DataFrame(data_2014[data_2014.actiontype=='Login'].index.sort_values().value_counts())
l_1 = [i.pop() for i in action_counts_2014[action_counts_2014.actiontype == 'Login'][['num_logins_orders']][1:100].values.tolist()]

# <codecell>

l_2 = val_counts_2014.reset_index().sort(columns = 'index').time[1:100].values.tolist()
l_1 == l_2
# they match

# <codecell>

# some plots of Orders 2014-2015
action_counts_2014[action_counts_2014.actiontype=='Order'].plot(figsize = (14,8))
action_counts_2015[action_counts_2015.actiontype=='Order'].plot(figsize = (14,8))

# <codecell>

# write data to csv now that its merged
dat_rll_only.to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/dat_rll_only.csv')

# <codecell>

# write orders to csv
dat_rll_only[(dat_rll_only.actiontype =='Order')].to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/rll_orders.csv')

# <codecell>

# write comments and likes to csv
dat_rll_only[(dat_rll_only.actiontype.isin(['like','comment']))].to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/rll_likes_comm.csv')

# <codecell>

# list of buyer segments
seg_list = sorted(dat_rll_only.buyer_segment_cd.unique().tolist())
print seg_list

# <codecell>

dat_rll_only.groupby(['buyer_segment_cd'])['email'].agg(['count'])

# <codecell>

# number of actions by segment
actions_by_seg = pd.crosstab(dat_rll_only.actiontype, dat_rll_only.buyer_segment_cd)
actions_by_seg

# <codecell>

avg_actions_by_seg_all = actions_by_seg_2.copy()
avg_actions_by_seg_all['num_users'] = [i for i in seg_num_users.num_users]
avg_actions_by_seg_all.Login.apply(lambda x: x/avg_actions_by_seg_all.num_users)
avg_actions_by_seg_all.Order.apply(lambda x: np.round(x/avg_actions_by_seg_all.num_users))

# <codecell>


# <codecell>

# dictionary of rll user only {email:segment}

rll_buyers_fb_users = dat_rll_only[(dat_rll_only.year.isin([2014,2015])) & (dat_rll_only.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
rll_deduped_users = rll_buyers_fb_users.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()
rll_users_dict = rll_deduped_users.set_index('email').T.to_dict('records')[0]
len(rll_users_dict)#860

# <codecell>

dat_rll_only.to_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/dat_rll_only_11152015.csv")

# <codecell>

#dat_rll_only[['email']].drop_duplicates().size #2831 total emails across fb users and rll users
d = dat_rll_only[['email','buyer_segment_cd']].drop_duplicates()
d = d.reset_index().drop('time',axis=1)
d
email_grouped = d.groupby('buyer_segment_cd').count()
seg_num_users = pd.DataFrame([i for i in email_grouped.email], index = seg_list, columns = ['num_users'])

print seg_num_users.sum() # 2831
print "Number of users by Segment\n", seg_num_users

# <codecell>

# number of users in each segment - only rll users - FB only - not all users!!
seg_num_users_fb = pd.DataFrame([0 for i in seg_list], index = seg_list, columns = ['num_users'])
# loop thru dictionary values (email:segment)
for i in seg_list:
    seg_num_users_fb.ix[i] = sum([1 for s in rll_users_dict.values() if s == i])
print seg_num_users_fb.sum() # 860
print "Number of users by Segment\n", seg_num_users_fb

print actions_by_seg

# <codecell>

# avg number of actions by segment

actions_by_seg_2 = actions_by_seg.T # number of actions by segment
seg_num_users_2 = seg_num_users.copy() #number of users in segment

avg_num_actions_df = pd.DataFrame(columns = seg_list) # df with avg number of actions per segment
avg_num_actions = [np.round(actions_by_seg_2.ix[[i]].values/seg_num_users_2.ix[[i]].values) for i in seg_list]  # this creates arrays w avgs
print avg_num_actions

# <codecell>

for e,i in enumerate(avg_num_actions):
    tempL = [item for sublist in i for item in sublist]
    print e
    colN = seg_list[e]
    avg_num_actions_df[colN] = tempL
avg_num_actions_df.index = ['Login','Order','comment','like']
avg_num_actions_df

# <codecell>

dat_users = dat_rll_only.drop('time', axis =1).reset_index()
dat_users = dat_users.set_index('buyer_segment_cd')
dat_users.ix[["A"]].shape
print dat_rll_only.pageid.unique()
#pd.crosstab(dat_users.ix[["B"]].email,

# <codecell>

rll_users_counts['abbstr101@aol.com']

# <codecell>

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





#daily_C
#daily_C.sort_index()
#daily_C[(daily_C.fb >0) & (daily_C.Order>0)]
#daily_C.groupby(daily_C.index.month).sum() #
# print tempC.shape
# print tempC.actiontype.value_counts()
# # 'actions_order' - actions and orders

## separate pop into users who order and like

#c_user_actions = pd.DataFrame(index = tempC.index, columns ={'actions_order' = [1 for i in tempC if tempC.actiontype =='Login'}

# <codecell>

#function to encode actiontypes
## encode categorization of states:
# like, comment, login, order

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

# <codecell>

 # apply codes
action_codes_c = tempC.actiontype.map(action_codes)
tempC['actiontype_codes'] = action_codes_c.values
tempC = tempC.sort_index(ascending=False)

# <codecell>

tempC.head()

# <codecell>

# reset index to email
#email = tempC.groupby(['email',tempC.index.month])
email = tempC.groupby('email')

# <codecell>

# email: [state, state, state,...]
user_states = email['actiontype_codes'].apply(lambda s: s.tolist()).reset_index()
# in case need this in a dict#
#user_states_dict = email['actiontype_codes'].apply(lambda s: s.tolist()).to_dict()

# <codecell>

# filter by users with >1 action
filter_rows = user_states.actiontype_codes.map(lambda x: len(x))
filter_rows = filter_rows[filter_rows >1]
# remove users who dont have at least 1 action
user_states = user_states.ix[filter_rows.index]
user_states = user_states.reset_index().drop('index', axis =1)

# <codecell>

from itertools import tee, izip

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

# <codecell>

l = [1, 2, 3, 4, 5, 6]
print [pair for pair in pairwise(l)]

# <codecell>



#l = user_states.actiontype_codes
#print l
user_pairs_list = list()
all_user_pairs = list()
for e,i in enumerate(user_states.actiontype_codes):
    user_pairs = []
#    print i
    user_pairs = pairwise(user_states.actiontype_codes[e])
    user_pairs_list = [p for p in user_pairs]
    user_counts = Counter(user_pairs_list)
    for k,v in user_counts.iteritems():
        states_dict[k]+= v

#     for pair in pairwise(i):

# #        user_pairs = user_pairs.append(pair[i])
#         print user_pairs
#     all_user_pairs.append(user_pairs)

# <codecell>


# list of possible state pairs
coordinates = list(itertools.product([1,2,3,4], [1,2,3,4]))
# dictionary of user states - to update with counts by user
states_dict = dict.fromkeys(coordinates,0)
states_dict



# <codecell>

print states_dict, "\n"
sorted_dict = dict(sorted(states_dict.items(), key=operator.itemgetter(0)))
sorted_dict

# <codecell>

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

# <codecell>

states_matrix

# <codecell>

#states_matrix.apply(lambda x: x/sum(x))
# calculate transition matrix probabilities
transition_matrix = states_matrix.apply(lambda row: row / row.sum(axis=0), axis=1)
transition_matrix

# <codecell>

# TO DO: visualize using network x

# <codecell>

# get timeline info for each user - count number of orders by month for each user
orders_c = tempC[tempC.actiontype =='Order']
orders_c_timeline = pd.crosstab(orders_c.email,  orders_c.index.month)
orders_c_timeline = orders_c_timeline.reset_index()
orders_c_timeline.head()

# <codecell>

# convert counts to 0 or 1's for each user
order_c_binom = orders_c_timeline.set_index('email').applymap(lambda x: x>0).applymap(lambda x: 1 if x else 0)
order_c_binom.head()

# <codecell>

L=p**x*(1-p)**(1-x)

# <codecell>

zip(user_states[1:]
for (x,y), c in Counter(zip(a, a[1:])).iteritems():
    b[x-1,y-1] = c
print b

# <codecell>

# pd.DataFrame(index = tempC.index.drop_duplicates(), columns = {'codes':[[code for code in tempC.actiontype_codes] for email in
#                                                                tempC.index.drop_duplicates()]]

#             tempL = [item for sublist in i for item in sublist]


# <codecell>

pd.crosstab(dat_rll_only.actiontype, dat_rll_only.buyer_segment_cd, aggfunc='count').apply(lambda r: np.round(r/r.sum()*100), axis=1)

# <codecell>

# number of logins/comments/likes by buyer segment - rll fb users only
dat_rll_only.groupby(['buyer_segment_cd', 'actiontype'])['email'].agg(['count']).to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/seg_stats_2014_2015.csv')

# <codecell>

dat_rll_only.groupby([dat_rll_only.index.year, dat_rll_only.index.month, 'actiontype'])['email'].agg(['count']).to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/action_stats_2014_2015.csv')

# <codecell>

users_C.head()

# <codecell>

# 2014 & 2015 stats only - include !=2013

# list of unique users with segment info and num_logins_orders
orders_logs = dat_rll_only[(dat_rll_only.actiontype.isin(['Order','Login'])) &(dat_rll_only.index.year!=2013) ]
orders_logs = orders_logs.drop('time',axis=1) #have as index

orders_logs = orders_logs.reset_index()
orders_logs = orders_logs.set_index('actiontype')
# count of logins by segment
total_logs_grouped = orders_logs.ix['Login'][['buyer_segment_cd','num_logins_orders']].groupby(['buyer_segment_cd']).agg(['count'])
log_unique_users_grouped = orders_logs.ix['Login'][['buyer_segment_cd','num_logins_orders']].drop_duplicates().groupby(['buyer_segment_cd']).agg(['count'])


logs_stats = pd.DataFrame({'n_users_logged': [i[0] for i in log_unique_users_grouped.values],
              'n_logins': [i[0] for i in total_logs_grouped.values]}, index = log_unique_users_grouped.index)

#logs_stats
# count = # of total orders
# sum = sum of total price paid
# mean - avg price paid by order
total_orders_grouped = orders_logs.ix['Order'][['buyer_segment_cd','price']].groupby(['buyer_segment_cd']).agg(['count','sum','mean'])
# sum = sum of total_amt by user (this matches sum of total price paid)
# count = count of unique users
# mean = avg total amt paid by user
orders_unique_users_grouped  = orders_logs.ix['Order'][['buyer_segment_cd','email','total_amt']].drop_duplicates().groupby(['buyer_segment_cd']).agg(['sum', 'count', 'mean'])


# print total_orders_grouped
# print orders_unique_users_grouped

#print [i[1] for i in orders_unique_users_grouped.values]
orders_stats = pd.DataFrame({'n_users_ordered': [i[1] for i in orders_unique_users_grouped.values],
              'n_orders': [i[0] for i in total_orders_grouped.values],
              'total_price_paid': [i[0] for i in orders_unique_users_grouped.values],
              'avg_user_paid': [i[2] for i in orders_unique_users_grouped.values],
              'avg_order_price': [i[2] for i in total_orders_grouped.values]
              },
             index = orders_unique_users_grouped.index)

print "Logins stats\n", logs_stats
print "Orders stats\n", orders_stats

#pd.concat([logs_stats, orders_stats]).plot(kind = 'bar', subplots=True, figsize = (12,7))

pd.merge(logs_stats, orders_stats, right_index=True, left_index=True).plot(subplots = True)

# <codecell>

orders_stats

# <codecell>


# <codecell>

# 2014-2015
orders_stats

# <codecell>


# <codecell>

orders_stats['percent_total_orders'] = orders_stats.n_orders/orders_stats.n_orders.sum()

# <codecell>

orders_stats

# <codecell>

logs_stats['avg_login_user'] = logs_stats.n_users_logged/(logs_stats.n_logins)

# <codecell>

logs_stats

# <codecell>

# list of unique users with segment info and num_logins_orders
fb_eng = dat_rll_only[(dat_rll_only.actiontype.isin(['like','comment']))]
fb_eng = fb_eng.drop('time',axis=1)
fb_eng = fb_eng.reset_index()
fb_eng = fb_eng.set_index('actiontype')

# <codecell>

likes_grouped = fb_eng.ix['like'][['buyer_segment_cd','spend_365']].groupby(['buyer_segment_cd']).agg(['count'])
comm_grouped = fb_eng.ix['comment'][['buyer_segment_cd','spend_365']].groupby(['buyer_segment_cd']).agg(['count'])

# spend365_sum: sum of all spend 365 by groups, need to filter out duplicate emails since spend365 is repeated multiple times per like/comment
spend365_sum = fb_eng.reset_index()[['buyer_segment_cd','spend_365']].drop_duplicates().groupby(['buyer_segment_cd']).agg(['sum'])


fb_stats =  pd.DataFrame({'n_likes': [i[0] for i in likes_grouped.values],
                    'n_comments': [i[0] for i in comm_grouped.values],
                    'total_eng': fb_stats.n_likes + fb_stats.n_comments,
              'spend_365':[i[0] for i in spend365_sum.values]}, index=likes_grouped.index)

fb_stats

# print dat_rll_only[(dat_rll_only.buyer_segment_cd =='F') & (dat_rll_only.actiontype.isin(['like','comment']))][['email','spend_365']].drop_duplicates().sum()


# <codecell>


# 2014 & 2015 stats only
fb_eng_14_5 =  fb_eng.reset_index()
fb_eng_14_5 = fb_eng_14_5.set_index('time')
fb_eng_14_5 = fb_eng_14_5[fb_eng_14_5.index.year != 2013]
fb_eng_14_5 = fb_eng_14_5.reset_index()
fb_eng_14_5 = fb_eng_14_5.set_index('actiontype')

#fb_eng[fb_eng.set_index('time').index.year != 2013]
likes_grouped = fb_eng_14_5.ix['like'][['buyer_segment_cd','spend_365']].groupby(['buyer_segment_cd']).agg(['count'])
comm_grouped = fb_eng_14_5.ix['comment'][['buyer_segment_cd','spend_365']].groupby(['buyer_segment_cd']).agg(['count'])

# spend365_sum: sum of all spend 365 by groups, need to filter out duplicate emails since spend365 is repeated multiple times per like/comment
spend365_sum = fb_eng_14_5.reset_index()[['buyer_segment_cd','spend_365']].drop_duplicates().groupby(['buyer_segment_cd']).agg(['sum'])

fb_stats =  pd.DataFrame({'n_likes': [i[0] for i in likes_grouped.values],
                    'n_comments': [i[0] for i in comm_grouped.values],
                    'total_eng': fb_stats.n_likes + fb_stats.n_comments,
              'spend_365':[i[0] for i in spend365_sum.values]}, index=likes_grouped.index)

fb_stats

# <codecell>


# count number of likes and comments by group for 2014 - 2015

fb_eng_ts = dat_rll_only[(dat_rll_only.actiontype.isin(['like','comment']))]
fb_eng_ts = fb_eng_ts.drop('time',axis=1)
#fb_eng_ts_14 = fb_eng_ts[fb_eng_ts.index.year ==2014]
#fb_eng_ts_14 = fb_eng_ts_14.reset_index()
fb_eng_ts_14_15 = fb_eng_ts.copy()
fb_eng_ts_14_15 = fb_eng_ts_14_15.set_index(fb_eng_ts_14_15.index.date)

# quick  plot of engagement counts across 2014 - 2015
fb_eng_ts_14_15.groupby([fb_eng_ts_14_15.index]).email.count().plot(subplots=True, title = '2014 likes and comments')

# FB likes Comments index
days_ts_14_15 = pd.to_datetime(fb_eng_ts_14_15.index.drop_duplicates())
days_ts_14_15 = days_ts_14_15.sort_values()
# df for likes
df_likes_14_15 = pd.DataFrame(index= days_ts_14_15, columns = seg_list)


# df for comments
df_comm_14_15 = pd.DataFrame(index= days_ts_14_15, columns = seg_list)

# build dataframes with buyer_seg columns and # of likes per day
# build dataframes with buyer_seg columns and # of comments per day
for e, seg in enumerate(df_likes_14_15.columns):
    # index of likes for that segment
    seg_ts = fb_eng_ts_14_15[(fb_eng_ts_14_15.actiontype == 'like') & (fb_eng_ts_14_15.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    # temp df for that segment with 1 for each like
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    # sum by date
    temp_g = temp.groupby(temp.index).sum()
    # merge by date and keep only y columns
    df_likes_14_15 = pd.merge(left = df_likes_14_15, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_likes_14_15 = df_likes_14_15.drop(str(seg+"_x"), axis =1)

    # do the same for comments
    seg_ts = fb_eng_ts_14_15[(fb_eng_ts_14_15.actiontype == 'comment') & (fb_eng_ts_14_15.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    temp_g = temp.groupby(temp.index).sum()
    df_comm_14_15 = pd.merge(left = df_comm_14_15, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_comm_14_15 = df_comm_14_15.drop(str(seg+"_x"), axis =1)

# Likes
# rename colums and fill na as 0
df_likes_14_15.columns = df_likes_14_15.columns.map(lambda x:x.replace("_y",""))
df_likes_14_15 = df_likes_14_15.fillna(0)
# comments
df_comm_14_15.columns = df_comm_14_15.columns.map(lambda x:x.replace("_y",""))
df_comm_14_15 = df_comm_14_15.fillna(0)


# <codecell>

# write to csv files
df_likes_14_15.to_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/likes_14_15.csv")
df_comm_14_15.to_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/comm_14_15.csv")

# <codecell>

# orders and logins for 2014-2015

orders_logs_ts = dat_rll_only[(dat_rll_only.actiontype.isin(['Order','Login']))]
orders_logs_ts = orders_logs_ts.drop('time',axis=1)
# orders_logs_ts_14 = orders_logs_ts[orders_logs_ts.index.year ==2014]
#fb_eng_ts_14 = fb_eng_ts_14.reset_index()
orders_logs_ts_14_15 = orders_logs_ts
orders_logs_ts_14_15 = orders_logs_ts_14_15.set_index(orders_logs_ts_14_15.index.date)
# plot of orders and logins across 2014-2015
orders_logs_ts_14_15.groupby([orders_logs_ts_14_15.index]).email.count().plot(subplots=True, title = '2013 - 2015 orders and logins')

# Logins and ORDERS - these start at 1/1/2014 = will filter out values so they match fb
days_ts_ol = pd.to_datetime(orders_logs_ts_14_15.index.drop_duplicates())
days_ts_ol = days_ts_ol.sort_values()
# df for likes
df_orders_14_15 = pd.DataFrame(index= days_ts_ol, columns = seg_list)
df_orders_14_15.head()

# df for comments
df_logins_14_15 = pd.DataFrame(index= days_ts_ol, columns = seg_list)


# build dataframes with buyer_seg columns and # of orders per day
# build dataframes with buyer_seg columns and # of logins per day
for e, seg in enumerate(df_orders.columns):
    # index of orders for that segment
    seg_ts = orders_logs_ts_14_15[(orders_logs_ts_14_15.actiontype == 'Order') & (orders_logs_ts_14_15.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    # temp df for that segment with 1 for each order
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    # sum by date
    temp_g = temp.groupby(temp.index).sum()
    # merge by date and keep only y columns
    df_orders_14_15 = pd.merge(left = df_orders_14_15, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_orders_14_15 = df_orders_14_15.drop(str(seg+"_x"), axis =1)

    # do the same for logins
    seg_ts = orders_logs_ts_14_15[(orders_logs_ts_14_15.actiontype == 'Login') & (orders_logs_ts_14_15.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    temp_g = temp.groupby(temp.index).sum()
    df_logins_14_15 = pd.merge(left = df_logins_14_15, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_logins_14_15 = df_logins_14_15.drop(str(seg+"_x"), axis =1)

# orders
# rename colums and fill na as 0
df_orders_14_15.columns = df_orders_14_15.columns.map(lambda x:x.replace("_y",""))
df_orders_14_15 = df_orders_14_15.fillna(0)
# logins
df_logins_14_15.columns = df_logins_14_15.columns.map(lambda x:x.replace("_y",""))
df_logins_14_15 = df_logins_14_15.fillna(0)

print df_orders_14_15.head()
print df_logins_14_15.head()

# <codecell>

print df_comm[df_comm.index.year == 2014].A.sum()
print df_comm_14_15[df_comm_14_15.index.year == 2014].A.sum()

# <codecell>

fb_eng_ts = dat_rll_only[(dat_rll_only.actiontype.isin(['like','comment']))]
fb_eng_ts = fb_eng_ts.drop('time',axis=1)
fb_eng_ts_14 = fb_eng_ts[fb_eng_ts.index.year ==2014]
#fb_eng_ts_14 = fb_eng_ts_14.reset_index()
fb_eng_ts_14 = fb_eng_ts_14.set_index(fb_eng_ts_14.index.date)
fb_eng_ts_14.groupby([fb_eng_ts_14.index]).email.count().plot(subplots=True, title = '2014 likes and comments')

# <codecell>

orders_logs_ts = dat_rll_only[(dat_rll_only.actiontype.isin(['Order','Login']))]
orders_logs_ts = orders_logs_ts.drop('time',axis=1)
orders_logs_ts_14 = orders_logs_ts[orders_logs_ts.index.year ==2014]
#fb_eng_ts_14 = fb_eng_ts_14.reset_index()
orders_logs_ts_14 = orders_logs_ts_14.set_index(orders_logs_ts_14.index.date)
orders_logs_ts_14.groupby([orders_logs_ts_14.index]).email.count().plot(subplots=True, title = '2014 orders and logins')

# <codecell>

# FB likes Comments index
days_ts = pd.to_datetime(fb_eng_ts_14.index.drop_duplicates())
days_ts = days_ts.sort_values()
# df for likes
df_likes = pd.DataFrame(index= days_ts, columns = seg_list)
print df_likes.head()

# df for comments
df_comm = pd.DataFrame(index= days_ts, columns = seg_list)

# <codecell>

# when fb starts
fb_start = days_ts[0].date()
fb_start

# <codecell>

print df_likes.head(5)

# <codecell>

# Logins and ORDERS - these start at 1/1/2014 = will filter out values so they match fb
days_ts_ol = pd.to_datetime(orders_logs_ts_14.index.drop_duplicates())
days_ts_ol = days_ts_ol.sort_values()
# df for likes
df_orders = pd.DataFrame(index= days_ts_ol, columns = seg_list)
df_orders.head()

# df for comments
df_logins = pd.DataFrame(index= days_ts_ol, columns = seg_list)

# <codecell>


# # This would be using full index of 2014 - no comments or likes that far back, only july onward
# ix_2014 = dat_rll_only[dat_rll_only.index.year == 2014].index.drop_duplicates()
# ix_2014 = ix_2014.sort_values()

# # df for likes
# df_likes = pd.DataFrame(index= ix_2014, columns = seg_list)
# print df_likes.head()

# # df for comments
# df_comm = pd.DataFrame(index= ix_2014, columns = seg_list)


# <codecell>

# build dataframes with buyer_seg columns and # of likes per day
# build dataframes with buyer_seg columns and # of comments per day
for e, seg in enumerate(df_likes.columns):
    # index of likes for that segment
    seg_ts = fb_eng_ts_14[(fb_eng_ts_14.actiontype == 'like') & (fb_eng_ts_14.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    # temp df for that segment with 1 for each like
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    # sum by date
    temp_g = temp.groupby(temp.index).sum()
    # merge by date and keep only y columns
    df_likes = pd.merge(left = df_likes, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_likes = df_likes.drop(str(seg+"_x"), axis =1)

    # do the same for comments
    seg_ts = fb_eng_ts_14[(fb_eng_ts_14.actiontype == 'comment') & (fb_eng_ts_14.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    temp_g = temp.groupby(temp.index).sum()
    df_comm = pd.merge(left = df_comm, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_comm = df_comm.drop(str(seg+"_x"), axis =1)

print df_comm.head()

# <codecell>

# build dataframes with buyer_seg columns and # of orders per day
# build dataframes with buyer_seg columns and # of logins per day
for e, seg in enumerate(df_orders.columns):
    # index of orders for that segment
    seg_ts = orders_logs_ts_14[(orders_logs_ts_14.actiontype == 'Order') & (orders_logs_ts_14.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    # temp df for that segment with 1 for each order
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    # sum by date
    temp_g = temp.groupby(temp.index).sum()
    # merge by date and keep only y columns
    df_orders = pd.merge(left = df_orders, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_orders = df_orders.drop(str(seg+"_x"), axis =1)

    # do the same for logins
    seg_ts = orders_logs_ts_14[(orders_logs_ts_14.actiontype == 'Login') & (orders_logs_ts_14.buyer_segment_cd == seg)].index
    seg_ts = pd.to_datetime(seg_ts)
    temp = pd.DataFrame(index = seg_ts, columns= [seg], data= [1 for i in xrange(len(seg_ts))])
    temp_g = temp.groupby(temp.index).sum()
    df_logins = pd.merge(left = df_logins, right = temp_g, left_index=True, right_index=True, how = 'outer', sort = False)
    df_logins = df_logins.drop(str(seg+"_x"), axis =1)

print df_orders.head()
print df_logins.head()

# <codecell>

df_orders.head()

# <codecell>

# Likes
# rename colums and fill na as 0
df_likes.columns = df_likes.columns.map(lambda x:x.replace("_y",""))
df_likes = df_likes.fillna(0)
print df_likes.head()
print df_likes.ix[1].sum()
fb_eng_ts_14[fb_eng_ts_14.actiontype == 'like'].ix[datetime(2014,7,3).date()].actiontype.count()

# <codecell>

# Comments
df_comm.columns = df_comm.columns.map(lambda x:x.replace("_y",""))
df_comm = df_comm.fillna(0)
print df_comm.head()
print df_comm.ix[1].sum()
fb_eng_ts_14[fb_eng_ts_14.actiontype == 'comment'].ix[datetime(2014,7,3).date()].actiontype.count()

# <codecell>

# where fb comments & likes start in the orders index
fb_start_ix = df_orders.index.searchsorted(fb_start)
fb_start_ix

# <codecell>

last_day = df_orders.index.sort_values()[-1].date()
# get index position of last day
last_day_ix = df_orders.index.searchsorted(last_day)
last_day_ix

# <codecell>

# Orders
# rename colums and fill na as 0
df_orders.columns = df_orders.columns.map(lambda x:x.replace("_y",""))
df_orders = df_orders.fillna(0)
print df_orders.ix[fb_start_ix: fb_start_ix+5]
print df_orders.ix[fb_start_ix].sum()
orders_logs_ts_14[orders_logs_ts_14.actiontype == 'Order'].ix[datetime(2014,7,5).date()].actiontype.count()

# <codecell>

# Logins
# rename colums and fill na as 0
df_logins.columns = df_logins.columns.map(lambda x:x.replace("_y",""))
df_logins = df_logins.fillna(0)
print df_logins.ix[fb_start_ix:fb_start_ix+4]
print df_logins.ix[fb_start_ix+4].sum()
orders_logs_ts_14[orders_logs_ts_14.actiontype == 'Login'].ix[datetime(2014,7,5).date()].actiontype.count()

# <codecell>

last_day = df_orders_14_15.index.sort_values()[-1].date()
# get index position of last day
last_day_ix = df_orders_14_15.index.searchsorted(last_day)
last_day_ix

fb_start
last_day

# <codecell>

# where fb comments & likes start in the orders index
fb_start_ix_14_15 = df_orders_14_15.index.searchsorted(fb_start)
last_day_14_15 = df_orders_14_15.index.sort_values()[-1].date()
# get index position of last day
last_day_ix_14_15 = df_orders_14_15.index.searchsorted(last_day)
last_day_ix_14_15

# <codecell>

# ******************************************** #
# filter out any orders & logins before 7/1/2014
# ******************************************** #
# since starts at 0 make sure to include last day
df_orders_2 = df_orders_14_15.ix[fb_start_ix_14_15 : last_day_ix_14_15+1]
df_logins_2 = df_logins_14_15.ix[fb_start_ix_14_15 : last_day_ix_14_15+1]

# <codecell>

f, ax = pyplot.subplots()
df_orders_2.A.plot(ax = ax)
x_labels = pd.Series([l.strftime('%b-%y') for l in df_orders_2.index])
x_labels = x_labels.drop_duplicates().tolist()
ax.set_xticklabels(labels = x_labels)
pyplot.tight_layout()
#pyplot.xticks([i for i in x_labels])
#str.format(df_orders_2.index, '%y%m')
#pyplot.xticks(['July2014','Aug2014','Sept2014','Oct2014','Nov2014','Dec2014])

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.A.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.A.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.A.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.A.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.B.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.B.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.B.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.B.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.C.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.C.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.C.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.C.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.D.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.D.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.D.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.D.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.F.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.F.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.F.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.F.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.G.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.G.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.G.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.G.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.L.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.L.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.L.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.L.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

palette = sns.color_palette("Set1", 4)
f, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(figsize = (18,6), ncols=4)
df_logins_2.N.plot(ax = ax_1, title = 'logins', color = palette[0])
df_likes.N.plot(ax=ax_2, title = 'likes', color = palette[1])
df_comm.N.plot(ax=ax_3, title = 'comments', color = palette[2])
df_orders_2.N.plot(ax = ax_4, title = 'orders',color = palette[3])
axes_l = [ax_1, ax_2, ax_3, ax_4]
for a in axes_l:
    a.set_xticklabels(labels = x_labels, rotation = 45)
pyplot.tight_layout()

# <codecell>

df_comm_14_15.A.plot()

# <codecell>

df_orders_2.plot(subplots=True)
df_likes_14_15.plot(subplots=True)
pyplot.tight_layout()

# <codecell>

f, ax = pyplot.subplots(figsize=(12, 8))
#palette= sns.color_palette()
palette = sns.color_palette("Set1", 9)
df_likes.plot(subplots=True, ax = ax, title ='likes', color = palette )
pyplot.show(f)
df_comm.plot(subplots=True, figsize=(12,8), title = 'comments')

# <codecell>


# <codecell>

fb_monthly_avg = fb_eng_ts_14_15.copy()
fb_monthly_avg.index.name = 'time'
fb_monthly_avg.index = pd.to_datetime(fb_monthly_avg.index)
#fb_monthly_avg = fb_monthly_avg.reset_index()
fb_monthly_avg = fb_monthly_avg.groupby([fb_monthly_avg.index.year, fb_monthly_avg.index.month,'buyer_segment_cd','actiontype'])
fb_monthly_avg.agg(['count'])
#fb_eng_ts_14_15.groupby([fb_eng_ts_14_15.index]).email.count()

# <codecell>


# <codecell>

orders_logs = orders_logs.drop('time',axis=1) #have as index

orders_logs = orders_logs.reset_index()
orders_logs = orders_logs.set_index('actiontype')
# count of logins by segment
total_logs_grouped = orders_logs.ix['Login'][['buyer_segment_cd','num_logins_orders']].groupby(['buyer_segment_cd']).agg(['count'])
log_unique_users_grouped = orders_logs.ix['Login'][['buyer_segment_cd','num_logins_orders']].drop_duplicates().groupby(['buyer_segment_cd']).agg(['count'])


logs_stats = pd.DataFrame({'n_users_logged': [i[0] for i in log_unique_users_grouped.values],
              'n_logins': [i[0] for i in total_logs_grouped.values]}, index = log_unique_users_grouped.index)

#logs_stats
# count = # of total orders
# sum = sum of total price paid
# mean - avg price paid by order
total_orders_grouped = orders_logs.ix['Order'][['buyer_segment_cd','price']].groupby(['buyer_segment_cd']).agg(['count','sum','mean'])
# sum = sum of total_amt by user (this matches sum of total price paid)
# count = count of unique users
# mean = avg total amt paid by user
orders_unique_users_grouped  = orders_logs.ix['Order'][['buyer_segment_cd','email','total_amt']].drop_duplicates().groupby(['buyer_segment_cd']).agg(['sum', 'count', 'mean'])


# print total_orders_grouped
# print orders_unique_users_grouped

#print [i[1] for i in orders_unique_users_grouped.values]
orders_stats = pd.DataFrame({'n_users_ordered': [i[1] for i in orders_unique_users_grouped.values],
              'n_orders': [i[0] for i in total_orders_grouped.values],
              'total_price_paid': [i[0] for i in orders_unique_users_grouped.values],
              'avg_user_paid': [i[2] for i in orders_unique_users_grouped.values],
              'avg_order_price': [i[2] for i in total_orders_grouped.values]
              },
             index = orders_unique_users_grouped.index)

print "Logins stats\n", logs_stats
print "Orders stats\n", orders_stats

#pd.concat([logs_stats, orders_stats]).plot(kind = 'bar', subplots=True, figsize = (12,7))

pd.merge(logs_stats, orders_stats, right_index=True, left_index=True).plot(subplots = True)

# <codecell>

b_dat[(b_dat.actiontype == 'Login')& (b_dat.buyer_segment_cd=='G')].count()

# <codecell>

users_seg_sum_orders[users_seg_sum_orders.actiontype =='Login'].groupby('buyer_segment_cd').get_group('G')

# <codecell>

orders_all.groupby(['buyer_segment_cd','actiontype'])[['price']].agg(['sum','mean','count'])

orders_all.groupby(['buyer_segment_cd','actiontype']).get_group(('G','Order'))#agg(['sum','mean','count'])

# <codecell>

orders_logs.groupby(['buyer_segment_cd','actiontype'])[['price','spend_365']].agg(['sum','mean','count'])

# <codecell>

orders_all.groupby(['buyer_segment_cd','actiontype'])[['price','spend_365','num_logins_orders']].agg(['sum','mean']).to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/seg_orders_summary.csv')

# <codecell>

# number of logins/comments/likes by buyer segment - all fb users only
data_all.groupby(['buyer_segment_cd', 'actiontype'])['email'].agg(['count']).to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/fb_all_seg_stats.csv')

# <codecell>

data_all.groupby(['buyer_segment_cd', 'actiontype'])['email'].agg(['count'])

# <codecell>

dat_rll_only.groupby(['buyer_segment_cd', 'actiontype'])['email'].agg('count')

# <codecell>

dat_rll_only.groupby(['buyer_segment_cd', dat_rll_only.index.year, 'actiontype'])['email'].agg(['count']).to_csv('/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/seg_year_stats.csv')

# <codecell>

orders_all[['buyer_segment_cd','num_logins_orders']].groupby([orders_all.index.year, 'buyer_segment_cd']).agg('count')

# <codecell>

seg_stats = dat_rll_only.groupby(['buyer_segment_cd', 'actiontype'])['email'].agg(['count']).reset_index(level=1)
print seg_stats[seg_stats.actiontype == 'Order']
print seg_stats[seg_stats.actiontype == 'Order'].describe()

# <codecell>

dat_rll_only.groupby([dat_rll_only.index.year,dat_rll_only.index.month,'buyer_segment_cd',
                      'actiontype'])['email'].agg(['count']).to_csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/out_files/seg_year_month.csv")

# <codecell>

dat_rll_only.groupby([dat_rll_only.index.year, 'actiontype',dat_rll_only.index.month,'buyer_segment_cd'
                      ])['email'].agg(['count']).ix[2013].ix['Order'].reset_index(level=0)[['count']].plot(kind = 'bar', figsize=(12,8),
                                                                                                           color = ['red','green','blue','green','yellow','orange'],
                                                                                                           stacked = True)

# <codecell>

dat_rll_only.groupby([dat_rll_only.index.year, 'actiontype',dat_rll_only.index.month,'buyer_segment_cd'
                      ])['email'].agg(['count']).ix[2014].ix['Order'].reset_index(level=0)[['count']].plot(kind = 'bar', figsize=(12,8),
                                                                                                           color = ['red','green','blue','green','yellow','orange'],
                                                                                                           stacked = True)

# <codecell>

t = dat_rll_only.set_index(dat_rll_only.index.dayofweek)
t = t[t.year == 2014]
t.index.name = 'dayofweek'

t = t.reset_index()
t.set_index(t.time)

gp = t.groupby(['buyer_segment_cd'])[['actiontype','dayofweek']]
gp.get_group('A')
# for g in gp.groups:
#     print gp.count().plot()



#t.groupby(['buyer_segment_cd','dayofweek'])[['actiontype',]].agg('count').plot()
#dat_rll_only[dat_rll_only.year==2014][['email','actiontype']]#.groupby(['email']).count()#.plot()

# <codecell>

orders_all[['buyer_segment_cd','num_logins_orders']].groupby([orders_all.index.year, 'buyer_segment_cd']).agg('count').plot(kind='bar',subplots = True)

# <codecell>

likes_comm = data_all[(data_all.actiontype.isin(['like','comment']))]
likes_comm.index = pd.to_datetime(likes_comm.index)
likes_comm.head()

# <codecell>


# <codecell>

likes_comm[['buyer_segment_cd_2','num_logins_orders']].groupby([likes_comm.index.year, 'buyer_segment_cd_2']).agg('count')

# <codecell>

f_dat[f_dat.email =='gjlustig@hotmail.com']

# <codecell>

likes_comm[['buyer_segment_cd_2','num_logins_orders']].groupby([likes_comm.index.year, likes_comm.index.month, 'buyer_segment_cd_2']).agg('count')

print likes_comm[likes_comm.year == 2015][['year','month']].sort(columns = 'month').drop_duplicates()

# <codecell>

data_all[(data_all.year == 2014) & (data_all.actiontype =='Order')][['buyer_segment_cd_2','num_logins_orders']].groupby('buyer_segment_cd_2').agg('count')

avg_orders_2014 = data_all[(data_all.year == 2014) & (data_all.actiontype =='Order')][['buyer_segment_cd_2','num_logins_orders']].groupby('buyer_segment_cd_2').agg('mean')

# <codecell>

data_all[(data_all.year == 2015) & (data_all.actiontype =='Order')][['buyer_segment_cd_2','num_logins_orders']].groupby('buyer_segment_cd_2').agg('count')

# <codecell>


# <codecell>

# create dataframes for each year with actiontypes and #orders_logins
data_2013_2 = data_all[data_all.year == 2013][['actiontype','num_logins_orders', 'buyer_segment_cd_2']]
df_2013_2 = data_2013_2.groupby([data_2013_2.index, 'actiontype','buyer_segment_cd_2']).agg('sum')
print df_2013_2.head()
# reset time index
df_2013_2 = df_2013_2.reset_index(level = 0)
print df_2013_2.head()

# <codecell>

# create dataframes for each year with actiontypes and #orders_logins
# data_2013_2 = data_all[data_all.year == 2013][['actiontype','num_logins_orders', 'buyer_segment_cd_2']]
# data_2014_2 = data_all[data_all.year == 2014][['actiontype','num_logins_orders','buyer_segment_cd_2']]
# data_2015_2 = data_all[data_all.year == 2015][['actiontype','num_logins_orders', 'buyer_segment_cd_2']]
df_2013_2.head()
orders_2013 = df_2013_2.ix['Order'].reset_index(level=0)
orders_2013 = orders_2013.set_index('time')
print orders_2013.head()

seg_list_orders = sorted(orders_2013.buyer_segment_cd_2.unique().tolist())
print seg_list_orders

for seg in seg_list_orders:
    seg_dat = orders_2013[orders_2013.buyer_segment_cd_2 ==seg]
    seg_dat.plot(figsize=(12,8), title = seg)
#pd.DataFrame(list(orders_2013.values), index=orders_2013.index).plot


# <codecell>

group_A_2013 = orders_2013[orders_2013.buyer_segment_cd_2 =='A']
group_A_2013.index = pd.to_datetime(group_A_2013.index)

# <codecell>

group_A_2013.head()

# <codecell>

group_A_2013.groupby(group_A_2013.index.month).agg(['sum', 'mean']).plot(subplots=True)

# <codecell>


# <codecell>

#orders_2013.set_index('time').plot(kind = 'bar')
# fig, pyplot.subplots = df_2013_2.ix['Order'].reset_index(level=0).set_index('time').groupby(['buyer_segment_cd_2']).plot()
# pyplot.show(fig)
#df_2013_2.ix['Order'].groupby(['buyer_segment_cd_2']).plot(stacked = True)

# df_2013_2[df_2013_2.actiontype == 'Order'].reset_index(level=1).groupby(['buyer_segment_cd_2']).plot(figsize = (12,18),
#                                                                                                      stacked = True,
#                                                                                                      colors=['red','green',
#                                                                                                               'blue','yellow'])

# group by date & actiontype for each year and count number of rows for each actiontype
# reset index level 1 (action type) so have count by day of each action type per day

# df_2013_2 = data_2013.groupby([data_2013.index, 'actiontype','buyer_segment_cd_2']).agg('count')
# df_2013_2 = df_2013_2.reset_index(level=1)

# df_2014 = data_2014.groupby([data_2014.index, 'actiontype']).agg('count')
# df_2014 = df_2014.reset_index(level=1)

# df_2015 = data_2015.groupby([data_2015.index, 'actiontype']).agg('count')
# df_2015 = df_2015.reset_index(level=1)

# <codecell>

pyplot.show()

# <codecell>

users_dict_2014_2015

# <codecell>

unique_emails = data_all.email[(data_all.buyer_segment_cd == 'None') & (data_all.year!=2013)].drop_duplicates().tolist()

# <codecell>

#[users_dict_2014_2015[0][email] for email in unique_emails]


[email for email in data_all[(data_all.buyer_segment_cd == "None")][['email']]]

#print '154750712@qq.com' in users_dict_2014_2015[0].keys()
#[users_dict_2014_2015[0][email] for email in data_all.email[data_all.buyer_segment_cd == 'None']]
# replace none segment_cd by looking it up in dictionary
# data_all.buyer_segment_cd[data_all.buyer_segment_cd == 'None'] =

# [email for email in data_all.email[data_all.buyer_segment_cd == 'None']
# lookup_seg = [email for email in data_all.email[data_all.buyer_segment_cd == 'None'].drop_duplicates()]
# print [(email, users_dict_2014_2015[0][email]) for email in lookup_seg]

# <codecell>

df_2013 = data_2013.groupby([data_2013.index, 'actiontype', data_2013.]).agg('count')
df_2013 = df_2013.reset_index(level=1)

# <codecell>

df_2013.to_csv("df_2013.csv")

# <codecell>

# states_actions_yearly_monthly = actions_states.groupby([actions_states.index.year, actions_states.index.month, 'actiontype']).agg('count')
# states_actions_yearly_monthly.index.levels[0].name = 'year'
# states_actions_yearly_monthly.index.levels[1].name = 'month'

# group actions by year + month
actions_yearly_monthly = actions.groupby([actions.index.year, actions.index.month, 'actiontype']).agg('count')
actions_yearly_monthly.index.levels[0].name = 'year'
actions_yearly_monthly.index.levels[1].name = 'month'

#states_actions_yearly_monthly = states_actions_yearly_monthly.reset_index(level=0)

# <codecell>

actions_yearly_monthly

# <codecell>

# states_actions_yearly_monthly = states_actions_yearly_monthly.reset_index(level=0)
# actions_yearly_monthly = actions_yearly_monthly.reset_index(level=0)

# reset monthly index, look at years
actions_yearly_monthly_long = actions_yearly_monthly.reset_index(level=1)
#actions_yearly_monthly_long = actions_yearly_monthly.reset_index(level=0)
actions_yearly_monthly_long

# <codecell>

actions_yearly_monthly_long.ix[2013].ix['Login'][['actiontype']].sum()

# <codecell>

actions_yearly_monthly_long.reset_index(level=0).to_csv('actions_yearly_monthly.csv')

# <codecell>

# 2015
actions_yearly_monthly_long.index.levels[0].name  = 'actionTypes'
actions_yearly_monthly_long.ix[2015].sort(columns = 'month').ix[['like','comment']]
#states_actions_yearly_monthly.ix[2015].reset_index(level=0)[[0,2]].sort(columns = 'month').ix[['like','comment']]

# <codecell>

#sum_actions = states_actions_yearly_monthly.ix[2015].reset_index(level=0)[[0,2]]
#sum_actions = states_actions_yearly_monthly.ix[2014].reset_index(level=0)[[0,2]]
# total_actions_2015 = sum_actions.reset_index().groupby('actionTypes').agg('sum')
#sum_actions.groupby('actiontype').get_group('comment')#agg('sum')#sort(columns = 'month').ix[['like','comment']]

# 2015 - likes and comments
actions_yearly_monthly_long.ix[2015].sort(columns = 'month').ix[['like','comment']]
sum_actions = actions_yearly_monthly_long.ix[2015].sort(columns = 'month').ix[['like','comment']] #.reset_index(level=0)[[0,2]]
sum_actions.index.name = 'actionTypes'
print "2015 sum likes and comments\n", sum_actions

total_actions_2015 = sum_actions.reset_index().groupby('actionTypes').agg('sum')
print "2015 likes + comments\n", total_actions_2015


# 2014
sum_actions = actions_yearly_monthly_long.ix[2014].ix[['like','comment']]
sum_actions.index.name = 'actionTypes'
print "2014 sum likes and comments\n", sum_actions

total_actions_2014 = sum_actions.reset_index().groupby('actionTypes').agg('sum')
print "2014 likes + comments\n", total_actions_2014

# <codecell>

# look at total comments and likes by year - 2014 & 2015 (2013 doesnt have any comments/likes)
likes_comments_2015 = total_actions_2015.ix[['comment','like']][['actiontype']]
likes_comments_2014 = total_actions_2014.ix[['comment','like']][['actiontype']]
print "2015\n", likes_comments_2015
print "2014\n", likes_comments_2014

# <codecell>

print '2015 as percents\n'
print likes_comments_2015/likes_comments_2015.sum()*100
print
print '2014 as percents\n'
print likes_comments_2014/likes_comments_2014.sum()*100

total_likes_comments = likes_comments_2014.sum() + likes_comments_2015.sum()
print "2014 + 2015 likes and comments\n", total_likes_comments

# <codecell>

print likes_comments_2014.sum()/total_likes_comments
print likes_comments_2015.sum()/total_likes_comments

# <codecell>

by_week = actions.groupby([actions.index.year, actions.index.weekofyear, 'actiontype']).actiontype.agg('count')
#by_state_week.index.levels[0].name = 'week'
by_week.index.levels[0].name = 'year'
by_week.index.levels[1].name = 'week'
by_week

# <codecell>

bi_weekly = np.arange(2, 54, 2)
bi_weekly_2013 = np.arange(2, by_week.ix[2013].reset_index(level=0)[['week']].max()[0]+1, 2)
bi_weekly_2014 = np.arange(2, by_week.ix[2014].reset_index(level=0)[['week']].max()[0]+1, 2)
bi_weekly_2015 = np.arange(2, by_week.ix[2015].reset_index(level=0)[['week']].max()[0]+1, 2)
# print bi_weekly
# print bi_weekly_2013
bi_counters = pd.DataFrame({'week_i': np.arange(1,53,2), 'week_j':np.arange(2,54,2)})
# print "bi_counters\n", bi_counters, "\n"
l_2013 = [2013 for i in xrange(1, len(bi_weekly_2013)+1)]
l_2014 = [2014 for i in xrange(1, len(bi_weekly_2014)+1)]
l_2015 = [2015 for i in xrange(1, len(bi_weekly_2015)+1)]

logins_bi_ts_2013 = pd.DataFrame({'week': bi_weekly_2013, 'bi_weekly_logins': np.zeros_like(np.arange(1,53,2)),
                                  'year': l_2013})
logins_bi_ts_2014 = pd.DataFrame({'week': bi_weekly_2014, 'bi_weekly_logins': np.zeros_like(np.arange(1,53,2)),
                                  'year': l_2014})
# 2015 - fewer weeks
lstweek_2015 = by_week.ix[2015].reset_index(level=0)[['week']].max()[0]
logins_bi_ts_2015 = pd.DataFrame({'week': bi_weekly_2015, 'bi_weekly_logins': np.zeros_like(np.arange(1,
                                lstweek_2015,2)),
                                  'year': l_2015})


# <codecell>

print bi_weekly_2015.shape
print len(l_2015)
print logins_bi_ts_2015

# <codecell>


for i,e in enumerate(by_week.index.levels[0]):
    print i
    print e
    year_i = by_week.ix[e]
    num_weeks = year_i.reset_index(level=0)[['week']].max()
    print "\nnum weeks\n", num_weeks[['week']][0]
    for j in xrange(1, num_weeks/2+1):
        print "\nj", j
        counter_1 = bi_counters.week_i[j-1]
        counter_2 = bi_counters.week_j[j-1]
        print "counter1", counter_1
        print "counter2", counter_2
        curr_login_count = by_week.ix[e].ix[counter_1].iloc['actiontype' =='Login']
        next_login_count = by_week.ix[e].ix[counter_2].iloc['actiontype' =='Login']
        print "curr login", curr_login_count
        print 'next login', next_login_count
        bi_weekly_logins = curr_login_count + next_login_count

        print "weeks added", counter_1, "+", counter_2, "=", bi_weekly_logins
        if e == 2013:
            logins_bi_ts_2013.bi_weekly_logins[j-1] = bi_weekly_logins
        elif e == 2014:
            logins_bi_ts_2014.bi_weekly_logins[j-1] = bi_weekly_logins
        elif e ==2015:
            logins_bi_ts_2015.bi_weekly_logins[j-1] = bi_weekly_logins

# <codecell>

logins_bi_ts_2013.to_csv('logins_bi_ts_2013.csv')
df = pd.concat([logins_bi_ts_2013, logins_bi_ts_2014, logins_bi_ts_2015])
df.head(10)
# fig, ax = pyplot.subplots(figsize=(8,6))
# df.groupby('year').plot(kind = 'bar', ax= ax)
#logins_bi_ts[['bi_weekly_logins']].ix[1]

# <codecell>

pd.melt(df, id_vars='year', value_vars=['week','bi_weekly_logins'], value_name='bi_weekly_logins').plot()

# <codecell>

p_df = pd.DataFrame({"class": [1,1,2,2,1], "a": [2,3,2,3,2]})
fig, ax = pyplot.subplots(figsize=(8,6))
bp = p_df.groupby('class').plot(kind='kde', ax=ax)

# <codecell>


# <codecell>

logins_bi_ts_2014.to_csv('logins_bi_ts_2014.csv')
logins_bi_ts_2014

# <codecell>

logins_bi_ts_2015.to_csv('logins_bi_ts_2015.csv')
logins_bi_ts_2015

# <codecell>

for i ,e in enumerate(by_week.index.levels[0]):
    print "\nyear index", i
    print "\nyear", e
    year_i = by_week.ix[e]
    print len(year_i.reset_index(level=0)[['week']].max()) + 1, '\n'
    num_weeks = year_i.reset_index(level=0)[['week']].max() +1
    for j in xrange(0, num_weeks) :
        print "week", j

# <codecell>

by_week.ix[2015].reset_index(level=0)[['week']].max()

# <codecell>

by_week.to_csv('by_week_v2.csv')

# <codecell>

states_actions_yearly_monthly = states_actions_yearly_monthly[[0,1,3]]

# <codecell>

print states_actions_yearly_monthly.head()
states_actions_yearly_monthly.set_index(['year']).plot(kind = 'barh', stacked = True, figsize = (12,8))

# <codecell>

by_state_week_2 = by_state_week.reset_index()
by_state_week_2.pref_state = by_state_week_2.pref_state.apply(lambda x: x.upper())

# <codecell>

by_state_week_2[['action_counts']] = by_state_week_2[[2]].copy()
by_state_week_2.drop(axis=1, labels = 0, inplace=True)

# <codecell>

by_state_week_2.set_index('week')

# <codecell>

by_state_week_2.plot(kind = 'bar', x = 'action_counts',y = 'week', stacked=True)

# <codecell>

orders.spend_lifetime[orders.actiontype=='comment']

# <codecell>

cols = data.columns
cols = [x.replace('temp.','') for x in cols]
data.columns = cols

# convert time column to datetime
data.time = pd.to_datetime(data.time)

# <codecell>

data = data.set_index('time')

# <codecell>

# only orders, likes and comments
orders = data[data.actiontype!='Login']
pd.value_counts(data['actiontype'])

# <codecell>

data.head(n=10)
#data[(data.email == '03weg@williams.edu') & (data.actiontype =='Order')]
orders.spend_lifetime[pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')] = orders.spend_lifetime[pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')].astype(np.float64)

# <codecell>

orders.spend_lifetime[pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')].plot()

# <codecell>

# look at buyer_segment_cd and distribution of data
n = orders.shape[0]
orders.buyer_segment_cd.value_counts().plot(kind = 'bar', title ='count of buyers by segment')

seg_percent_total = orders.buyer_segment_cd.value_counts().apply(lambda x: x*1.0/n)
seg_percent_total.plot(kind='bar')
print seg_percent_total
# seg_A = orders[orders.buyer_segment_cd =='A'][['total_amt','spend_lifetime','orders_lifetime','price']] .describe().T.plot()
# df = pd.DataFrame({'mean': seg_A.mean(), 'median': seg_A.median(),
#                    '25%': seg_A.quantile(0.25), '50%': seg_A.quantile(0.5),
#                    '75%': seg_A.quantile(0.75)})

# df.plot()
# df

# <codecell>

#orders[orders.spend_lifetime !="None"][['buyer_segment_cd','spend_lifetime']].dropna().groupby('buyer_segment_cd').plot()

grouped_seg = orders[orders.spend_lifetime !="None"][['buyer_segment_cd','spend_lifetime',
                                                      'orders_lifetime','total_amt']].dropna().groupby('buyer_segment_cd')

# <codecell>

# for key, grp in grouped_seg:
#     pyplot.plot(grp['spend_lifetime'], label = key)
# pyplot.legend(loc='best')


# fig = pyplot.subplots(figsize = (12,8))
grouped_seg.get_group('A')[['spend_lifetime','orders_lifetime']].plot(color='blue', subplots=True, figsize = (10,6), title='Group A')
grouped_seg.get_group('L')[['spend_lifetime','orders_lifetime']].plot(color='red', subplots = True, figsize = (10,6), title = 'Group L')
grouped_seg.get_group('N')[['spend_lifetime','orders_lifetime']].apply(lambda x: x.astype(np.int64)).plot(color='purple', subplots = True, figsize = (10,6), title = 'Group N')

# <codecell>

orders.groupby('buyer_segment_cd').get_group('A')[['total_amt']].plot(color='green', title = 'Group A')
orders.groupby('buyer_segment_cd').get_group('L')[['total_amt']].plot(color='green', title = 'Group L')
orders.groupby('buyer_segment_cd').get_group('N')[['total_amt']].plot(color='green', title = 'Group N')
orders.groupby('buyer_segment_cd').get_group('None')[['total_amt']].plot(color='green', title = 'Group None')

# <codecell>

#orders.head()
pd.melt(orders, id_vars=['email','fb_appscopedid','buyer_segment_cd'], value_vars=['num_logins_orders','spend_lifetime',
                                                                                   'orders_lifetime','login_days_365',
                                                                                   'total_amt']).drop_duplicates().sort(columns='email')
# create dataframe of unique users with: lifetime_spend, buyer_segment_cd, orders_num
# for k in grouped_seg.groups:
#     if k!='N':
#         grouped_seg.get_group(k).plot()

# <codecell>

users = orders[pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')].groupby('email')

#orders[['buyer_segment_cd','spend_lifetime']][pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')].plot()

# <codecell>

segments = orders[pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')].groupby('buyer_segment_cd')

# <codecell>

segments = orders[pd.notnull(orders.spend_lifetime) & (orders.actiontype == 'like')].groupby('buyer_segment_cd')

# <codecell>

print pd.DataFrame(amt_list).describe()

# <codecell>

print orders.shape
print orders.head(n=5)

# <codecell>

print orders.price.groupby(orders.index.year).sum()

# <codecell>

# look at distribution of prices in years
orders.price.groupby(orders.index.year).sum().plot(kind = 'bar')
pyplot.show()

# <codecell>

orders[['price']].groupby(orders.index.year).sum().plot(kind = 'bar')
pyplot.show()

# <codecell>

orders[['num_logins_orders']].groupby(orders.index.year).count().plot(kind = 'bar', stacked= True)

# <codecell>

orders_by_email = orders[['email','actiontype','num_logins_orders','price']][orders.num_logins_orders>0].groupby('email')

# <codecell>

orders.groupby('email')['num_logins_orders', 'price'].agg(['sum','mean'])
#print orders[orders['actiontype']=='Order']#.spend_lifetime#[orders.num_logins_orders >0]

# <codecell>

action_groups = orders.groupby(['actiontype','fb_appscopedid'])
action_groups[

# <codecell>

orders.time[0]

# <codecell>

data.time.groupby(data.time.timeofday).mean()

# <codecell>

data.groupby(['actiontype']).aggregate(['sum','mean'])

# <codecell>

np.unique(data.actiontype)

# <codecell>

data.groupby(['fb_appscopedid','actiontype']).aggregate('count')

