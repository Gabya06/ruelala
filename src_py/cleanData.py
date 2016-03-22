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

import itertools
from collections import Counter
import operator

os.chdir("/Users/Gabi/dev/Shareablee/RueLaLa/")


# read data
# posts of RLL customers (not necessarily on ruelala's pageid)
fb_rll_data = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/datasets/fb_rll_fixed.csv" ,
	dtype = {'fb_rll_temp.email': np.str, 'fb_rll_temp.actiontype':np.str,
	'fb_rll_temp.month': np.int64, 'fb_rll_temp.total_amt':np.int64,
	'fb_rll_temp.fb_appscopedid' : np.str})
# this only relating to posts for RUELALA - filtered out pageid
rll_fb_data = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/datasets/ruelala_fb_1192015.csv" ,
	dtype = {'fb_rll_temp.email': np.str, 'fb_rll_temp.actiontype':np.str,
	'fb_rll_temp.month': np.int64, 'fb_rll_temp.total_amt':np.int64,
	'fb_rll_temp.fb_appscopedid' : np.str})

buyer90_data = pd.read_csv("/Users/Gabi/dev/Shareablee/RueLaLa/datasets/buyer90_filtered.csv" ,
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



f_dat = fb_rll_data[['email','actiontype','buyer_segment_cd','time','month','price','total_amt','num_logins_orders','user_type','fb_appscopedid',
             'spend_365','spend_lifetime','orders_lifetime','login_days_365','last_visit_spend','pref_city','pref_state','pageid','postymd']]
b_dat = buyer90_data[['email','actiontype','buyer_segment_cd','time','month','price','total_amt','num_logins_orders','user_type','fb_appscopedid',
             'spend_365','spend_lifetime','orders_lifetime','login_days_365','last_visit_spend','pref_city','pref_state','pageid','postymd']]
rll_dat = rll_fb_data[['email','actiontype','buyer_segment_cd','time','month','price','total_amt','num_logins_orders','user_type','fb_appscopedid',
             'spend_365','spend_lifetime','orders_lifetime','login_days_365','last_visit_spend','pref_city','pref_state','pageid','postymd']]


print "f dat size\n", f_dat.shape #(1000000 X 19)
print "b dat size\n", b_dat.shape #(1000000 X 19)
print "rll dat size\n", rll_dat.shape #(2808 X 19)


# remove 2013 data from buyer data
b_dat.time = pd.to_datetime(b_dat.time)
b_dat = b_dat.set_index('time')
b_dat = b_dat[b_dat.index.year != 2013]
b_dat = b_dat.reset_index()


# concatenate dataframes - ignore 2013 buyer data
data_all = pd.DataFrame()
data_all = pd.concat([data_all, f_dat, b_dat])
# convert time column to datetime
data_all.time = pd.to_datetime(data_all.time)
print "combined fb all + buyer all \n", data_all.shape #(2000000, 19) wt 2013 and  (1692573,19) w.0 2013


# reset index and use only dates
#data_all.time[0].map(lambda x: x.date())
data_all.time = data_all.time.map(lambda x: x.date())
data_all.time.head()

# convert back to datetime - this  takes a longggg time
#data_all.time = data_all.time.map(lambda x: pd.Timestamp(x))
data_all.time = data_all.time.map(lambda x: pd.to_datetime(x))


data_all = data_all.set_index(data_all.time)
data_all['year'] = data_all.index.map(lambda x:x.year)

# filter out orders and logins since they dont have segment info (only likes and comments do)
users_2014 = data_all[(data_all.year == 2014) & (data_all.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
dedup_users_2014 = users_2014.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()#.to_dict()

users_2015 = data_all[(data_all.year == 2015) & (data_all.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
dedup_users_2015 = users_2015.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()#.to_dict()

deduped_users_2014_2015 = pd.concat([dedup_users_2015, dedup_users_2014]).drop_duplicates()
# put all fb users in a dictionary
users_dict_2014_2015 = deduped_users_2014_2015.set_index('email').T.to_dict('records')
# built dataframe for emails and buyer-segment-cd
# replace None buyer segments with segment inf
email_seg = data_all[['email','buyer_segment_cd']].reset_index(level=0)[['email','buyer_segment_cd']]
for row in email_seg.values:
    if row[1] =="None":
        row[1] = users_dict_2014_2015[0][row[0]]

 # data_all: fb users for all properties
# create new field called buyer_segment_cd_2 where None values are replaced by looked up segment info
data_all['buyer_segment_cd_2'] = [i for i in email_seg.buyer_segment_cd]

# check there s buyer segment in new column
print data_all[data_all.buyer_segment_cd_2=="None"]


# replace segment info now that it s filled in
# since data_all has additional content data this will have all segment info about rll users
data_all.buyer_segment_cd = data_all.buyer_segment_cd_2

print data_all[data_all.buyer_segment_cd == 'None']
# drop column
data_all = data_all.drop('buyer_segment_cd_2', axis=1)

# built dataframe for emails and buyer-segment-cd
# replace None buyer segments with segment info in Buyer_data since will also join this with RLL Fb only data
b_email_seg = b_dat[['email','buyer_segment_cd']]#.reset_index(level=0)[['email','buyer_segment_cd']]
for row in b_email_seg.values:
    if row[1] =="None":
        row[1] = users_dict_2014_2015[0][row[0]]

# create new field called buyer_segment_cd_2 where None values are replaced by looked up segment info
b_dat['buyer_segment_cd_2'] = [i for i in b_email_seg.buyer_segment_cd]
b_dat[b_dat.buyer_segment_cd=="None"][['buyer_segment_cd','buyer_segment_cd_2']].head(n=10)

# replace column with fixed buyer segment info
b_dat.buyer_segment_cd = b_dat.buyer_segment_cd_2
# drop other col
b_dat[b_dat.buyer_segment_cd=="None"]
b_dat = b_dat.drop('buyer_segment_cd_2', axis =1)

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

dat_rll_only = dat_rll_only.sort_index()
dat_rll_only = dat_rll_only.ix[datetime(2014,7,1):datetime(2015,6,30)]

# ******************************************************************** #
# MAKE SURE TO Filter out ruelala page_id***
rll_pageid = 22505733956
dat_rll_only[~dat_rll_only.pageid.isin(['None',rll_pageid])]

dat_rll_only['year'] = dat_rll_only.index.year


# dictionary of rll user only {email:segment}

rll_buyers_fb_users = dat_rll_only[(dat_rll_only.year.isin([2014,2015])) & (dat_rll_only.actiontype.isin(['comment','like']))][['email','buyer_segment_cd']]
rll_deduped_users = rll_buyers_fb_users.reset_index(level=0)[['email','buyer_segment_cd']].drop_duplicates()
rll_users_dict = rll_deduped_users.set_index('email').T.to_dict('records')[0]
print len(rll_users_dict)#860

dat_rll_only.to_csv("/Users/Gabi/dev/Shareablee/RueLaLa/cleanData/dat_rll_only.csv")
