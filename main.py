from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency
from helpers import *
import matplotlib.pyplot as plt
import matplotlib


# In[2]:

df_user = pd.read_csv('data_sample/user_data_sample.csv')
df_song = pd.read_csv('data_sample/end_song_sample.csv')
df = pd.merge(df_user, df_song, on='user_id',  how='left')


# In[3]:

grouped = df.groupby('gender')
# rows with 'unknown' gender is discarded
male = grouped.get_group('male')
female = grouped.get_group('female')
# At this point we have two DataFrames: `male` and `female`. 
# They contain all the columns merged from the two tables. 


# In[4]:

skip_cutoff = 10 * 1e3
skip10_m = male['ms_played'][male['ms_played'] < skip_cutoff].count()/len(male) * 100
skip10_f = female['ms_played'][female['ms_played'] < skip_cutoff].count()/len(female) * 100
print "  male [0-10] sec percentage = {} %".format(round(skip10_m, 1))
print "female [0-10] sec percentage = {} %".format(round(skip10_f, 1))


# 33 % of all records are between [0 - 10] secs -- i.e a third of all tracks are being skipped almost at equal proportion across male and female listeners.

# In[5]:

width = 10
# Impose and use same bin ranges for male and female hist()
# ignore the [0-10] sec quick-skip range
bin_range = np.linspace(10,500,(500/width))
# convert ms to secs
sec_played_m = male['ms_played']/1000.
sec_played_f = female['ms_played']/1000.

plt.hist(sec_played_m, bins=bin_range, width=width, alpha=0.3, color='b', normed=True)
plt.hist(sec_played_f, bins=bin_range, width=width, alpha=0.3, color='r', normed=True)

# plt.close()
n_m ,_, _ = plt.hist(sec_played_m, bins=bin_range, width=width, alpha=0.3, color='b')
n_f ,_ ,_ = plt.hist(sec_played_f, bins=bin_range, width=width, alpha=0.3, color='r')

plt.gca().legend(['male','female'])
plt.gcf().set_size_inches((15,5))
plt.grid('on')
plt.xlabel('secs played')
plt.xlim((0, 500));
plt.title('raw counts for track listen times ([0-10] sec omitted)')
# plt.ylim((0, 40000));


# **Comments:**  
# * The normalized histograms, exhibits a higher kurtosis for *female track listen times* than its male counterpart -- the frequency between [180 - 240] sec range is higher for females.  
# For a more robust analysis, $\chi^2$-test for independence will be employed.   
# 
# 
# 
# * The histogram looks like a superposition of a _Gaussian Distribution_ and an *Exponential-like Distribution*:
# The Gaussian Distribution seems to reflect the distibution of track lengths where the user listened to the whole track.  
# The Exponential Distribution likely originates from the listeners skipping songs. In this case, the random variable would be defined as the part of the song played before it was skipped. The distribution might deviate from the analytical exponential distribution profile leaning more heavily on the smaller values.
# 
# 
# ------------------

# ## $\chi^2$-test For Independence -- for track listening time across genders
# The question being considered is whether the *total track listening time* varies across `male` and `female` listeners.  
# 
# $H_o$: *Total track listening time*  doesn't vary across `male` and `female` listeners. (`male` and `female` have same pdfs)

# In[6]:

# my_chisquare_p_value(n_m, n_f)
_, p_value, _, _ = chi2_contingency(np.vstack((n_m,n_f)))
print p_value


# Conclusion:  
# * Reject $H_o$!...   
# * Consumer listening habits are different across the genders.
# 
# ---------

# ## $\chi^2$-test For Independence -- for age groups across genders
# The question being considered is whether the *total track listening time* varies across `male` and `female` listeners.  
# 
# $H_o$: *Total track listening time*  varies across `male` and `female` listeners. (`male` and `female` have different pdfs)

# In[7]:

# omit the [0-10] sec range

grouped_m = male[male['ms_played'] > skip_cutoff].groupby('age_range')
grouped_f = female[female['ms_played'] > skip_cutoff].groupby('age_range')
age_m = grouped_m.groups.keys()
age_f = grouped_f.groups.keys()
print sorted(age_f)
print sorted(age_m)


# In[8]:

m1 = grouped_m.mean()['ms_played']
f1 = grouped_f.mean()['ms_played']


# In[9]:

my_xticks = sorted(age_m)
ind = np.asarray(range(len(np.asarray(m1))))
width = .45
ax = plt.gca()
ax.bar(ind - width, np.asarray(m1), width=width, color='k', alpha=.3)
ax.bar(ind,         np.asarray(f1), width=width, color='r', alpha=.3)

plt.gcf().set_size_inches((15,5))
plt.xticks(ind, my_xticks,rotation=45)
plt.grid('on')
plt.ylim([0, 350000])
plt.legend(['male','female']);


# In[10]:

_, p_value, _, _ = chi2_contingency(np.vstack((m1,f1)))
print p_value


# ---
# ### Break DataFrame Into Sessions
#  
# > * group `df` by `user_id`,   
# > * sort `df` wrt column `end_timestamp`,   
# > * create a new column `end_time_diff` that carries the value of end_time difference between two successive listening events.
# 

# In[11]:

df['end_timestamp'] += -min(df['end_timestamp'])

df_new = pd.DataFrame({})
grouped = df.groupby('user_id')
user_ids = grouped.groups.keys()  # len(user_ids) = 9565

# build a new df from individual 'user_id' groups.
for i, key in enumerate(user_ids[:1000]):
    user = df[df['user_id'] == key].sort_values('end_timestamp')
    user['end_time_diff'] = (user['end_timestamp'] - user['end_timestamp'].shift(1)).fillna(-1)
    df_new = pd.concat([df_new, user])


# In[12]:

df_new = df_new.reset_index(drop=True)
df_new[['gender','user_id','ms_played', 'end_timestamp', 'end_time_diff']].head(5)


# >   Define *a session start event* if more 15 mins has elapsed since last activity. 

# In[13]:

sess_break_cutoff = 15*60 # 15 minute break means 
df_new['session'] = df_new.apply(lambda row: 'start'                                  if (row['end_time_diff'] > sess_break_cutoff)                                  or (row['end_time_diff'] == -1.00)                                  else '.', axis=1)


# In[14]:

session_start_indices = df_new[df_new['session'] == 'start'].index
session_end_indices = (session_start_indices[1:] - 1).append(pd.Int64Index([df_new.index.max()]))

df_s1 = df_new.loc[session_start_indices]['end_timestamp']
df_s2 = df_new.loc[session_end_indices]['end_timestamp'] 
df_s3 = df_new.loc[session_start_indices]['ms_played']/1000

# Need to re-define the index labels so we can do algebra 
df_s2.index = session_start_indices
df_s3.index = session_start_indices


# In[15]:

# create a new column for 'session_length'
df_new['session_length'] = df_s2 - df_s1 + df_s3


# In[39]:

df_new.tail(10) # notice that within the same session the context can change.


# > `df_new` carries the `'session'` and `'session_length'` columns
# 
# ---
# 

# ## Mean Session Lengths wrt Countries
# 
# Questions to be explored:
# 1. Does the session length vary with respect to `country`?
# 2. Does the session length vary with respect to `gender`?
# 3. Does the session length vary with respect to `context`?
# 4. Does the session length vary with respect to `product` (paid or free)?
# 5. Does the session length vary with respect to `age_range`?
# 

# In[42]:

def bar_plot_session_length(gr_by='country', rotation=0):
    df_sessions = df_new[df_new['session'] == 'start']
    gr = df_sessions['session_length'].groupby(df[gr_by])
    
    country = gr.groups.keys()
    intermediate = sorted(zip(range(len(gr.mean())), gr.mean()/60), key=lambda x: x[1], reverse=True)
    ind, y = zip(*intermediate)
    ind, y, country= np.asarray(ind), np.asarray(y), np.asarray(country)
    
    ax = plt.gca()
    width = 1
    x = np.asarray(range(len(y)))
    ax.bar(x, y, color='b', alpha=.4)

    plt.gcf().set_size_inches((17,5))
    plt.xticks(x+.5, country[ind], rotation=rotation)
    plt.grid('on')
    plt.xlim([0, max(x)+1])
    plt.title('average session lengths wrt to '+ gr_by +' [min]');



bar_plot_session_length(gr_by='country', rotation=90)
bar_plot_session_length(gr_by='gender', rotation=0)
bar_plot_session_length(gr_by='context', rotation=0)
bar_plot_session_length('product')
bar_plot_session_length('age_range')






