## Project Dataset Introduction: San Francisco 311 Cases 

Provided by [DataSF](https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6), this set is a monstrosity containing about 4.25 Million rows and counting. For those not familiar, 311 is a general customer service number for city government, most commonly associated with non-emergency complaints. 311 cases can be created via phone, mobile, and web. The dataset covers the time period from July 2008 until present. 

In order to do data exploration and analysis of this dataset we needed to make a working sample to reduce memory and cpu usage upfront: 
    <pre><code>awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01 || FNR==1) print $0}' filename</code></pre>

Further information about the dataset can be found [here.](https://support.datasf.org/help/311-case-data-faq)


## One of the fascinating aspects of city living is human behavior.
One behavior every human seems to enjoy is complaining, in some shape way or form. Whether its done in exercise of rights to recourse, for entertainment, or out of civic duty, we encounter a situtation that make us uncomfortable... and we file a 311 case on it. I wanted to focus on the subset of cases that truly reflect the spirit of 'complaints,' in particular those concerned with the behavior of others, and generally require some sort of short term physical response. Sadly, it seems there aren't many cases filed to commend folks for good behavior, so we will be looking at mostly negative or unpleasant situatioons here. Accordingly, we have attempted to exclude cases concerning general administrative requests, such as building permits, tree maintenance,  and the like. In addition, despite it being filled with negative comments, I also chose to exclude the muni category, insofar as the Muni (city bus & train operators) is its own organization with its own culture, that I don't care to upset by pointout the exceedingly high volume of complaints.

From my personal observation, corrorated by many of my peers, once the 311 case is filed it goes into a black box, and we can only hope to guess at if or when the matter will be addressed. This can be very frustrating for the complainant which in turn probably results in corrsponding frustration for the people inside the black box, who receive many repeated complaints, doubtless with inversely proportional civility. 
### If only there were a way to *predict* how long it would take to resolve each issue... 
Well, luckily, there *are* ways, and we shall see what fruit they may bear. 

Datacleaning highlight: encode, convert or cast *every single feature*




```python
!pwd

```

    /home/dliu/lambda/unit-2/DS-Unit-2-Build-dmhliu



```python
import pandas as pd
import numpy as np
import math 
import pprint
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

pp = pprint.PrettyPrinter(indent=4)
pd.options.display.max_columns = 100
pd.set_option('display.max_rows', 100)

```


```python
# Get Pandas Profiling Report
import pandas_profiling
#pandas_profiling.ProfileReport(df)
```


```python
df = pd.read_csv('reduced1.csv', header=0, error_bad_lines=False, engine="python", warn_bad_lines=False);
```


```python
## experiment in oop
## instantiate  a wrangler and setup drops and encoding. 
## run methods on the internal working copy to iterate 
## 

class Wrangler:        
    """the intent is for each instance of wrangler to contain lists of 
    specific transformations to be performed on an dataframe
    by the class methods. """
    #class vars
    dropcols = list()    #list to store columns to drop
    droprows = dict()    #row boolean filters

    encoders = dict()    # colname, function to be applied to elements
    
    dropcols_post = list()  #keep a distinct list to drop after enconding

    def calc_lt_005_cat(self, df): 
        """ return a list of the bottom .5% category names for binning"""
        return df.Category.value_counts()[df.Category.value_counts(normalize=True).values < .005]

    def __init__(self, data):
        self.raw_df = data.copy()         #preserve the original dataframe
        self.working = self.raw_df.copy() #this will be passed around or be used as default by class methods
        
    #methods
    def set_data(self,newdf):
        """reset dataframe to new df, leaving configuration intact"""
        print('set data to df', df.shape)
        self.raw_df= newdf.copy()
        self.working = self.raw_df.copy()
        return self.working
    
    def add_to_dropcols(self,labels):
        for l in labels:
            if l in self.dropcols:
                print('error column already in dropcols list')
                break
                return self.dropcols
        self.dropcols.extend(labels)
        return self.dropcols

    def get_dropcols(self):         #why use accessor methods in python? 
        
        if not self.dropcols: 
            print('no columns to drop')
        return self.dropcols
    
    def clear_dropcols(self):
        self.dropcols=list()
        return True
    def get_nancols(self, df=None, store=False):
        if df is None: 
            df=self.working
        cols =df.columns
        nc=df[cols].isnull().sum().index.tolist()
        if store:
            self.nancols = nc
        return nc
    def add_to_droprows(self, name, expr):
        self.droprows[name] = expr 
        return self.droprows.keys()

   
    def add_encoding(self, label, mapper):
        try:
            label in self.working.columns
        except:
            print(label, 'not found in working copy, may have been dropped')
            assert label in self.raw_df.columns
        self.encoders[label]= mapper
    ###TODO: 
    def get_params(self):
        pass
        #return all wrangler settings 
    def set_nanval(self):   
        pass 
       # self.nanvals add self to 

#"internal"
    def drop_rows_by_mask(self,df=None,labels=None):
        """takes a dataframe, and list of labels to drop INPLACE,
        returns the modified dataframe"""
        if df is None:
            df=self.working
        if labels is None:
            print('dropping all in droprows..')
            if self.droprows:
                labels = list(self.droprows.keys())
        mask = self.droprows[labels[0]]  #get first boolean mask
        for l in labels: 
            print('\napplying mask: ',l )
            mask = mask | self.droprows[l]   #or them all together
        df.drop(index=df[mask].index,inplace=True)
        return df 
    
    def drop_dupes(self, df=None):
        """INPLACE: drop duplicate rows in df,
        return modified copy"""
        if df is None:
            df=self.working
        todropindex = df[df.duplicated()].index
        print('\n dropping',todropindex.shape, 'rows')
        df.drop(todropindex,axis=0, inplace=True)
        return df

    def drop_columns(self,df=None, list=None):
        """INPLACE if df provided, use working
        if no list is provided, drop all columns in list
        if list is provided, drop only the columns in argument list
        and add them to the dropcols list
        return modified dataframe"""
        
        if df is None:
            df = self.working  
        if list:                          #list of droplabels is passed, added to self.dropcols,then dropped.
            for l in list:
                if l in self.dropcols:
                    print('error column already in dropcols list')
                    break
                    return self.dropcols
            self.dropcols.extend(list)
        else:
            list= self.dropcols
                                   #TODO: check dropcols present in df.columns
        return df.drop(labels=list, axis=1, inplace=True)
    
    def drop_columns_post(self):
        """INPLACE drop any columsn in list dropcols_post 
        dont return anything"""
        drop_columns(list=dropcols_post)
        
    def bin_othercats(self):
        """to help reduce cardinality bin the lower .5pct of categories as other
            CAUTION need to do this after dropping bad data rows, it will change every time
            the df is resampled.... """
        df = self.working
        othercats = self.calc_lt_005_cat(df).index.tolist()
        df['Category'] = df.Category.map(lambda cat : 'other' if cat in othercats else cat)
        
    def encode(self,df=None, label=None, fun=None):
        """label is key for dict AND is column label 
        fun is a function for pd.Series.map() 
        """
        if df is None: 
            df=self.working
            print('\n Encoding, changing working copy..')
        if fun is None:
            fun = list(self.encoders.keys()) 
            for k in fun:
                print('  ... encoding column: ',k)
                df[k] = df[k].map(self.encoders[k])
        else:
            df[label] = df[label].map(fun)
        return df
    
    def to_datetime(self, cols=None, df=None):
        if df is None:
            df=self.working
            print('\nworking df is being changed..')
        if cols is None:
            cols=self.dt_cols
        for c in cols:
            print('converting',c,'to datetime')
            try:
                df[c] = pd.to_datetime(df[c],infer_datetime_format=True)  #inplace 
            except: 
                print('error - possible this column needs cleaning')
        return df

       
    def make_feature(self,newlabel,input,fun):      #make or overwrite column newlabel
        df=self.working
        try:
            df[newlabel] = df[input].apply(fun, axis=1)  #or map or tranform?
            print('\nadded feature:', newlabel)
        except:
            print('there was a problem, not added!!')
            return False
        return True
    def calc_open_cases(self, sometime):  #input time
        df= self.working[['CaseID','Opened','Closed']]
        opened_prior = df['Opened'] < sometime        # cases opened before it,
        not_closed_prior = ~(df['Closed'] < sometime) # not closed, 
        open_at_thattime = opened_prior & not_closed_prior  #and 
        return open_at_thattime.sum()
    
    order_default = [drop_rows_by_mask,
                    drop_columns, 
                    drop_dupes,
                    encode,
                    drop_columns_post]   #list of methods in order of application

    def wrangle(self, df=None):

        if df is None:
            df =self.raw_df.copy() # start from the beginning 
            result = df
            print('will apply :', self.order_default)
            for f in self.order_default:
                print('level')
                result = f(result)
        return result
    
```


```python
##column lists after exploration and research

delete = [s for s in df.columns.values if 'DELETE - ' in s]   #columns have delete in name, the appear to be older version of data
boundary = [s for s in df.columns.values if 'Boundary' in s]   #columns have boundary in name, dont need, redundant with neighborhood
floatcols = df.select_dtypes(include='float64').columns

```


```python
##define settings here 
wrangler = Wrangler(df)   
wrangler.clear_dropcols()

wrangler.add_to_dropcols(delete)
wrangler.add_to_dropcols(['Point'])  ##redundant w/ lat long, but may use for geospatial later
wrangler.add_to_dropcols(['Parks Alliance CPSI (27+TL sites)','Supervisor District'])
wrangler.add_to_dropcols(['Central Market/Tenderloin Boundary Polygon - Updated'])
wrangler.add_to_dropcols(['Civic Center Harm Reduction Project Boundary',
       'Fix It Zones as of 2017-11-06 ', 'Invest In Neighborhoods (IIN) Areas',
       'Fix It Zones as of 2018-02-07','SF Find Neighborhoods',
       'CBD, BID and GBD Boundaries as of 2017','Current Supervisor Districts',
       'Central Market/Tenderloin Boundary', 'Areas of Vulnerability, 2016',
       'HSOC Zones as of 2018-06-05', 'OWED Public Spaces'])

missing_to_0 = lambda x : 0 if math.isnan(x) else int(x)   #convert float to int, nan to zeros
missing_to_unk =  lambda x : x if type(x) == str else 'missing'  #for strings 

wrangler.add_encoding('Request Details', missing_to_unk)
wrangler.add_encoding('Neighborhood', missing_to_unk)
wrangler.add_encoding('Police District', missing_to_unk)

wrangler.add_encoding('Analysis Neighborhoods', missing_to_0)  #convert neigborhoods
wrangler.add_encoding('Neighborhoods', missing_to_0) 
wrangler.add_encoding('Street', missing_to_unk )
wrangler.add_encoding('Media URL', lambda x : 'present' if type(x) == str else 'missing')

```


```python
### special alert, some of our nasty data yields 

def caseid_to_int (x):    #convert good caseids to int, bad to zero for drop
    try:
        i = int(x)
    except:
        i = 0   
        return 0      #strings are converted to zero for dropping 
    return i

wrangler.droprows['drop_missing_caseid'] = df.CaseID.map(caseid_to_int) == 0    #mask to drop rows with no caseid
wrangler.add_encoding('CaseID', caseid_to_int)        #can be done any order


```


```python

```


```python

## Rows to drop 

nopen = df.Opened.isnull()    # bad records null, open date

wrangler.droprows['drop null open date'] = nopen #migh be fixed by  missing caseID?

nclosed= df.Closed.isnull()   # some are really still open this doenst need to be dropped.



excluded_cats =['SFHA Requests',     #these are adminstrative, e.g temp signs, or belong to other departments and are not centered around human behavior
                'Sidewalk or Curb',
                'Temporary Sign Request',
                'Tree Maintenance',
                'Rec and Park Requests']
#more infrastructure generic requests 
gen_req =['General Request - PUBLIC WORKS',    # is street cleaning in here?

        'Sewer Issues', 'Streetlights',
        'Sign Repair', 
        'General Request - PUC',
        'General Request - COUNTY CLERK',
        'General Request - 311CUSTOMERSERVICECENTER',               
        'General Request - MTA']
#
exclude_gen = df.Category.isin(gen_req)
exclude_svc = df.Category.isin(excluded_cats)  
muni = df.Category== 'MUNI Feedback'   #bus complaints we dont want, DPW doesnt service 

wrangler.add_to_droprows('muni', muni);
wrangler.add_to_droprows('svc_req', exclude_svc)
wrangler.add_to_droprows('gen_req',exclude_gen)
```




    dict_keys(['drop_missing_caseid', 'drop null open date', 'muni', 'svc_req', 'gen_req'])




```python
(df[muni]['Request Type'] == 'MUNI - Commendation').mean()  ## at least 3% of these are positive. 
df[muni]['Request Type'].value_counts(ascending=False,normalize=True)   #LOL DROP
```




    MUNI - Conduct_Inattentiveness_Negligence                         0.240359
    MUNI - Services_Service_Delivery_Facilities                       0.160987
    MUNI - Conduct_Discourteous_Insensitive_Inappropriate_Conduct     0.112108
    MUNI - Conduct_Unsafe_Operation                                   0.078027
    MUNI - Services_Miscellaneous                                     0.055157
    MUNI  - Services_Service_Delivery_Facilities                      0.051121
    MUNI - Services_Service_Planning                                  0.046188
    MUNI  -                                                           0.045291
    MUNI  - Conduct_Discourteous_Insensitive_Inappropriate_Conduct    0.039462
    MUNI - Commendation                                               0.039013
    MUNI  - Conduct_Inattentiveness_Negligence                        0.035426
    MUNI  - Services_Miscellaneous                                    0.031839
    MUNI  - Conduct_Unsafe_Operation                                  0.024664
    MUNI  - Services_Service_Planning                                 0.021525
    MUNI - Services_Criminal_Activity                                 0.016592
    MUNI  - Services_Criminal_Activity                                0.001794
    SSP SFMTA Feedback                                                0.000448
    Name: Request Type, dtype: float64



### So.... I was wrong about it being ALL BAD...

Fully **3%** of the MUNI cases are commendations for MUNI employees. Sounds about right. I would draw your attention to the subject of the remaining cases but I promised not to bash them...


```python
## these cols contain our timestamps as strings. we weill convert them to datetime object type for (in)convenience

dt_cols = ['Opened','Closed','Updated']
wrangler.dt_cols = dt_cols

```


```python
## run  some of the setup functionality 
wrangler.drop_rows_by_mask()
wrangler.drop_dupes()
wrangler.to_datetime();
wrangler.bin_othercats()   
wrangler.encode();
wrangler.drop_columns(list=wrangler.dropcols_post)
wrangler.make_feature('ttr',['Opened','Closed'],lambda x : x[1]-x[0])
wrangler.working.Latitude = wrangler.working.Latitude.astype(float)    # :(
wrangler.working.Longitude = wrangler.working.Longitude.astype(float)    # :(
wrangler.working['case_year'] = pd.DatetimeIndex(wrangler.working.Opened).year
wrangler.working['case_month'] = pd.DatetimeIndex(wrangler.working.Opened).month
#add feature open cases, number of cases open at the time current ticket is open
#not super happy with this, since we have dropped many rows from consideration, so this statistic should be
#interpreted carefully

wrangler.working['workload'] = wrangler.working['Opened'].apply(wrangler.calc_open_cases)


```

    dropping all in droprows..
    
    applying mask:  drop_missing_caseid
    
    applying mask:  drop null open date
    
    applying mask:  muni
    
    applying mask:  svc_req
    
    applying mask:  gen_req
    
     dropping (0,) rows
    
    working df is being changed..
    converting Opened to datetime
    converting Closed to datetime
    converting Updated to datetime
    
     Encoding, changing working copy..
      ... encoding column:  Request Details
      ... encoding column:  Neighborhood
      ... encoding column:  Police District
      ... encoding column:  Analysis Neighborhoods
      ... encoding column:  Neighborhoods
      ... encoding column:  Street
      ... encoding column:  Media URL
      ... encoding column:  CaseID
    
    added feature: ttr



```python

def calc_open_cases(sometime):  #input time
    df= wrangler.working[['CaseID','Opened','Closed']]
    opened_prior = df['Opened'] < sometime        # cases opened before it,
    not_closed_prior = ~(df['Closed'] < sometime) # not closed, 
    open_at_thattime = opened_prior & not_closed_prior  #and 
    return open_at_thattime.sum()
    
#workload = wrangler.working['Opened'].apply(calc_open_cases)

```


```python

```


```python
df['Analysis Neighborhoods'].value_counts(normalize=True,dropna=False)
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(10,5))
ax = sns.countplot(x='Analysis Neighborhoods',data=wrangler.working)

fig.show()
```

    /home/dliu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:7: UserWarning:
    
    Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
    



![png](output_18_1.png)



```python
##look at negativ deltas, 
df = wrangler.working 
neg_ttr = df.ttr.map(lambda td : td.total_seconds() < 0)
df[neg_ttr]['CaseID'].nunique() ,df[neg_ttr].shape       #one record per caseid here



```




    (329, (329, 25))




```python
wrangler.working.columns

```




    Index(['CaseID', 'Opened', 'Closed', 'Updated', 'Status', 'Status Notes',
           'Responsible Agency', 'Category', 'Request Type', 'Request Details',
           'Address', 'Street', 'Neighborhood', 'Police District', 'Latitude',
           'Longitude', 'Source', 'Media URL', 'Current Police Districts',
           'Analysis Neighborhoods', 'Neighborhoods', 'ttr', 'case_year',
           'case_month', 'workload'],
          dtype='object')




```python
wrangler.working['Request Type'].value_counts()
```




    Bulky Items                                                5463
    General Cleaning                                           4737
    Encampment Reports                                         1952
    Human or Animal Waste                                      1669
    Graffiti on Building_commercial                             842
                                                               ... 
    Construction Zone Tow-away Permits for Water Department       1
    Graffiti on Fire_hydrant_puc                                  1
    Building - Fire_Extinguishers_Missing_Expired                 1
    Construction Zone Tow-away Permits for Ranger Pipeline        1
    Construction Zone Tow-away Permits for DPW/Radio              1
    Name: Request Type, Length: 281, dtype: int64




```python
#pandas_profiling.ProfileReport(wrangler.working)
wrangler.working.Category.value_counts().index
```




    Index(['Street and Sidewalk Cleaning', 'Graffiti', 'Abandoned Vehicle',
           'Encampments', 'Parking Enforcement', 'other', 'Damaged Property',
           'Litter Receptacles', 'Street Defects', 'Illegal Postings',
           'Homeless Concerns', 'Blocked Street or SideWalk', 'Noise Report',
           '311 External Request'],
          dtype='object')




```python
wrangler.working.groupby(by=['case_year','case_month','Request Type'])[['CaseID','ttr']].agg({'CaseID': 'count'})
#produce stats by month and year
monthly_counts_by_request_type = wrangler.working.groupby(by=['case_year','case_month','Request Type']).agg({'CaseID': 'count', 'workload': np.mean, 'ttr': max})
monthly_counts_by_request_type.columns = ['case_count', 'avg_opencases', 'avg_ttr']
monthly_counts_by_request_type.sort_index(inplace=True)

monthly_counts = wrangler.working.groupby(by=['case_year','case_month']).agg({'CaseID': 'count', 'workload': np.mean, 'ttr': max})

```


```python
monthly_counts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>CaseID</th>
      <th>workload</th>
      <th>ttr</th>
    </tr>
    <tr>
      <th>case_year</th>
      <th>case_month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2008</th>
      <th>7</th>
      <td>91</td>
      <td>17.010989</td>
      <td>129 days 12:36:18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>93</td>
      <td>23.989247</td>
      <td>414 days 18:45:17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>87</td>
      <td>24.827586</td>
      <td>4269 days 00:36:47</td>
    </tr>
    <tr>
      <th>10</th>
      <td>76</td>
      <td>26.052632</td>
      <td>299 days 20:04:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>73</td>
      <td>27.547945</td>
      <td>235 days 00:55:26</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2020</th>
      <th>2</th>
      <td>472</td>
      <td>411.254237</td>
      <td>97 days 20:26:39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>351</td>
      <td>405.595442</td>
      <td>64 days 04:24:48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>294</td>
      <td>415.639456</td>
      <td>56 days 14:08:53</td>
    </tr>
    <tr>
      <th>5</th>
      <td>374</td>
      <td>432.756684</td>
      <td>32 days 22:09:44</td>
    </tr>
    <tr>
      <th>6</th>
      <td>71</td>
      <td>292.352113</td>
      <td>2 days 18:48:00</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 3 columns</p>
</div>




```python
monthly_counts_by_request_type.loc[(2009)]

monthly_counts.columns =['case_count', 'avg_opencases', 'avg_ttr']  # test regression on this  or remove
monthly_counts.plot(legend=False, figsize=(15,4))
```


```python
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates


daily = wrangler.working.copy()
daily['date'] = pd.DatetimeIndex(daily.Opened).date

daily_load=daily[['date','workload']].groupby('date').max()
daily_load.plot(legend=False, figsize=(15,3))
plt.title('daily workload: (#open cases per day)')
plt.show()

new_cases_daily = daily[['date','CaseID']].groupby('date').count() #daily new opens

new_cases_daily.plot(legend=False, figsize=(15,3))


```


![png](output_26_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x7f07c3104510>




![png](output_26_2.png)


### Linear trend within the times series :

Although the workload appears to be time serie attributes, we can see that there is a linear component as well:


```python
import plotly.express as px
from datetime import datetime

df_dl =daily_load.groupby('date').max()

df_dl['day_offset'] = (df_dl.index - df_dl.index[0]).days  #get an int represention of days since the first value


#need to create label list since we have to serialize the dates as ints for px to do the least squares
datelist = pd.date_range(datetime.today(), periods=df_dl.shape[0]).tolist()  #

fig = px.scatter(df_dl,x='day_offset', y='workload',trendline='ols')


fig.update_xaxes(tickangle=45,
                 tickmode = 'array',
                 tickvals = df_dl['day_offset'][0::40],
                 ticktext= [d.strftime('%Y-%m-%d') for d in datelist])

   
fig.show()
```


<div>


            <div id="acc44288-4978-40f6-8686-1f23e606d105" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("acc44288-4978-40f6-8686-1f23e606d105")) {
                    Plotly.newPlot(
                        'acc44288-4978-40f6-8686-1f23e606d105',
                        [{"hovertemplate": "day_offset=%{x}<br>workload=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scattergl", "x": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 309, 311, 313, 314, 315, 316, 318, 319, 321, 322, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 433, 434, 435, 436, 438, 439, 440, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 661, 662, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 860, 861, 862, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 901, 902, 903, 904, 905, 907, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 972, 973, 974, 975, 976, 978, 979, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1082, 1084, 1085, 1086, 1087, 1088, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1112, 1113, 1114, 1115, 1116, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1252, 1253, 1254, 1255, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1406, 1407, 1408, 1409, 1410, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1600, 1601, 1602, 1603, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358], "xaxis": "x", "y": [5, 4, 6, 7, 9, 9, 12, 14, 11, 14, 15, 15, 17, 20, 24, 22, 25, 25, 25, 25, 23, 18, 19, 21, 20, 20, 27, 25, 22, 23, 23, 18, 20, 20, 22, 24, 25, 25, 26, 25, 23, 24, 26, 26, 26, 24, 24, 25, 28, 27, 29, 27, 29, 31, 28, 27, 21, 23, 24, 27, 27, 27, 28, 29, 30, 19, 20, 22, 20, 16, 17, 21, 21, 17, 19, 20, 24, 28, 29, 31, 34, 36, 39, 34, 37, 30, 28, 26, 26, 27, 26, 28, 23, 23, 24, 23, 26, 28, 29, 25, 23, 26, 30, 31, 29, 27, 24, 23, 22, 22, 24, 28, 28, 28, 29, 27, 27, 27, 30, 31, 32, 33, 35, 35, 33, 31, 30, 28, 29, 25, 30, 28, 27, 25, 25, 27, 24, 25, 23, 25, 24, 22, 26, 27, 28, 28, 28, 27, 30, 34, 32, 32, 33, 33, 31, 34, 34, 31, 32, 33, 34, 33, 35, 33, 33, 29, 29, 30, 34, 32, 33, 37, 37, 39, 42, 45, 41, 37, 40, 42, 46, 46, 45, 44, 45, 46, 42, 45, 47, 48, 46, 45, 50, 51, 50, 54, 55, 57, 56, 56, 58, 57, 58, 58, 53, 54, 55, 59, 58, 57, 54, 55, 56, 55, 56, 57, 57, 57, 54, 52, 53, 54, 55, 54, 55, 53, 55, 55, 58, 58, 56, 49, 55, 55, 56, 57, 58, 53, 55, 54, 54, 55, 56, 53, 56, 57, 56, 56, 56, 57, 51, 49, 47, 52, 57, 56, 56, 54, 61, 60, 59, 59, 56, 53, 56, 54, 52, 56, 59, 55, 56, 58, 57, 58, 57, 59, 60, 61, 64, 64, 69, 70, 69, 65, 62, 63, 65, 67, 69, 70, 68, 65, 65, 71, 69, 68, 68, 69, 77, 77, 76, 80, 82, 84, 85, 90, 92, 94, 97, 70, 73, 74, 73, 70, 65, 70, 70, 70, 68, 68, 67, 71, 73, 73, 73, 72, 70, 72, 76, 79, 79, 82, 82, 84, 78, 81, 86, 82, 82, 83, 82, 83, 87, 86, 83, 83, 85, 85, 87, 89, 88, 89, 90, 87, 86, 90, 90, 85, 85, 87, 91, 94, 94, 93, 90, 92, 93, 93, 93, 90, 89, 87, 86, 90, 89, 85, 86, 85, 84, 84, 89, 85, 83, 81, 83, 80, 81, 80, 82, 72, 77, 76, 79, 80, 83, 81, 80, 80, 81, 80, 80, 78, 81, 80, 81, 84, 80, 78, 79, 81, 82, 86, 83, 82, 86, 88, 91, 99, 95, 90, 89, 92, 95, 91, 90, 84, 67, 74, 73, 69, 68, 70, 72, 71, 75, 72, 71, 72, 75, 75, 77, 80, 83, 82, 81, 82, 80, 83, 83, 79, 77, 77, 75, 75, 78, 77, 70, 71, 68, 72, 74, 77, 76, 77, 75, 76, 74, 72, 72, 72, 71, 73, 77, 78, 80, 84, 85, 85, 85, 88, 87, 87, 87, 83, 74, 76, 78, 78, 80, 77, 76, 73, 71, 73, 70, 71, 68, 69, 72, 72, 74, 77, 76, 71, 73, 71, 74, 73, 78, 76, 77, 75, 77, 74, 75, 80, 81, 79, 82, 81, 81, 81, 82, 82, 83, 83, 83, 83, 85, 86, 92, 96, 96, 97, 95, 96, 102, 101, 93, 89, 88, 88, 88, 85, 86, 89, 89, 89, 91, 92, 93, 90, 87, 88, 89, 91, 92, 93, 90, 93, 93, 91, 93, 97, 97, 99, 97, 96, 97, 94, 96, 91, 86, 83, 80, 80, 81, 86, 87, 90, 88, 90, 91, 96, 94, 81, 89, 84, 82, 82, 83, 82, 83, 81, 81, 77, 79, 83, 83, 76, 79, 82, 81, 84, 85, 82, 77, 78, 78, 80, 85, 84, 84, 81, 80, 87, 88, 85, 85, 85, 86, 87, 91, 92, 94, 98, 102, 104, 106, 108, 113, 113, 76, 77, 77, 77, 74, 75, 77, 76, 75, 75, 80, 81, 73, 76, 77, 78, 79, 79, 80, 83, 86, 86, 88, 89, 86, 81, 82, 80, 81, 83, 79, 80, 75, 80, 81, 84, 87, 90, 85, 83, 82, 78, 79, 79, 87, 89, 88, 84, 85, 85, 88, 88, 91, 87, 91, 90, 93, 94, 92, 93, 92, 91, 91, 90, 90, 86, 86, 84, 83, 81, 81, 88, 89, 86, 86, 89, 91, 96, 101, 93, 89, 89, 86, 88, 87, 90, 90, 81, 88, 91, 95, 96, 94, 93, 92, 86, 90, 92, 93, 91, 87, 90, 83, 84, 88, 90, 84, 83, 85, 85, 83, 85, 89, 93, 94, 95, 98, 98, 97, 101, 99, 97, 99, 97, 99, 101, 102, 102, 100, 100, 102, 101, 100, 99, 100, 96, 99, 101, 96, 97, 96, 95, 93, 92, 95, 96, 97, 97, 100, 98, 98, 98, 101, 99, 100, 102, 100, 97, 96, 95, 93, 95, 94, 94, 94, 97, 100, 98, 100, 103, 103, 104, 100, 98, 99, 98, 98, 94, 96, 97, 102, 102, 99, 99, 100, 100, 101, 103, 96, 99, 97, 97, 94, 95, 93, 92, 90, 91, 92, 94, 98, 94, 93, 91, 93, 94, 92, 93, 97, 96, 96, 98, 100, 100, 99, 100, 99, 107, 104, 103, 101, 102, 101, 102, 104, 105, 106, 101, 102, 100, 101, 101, 99, 100, 100, 104, 108, 106, 102, 100, 101, 101, 104, 101, 101, 104, 107, 108, 109, 109, 111, 110, 111, 111, 111, 110, 111, 116, 116, 114, 114, 115, 113, 114, 117, 119, 119, 117, 114, 115, 119, 119, 119, 121, 122, 122, 123, 119, 119, 118, 115, 114, 114, 115, 114, 116, 113, 114, 115, 114, 114, 116, 118, 117, 118, 118, 122, 122, 123, 125, 116, 115, 114, 114, 115, 114, 112, 116, 115, 115, 114, 114, 118, 118, 116, 110, 101, 99, 98, 96, 96, 96, 96, 98, 95, 96, 96, 99, 102, 100, 101, 96, 94, 98, 99, 96, 95, 97, 94, 97, 98, 99, 100, 101, 100, 100, 101, 101, 101, 98, 98, 99, 100, 101, 103, 104, 105, 103, 103, 103, 98, 101, 99, 102, 103, 102, 99, 100, 101, 95, 96, 96, 97, 97, 98, 99, 98, 98, 100, 97, 95, 95, 95, 95, 95, 95, 98, 97, 98, 102, 98, 100, 98, 97, 96, 100, 102, 99, 100, 103, 103, 110, 111, 106, 100, 99, 100, 99, 99, 99, 103, 104, 99, 101, 100, 98, 94, 94, 95, 97, 98, 102, 102, 103, 101, 103, 101, 102, 98, 96, 96, 94, 96, 95, 98, 97, 96, 98, 98, 101, 101, 97, 94, 97, 100, 100, 101, 109, 106, 107, 113, 111, 109, 107, 107, 111, 112, 105, 107, 111, 111, 112, 108, 104, 107, 110, 109, 110, 110, 112, 110, 106, 110, 109, 108, 115, 120, 116, 114, 119, 121, 122, 123, 119, 120, 108, 110, 108, 112, 115, 117, 114, 112, 107, 108, 109, 112, 114, 113, 109, 107, 106, 103, 102, 102, 102, 105, 102, 101, 100, 105, 105, 102, 103, 103, 101, 101, 99, 102, 99, 98, 97, 101, 101, 104, 105, 106, 69, 63, 59, 58, 63, 61, 59, 59, 58, 59, 62, 66, 68, 68, 69, 64, 64, 67, 63, 64, 57, 56, 57, 59, 58, 56, 53, 54, 52, 52, 59, 56, 60, 60, 61, 60, 60, 61, 61, 60, 58, 55, 54, 55, 55, 54, 63, 60, 61, 58, 65, 64, 63, 66, 61, 59, 64, 68, 67, 64, 63, 60, 57, 56, 57, 57, 54, 57, 56, 59, 61, 56, 52, 52, 51, 51, 49, 49, 49, 49, 43, 43, 44, 43, 42, 44, 46, 45, 45, 53, 51, 47, 49, 48, 49, 49, 49, 46, 45, 49, 50, 50, 55, 59, 50, 53, 51, 53, 52, 51, 55, 51, 49, 51, 50, 51, 54, 53, 58, 59, 59, 56, 54, 54, 54, 55, 51, 50, 49, 50, 52, 49, 53, 45, 47, 44, 44, 45, 50, 53, 54, 50, 50, 51, 48, 47, 50, 50, 44, 48, 49, 43, 45, 43, 40, 41, 41, 41, 41, 40, 48, 50, 50, 53, 54, 54, 56, 57, 61, 62, 61, 64, 67, 67, 68, 69, 72, 80, 83, 85, 86, 89, 93, 96, 99, 104, 109, 111, 113, 111, 115, 121, 124, 127, 130, 132, 133, 131, 132, 135, 136, 137, 137, 138, 140, 144, 153, 156, 156, 158, 159, 122, 52, 48, 48, 53, 62, 62, 58, 58, 56, 53, 56, 56, 56, 57, 54, 55, 55, 56, 59, 59, 56, 55, 54, 53, 53, 57, 54, 58, 59, 61, 64, 59, 59, 60, 58, 56, 60, 65, 63, 63, 58, 57, 62, 65, 64, 65, 67, 70, 70, 69, 69, 71, 69, 67, 66, 66, 67, 68, 65, 70, 67, 70, 71, 75, 73, 73, 75, 78, 76, 74, 74, 74, 74, 77, 75, 73, 78, 78, 74, 75, 75, 74, 73, 74, 74, 77, 78, 78, 78, 79, 83, 84, 86, 86, 90, 90, 91, 92, 93, 93, 93, 95, 87, 87, 85, 90, 93, 92, 91, 88, 85, 89, 90, 90, 93, 91, 84, 85, 84, 85, 86, 87, 89, 88, 91, 91, 91, 91, 87, 85, 85, 86, 88, 82, 85, 87, 88, 86, 87, 89, 88, 84, 85, 84, 87, 91, 92, 93, 93, 93, 89, 90, 89, 89, 93, 92, 94, 92, 92, 94, 85, 84, 87, 86, 87, 86, 84, 82, 85, 86, 83, 83, 85, 88, 86, 87, 85, 87, 88, 92, 85, 87, 89, 90, 92, 91, 89, 89, 90, 90, 90, 92, 94, 94, 95, 95, 98, 101, 99, 99, 105, 110, 109, 104, 104, 108, 115, 115, 115, 116, 115, 116, 112, 110, 108, 113, 113, 107, 106, 106, 109, 108, 108, 107, 107, 105, 105, 109, 111, 110, 109, 106, 105, 106, 106, 100, 100, 103, 103, 107, 107, 109, 110, 108, 109, 107, 102, 102, 100, 97, 98, 101, 104, 107, 107, 110, 111, 101, 102, 106, 103, 101, 105, 109, 109, 112, 110, 115, 114, 113, 104, 103, 105, 108, 108, 109, 106, 108, 111, 110, 104, 108, 108, 107, 112, 109, 113, 112, 111, 115, 118, 117, 114, 119, 122, 119, 122, 118, 118, 119, 118, 115, 114, 116, 119, 122, 121, 118, 118, 122, 125, 122, 122, 117, 118, 117, 118, 118, 121, 121, 124, 124, 125, 126, 125, 130, 131, 132, 126, 124, 127, 127, 129, 131, 132, 133, 134, 136, 131, 132, 133, 131, 129, 128, 129, 128, 127, 130, 128, 127, 127, 129, 131, 130, 131, 130, 135, 136, 136, 137, 140, 139, 134, 136, 135, 139, 140, 137, 129, 130, 129, 133, 131, 130, 127, 128, 129, 129, 129, 129, 130, 129, 130, 134, 138, 140, 141, 141, 140, 143, 141, 138, 140, 139, 140, 146, 145, 144, 143, 144, 145, 144, 141, 141, 143, 145, 146, 143, 137, 140, 136, 137, 135, 134, 139, 140, 145, 147, 148, 146, 149, 150, 153, 146, 144, 142, 143, 143, 144, 146, 144, 142, 146, 148, 149, 147, 144, 150, 150, 147, 148, 147, 147, 145, 145, 142, 142, 145, 146, 147, 142, 145, 140, 143, 144, 143, 148, 145, 148, 147, 149, 149, 154, 155, 157, 156, 150, 150, 151, 155, 156, 152, 152, 153, 156, 157, 155, 153, 153, 157, 151, 150, 149, 148, 145, 144, 141, 145, 146, 147, 151, 152, 150, 147, 151, 149, 148, 150, 147, 148, 149, 155, 159, 154, 151, 148, 149, 148, 149, 154, 157, 158, 145, 143, 143, 140, 144, 145, 146, 146, 148, 145, 146, 149, 150, 151, 148, 145, 146, 148, 150, 149, 154, 151, 150, 147, 151, 156, 158, 154, 151, 152, 149, 154, 156, 158, 162, 165, 166, 165, 159, 161, 163, 159, 151, 154, 157, 156, 156, 163, 161, 160, 156, 155, 155, 156, 158, 155, 157, 153, 160, 160, 158, 151, 151, 153, 153, 155, 152, 151, 152, 151, 151, 155, 157, 158, 161, 164, 166, 168, 174, 174, 174, 176, 180, 182, 185, 183, 183, 187, 192, 194, 193, 191, 192, 163, 153, 152, 156, 158, 158, 155, 156, 157, 159, 162, 161, 160, 155, 156, 159, 160, 166, 168, 168, 161, 164, 165, 168, 169, 169, 172, 169, 169, 167, 166, 166, 167, 164, 163, 160, 161, 159, 161, 163, 161, 161, 154, 156, 157, 160, 156, 158, 162, 163, 163, 166, 165, 166, 168, 169, 169, 166, 168, 170, 170, 171, 170, 168, 173, 173, 176, 175, 175, 172, 171, 170, 172, 174, 174, 171, 172, 173, 172, 172, 178, 182, 185, 186, 183, 181, 180, 182, 179, 182, 183, 185, 186, 189, 193, 190, 196, 194, 193, 191, 193, 194, 194, 192, 191, 192, 194, 193, 192, 194, 196, 194, 199, 198, 200, 205, 199, 200, 201, 202, 203, 207, 210, 211, 213, 211, 213, 213, 215, 220, 220, 218, 215, 215, 218, 220, 226, 223, 220, 224, 223, 225, 227, 227, 229, 227, 228, 228, 229, 229, 226, 228, 226, 228, 223, 226, 226, 233, 232, 229, 231, 232, 230, 231, 236, 240, 239, 240, 239, 242, 243, 247, 246, 249, 251, 252, 252, 255, 255, 258, 261, 261, 262, 264, 268, 270, 261, 264, 264, 264, 264, 267, 264, 262, 258, 251, 245, 245, 249, 253, 251, 249, 246, 240, 241, 245, 250, 251, 248, 253, 244, 247, 250, 253, 251, 242, 240, 230, 237, 236, 238, 239, 234, 229, 229, 229, 228, 230, 228, 225, 233, 236, 238, 237, 237, 230, 231, 233, 235, 236, 237, 232, 230, 231, 223, 224, 222, 222, 222, 224, 224, 229, 226, 226, 224, 228, 224, 223, 223, 225, 228, 229, 231, 230, 230, 235, 237, 238, 244, 239, 245, 245, 246, 245, 246, 248, 248, 247, 245, 245, 249, 248, 253, 254, 248, 249, 248, 253, 257, 258, 255, 257, 257, 256, 256, 256, 257, 259, 257, 256, 260, 260, 261, 262, 262, 266, 264, 267, 266, 263, 268, 263, 261, 265, 264, 263, 262, 264, 264, 264, 262, 263, 265, 268, 274, 268, 268, 268, 273, 269, 273, 277, 280, 280, 282, 282, 282, 284, 286, 283, 286, 284, 280, 279, 283, 282, 283, 281, 279, 283, 290, 288, 287, 285, 287, 286, 285, 289, 293, 293, 274, 269, 267, 265, 268, 273, 275, 277, 277, 276, 278, 280, 284, 279, 279, 282, 285, 289, 290, 289, 287, 285, 288, 289, 289, 293, 295, 296, 300, 297, 294, 292, 296, 295, 287, 286, 292, 289, 282, 283, 285, 278, 277, 275, 272, 271, 272, 271, 256, 255, 241, 238, 234, 232, 230, 222, 215, 214, 209, 203, 205, 206, 200, 201, 194, 199, 195, 199, 201, 190, 163, 166, 163, 155, 161, 162, 155, 145, 147, 142, 144, 145, 147, 145, 140, 141, 134, 139, 144, 140, 136, 143, 146, 146, 147, 148, 146, 151, 154, 147, 149, 151, 157, 160, 158, 158, 157, 157, 159, 156, 157, 155, 150, 155, 157, 158, 161, 162, 159, 160, 159, 160, 162, 158, 156, 160, 158, 155, 156, 156, 161, 160, 165, 167, 168, 166, 166, 167, 168, 165, 168, 168, 170, 174, 174, 173, 170, 165, 165, 165, 166, 173, 172, 168, 168, 169, 152, 154, 155, 162, 161, 166, 167, 169, 172, 172, 168, 167, 171, 172, 176, 178, 180, 180, 179, 180, 184, 184, 185, 180, 176, 179, 181, 172, 171, 174, 179, 186, 184, 174, 176, 176, 179, 181, 181, 179, 180, 182, 184, 184, 184, 188, 188, 187, 184, 186, 185, 187, 187, 188, 190, 191, 191, 194, 192, 190, 189, 186, 191, 189, 189, 195, 199, 198, 197, 198, 201, 203, 207, 203, 202, 200, 199, 196, 201, 200, 203, 201, 203, 203, 201, 203, 202, 199, 197, 197, 197, 198, 198, 197, 194, 199, 201, 199, 198, 199, 205, 198, 196, 204, 207, 209, 210, 205, 209, 208, 209, 211, 212, 217, 217, 218, 218, 213, 221, 220, 219, 216, 213, 214, 211, 209, 212, 213, 215, 213, 211, 210, 207, 209, 209, 212, 208, 208, 206, 206, 208, 209, 212, 210, 210, 212, 216, 217, 219, 225, 218, 217, 217, 215, 218, 219, 220, 219, 217, 219, 214, 209, 209, 213, 211, 218, 220, 220, 224, 222, 224, 222, 230, 231, 225, 224, 226, 237, 236, 233, 235, 232, 235, 232, 232, 232, 229, 231, 230, 231, 231, 235, 231, 224, 226, 224, 223, 223, 228, 224, 225, 221, 208, 210, 209, 210, 209, 211, 213, 211, 208, 218, 216, 215, 215, 214, 213, 216, 221, 224, 225, 226, 233, 231, 231, 232, 226, 229, 227, 225, 223, 220, 222, 226, 229, 230, 231, 230, 226, 230, 230, 230, 231, 237, 235, 234, 234, 237, 198, 198, 198, 198, 196, 199, 203, 204, 197, 200, 198, 199, 200, 199, 197, 191, 190, 177, 185, 187, 188, 183, 197, 196, 192, 185, 189, 191, 187, 184, 186, 187, 189, 189, 189, 191, 186, 191, 191, 194, 195, 196, 199, 201, 204, 205, 199, 201, 202, 201, 197, 202, 203, 199, 201, 204, 206, 206, 200, 203, 202, 200, 202, 202, 202, 206, 201, 203, 202, 203, 201, 198, 200, 201, 203, 206, 206, 199, 199, 200, 201, 201, 202, 201, 203, 204, 192, 197, 198, 197, 197, 197, 201, 201, 203, 205, 207, 206, 211, 210, 210, 211, 215, 219, 221, 214, 216, 218, 217, 216, 222, 221, 220, 223, 219, 220, 220, 226, 226, 225, 228, 225, 220, 221, 220, 219, 223, 224, 228, 228, 226, 225, 206, 202, 202, 209, 215, 219, 228, 225, 223, 226, 229, 233, 234, 234, 242, 245, 248, 254, 264, 267, 279, 285, 285, 289, 294, 296, 300, 306, 306, 311, 317, 325, 327, 333, 339, 339, 344, 346, 344, 348, 352, 360, 363, 363, 370, 375, 379, 382, 387, 393, 398, 400, 403, 402, 406, 408, 201, 204, 209, 211, 213, 218, 219, 219, 219, 217, 211, 212, 218, 220, 218, 215, 220, 217, 220, 224, 224, 225, 223, 226, 222, 228, 229, 233, 232, 233, 234, 237, 243, 242, 242, 241, 239, 235, 237, 236, 241, 246, 255, 255, 258, 254, 255, 256, 262, 259, 250, 249, 240, 238, 240, 224, 230, 234, 235, 232, 228, 226, 224, 219, 217, 219, 214, 212, 212, 207, 212, 213, 218, 223, 220, 221, 226, 224, 226, 226, 235, 236, 237, 244, 239, 238, 237, 239, 242, 246, 253, 259, 259, 248, 246, 250, 250, 248, 245, 242, 238, 243, 242, 238, 238, 239, 241, 241, 245, 244, 248, 246, 243, 246, 245, 253, 252, 253, 259, 260, 264, 258, 251, 247, 248, 248, 247, 246, 250, 254, 257, 256, 254, 252, 252, 247, 241, 243, 242, 246, 251, 253, 254, 256, 259, 258, 260, 259, 259, 262, 260, 264, 263, 263, 264, 271, 271, 268, 272, 274, 273, 277, 281, 282, 281, 275, 277, 278, 271, 269, 272, 273, 272, 274, 276, 274, 272, 275, 271, 272, 272, 271, 278, 278, 278, 275, 272, 271, 277, 269, 267, 268, 264, 263, 264, 271, 274, 272, 275, 272, 270, 265, 267, 263, 263, 266, 265, 265, 268, 271, 276, 276, 284, 285, 286, 287, 291, 292, 290, 296, 297, 295, 297, 299, 300, 296, 294, 291, 293, 299, 308, 300, 300, 301, 304, 305, 303, 306, 301, 299, 297, 298, 299, 289, 295, 300, 300, 303, 307, 307, 307, 310, 312, 310, 311, 309, 303, 302, 307, 309, 312, 315, 312, 311, 312, 311, 316, 313, 318, 319, 325, 328, 327, 321, 323, 323, 325, 320, 325, 330, 332, 339, 340, 341, 337, 337, 337, 336, 339, 344, 349, 348, 345, 344, 342, 349, 345, 336, 340, 340, 338, 345, 346, 349, 344, 341, 344, 345, 349, 348, 352, 359, 359, 359, 350, 357, 358, 358, 358, 360, 361, 366, 366, 367, 367, 368, 371, 367, 374, 373, 366, 367, 368, 366, 370, 373, 373, 379, 379, 382, 385, 385, 385, 381, 380, 378, 373, 379, 374, 376, 370, 368, 362, 359, 366, 367, 370, 368, 368, 366, 353, 345, 345, 356, 352, 351, 353, 354, 354, 353, 350, 348, 347, 342, 341, 343, 348, 346, 348, 350, 349, 344, 351, 352, 356, 356, 355, 358, 358, 357, 360, 361, 361, 360, 357, 355, 355, 356, 361, 358, 357, 350, 348, 349, 350, 351, 352, 353, 342, 344, 348, 351, 350, 348, 347, 349, 352, 354, 352, 348, 354, 359, 358, 357, 356, 359, 360, 355, 354, 353, 358, 361, 365, 367, 371, 368, 368, 366, 362, 361, 362, 365, 364, 360, 358, 360, 360, 358, 357, 359, 362, 364, 367, 373, 381, 386, 369, 365, 365, 366, 368, 372, 375, 376, 379, 377, 382, 383, 392, 383, 380, 378, 378, 377, 376, 382, 383, 382, 382, 381, 386, 386, 386, 384, 385, 389, 387, 387, 383, 389, 393, 390, 392, 396, 395, 394, 397, 399, 398, 390, 387, 391, 397, 398, 400, 399, 399, 393, 391, 387, 387, 388, 385, 386, 383, 386, 382, 385, 385, 395, 395, 395, 392, 397, 394, 392, 391, 393, 396, 397, 397, 399, 395, 396, 399, 399, 398, 401, 406, 407, 407, 406, 407, 413, 418, 412, 415, 410, 408, 404, 416, 416, 413, 415, 420, 423, 414, 417, 422, 428, 428, 432, 432, 428, 431, 440, 445, 445, 443, 436, 436, 438, 437, 437, 438, 436, 435, 436, 435, 438, 437, 429, 424, 424, 422, 425, 427, 434, 436, 431, 434, 438, 441, 443, 450, 449, 449, 452, 455, 449, 449, 455, 453, 452, 457, 454, 456, 457, 461, 461, 455, 446, 454, 461, 459, 463, 455, 457, 454, 449, 455, 456, 455, 463, 462, 467, 477, 476, 471, 473, 472, 470, 468, 467, 465, 468, 467, 464, 464, 465, 471, 470, 476, 478, 472, 474, 473, 472, 472, 471, 474, 483, 489, 486, 484, 482, 485, 490, 490, 487, 486, 482, 485, 489, 491, 491, 492, 485, 482, 486, 481, 483, 488, 491, 487, 480, 480, 484, 480, 478, 481, 485, 486, 488, 486, 489, 491, 497, 499, 497, 482, 478, 484, 484, 487, 481, 475, 478, 476, 473, 475, 475, 470, 472, 474, 467, 472, 474, 477, 480, 483, 483, 485, 489, 487, 487, 489, 491, 491, 491, 495, 498, 496, 496, 496, 496, 498, 499, 498, 501, 503, 505, 505, 507, 506, 507, 504, 498, 499, 500, 506, 510, 508, 502, 509, 506, 506, 510, 510, 513, 510, 512, 510, 510, 514, 514, 514, 509, 503, 501, 506, 505, 509, 515, 513, 511, 514, 510, 513, 513, 516, 521, 521, 520, 523, 526, 533, 527, 519, 519, 521, 526, 528, 533, 526, 527, 521, 518, 519, 522, 520, 520, 512, 503, 501, 494, 492, 491, 488, 489, 486, 472, 470, 471, 471, 472, 478, 478, 476, 475, 473, 476, 477, 475, 477, 471, 461, 462, 464, 467, 465, 455, 444, 445, 446, 443, 441, 435, 423, 418, 418, 414, 418, 423, 416, 406, 405, 397, 392, 394, 391, 390, 388, 384, 373, 374, 371, 367, 370, 365, 364, 355, 355, 352, 355, 354, 361, 364, 365, 366, 361, 357, 363, 369, 366, 358, 361, 363, 359, 357, 352, 348, 347, 346, 346, 347, 337, 340, 336, 336, 335, 334, 333, 332, 331, 328, 330, 334, 335, 337, 337, 343, 343, 338, 341, 345, 340, 336, 333, 331, 324, 328, 337, 337, 332, 332, 333, 338, 341, 343, 346, 341, 345, 345, 348, 348, 352, 343, 343, 337, 342, 343, 352, 354, 353, 346, 347, 347, 355, 355, 355, 351, 351, 351, 357, 357, 358, 360, 351, 352, 356, 356, 353, 350, 356, 354, 360, 353, 354, 350, 351, 350, 355, 351, 349, 353, 354, 360, 364, 360, 353, 356, 348, 347, 348, 349, 346, 347, 349, 352, 354, 355, 362, 357, 359, 356, 355, 352, 356, 356, 354, 353, 356, 361, 358, 359, 361, 368, 370, 367, 374, 374, 367, 370, 372, 374, 375, 378, 380, 383, 384, 376, 376, 375, 375, 380, 384, 385, 378, 372, 371, 376, 374, 375, 384, 377, 377, 368, 369, 371, 372, 377, 373, 374, 371, 370, 372, 376, 376, 378, 376, 377, 377, 379, 380, 383, 385, 382, 382, 375, 377, 383, 379, 373, 379, 378, 384, 386, 391, 395, 396, 392, 393, 397, 393, 397, 394, 389, 382, 380, 390, 388, 390, 391, 388, 381, 376, 377, 380, 384, 382, 381, 381, 381, 380, 378, 380, 381, 386, 387, 385, 387, 385, 390, 392, 392, 380, 380, 380, 380, 379, 389, 396, 391, 388, 387, 393, 399, 396, 401, 398, 389, 389, 393, 389, 393, 395, 395, 394, 396, 404, 405, 397, 399, 400, 392, 388, 387, 387, 391, 399, 398, 402, 398, 407, 406, 409, 405, 406, 405, 402, 399, 402, 402, 408, 405, 402, 402, 401, 405, 404, 401, 395, 396, 401, 395, 396, 398, 403, 402, 404, 408, 411, 409, 411, 408, 410, 412, 416, 419, 413, 412, 407, 406, 405, 404, 400, 402, 402, 402, 407, 411, 401, 403, 407, 407, 406, 404, 401, 398, 400, 409, 406, 408, 408, 406, 412, 413, 410, 413, 410, 405, 403, 400, 397, 397, 401, 395, 395, 395, 396, 401, 403, 406, 404, 402, 404, 403, 401, 402, 408, 404, 409, 413, 417, 422, 422, 428, 427, 424, 421, 423, 423, 424, 426, 421, 423, 418, 419, 421, 428, 429, 424, 414, 411, 409, 410, 411, 408, 413, 411, 409, 407, 401, 401, 396, 393, 398, 396, 394, 394, 401, 403, 403, 400, 400, 404, 405, 409, 413, 404, 409, 412, 415, 418, 421, 427, 424, 422, 418, 411, 413, 415, 419, 417, 417, 414, 413, 415, 419, 423, 419, 418, 416, 419, 418, 420, 421, 421, 421, 420, 419, 421, 423, 432, 429, 429, 428, 428, 429, 440, 442, 436, 438, 437, 439, 428, 435, 430, 430, 430, 428, 430, 435, 435, 440, 442, 446, 446, 444, 449, 453, 452, 445, 296, 253, 220, 226], "yaxis": "y"}, {"hovertemplate": "<b>OLS trendline</b><br>workload = 0.0987404 * day_offset + -11.2688<br>R<sup>2</sup>=0.845667<br><br>day_offset=%{x}<br>workload=%{y} <b>(trend)</b><extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 309, 311, 313, 314, 315, 316, 318, 319, 321, 322, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 433, 434, 435, 436, 438, 439, 440, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 661, 662, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 860, 861, 862, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 901, 902, 903, 904, 905, 907, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 972, 973, 974, 975, 976, 978, 979, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1082, 1084, 1085, 1086, 1087, 1088, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1112, 1113, 1114, 1115, 1116, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1252, 1253, 1254, 1255, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1406, 1407, 1408, 1409, 1410, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1600, 1601, 1602, 1603, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358], "xaxis": "x", "y": [-11.268770952768818, -11.170030545233013, -11.071290137697208, -10.873809322625595, -10.77506891508979, -10.676328507553984, -10.577588100018179, -10.478847692482374, -10.380107284946568, -10.281366877410761, -10.182626469874956, -10.08388606233915, -9.985145654803345, -9.88640524726754, -9.787664839731734, -9.688924432195929, -9.590184024660122, -9.491443617124316, -9.392703209588511, -9.293962802052706, -9.1952223945169, -9.096481986981095, -8.99774157944529, -8.899001171909482, -8.800260764373677, -8.701520356837872, -8.602779949302066, -8.50403954176626, -8.405299134230455, -8.30655872669465, -8.207818319158843, -8.010337504087232, -7.911597096551427, -7.8128566890156215, -7.714116281479816, -7.61537587394401, -7.516635466408204, -7.417895058872398, -7.319154651336593, -7.2204142438007874, -7.121673836264982, -7.022933428729176, -6.92419302119337, -6.825452613657565, -6.726712206121759, -6.627971798585953, -6.529231391050148, -6.430490983514342, -6.331750575978536, -6.134269760906926, -6.035529353371119, -5.936788945835314, -5.838048538299509, -5.739308130763702, -5.640567723227897, -5.541827315692092, -5.443086908156285, -5.34434650062048, -5.048125278013063, -4.949384870477258, -4.850644462941452, -4.751904055405646, -4.653163647869841, -4.554423240334035, -4.45568283279823, -4.356942425262424, -4.258202017726618, -4.159461610190813, -4.060721202655007, -3.7644999800475896, -3.665759572511784, -3.567019164975979, -3.4682787574401734, -3.369538349904367, -3.270797942368562, -3.1720575348327564, -3.073317127296951, -2.9745767197611457, -2.8758363122253385, -2.777095904689533, -2.678355497153728, -2.5796150896179224, -2.480874682082117, -2.3821342745463117, -2.2833938670105045, -2.184653459474699, -2.0859130519388938, -1.9871726444030884, -1.888432236867283, -1.7896918293314776, -1.6909514217956723, -1.592211014259865, -1.4934706067240597, -1.295989791652449, -1.1972493841166436, -1.0985089765808382, -0.9997685690450329, -0.9010281615092257, -0.8022877539734203, -0.703547346437615, -0.5060665313660042, -0.40732612383019884, -0.30858571629439346, -0.2098453087585863, -0.11110490122278094, -0.012364493686975564, 0.08637591384882981, 0.18511632138463519, 0.28385672892044056, 0.3825971364562477, 0.4813375439920531, 0.5800779515278585, 0.6788183590636638, 0.7775587665994692, 0.8762991741352746, 1.073779989206887, 1.1725203967426925, 1.2712608042784979, 1.3700012118143032, 1.4687416193501086, 1.567482026885914, 1.6662224344217194, 1.7649628419575265, 1.8637032494933319, 1.9624436570291373, 2.0611840645649426, 2.159924472100748, 2.3574052871723588, 2.456145694708166, 2.5548861022439713, 2.6536265097797767, 2.752366917315582, 2.8511073248513874, 2.949847732387193, 3.048588139923, 3.1473285474588053, 3.2460689549946107, 3.344809362530416, 3.4435497700662214, 3.542290177602027, 3.641030585137832, 3.7397709926736393, 3.8385114002094447, 3.93725180774525, 4.0359922152810555, 4.134732622816861, 4.233473030352666, 4.332213437888472, 4.430953845424279, 4.529694252960084, 4.6284346604958895, 4.727175068031695, 4.825915475567502, 4.924655883103306, 5.023396290639113, 5.122136698174916, 5.2208771057107235, 5.319617513246527, 5.418357920782334, 5.517098328318141, 5.615838735853945, 5.714579143389752, 5.813319550925556, 5.912059958461363, 6.0108003659971665, 6.109540773532974, 6.4057619961403915, 6.504502403676195, 6.603242811212002, 6.800723626283613, 6.89946403381942, 6.998204441355224, 7.096944848891031, 7.1956852564268345, 7.294425663962642, 7.393166071498449, 7.491906479034252, 7.59064688657006, 7.689387294105863, 7.78812770164167, 7.886868109177474, 7.985608516713281, 8.084348924249088, 8.183089331784892, 8.281829739320699, 8.380570146856503, 8.47931055439231, 8.578050961928113, 8.67679136946392, 8.775531776999728, 8.874272184535531, 8.973012592071338, 9.071752999607142, 9.17049340714295, 9.269233814678753, 9.36797422221456, 9.466714629750367, 9.56545503728617, 9.664195444821978, 9.762935852357781, 9.861676259893589, 9.960416667429392, 10.0591570749652, 10.157897482501006, 10.355378297572617, 10.45411870510842, 10.552859112644228, 10.651599520180032, 10.750339927715839, 10.849080335251646, 10.94782074278745, 11.046561150323257, 11.14530155785906, 11.244041965394867, 11.342782372930671, 11.441522780466478, 11.540263188002285, 11.639003595538089, 11.737744003073896, 11.8364844106097, 11.935224818145507, 12.033965225681314, 12.132705633217117, 12.231446040752925, 12.330186448288728, 12.428926855824535, 12.527667263360339, 12.626407670896146, 12.725148078431953, 12.823888485967757, 12.922628893503564, 13.021369301039368, 13.120109708575175, 13.218850116110978, 13.317590523646786, 13.515071338718396, 13.613811746254203, 13.712552153790007, 13.811292561325814, 13.910032968861618, 14.008773376397425, 14.107513783933232, 14.206254191469036, 14.304994599004843, 14.403735006540646, 14.502475414076454, 14.601215821612257, 14.798696636683871, 14.897437044219675, 14.996177451755482, 15.094917859291286, 15.193658266827093, 15.292398674362897, 15.391139081898704, 15.48987948943451, 15.588619896970314, 15.687360304506122, 15.786100712041925, 15.884841119577732, 15.983581527113536, 16.082321934649343, 16.18106234218515, 16.279802749720954, 16.37854315725676, 16.477283564792565, 16.57602397232837, 16.67476437986418, 16.773504787399983, 16.87224519493579, 17.0697260100074, 17.168466417543204, 17.26720682507901, 17.36594723261482, 17.464687640150622, 17.56342804768643, 17.662168455222233, 17.76090886275804, 17.859649270293843, 17.95838967782965, 18.057130085365458, 18.15587049290126, 18.25461090043707, 18.353351307972872, 18.45209171550868, 18.550832123044483, 18.64957253058029, 18.748312938116097, 18.8470533456519, 18.945793753187708, 19.04453416072351, 19.242014975795122, 19.439495790866737, 19.636976605938347, 19.73571701347415, 19.834457421009958, 19.93319782854576, 20.130678643617376, 20.22941905115318, 20.42689986622479, 20.525640273760597, 20.6243806812964, 20.723121088832208, 20.821861496368015, 20.920601903903822, 21.11808271897543, 21.216823126511237, 21.315563534047044, 21.41430394158285, 21.51304434911865, 21.61178475665446, 21.710525164190265, 21.809265571726073, 21.908005979261873, 22.00674638679768, 22.204227201869294, 22.3029676094051, 22.50044842447671, 22.599188832012516, 22.697929239548323, 22.79666964708413, 22.89541005461993, 22.994150462155737, 23.092890869691544, 23.19163127722735, 23.29037168476315, 23.38911209229896, 23.487852499834766, 23.586592907370573, 23.68533331490638, 23.882814129977987, 23.981554537513794, 24.0802949450496, 24.17903535258541, 24.27777576012121, 24.376516167657016, 24.475256575192823, 24.57399698272863, 24.672737390264437, 24.771477797800237, 24.870218205336045, 24.96895861287185, 25.16643942794346, 25.265179835479266, 25.363920243015073, 25.46266065055088, 25.561401058086687, 25.660141465622488, 25.758881873158295, 25.857622280694102, 25.95636268822991, 26.055103095765716, 26.153843503301516, 26.252583910837323, 26.35132431837313, 26.548805133444738, 26.647545540980545, 26.746285948516352, 26.84502635605216, 26.943766763587966, 27.042507171123766, 27.23998798619538, 27.338728393731188, 27.437468801266995, 27.536209208802795, 27.73369002387441, 27.832430431410216, 27.931170838946016, 28.029911246481824, 28.12865165401763, 28.227392061553438, 28.326132469089245, 28.523613284160852, 28.62235369169666, 28.721094099232467, 28.819834506768274, 28.918574914304074, 29.01731532183988, 29.116055729375688, 29.214796136911495, 29.313536544447302, 29.412276951983102, 29.51101735951891, 29.609757767054717, 29.708498174590524, 29.807238582126324, 29.90597898966213, 30.00471939719794, 30.103459804733745, 30.202200212269553, 30.300940619805353, 30.39968102734116, 30.498421434876967, 30.597161842412774, 30.69590224994858, 30.79464265748438, 30.89338306502019, 30.992123472555996, 31.189604287627603, 31.28834469516341, 31.485825510235024, 31.58456591777083, 31.68330632530663, 31.78204673284244, 31.979527547914053, 32.07826795544986, 32.17700836298566, 32.374489178057274, 32.47322958559308, 32.57196999312888, 32.67071040066469, 32.769450808200496, 32.8681912157363, 33.06567203080791, 33.16441243834372, 33.263152845879524, 33.36189325341533, 33.46063366095114, 33.55937406848694, 33.658114476022746, 33.75685488355855, 33.85559529109436, 33.95433569863016, 34.05307610616597, 34.25055692123758, 34.34929732877339, 34.44803773630919, 34.84299936645242, 34.94173977398822, 35.040480181524025, 35.13922058905983, 35.23796099659564, 35.336701404131446, 35.435441811667246, 35.53418221920305, 35.63292262673886, 35.73166303427467, 35.83040344181047, 35.929143849346275, 36.02788425688208, 36.12662466441789, 36.225365071953696, 36.324105479489496, 36.422845887025304, 36.52158629456111, 36.62032670209692, 36.719067109632725, 36.817807517168525, 36.91654792470433, 37.11402873977595, 37.21276914731175, 37.311509554847554, 37.41024996238336, 37.50899036991917, 37.607730777454975, 37.706471184990775, 37.80521159252658, 37.90395200006239, 38.0026924075982, 38.101432815134004, 38.200173222669804, 38.29891363020561, 38.39765403774142, 38.496394445277225, 38.595134852813025, 38.69387526034883, 38.79261566788464, 38.89135607542045, 39.088836890492054, 39.18757729802786, 39.28631770556367, 39.385058113099475, 39.48379852063528, 39.58253892817108, 39.68127933570689, 39.7800197432427, 39.878760150778504, 39.97750055831431, 40.07624096585011, 40.17498137338592, 40.273721780921726, 40.37246218845753, 40.47120259599333, 40.56994300352914, 40.66868341106495, 40.767423818600754, 40.86616422613656, 40.96490463367236, 41.06364504120817, 41.162385448743976, 41.26112585627978, 41.35986626381559, 41.5573470788872, 41.656087486423004, 41.75482789395881, 41.85356830149461, 41.95230870903042, 42.051049116566226, 42.14978952410203, 42.24852993163784, 42.34727033917364, 42.44601074670945, 42.544751154245255, 42.64349156178106, 42.74223196931687, 42.84097237685267, 42.939712784388476, 43.03845319192428, 43.13719359946009, 43.23593400699589, 43.3346744145317, 43.433414822067505, 43.53215522960331, 43.63089563713912, 43.72963604467492, 43.828376452210726, 43.92711685974653, 44.02585726728234, 44.12459767481815, 44.22333808235395, 44.322078489889755, 44.42081889742556, 44.51955930496137, 44.618299712497176, 44.717040120032976, 44.81578052756878, 44.91452093510459, 45.0132613426404, 45.1120017501762, 45.210742157712005, 45.30948256524781, 45.40822297278362, 45.50696338031943, 45.60570378785523, 45.704444195391034, 45.80318460292684, 45.90192501046265, 46.000665417998455, 46.099405825534255, 46.19814623307006, 46.29688664060587, 46.39562704814168, 46.49436745567748, 46.593107863213284, 46.7905886782849, 46.889329085820705, 46.988069493356505, 47.08680990089231, 47.18555030842812, 47.28429071596393, 47.383031123499734, 47.481771531035534, 47.58051193857134, 47.67925234610715, 47.777992753642955, 47.876733161178755, 47.97547356871456, 48.07421397625037, 48.17295438378618, 48.271694791321984, 48.370435198857784, 48.46917560639359, 48.5679160139294, 48.666656421465206, 48.76539682900101, 48.86413723653681, 48.96287764407262, 49.06161805160843, 49.160358459144234, 49.259098866680034, 49.35783927421584, 49.45657968175165, 49.555320089287456, 49.65406049682326, 49.75280090435906, 49.85154131189487, 49.95028171943068, 50.049022126966484, 50.14776253450229, 50.24650294203809, 50.3452433495739, 50.443983757109706, 50.54272416464551, 50.64146457218132, 50.74020497971712, 50.83894538725293, 51.03642620232454, 51.13516660986034, 51.23390701739615, 51.332647424931956, 51.43138783246776, 51.53012824000357, 51.62886864753937, 51.72760905507518, 51.826349462610985, 51.92508987014679, 52.0238302776826, 52.1225706852184, 52.221311092754206, 52.32005150029001, 52.51753231536162, 52.61627272289743, 52.715013130433235, 52.813753537969035, 52.91249394550485, 53.01123435304065, 53.10997476057646, 53.20871516811226, 53.30745557564806, 53.40619598318388, 53.50493639071968, 53.60367679825549, 53.70241720579129, 53.80115761332709, 53.998638428398706, 54.09737883593452, 54.29485965100612, 54.393600058541935, 54.492340466077735, 54.59108087361355, 54.68982128114935, 54.78856168868515, 54.887302096220964, 54.986042503756764, 55.084782911292564, 55.18352331882838, 55.28226372636418, 55.38100413389999, 55.57848494897159, 55.67722535650741, 55.77596576404321, 55.87470617157902, 55.97344657911482, 56.07218698665062, 56.170927394186435, 56.269667801722235, 56.36840820925805, 56.46714861679385, 56.56588902432965, 56.664629431865464, 56.763369839401264, 56.86211024693708, 56.96085065447288, 57.05959106200868, 57.15833146954449, 57.35581228461611, 57.45455269215191, 57.55329309968771, 57.65203350722352, 57.75077391475932, 57.84951432229512, 57.948254729830936, 58.046995137366736, 58.14573554490255, 58.24447595243835, 58.34321635997415, 58.441956767509964, 58.540697175045764, 58.63943758258158, 58.73817799011738, 58.83691839765318, 58.93565880518899, 59.03439921272479, 59.13313962026061, 59.23188002779641, 59.33062043533221, 59.42936084286802, 59.52810125040382, 59.626841657939636, 59.725582065475436, 59.824322473011236, 59.92306288054705, 60.02180328808285, 60.120543695618665, 60.219284103154465, 60.318024510690265, 60.41676491822608, 60.51550532576188, 60.61424573329769, 60.71298614083349, 60.81172654836929, 60.91046695590511, 61.00920736344091, 61.10794777097671, 61.20668817851252, 61.30542858604832, 61.404168993584136, 61.502909401119936, 61.601649808655736, 61.70039021619155, 61.79913062372735, 61.897871031263165, 61.996611438798965, 62.095351846334765, 62.19409225387058, 62.29283266140638, 62.39157306894219, 62.58905388401379, 62.68779429154961, 62.78653469908541, 62.88527510662122, 62.98401551415702, 63.08275592169282, 63.18149632922864, 63.28023673676444, 63.37897714430025, 63.47771755183605, 63.57645795937185, 63.675198366907665, 63.773938774443465, 63.97141958951508, 64.07015999705088, 64.1689004045867, 64.2676408121225, 64.3663812196583, 64.46512162719411, 64.56386203472991, 64.66260244226572, 64.76134284980152, 64.86008325733732, 64.95882366487314, 65.05756407240894, 65.15630447994475, 65.25504488748055, 65.35378529501635, 65.45252570255217, 65.55126611008797, 65.65000651762378, 65.74874692515958, 65.84748733269538, 65.9462277402312, 66.044968147767, 66.14370855530281, 66.24244896283861, 66.34118937037441, 66.43992977791022, 66.53867018544602, 66.63741059298184, 66.73615100051764, 66.83489140805344, 66.93363181558925, 67.03237222312505, 67.13111263066085, 67.22985303819667, 67.32859344573247, 67.42733385326828, 67.52607426080408, 67.62481466833988, 67.7235550758757, 67.8222954834115, 67.92103589094731, 68.01977629848311, 68.11851670601891, 68.21725711355472, 68.31599752109052, 68.41473792862634, 68.51347833616214, 68.61221874369794, 68.71095915123375, 68.80969955876955, 68.90843996630537, 69.00718037384117, 69.10592078137697, 69.20466118891278, 69.30340159644858, 69.4021420039844, 69.5008824115202, 69.599622819056, 69.69836322659181, 69.79710363412761, 69.89584404166342, 69.99458444919922, 70.09332485673502, 70.19206526427084, 70.29080567180664, 70.38954607934244, 70.48828648687825, 70.58702689441405, 70.68576730194987, 70.78450770948567, 70.88324811702147, 70.98198852455728, 71.08072893209308, 71.1794693396289, 71.2782097471647, 71.3769501547005, 71.47569056223631, 71.57443096977211, 71.67317137730792, 71.77191178484372, 71.87065219237952, 71.96939259991534, 72.06813300745114, 72.16687341498695, 72.26561382252275, 72.36435423005855, 72.46309463759437, 72.56183504513017, 72.66057545266598, 72.75931586020178, 72.85805626773758, 72.9567966752734, 73.0555370828092, 73.154277490345, 73.25301789788081, 73.35175830541661, 73.45049871295242, 73.64797952802402, 73.74671993555984, 73.84546034309564, 74.04294115816725, 74.14168156570305, 74.24042197323887, 74.33916238077467, 74.43790278831048, 74.53664319584628, 74.63538360338208, 74.7341240109179, 74.8328644184537, 74.93160482598951, 75.03034523352531, 75.12908564106111, 75.22782604859692, 75.32656645613272, 75.52404727120434, 75.62278767874014, 75.72152808627595, 75.82026849381175, 75.91900890134757, 76.01774930888337, 76.11648971641917, 76.21523012395498, 76.31397053149078, 76.41271093902658, 76.5114513465624, 76.6101917540982, 76.80767256916981, 76.90641297670561, 77.00515338424142, 77.10389379177722, 77.20263419931304, 77.30137460684884, 77.40011501438464, 77.49885542192045, 77.69633623699207, 77.79507664452787, 77.89381705206367, 77.99255745959948, 78.09129786713528, 78.2887786822069, 78.48625949727851, 78.58499990481431, 78.68374031235012, 78.78248071988592, 78.88122112742172, 78.97996153495754, 79.07870194249334, 79.17744235002914, 79.27618275756495, 79.37492316510075, 79.47366357263657, 79.57240398017237, 79.86862520277978, 79.9673656103156, 80.0661060178514, 80.1648464253872, 80.26358683292301, 80.36232724045881, 80.46106764799462, 80.55980805553043, 80.65854846306623, 80.75728887060204, 80.85602927813784, 80.95476968567365, 81.05351009320945, 81.15225050074525, 81.25099090828107, 81.34973131581687, 81.44847172335268, 81.54721213088848, 81.64595253842428, 81.7446929459601, 81.8434333534959, 81.94217376103171, 82.04091416856751, 82.13965457610331, 82.23839498363913, 82.33713539117493, 82.43587579871073, 82.53461620624654, 82.63335661378234, 82.73209702131815, 82.83083742885395, 82.92957783638975, 83.02831824392557, 83.12705865146137, 83.22579905899718, 83.32453946653298, 83.42327987406878, 83.5220202816046, 83.6207606891404, 83.71950109667621, 83.81824150421201, 83.91698191174781, 84.01572231928363, 84.11446272681943, 84.21320313435524, 84.31194354189104, 84.41068394942684, 84.50942435696265, 84.70690517203427, 84.80564557957007, 84.90438598710587, 85.00312639464168, 85.10186680217748, 85.2993476172491, 85.3980880247849, 85.59556883985651, 85.69430924739231, 85.79304965492813, 85.89179006246393, 85.99053046999974, 86.08927087753554, 86.18801128507134, 86.28675169260715, 86.38549210014295, 86.48423250767877, 86.58297291521457, 86.68171332275037, 86.78045373028618, 86.87919413782198, 86.9779345453578, 87.0766749528936, 87.1754153604294, 87.27415576796521, 87.37289617550101, 87.47163658303683, 87.57037699057263, 87.66911739810843, 87.76785780564424, 87.86659821318004, 87.96533862071585, 88.06407902825165, 88.16281943578745, 88.26155984332327, 88.36030025085907, 88.45904065839487, 88.55778106593068, 88.7552618810023, 88.8540022885381, 88.9527426960739, 89.05148310360971, 89.15022351114551, 89.24896391868133, 89.34770432621713, 89.44644473375293, 89.54518514128874, 89.64392554882454, 89.74266595636036, 89.84140636389616, 89.94014677143196, 90.03888717896777, 90.13762758650357, 90.23636799403938, 90.33510840157518, 90.43384880911098, 90.5325892166468, 90.6313296241826, 90.73007003171841, 90.82881043925421, 90.92755084679001, 91.02629125432583, 91.12503166186163, 91.22377206939744, 91.32251247693324, 91.42125288446904, 91.51999329200486, 91.61873369954066, 91.71747410707646, 91.81621451461227, 91.91495492214807, 92.01369532968388, 92.11243573721968, 92.21117614475548, 92.3099165522913, 92.4086569598271, 92.50739736736291, 92.60613777489871, 92.70487818243451, 92.80361858997033, 92.90235899750613, 93.00109940504194, 93.09983981257774, 93.19858022011354, 93.29732062764936, 93.39606103518516, 93.69228225779257, 93.79102266532838, 93.88976307286418, 93.9885034804, 94.0872438879358, 94.1859842954716, 94.28472470300741, 94.38346511054321, 94.48220551807901, 94.58094592561483, 94.67968633315063, 94.77842674068644, 94.87716714822224, 94.97590755575804, 95.07464796329386, 95.17338837082966, 95.27212877836547, 95.37086918590127, 95.56835000097288, 95.7658308160445, 95.8645712235803, 95.9633116311161, 96.06205203865191, 96.16079244618771, 96.45701366879513, 96.55575407633094, 96.65449448386674, 96.75323489140256, 96.85197529893836, 96.95071570647416, 97.04945611400997, 97.14819652154577, 97.24693692908158, 97.34567733661738, 97.44441774415318, 97.543158151689, 97.6418985592248, 97.7406389667606, 97.83937937429641, 97.93811978183221, 98.03686018936803, 98.13560059690383, 98.23434100443963, 98.33308141197544, 98.53056222704706, 98.62930263458286, 98.72804304211866, 98.82678344965447, 98.92552385719027, 99.12300467226189, 99.22174507979769, 99.3204854873335, 99.4192258948693, 99.51796630240511, 99.61670670994091, 99.71544711747671, 99.81418752501253, 99.91292793254833, 100.01166834008414, 100.11040874761994, 100.20914915515574, 100.30788956269156, 100.40662997022736, 100.50537037776317, 100.60411078529897, 100.70285119283477, 100.80159160037059, 100.90033200790639, 100.99907241544219, 101.097812822978, 101.1965532305138, 101.29529363804961, 101.39403404558541, 101.49277445312121, 101.59151486065703, 101.69025526819283, 101.78899567572864, 101.88773608326444, 102.08521689833606, 102.18395730587186, 102.28269771340767, 102.38143812094347, 102.48017852847927, 102.57891893601509, 102.67765934355089, 102.7763997510867, 102.8751401586225, 102.9738805661583, 103.07262097369411, 103.17136138122991, 103.27010178876573, 103.36884219630153, 103.46758260383733, 103.56632301137314, 103.66506341890894, 103.76380382644474, 103.86254423398056, 103.96128464151636, 104.06002504905217, 104.15876545658797, 104.25750586412377, 104.35624627165959, 104.45498667919539, 104.5537270867312, 104.652467494267, 104.7512079018028, 104.84994830933861, 104.94868871687441, 105.04742912441023, 105.14616953194603, 105.24490993948183, 105.34365034701764, 105.44239075455344, 105.54113116208926, 105.63987156962506, 105.73861197716086, 105.83735238469667, 105.93609279223247, 106.03483319976829, 106.13357360730409, 106.23231401483989, 106.3310544223757, 106.4297948299115, 106.52853523744731, 106.62727564498311, 106.72601605251891, 106.82475646005473, 106.92349686759053, 107.02223727512633, 107.12097768266214, 107.21971809019794, 107.31845849773376, 107.41719890526956, 107.51593931280536, 107.61467972034117, 107.71342012787697, 107.81216053541279, 107.91090094294859, 108.00964135048439, 108.1083817580202, 108.207122165556, 108.30586257309182, 108.40460298062762, 108.50334338816342, 108.60208379569923, 108.70082420323503, 108.79956461077084, 108.89830501830664, 108.99704542584244, 109.09578583337826, 109.19452624091406, 109.29326664844987, 109.39200705598567, 109.49074746352147, 109.58948787105729, 109.68822827859309, 109.78696868612889, 109.8857090936647, 109.9844495012005, 110.08318990873632, 110.18193031627212, 110.28067072380792, 110.37941113134373, 110.47815153887953, 110.57689194641534, 110.67563235395114, 110.77437276148694, 110.87311316902276, 110.97185357655856, 111.07059398409437, 111.16933439163017, 111.26807479916597, 111.36681520670179, 111.46555561423759, 111.5642960217734, 111.6630364293092, 111.761776836845, 111.86051724438082, 111.95925765191662, 112.05799805945243, 112.15673846698823, 112.35421928205984, 112.45295968959564, 112.55170009713146, 112.65044050466726, 112.84792131973887, 112.94666172727467, 113.04540213481047, 113.14414254234629, 113.24288294988209, 113.3416233574179, 113.4403637649537, 113.5391041724895, 113.63784458002532, 113.73658498756112, 113.83532539509693, 113.93406580263273, 114.03280621016853, 114.13154661770434, 114.23028702524014, 114.32902743277596, 114.42776784031176, 114.52650824784756, 114.62524865538337, 114.72398906291917, 114.82272947045499, 115.02021028552659, 115.1189506930624, 115.2176911005982, 115.31643150813402, 115.41517191566982, 115.51391232320562, 115.61265273074143, 115.71139313827723, 115.81013354581304, 115.90887395334885, 116.00761436088465, 116.10635476842046, 116.20509517595626, 116.30383558349206, 116.40257599102787, 116.50131639856367, 116.60005680609949, 116.69879721363529, 116.7975376211711, 116.89627802870689, 117.09375884377852, 117.19249925131433, 117.29123965885012, 117.38998006638593, 117.48872047392175, 117.58746088145753, 117.68620128899335, 117.78494169652916, 117.88368210406495, 117.98242251160076, 118.08116291913657, 118.17990332667236, 118.27864373420817, 118.57486495681559, 118.6736053643514, 118.77234577188722, 118.871086179423, 118.96982658695882, 119.06856699449463, 119.16730740203042, 119.26604780956623, 119.36478821710205, 119.46352862463786, 119.56226903217365, 119.66100943970946, 119.85849025478106, 119.95723066231687, 120.05597106985269, 120.15471147738847, 120.25345188492429, 120.3521922924601, 120.45093269999592, 120.5496731075317, 120.64841351506752, 120.74715392260333, 120.84589433013912, 120.94463473767493, 121.04337514521075, 121.14211555274653, 121.24085596028235, 121.33959636781816, 121.43833677535395, 121.53707718288976, 121.63581759042557, 121.73455799796139, 121.83329840549717, 121.93203881303299, 122.0307792205688, 122.12951962810459, 122.2282600356404, 122.32700044317622, 122.425740850712, 122.52448125824782, 122.62322166578363, 122.72196207331945, 122.82070248085523, 122.91944288839105, 123.01818329592686, 123.11692370346265, 123.21566411099846, 123.31440451853427, 123.41314492607006, 123.51188533360587, 123.61062574114169, 123.70936614867747, 123.80810655621329, 123.9068469637491, 124.00558737128492, 124.1043277788207, 124.20306818635652, 124.30180859389233, 124.40054900142812, 124.49928940896393, 124.59802981649975, 124.79551063157135, 124.89425103910716, 124.99299144664298, 125.09173185417876, 125.19047226171458, 125.28921266925039, 125.38795307678618, 125.48669348432199, 125.5854338918578, 125.68417429939359, 125.7829147069294, 125.88165511446522, 126.07913592953682, 126.17787633707263, 126.27661674460845, 126.37535715214423, 126.47409755968005, 126.57283796721586, 126.67157837475165, 126.77031878228746, 126.86905918982328, 126.96779959735906, 127.06654000489488, 127.16528041243069, 127.2640208199665, 127.36276122750229, 127.56024204257392, 127.6589824501097, 127.75772285764552, 127.85646326518133, 127.95520367271712, 128.15268448778875, 128.25142489532456, 128.35016530286035, 128.44890571039616, 128.54764611793198, 128.64638652546776, 128.74512693300358, 128.8438673405394, 128.94260774807518, 129.041348155611, 129.1400885631468, 129.23882897068262, 129.3375693782184, 129.43630978575422, 129.53505019329003, 129.63379060082582, 129.73253100836163, 129.83127141589745, 129.93001182343323, 130.02875223096905, 130.12749263850486, 130.22623304604065, 130.32497345357646, 130.42371386111228, 130.5224542686481, 130.62119467618388, 130.7199350837197, 130.8186754912555, 130.9174158987913, 131.0161563063271, 131.11489671386292, 131.2136371213987, 131.31237752893452, 131.41111793647033, 131.50985834400615, 131.60859875154193, 131.70733915907775, 131.80607956661356, 131.90481997414935, 132.00356038168516, 132.10230078922098, 132.20104119675676, 132.29978160429258, 132.3985220118284, 132.4972624193642, 132.5960028269, 132.6947432344358, 132.79348364197162, 132.8922240495074, 132.99096445704322, 133.08970486457903, 133.18844527211482, 133.28718567965063, 133.38592608718645, 133.48466649472223, 133.58340690225805, 133.68214730979386, 133.78088771732968, 133.87962812486546, 133.97836853240128, 134.0771089399371, 134.17584934747288, 134.2745897550087, 134.3733301625445, 134.4720705700803, 134.5708109776161, 134.66955138515192, 134.76829179268773, 134.86703220022352, 135.06451301529515, 135.16325342283093, 135.26199383036675, 135.36073423790256, 135.45947464543835, 135.55821505297416, 135.65695546050998, 135.85443627558158, 135.9531766831174, 136.0519170906532, 136.150657498189, 136.2493979057248, 136.34813831326062, 136.4468787207964, 136.54561912833222, 136.64435953586803, 136.74309994340382, 136.84184035093963, 136.94058075847545, 137.03932116601126, 137.13806157354705, 137.23680198108286, 137.33554238861868, 137.43428279615446, 137.53302320369028, 137.6317636112261, 137.73050401876188, 137.8292444262977, 137.9279848338335, 138.02672524136932, 138.1254656489051, 138.22420605644092, 138.32294646397673, 138.42168687151252, 138.52042727904833, 138.61916768658415, 138.71790809411993, 138.81664850165575, 138.91538890919156, 139.01412931672735, 139.11286972426316, 139.21161013179898, 139.3103505393348, 139.40909094687058, 139.5078313544064, 139.6065717619422, 139.705312169478, 139.8040525770138, 139.90279298454962, 140.0015333920854, 140.10027379962122, 140.19901420715703, 140.29775461469285, 140.39649502222863, 140.49523542976445, 140.59397583730026, 140.69271624483605, 140.79145665237186, 140.89019705990768, 140.98893746744346, 141.08767787497928, 141.1864182825151, 141.2851586900509, 141.3838990975867, 141.4826395051225, 141.58137991265832, 141.6801203201941, 141.77886072772992, 141.87760113526573, 141.97634154280152, 142.07508195033734, 142.17382235787315, 142.27256276540894, 142.37130317294475, 142.47004358048056, 142.56878398801638, 142.76626480308798, 142.8650052106238, 142.96374561815958, 143.0624860256954, 143.1612264332312, 143.259966840767, 143.3587072483028, 143.45744765583862, 143.55618806337444, 143.65492847091022, 143.75366887844604, 143.85240928598185, 143.95114969351764, 144.04989010105345, 144.14863050858926, 144.24737091612505, 144.34611132366086, 144.44485173119668, 144.5435921387325, 144.64233254626828, 144.7410729538041, 144.8398133613399, 144.9385537688757, 145.0372941764115, 145.13603458394732, 145.2347749914831, 145.33351539901892, 145.43225580655474, 145.53099621409052, 145.62973662162634, 145.82721743669796, 145.92595784423375, 146.02469825176956, 146.12343865930538, 146.22217906684116, 146.32091947437698, 146.4196598819128, 146.7158811045202, 146.81462151205602, 146.9133619195918, 147.01210232712762, 147.20958314219922, 147.30832354973504, 147.40706395727085, 147.50580436480664, 147.60454477234245, 147.70328517987826, 147.80202558741408, 147.90076599494986, 147.99950640248568, 148.0982468100215, 148.19698721755728, 148.2957276250931, 148.3944680326289, 148.4932084401647, 148.5919488477005, 148.69068925523632, 148.88817007030792, 148.98691047784374, 149.08565088537955, 149.18439129291534, 149.28313170045115, 149.38187210798696, 149.48061251552275, 149.57935292305856, 149.67809333059438, 149.77683373813016, 149.87557414566598, 149.9743145532018, 150.0730549607376, 150.1717953682734, 150.2705357758092, 150.36927618334502, 150.4680165908808, 150.56675699841662, 150.66549740595244, 150.76423781348822, 150.86297822102404, 150.96171862855985, 151.06045903609566, 151.15919944363145, 151.25793985116727, 151.35668025870308, 151.45542066623887, 151.55416107377468, 151.6529014813105, 151.75164188884628, 151.8503822963821, 151.9491227039179, 152.0478631114537, 152.1466035189895, 152.24534392652532, 152.34408433406114, 152.44282474159692, 152.54156514913274, 152.64030555666855, 152.73904596420434, 152.83778637174015, 152.93652677927597, 153.03526718681175, 153.13400759434757, 153.23274800188338, 153.3314884094192, 153.43022881695498, 153.5289692244908, 153.6277096320266, 153.7264500395624, 153.8251904470982, 153.92393085463402, 154.0226712621698, 154.12141166970562, 154.22015207724144, 154.31889248477722, 154.51637329984885, 154.61511370738467, 154.71385411492045, 154.81259452245627, 154.91133492999208, 155.01007533752787, 155.10881574506368, 155.2075561525995, 155.30629656013528, 155.4050369676711, 155.5037773752069, 155.60251778274272, 155.7012581902785, 155.79999859781432, 155.89873900535014, 155.99747941288592, 156.09621982042174, 156.19496022795755, 156.29370063549334, 156.39244104302915, 156.49118145056497, 156.58992185810078, 156.68866226563657, 156.78740267317238, 156.8861430807082, 156.98488348824398, 157.0836238957798, 157.1823643033156, 157.2811047108514, 157.3798451183872, 157.47858552592302, 157.5773259334588, 157.67606634099462, 157.77480674853044, 157.87354715606625, 157.97228756360204, 158.07102797113785, 158.16976837867367, 158.26850878620945, 158.36724919374527, 158.46598960128108, 158.56473000881687, 158.66347041635268, 158.7622108238885, 158.8609512314243, 158.9596916389601, 159.0584320464959, 159.15717245403172, 159.2559128615675, 159.35465326910332, 159.45339367663914, 159.55213408417492, 159.65087449171074, 159.74961489924655, 159.84835530678237, 160.04583612185397, 160.14457652938978, 160.24331693692557, 160.34205734446138, 160.4407977519972, 160.53953815953298, 160.6382785670688, 160.7370189746046, 160.8357593821404, 160.9344997896762, 161.03324019721202, 161.13198060474784, 161.23072101228362, 161.32946141981944, 161.42820182735525, 161.52694223489104, 161.62568264242685, 161.72442304996267, 161.82316345749845, 161.92190386503427, 162.02064427257008, 162.1193846801059, 162.21812508764168, 162.3168654951775, 162.4156059027133, 162.5143463102491, 162.6130867177849, 162.71182712532072, 162.8105675328565, 162.90930794039232, 163.00804834792814, 163.10678875546395, 163.20552916299974, 163.30426957053555, 163.40300997807137, 163.50175038560715, 163.60049079314297, 163.69923120067878, 163.79797160821457, 163.89671201575038, 163.9954524232862, 164.09419283082198, 164.1929332383578, 164.2916736458936, 164.39041405342942, 164.4891544609652, 164.58789486850102, 164.68663527603684, 164.78537568357262, 164.88411609110844, 164.98285649864425, 165.08159690618004, 165.18033731371585, 165.27907772125167, 165.37781812878748, 165.47655853632327, 165.57529894385908, 165.6740393513949, 165.77277975893068, 165.8715201664665, 165.9702605740023, 166.0690009815381, 166.1677413890739, 166.26648179660972, 166.36522220414554, 166.46396261168132, 166.56270301921714, 166.66144342675295, 166.76018383428874, 166.85892424182455, 166.95766464936037, 167.05640505689615, 167.15514546443197, 167.25388587196778, 167.35262627950357, 167.45136668703938, 167.5501070945752, 167.648847502111, 167.7475879096468, 167.8463283171826, 167.94506872471842, 168.0438091322542, 168.14254953979002, 168.24128994732584, 168.34003035486163, 168.43877076239744, 168.53751116993325, 168.63625157746907, 168.73499198500485, 168.83373239254067, 168.93247280007648, 169.03121320761227, 169.12995361514808, 169.2286940226839, 169.32743443021968, 169.4261748377555, 169.5249152452913, 169.6236556528271, 169.7223960603629, 169.82113646789873, 169.91987687543454, 170.01861728297033, 170.11735769050614, 170.21609809804195, 170.31483850557774, 170.41357891311355, 170.51231932064937, 170.61105972818515, 170.70980013572097, 170.80854054325678, 170.9072809507926, 171.1047617658642, 171.2035021734, 171.3022425809358, 171.4009829884716, 171.49972339600743, 171.5984638035432, 171.69720421107903, 171.79594461861484, 171.89468502615065, 171.99342543368644, 172.09216584122225, 172.19090624875807, 172.28964665629385, 172.38838706382967, 172.48712747136548, 172.58586787890127, 172.68460828643708, 172.7833486939729, 172.88208910150868, 172.9808295090445, 173.0795699165803, 173.17831032411613, 173.2770507316519, 173.37579113918773, 173.47453154672354, 173.57327195425933, 173.67201236179514, 173.77075276933095, 173.86949317686674, 173.96823358440255, 174.06697399193837, 174.16571439947418, 174.26445480700997, 174.36319521454578, 174.4619356220816, 174.56067602961738, 174.6594164371532, 174.758156844689, 174.8568972522248, 174.9556376597606, 175.05437806729643, 175.15311847483224, 175.25185888236803, 175.35059928990384, 175.44933969743965, 175.54808010497544, 175.64682051251125, 175.74556092004707, 175.84430132758285, 175.94304173511867, 176.04178214265448, 176.14052255019027, 176.23926295772608, 176.3380033652619, 176.4367437727977, 176.6342245878693, 176.73296499540513, 176.8317054029409, 176.93044581047673, 177.02918621801254, 177.12792662554833, 177.22666703308414, 177.32540744061995, 177.42414784815577, 177.52288825569156, 177.62162866322737, 177.72036907076318, 177.81910947829897, 177.91784988583478, 178.0165902933706, 178.11533070090638, 178.2140711084422, 178.312811515978, 178.41155192351383, 178.5102923310496, 178.60903273858543, 178.70777314612124, 178.80651355365703, 178.90525396119284, 179.00399436872866, 179.10273477626444, 179.20147518380026, 179.30021559133607, 179.39895599887186, 179.49769640640767, 179.59643681394348, 179.6951772214793, 179.79391762901508, 179.8926580365509, 179.9913984440867, 180.0901388516225, 180.1888792591583, 180.28761966669413, 180.3863600742299, 180.48510048176573, 180.58384088930154, 180.68258129683736, 180.78132170437314, 180.88006211190896, 180.97880251944477, 181.07754292698056, 181.17628333451637, 181.27502374205218, 181.37376414958797, 181.47250455712378, 181.5712449646596, 181.6699853721954, 181.7687257797312, 181.867466187267, 181.96620659480283, 182.0649470023386, 182.16368740987443, 182.26242781741024, 182.36116822494603, 182.45990863248184, 182.55864904001766, 182.65738944755344, 182.75612985508926, 182.85487026262507, 182.95361067016088, 183.05235107769667, 183.15109148523248, 183.2498318927683, 183.34857230030408, 183.4473127078399, 183.5460531153757, 183.6447935229115, 183.7435339304473, 183.84227433798313, 183.94101474551894, 184.03975515305473, 184.13849556059054, 184.23723596812636, 184.33597637566214, 184.43471678319796, 184.53345719073377, 184.63219759826956, 184.73093800580537, 184.82967841334118, 184.92841882087697, 185.02715922841278, 185.1258996359486, 185.2246400434844, 185.3233804510202, 185.422120858556, 185.52086126609183, 185.6196016736276, 185.71834208116343, 185.81708248869924, 185.91582289623503, 186.01456330377084, 186.11330371130666, 186.21204411884247, 186.31078452637826, 186.40952493391407, 186.60700574898567, 186.70574615652149, 186.8044865640573, 186.90322697159309, 187.0019673791289, 187.1007077866647, 187.19944819420053, 187.2981886017363, 187.39692900927213, 187.49566941680794, 187.59440982434373, 187.69315023187954, 187.79189063941536, 187.89063104695114, 187.98937145448696, 188.08811186202277, 188.18685226955856, 188.28559267709437, 188.38433308463019, 188.483073492166, 188.58181389970179, 188.6805543072376, 188.7792947147734, 188.8780351223092, 188.976775529845, 189.07551593738083, 189.1742563449166, 189.27299675245243, 189.37173715998824, 189.47047756752406, 189.56921797505984, 189.66795838259566, 189.76669879013147, 189.86543919766726, 189.96417960520307, 190.06292001273889, 190.16166042027467, 190.26040082781049, 190.3591412353463, 190.4578816428821, 190.5566220504179, 190.65536245795371, 190.75410286548953, 190.85284327302531, 190.95158368056113, 191.05032408809694, 191.14906449563273, 191.24780490316854, 191.34654531070436, 191.44528571824014, 191.54402612577596, 191.64276653331177, 191.7415069408476, 191.84024734838337, 191.9389877559192, 192.037728163455, 192.1364685709908, 192.2352089785266, 192.33394938606241, 192.4326897935982, 192.53143020113401, 192.63017060866983, 192.72891101620564, 192.82765142374143, 192.92639183127724, 193.02513223881306, 193.12387264634884, 193.22261305388466, 193.32135346142047, 193.42009386895626, 193.51883427649207, 193.6175746840279, 193.7163150915637, 193.8150554990995, 193.9137959066353, 194.01253631417111, 194.1112767217069, 194.21001712924271, 194.30875753677853, 194.40749794431431, 194.50623835185013, 194.60497875938594, 194.70371916692173, 194.80245957445754, 194.90119998199336, 194.99994038952917, 195.09868079706496, 195.19742120460077, 195.2961616121366, 195.39490201967237, 195.4936424272082, 195.592382834744, 195.6911232422798, 195.7898636498156, 195.88860405735142, 195.98734446488723, 196.08608487242302, 196.18482527995883, 196.28356568749464, 196.38230609503043, 196.48104650256624, 196.57978691010206, 196.67852731763784, 196.77726772517366, 196.87600813270947, 196.9747485402453, 197.07348894778107, 197.1722293553169, 197.2709697628527, 197.3697101703885, 197.4684505779243, 197.56719098546012, 197.6659313929959, 197.76467180053172, 197.86341220806753, 197.96215261560332, 198.06089302313913, 198.15963343067494, 198.25837383821076, 198.35711424574654, 198.45585465328236, 198.55459506081817, 198.65333546835396, 198.75207587588977, 198.8508162834256, 198.94955669096137, 199.0482970984972, 199.147037506033, 199.24577791356882, 199.3445183211046, 199.44325872864042, 199.54199913617623, 199.64073954371202, 199.73947995124783, 199.83822035878364, 199.93696076631943, 200.03570117385524, 200.13444158139106, 200.23318198892684, 200.33192239646266, 200.43066280399847, 200.5294032115343, 200.62814361907007, 200.7268840266059, 200.8256244341417, 200.9243648416775, 201.0231052492133, 201.12184565674912, 201.2205860642849, 201.31932647182072, 201.41806687935653, 201.51680728689234, 201.61554769442813, 201.71428810196394, 201.81302850949976, 201.91176891703554, 202.01050932457136, 202.10924973210717, 202.20799013964296, 202.30673054717877, 202.4054709547146, 202.5042113622504, 202.6029517697862, 202.701692177322, 202.80043258485782, 202.8991729923936, 202.99791339992942, 203.09665380746523, 203.19539421500102, 203.29413462253683, 203.39287503007264, 203.49161543760843, 203.59035584514424, 203.68909625268006, 203.78783666021587, 203.88657706775166, 203.98531747528747, 204.0840578828233, 204.18279829035907, 204.2815386978949, 204.3802791054307, 204.4790195129665, 204.5777599205023, 204.67650032803812, 204.77524073557393, 204.87398114310972, 204.97272155064553, 205.07146195818135, 205.17020236571713, 205.26894277325295, 205.36768318078876, 205.46642358832455, 205.56516399586036, 205.66390440339617, 205.762644810932, 205.86138521846777, 205.9601256260036, 206.0588660335394, 206.1576064410752, 206.256346848611, 206.35508725614682, 206.4538276636826, 206.55256807121842, 206.65130847875423, 206.75004888629002, 206.84878929382583, 206.94752970136165, 207.04627010889746, 207.14501051643325, 207.24375092396906, 207.34249133150487, 207.44123173904066, 207.53997214657647, 207.6387125541123, 207.73745296164807, 207.8361933691839, 207.9349337767197, 208.03367418425552, 208.1324145917913, 208.23115499932712, 208.32989540686293, 208.42863581439872, 208.52737622193453, 208.62611662947035, 208.72485703700613, 208.82359744454195, 208.92233785207776, 209.02107825961357, 209.11981866714936, 209.21855907468517, 209.317299482221, 209.41603988975677, 209.5147802972926, 209.6135207048284, 209.7122611123642, 209.8110015199, 209.90974192743582, 210.0084823349716, 210.10722274250742, 210.20596315004323, 210.30470355757905, 210.40344396511483, 210.50218437265065, 210.60092478018646, 210.69966518772225, 210.79840559525806, 210.89714600279387, 210.99588641032966, 211.09462681786547, 211.1933672254013, 211.2921076329371, 211.3908480404729, 211.4895884480087, 211.58832885554452, 211.6870692630803, 211.78580967061612, 211.88455007815193, 211.98329048568772, 212.08203089322353, 212.18077130075935, 212.27951170829516, 212.37825211583095, 212.47699252336676, 212.57573293090257, 212.67447333843836, 212.77321374597418, 212.87195415351, 212.97069456104578, 213.0694349685816, 213.1681753761174, 213.2669157836532, 213.365656191189, 213.46439659872482, 213.56313700626063, 213.66187741379642, 213.76061782133223, 213.85935822886805, 213.95809863640383, 214.05683904393965, 214.15557945147546, 214.25431985901125, 214.35306026654706, 214.45180067408288, 214.5505410816187, 214.64928148915448, 214.7480218966903, 214.8467623042261, 214.9455027117619, 215.0442431192977, 215.14298352683352, 215.2417239343693, 215.34046434190512, 215.43920474944093, 215.53794515697672, 215.63668556451253, 215.73542597204835, 215.83416637958416, 215.93290678711995, 216.03164719465576, 216.13038760219158, 216.22912800972736, 216.32786841726318, 216.426608824799, 216.52534923233478, 216.6240896398706, 216.7228300474064, 216.82157045494222, 216.920310862478, 217.01905127001382, 217.11779167754963, 217.21653208508542, 217.31527249262123, 217.41401290015705, 217.51275330769283, 217.61149371522865, 217.71023412276446, 217.80897453030028, 217.90771493783606, 218.00645534537188, 218.1051957529077, 218.20393616044348, 218.3026765679793, 218.4014169755151, 218.5001573830509, 218.5988977905867, 218.69763819812252, 218.7963786056583, 218.89511901319412, 218.99385942072993, 219.09259982826575, 219.19134023580153, 219.29008064333735, 219.38882105087316, 219.48756145840895, 219.58630186594476, 219.68504227348058, 219.88252308855218, 219.981263496088, 220.0800039036238, 220.1787443111596, 220.2774847186954, 220.37622512623122, 220.474965533767, 220.57370594130282, 220.67244634883863, 220.77118675637442, 220.86992716391023, 220.96866757144605, 221.06740797898186, 221.16614838651765, 221.26488879405346, 221.36362920158928, 221.46236960912506, 221.56111001666088, 221.6598504241967, 221.75859083173248, 221.8573312392683, 221.9560716468041, 222.0548120543399, 222.1535524618757, 222.25229286941152, 222.35103327694733, 222.44977368448312, 222.54851409201893, 222.64725449955475, 222.74599490709053, 222.84473531462635, 222.94347572216216, 223.04221612969795, 223.14095653723376, 223.23969694476958, 223.3384373523054, 223.43717775984118, 223.535918167377, 223.6346585749128, 223.7333989824486, 223.8321393899844, 223.93087979752022, 224.029620205056, 224.12836061259182, 224.22710102012763, 224.32584142766345, 224.42458183519923, 224.52332224273505, 224.62206265027086, 224.72080305780665, 224.81954346534246, 224.91828387287828, 225.01702428041406, 225.11576468794988, 225.2145050954857, 225.31324550302148, 225.4119859105573, 225.5107263180931, 225.60946672562892, 225.7082071331647, 225.80694754070052, 225.90568794823633, 226.00442835577212, 226.10316876330793, 226.20190917084375, 226.30064957837953, 226.39938998591535, 226.49813039345116, 226.59687080098698, 226.69561120852276, 226.79435161605858, 226.8930920235944, 226.99183243113018, 227.090572838666, 227.1893132462018, 227.2880536537376, 227.3867940612734, 227.48553446880922, 227.58427487634503, 227.68301528388082, 227.78175569141663, 227.88049609895245, 227.97923650648823, 228.07797691402405, 228.17671732155986, 228.27545772909565, 228.37419813663146, 228.57167895170306, 228.67041935923888, 228.7691597667747, 228.8679001743105, 228.9666405818463, 229.0653809893821, 229.16412139691792, 229.2628618044537, 229.36160221198952, 229.46034261952533, 229.55908302706112, 229.65782343459693, 229.75656384213275, 229.85530424966856, 229.95404465720435, 230.05278506474016, 230.15152547227598, 230.25026587981176, 230.34900628734758, 230.4477466948834, 230.54648710241918, 230.645227509955, 230.7439679174908, 230.8427083250266, 230.9414487325624, 231.04018914009822, 231.13892954763404, 231.23766995516982, 231.33641036270564, 231.43515077024145, 231.53389117777724, 231.63263158531305, 231.73137199284886, 231.83011240038465, 231.92885280792046, 232.02759321545628, 232.1263336229921, 232.22507403052788, 232.3238144380637, 232.4225548455995, 232.5212952531353, 232.6200356606711, 232.71877606820692, 232.8175164757427, 232.91625688327852, 233.01499729081434, 233.11373769835015, 233.21247810588594, 233.31121851342175, 233.40995892095756, 233.50869932849335, 233.60743973602916, 233.70618014356498, 233.80492055110076, 233.90366095863658, 234.0024013661724, 234.10114177370818, 234.199882181244, 234.2986225887798, 234.39736299631562, 234.4961034038514, 234.59484381138722, 234.69358421892304, 234.79232462645882, 234.89106503399464, 234.98980544153045, 235.08854584906624, 235.18728625660205, 235.28602666413786, 235.38476707167368, 235.48350747920946, 235.58224788674528, 235.6809882942811, 235.77972870181688, 235.8784691093527, 235.9772095168885, 236.0759499244243, 236.1746903319601, 236.27343073949592, 236.37217114703174, 236.47091155456752, 236.56965196210334, 236.66839236963915, 236.76713277717494, 236.86587318471075, 236.96461359224656, 237.06335399978235, 237.16209440731816, 237.26083481485398, 237.35957522238976, 237.45831562992558, 237.5570560374614, 237.6557964449972, 237.754536852533, 237.8532772600688, 237.95201766760462, 238.0507580751404, 238.14949848267622, 238.24823889021204, 238.34697929774782, 238.44571970528364, 238.54446011281945, 238.64320052035526, 238.74194092789105, 238.84068133542686, 238.93942174296268, 239.03816215049847, 239.13690255803428, 239.2356429655701, 239.4331237806417, 239.5318641881775, 239.63060459571332, 239.7293450032491, 239.82808541078492, 239.92682581832074, 240.02556622585652, 240.12430663339234, 240.22304704092815, 240.32178744846394, 240.42052785599975, 240.51926826353557, 240.61800867107135, 240.71674907860717, 240.81548948614298, 240.9142298936788, 241.01297030121458, 241.1117107087504, 241.2104511162862, 241.309191523822, 241.4079319313578, 241.50667233889362, 241.6054127464294, 241.70415315396522, 241.80289356150104, 241.90163396903685, 242.00037437657264, 242.09911478410845, 242.19785519164427, 242.29659559918005, 242.39533600671587, 242.49407641425168, 242.59281682178747, 242.69155722932328, 242.7902976368591, 242.8890380443949, 242.9877784519307, 243.0865188594665, 243.18525926700232, 243.2839996745381, 243.38274008207392, 243.48148048960974, 243.58022089714552, 243.67896130468134, 243.77770171221715, 243.87644211975294, 243.97518252728875, 244.07392293482457, 244.17266334236038, 244.27140374989617, 244.37014415743198, 244.4688845649678, 244.56762497250358, 244.6663653800394, 244.76510578757518, 244.86384619511102, 244.9625866026468, 245.0613270101826, 245.16006741771844, 245.25880782525422, 245.35754823279, 245.45628864032585, 245.55502904786164, 245.65376945539748, 245.75250986293327, 245.85125027046905, 245.9499906780049, 246.04873108554068, 246.14747149307647, 246.2462119006123, 246.3449523081481, 246.44369271568388, 246.54243312321972, 246.6411735307555, 246.7399139382913, 246.83865434582714, 246.93739475336292, 247.0361351608987, 247.13487556843455, 247.23361597597034, 247.33235638350612, 247.43109679104197, 247.52983719857775, 247.62857760611354, 247.72731801364938, 247.82605842118517, 247.924798828721, 248.0235392362568, 248.12227964379258, 248.22102005132842, 248.3197604588642, 248.4185008664, 248.51724127393584, 248.61598168147162, 248.7147220890074, 248.81346249654325, 248.91220290407904, 249.01094331161482, 249.10968371915067, 249.20842412668645, 249.30716453422224, 249.40590494175808, 249.50464534929387, 249.60338575682965, 249.7021261643655, 249.80086657190128, 249.89960697943707, 249.9983473869729, 250.0970877945087, 250.19582820204454, 250.29456860958032, 250.3933090171161, 250.49204942465195, 250.59078983218774, 250.68953023972352, 250.78827064725937, 250.88701105479515, 250.98575146233094, 251.08449186986678, 251.18323227740257, 251.28197268493835, 251.3807130924742, 251.47945350000998, 251.57819390754577, 251.6769343150816, 251.7756747226174, 251.87441513015318, 251.97315553768902, 252.0718959452248, 252.17063635276065, 252.26937676029644, 252.36811716783222, 252.46685757536807, 252.56559798290385, 252.66433839043964, 252.76307879797548, 252.86181920551127, 252.96055961304705, 253.0593000205829, 253.15804042811868, 253.25678083565447, 253.3555212431903, 253.4542616507261, 253.55300205826188, 253.65174246579772, 253.7504828733335, 253.8492232808693, 253.94796368840514, 254.04670409594092, 254.1454445034767, 254.24418491101255, 254.34292531854834, 254.44166572608418, 254.54040613361997, 254.63914654115575, 254.7378869486916, 254.83662735622738, 254.93536776376317, 255.034108171299, 255.1328485788348, 255.23158898637058, 255.33032939390642, 255.4290698014422, 255.527810208978, 255.62655061651384, 255.72529102404962, 255.8240314315854, 255.92277183912125, 256.02151224665704, 256.1202526541928, 256.21899306172867, 256.31773346926445, 256.41647387680024, 256.5152142843361, 256.61395469187187, 256.7126950994077, 256.8114355069435, 256.9101759144793, 257.0089163220151, 257.1076567295509, 257.2063971370867, 257.30513754462254, 257.4038779521583, 257.5026183596941, 257.60135876722995, 257.70009917476574, 257.7988395823015, 257.89757998983737, 257.99632039737315, 258.09506080490894, 258.1938012124448, 258.29254161998057, 258.39128202751635, 258.4900224350522, 258.588762842588, 258.68750325012377, 258.7862436576596, 258.8849840651954, 258.98372447273124, 259.082464880267, 259.1812052878028, 259.27994569533865, 259.37868610287444, 259.4774265104102, 259.57616691794607, 259.67490732548185, 259.77364773301764, 259.8723881405535, 259.97112854808927, 260.06986895562505, 260.1686093631609, 260.2673497706967, 260.36609017823247, 260.4648305857683, 260.5635709933041, 260.6623114008399, 260.7610518083757, 260.8597922159115, 260.95853262344735, 261.05727303098314, 261.1560134385189, 261.25475384605477, 261.35349425359055, 261.45223466112634, 261.5509750686622, 261.64971547619797, 261.74845588373375, 261.8471962912696, 261.9459366988054, 262.04467710634117, 262.143417513877, 262.2421579214128, 262.3408983289486, 262.4396387364844, 262.5383791440202, 262.637119551556, 262.73585995909184, 262.8346003666276, 262.9333407741634, 263.03208118169925, 263.13082158923504, 263.2295619967709, 263.32830240430667, 263.42704281184245, 263.5257832193783, 263.6245236269141, 263.72326403444987, 263.8220044419857, 263.9207448495215, 264.0194852570573, 264.1182256645931, 264.2169660721289, 264.3157064796647, 264.41444688720054, 264.5131872947363, 264.6119277022721, 264.71066810980795, 264.80940851734374, 264.9081489248795, 265.00688933241537, 265.10562973995115, 265.20437014748694, 265.3031105550228, 265.40185096255857, 265.5005913700944, 265.5993317776302, 265.698072185166, 265.7968125927018, 265.8955530002376, 265.9942934077734, 266.09303381530924, 266.191774222845, 266.2905146303808, 266.38925503791666, 266.48799544545244, 266.5867358529882, 266.68547626052407, 266.78421666805986, 266.88295707559564, 266.9816974831315, 267.08043789066727, 267.17917829820306, 267.2779187057389, 267.3766591132747, 267.4753995208105, 267.5741399283463, 267.6728803358821, 267.77162074341794, 267.8703611509537, 267.9691015584895, 268.06784196602536, 268.16658237356114, 268.2653227810969, 268.36406318863277, 268.46280359616856, 268.56154400370434, 268.6602844112402, 268.75902481877597, 268.85776522631176, 268.9565056338476, 269.0552460413834, 269.15398644891917, 269.252726856455, 269.3514672639908, 269.4502076715266, 269.5489480790624, 269.6476884865982, 269.74642889413406, 269.84516930166984, 269.9439097092056, 270.04265011674147, 270.14139052427726, 270.24013093181304, 270.3388713393489, 270.43761174688467, 270.53635215442046, 270.6350925619563, 270.7338329694921, 270.83257337702787, 270.9313137845637, 271.0300541920995, 271.1287945996353, 271.2275350071711, 271.3262754147069, 271.4250158222427, 271.52375622977854, 271.6224966373143, 271.7212370448501, 271.81997745238596, 271.91871785992174, 272.0174582674576, 272.11619867499337, 272.21493908252916, 272.313679490065, 272.4124198976008, 272.51116030513657, 272.6099007126724, 272.7086411202082, 272.807381527744, 272.9061219352798, 273.0048623428156, 273.1036027503514, 273.20234315788724, 273.301083565423, 273.3998239729588, 273.49856438049466, 273.59730478803044, 273.6960451955662, 273.79478560310207, 273.89352601063786, 273.99226641817364, 274.0910068257095, 274.18974723324527, 274.2884876407811, 274.3872280483169, 274.4859684558527, 274.5847088633885, 274.6834492709243, 274.7821896784601, 274.88093008599594, 274.9796704935317, 275.0784109010675, 275.17715130860336, 275.27589171613914, 275.3746321236749, 275.47337253121077, 275.57211293874656, 275.67085334628234, 275.7695937538182, 275.86833416135397, 275.96707456888976, 276.0658149764256, 276.1645553839614, 276.2632957914972, 276.362036199033, 276.4607766065688, 276.55951701410464, 276.6582574216404, 276.7569978291762, 276.85573823671206, 276.95447864424784, 277.0532190517836, 277.15195945931947, 277.25069986685526, 277.34944027439104, 277.4481806819269, 277.54692108946267, 277.64566149699846, 277.7444019045343, 277.8431423120701, 277.94188271960587, 278.0406231271417, 278.2381039422133, 278.33684434974913, 278.4355847572849, 278.53432516482076, 278.63306557235654, 278.73180597989233, 278.83054638742817, 278.92928679496396, 279.02802720249974, 279.1267676100356, 279.22550801757137, 279.32424842510716, 279.422988832643, 279.5217292401788, 279.6204696477146, 279.7192100552504, 279.8179504627862, 279.916690870322, 280.01543127785783, 280.1141716853936, 280.2129120929294, 280.31165250046524, 280.41039290800103, 280.5091333155368, 280.60787372307266, 280.70661413060844, 280.8053545381443, 280.9040949456801, 281.00283535321586, 281.1015757607517, 281.2003161682875, 281.2990565758233, 281.3977969833591, 281.4965373908949, 281.5952777984307, 281.69401820596653, 281.7927586135023, 281.8914990210381, 281.99023942857394, 282.08897983610973, 282.1877202436455, 282.28646065118136, 282.38520105871714, 282.48394146625293, 282.5826818737888, 282.68142228132456, 282.7801626888604, 282.8789030963962, 282.977643503932, 283.0763839114678, 283.1751243190036, 283.2738647265394, 283.37260513407523, 283.471345541611, 283.5700859491468, 283.66882635668264, 283.76756676421843, 283.8663071717542, 283.96504757929006, 284.06378798682584, 284.16252839436163, 284.2612688018975, 284.36000920943326, 284.45874961696904, 284.5574900245049, 284.6562304320407, 284.75497083957646, 284.8537112471123, 284.9524516546481, 285.05119206218393, 285.1499324697197, 285.2486728772555, 285.34741328479134, 285.44615369232713, 285.5448940998629, 285.64363450739876, 285.74237491493454, 285.84111532247033, 285.9398557300062, 286.03859613754196, 286.13733654507774, 286.2360769526136, 286.3348173601494, 286.43355776768516, 286.532298175221, 286.6310385827568, 286.7297789902926, 286.8285193978284, 286.9272598053642, 287.0260002129, 287.12474062043583, 287.2234810279716, 287.32222143550746, 287.42096184304324, 287.51970225057903, 287.6184426581149, 287.71718306565066, 287.81592347318644, 287.9146638807223, 288.0134042882581, 288.11214469579386, 288.2108851033297, 288.3096255108655, 288.4083659184013, 288.5071063259371, 288.6058467334729, 288.7045871410087, 288.80332754854453, 288.9020679560803, 289.0008083636161, 289.09954877115194, 289.19828917868773, 289.2970295862235, 289.39576999375936, 289.49451040129514, 289.593250808831, 289.6919912163668, 289.79073162390256, 289.8894720314384, 289.9882124389742, 290.08695284651, 290.1856932540458, 290.2844336615816, 290.3831740691174, 290.48191447665323, 290.580654884189, 290.6793952917248, 290.77813569926064, 290.87687610679643, 290.9756165143322, 291.07435692186806, 291.17309732940384, 291.27183773693963, 291.3705781444755, 291.46931855201126, 291.5680589595471, 291.6667993670829, 291.7655397746187, 291.8642801821545, 291.9630205896903, 292.0617609972261, 292.16050140476193, 292.2592418122977, 292.3579822198335, 292.45672262736934, 292.55546303490513, 292.6542034424409, 292.75294384997676, 292.85168425751255, 292.95042466504833, 293.0491650725842, 293.14790548011996, 293.24664588765575, 293.3453862951916, 293.4441267027274, 293.54286711026316, 293.641607517799, 293.7403479253348, 293.83908833287063, 293.9378287404064, 294.0365691479422, 294.13530955547805, 294.23404996301383, 294.3327903705496, 294.43153077808546, 294.53027118562125, 294.62901159315703, 294.7277520006929, 294.82649240822866, 294.92523281576445, 295.0239732233003, 295.1227136308361, 295.22145403837186, 295.3201944459077, 295.4189348534435, 295.5176752609793, 295.6164156685151, 295.7151560760509, 295.8138964835867, 295.91263689112253, 296.0113772986583, 296.11011770619416, 296.20885811372995, 296.30759852126573, 296.4063389288016, 296.50507933633736, 296.60381974387315, 296.702560151409, 296.8013005589448, 296.90004096648056, 296.9987813740164, 297.0975217815522, 297.196262189088, 297.2950025966238, 297.3937430041596, 297.4924834116954, 297.59122381923123, 297.689964226767, 297.7887046343028, 297.88744504183865, 297.98618544937443, 298.0849258569103, 298.18366626444606, 298.28240667198185, 298.3811470795177, 298.4798874870535, 298.57862789458926, 298.6773683021251, 298.7761087096609, 298.8748491171967, 298.9735895247325, 299.0723299322683, 299.1710703398041, 299.26981074733993, 299.3685511548757, 299.4672915624115, 299.56603196994735, 299.66477237748313, 299.7635127850189, 299.86225319255476, 299.96099360009055, 300.05973400762633, 300.1584744151622, 300.25721482269796, 300.3559552302338, 300.4546956377696, 300.5534360453054, 300.6521764528412, 300.750916860377, 300.8496572679128, 300.94839767544863, 301.0471380829844, 301.1458784905202, 301.24461889805605, 301.34335930559183, 301.4420997131276, 301.54084012066346, 301.63958052819925, 301.73832093573503, 301.8370613432709, 301.93580175080666, 302.03454215834245, 302.1332825658783, 302.2320229734141, 302.33076338094986, 302.4295037884857, 302.5282441960215, 302.62698460355733, 302.7257250110931, 302.8244654186289, 302.92320582616475, 303.02194623370053, 303.1206866412363, 303.21942704877216, 303.31816745630795, 303.41690786384373, 303.5156482713796, 303.61438867891536, 303.71312908645115, 303.811869493987, 303.9106099015228, 304.00935030905856, 304.1080907165944, 304.2068311241302, 304.305571531666, 304.4043119392018, 304.5030523467376, 304.6017927542734, 304.70053316180923, 304.799273569345, 304.89801397688086, 304.99675438441665, 305.09549479195243, 305.1942351994883, 305.29297560702406, 305.39171601455985, 305.4904564220957, 305.5891968296315, 305.68793723716726, 305.7866776447031, 305.8854180522389, 305.9841584597747, 306.0828988673105, 306.1816392748463, 306.2803796823821, 306.37912008991793, 306.4778604974537, 306.5766009049895, 306.67534131252535, 306.77408172006113, 306.872822127597, 306.97156253513276, 307.07030294266855, 307.1690433502044, 307.2677837577402, 307.36652416527596, 307.4652645728118, 307.5640049803476, 307.6627453878834, 307.7614857954192, 307.860226202955, 307.9589666104908, 308.05770701802663, 308.1564474255624, 308.2551878330982, 308.35392824063405, 308.45266864816983, 308.5514090557056, 308.65014946324146, 308.74888987077725, 308.84763027831303, 308.9463706858489, 309.04511109338466, 309.1438515009205, 309.2425919084563, 309.3413323159921, 309.4400727235279, 309.5388131310637, 309.6375535385995, 309.73629394613533, 309.8350343536711, 309.9337747612069, 310.03251516874275, 310.13125557627853, 310.2299959838143, 310.32873639135016, 310.42747679888595, 310.52621720642173, 310.6249576139576, 310.72369802149336, 310.82243842902915, 310.921178836565, 311.0199192441008, 311.11865965163656, 311.2174000591724, 311.3161404667082, 311.41488087424403, 311.5136212817798, 311.6123616893156, 311.71110209685145, 311.80984250438723, 311.908582911923, 312.00732331945886, 312.10606372699465, 312.20480413453043, 312.3035445420663, 312.40228494960206, 312.50102535713785, 312.5997657646737, 312.6985061722095, 312.79724657974526, 312.8959869872811, 312.9947273948169, 313.0934678023527, 313.1922082098885, 313.2909486174243, 313.38968902496015, 313.48842943249593, 313.5871698400317, 313.68591024756756, 313.78465065510335, 313.88339106263913, 313.982131470175, 314.08087187771076, 314.17961228524655, 314.2783526927824, 314.3770931003182, 314.47583350785396, 314.5745739153898, 314.6733143229256, 314.7720547304614, 314.8707951379972, 314.969535545533, 315.0682759530688, 315.16701636060463, 315.2657567681404, 315.3644971756762, 315.46323758321205, 315.56197799074783, 315.6607183982837, 315.75945880581946, 315.85819921335525, 315.9569396208911, 316.0556800284269, 316.15442043596266, 316.2531608434985, 316.3519012510343, 316.4506416585701, 316.5493820661059, 316.6481224736417, 316.7468628811775, 316.84560328871333, 316.9443436962491, 317.0430841037849, 317.14182451132075, 317.24056491885653, 317.3393053263923, 317.43804573392816, 317.53678614146395, 317.63552654899974, 317.7342669565356, 317.83300736407136, 317.9317477716072, 318.030488179143, 318.1292285866788, 318.2279689942146, 318.3267094017504, 318.4254498092862, 318.52419021682203, 318.6229306243578, 318.7216710318936, 318.82041143942945, 318.91915184696524, 319.017892254501, 319.11663266203686, 319.21537306957265, 319.31411347710844, 319.4128538846443, 319.51159429218006, 319.61033469971585, 319.7090751072517, 319.8078155147875, 319.90655592232326, 320.0052963298591, 320.1040367373949, 320.20277714493074, 320.3015175524665, 320.4002579600023, 320.49899836753815, 320.59773877507394, 320.6964791826097, 320.79521959014556, 320.89395999768135, 320.99270040521714, 321.091440812753, 321.19018122028876, 321.28892162782455, 321.3876620353604, 321.4864024428962, 321.58514285043196, 321.6838832579678, 321.7826236655036, 321.8813640730394, 321.9801044805752, 322.078844888111, 322.17758529564685, 322.27632570318264, 322.3750661107184, 322.47380651825426, 322.57254692579005, 322.67128733332584, 322.7700277408617, 322.86876814839746, 322.96750855593325, 323.0662489634691, 323.1649893710049, 323.26372977854066, 323.3624701860765, 323.4612105936123, 323.5599510011481, 323.6586914086839, 323.7574318162197, 323.8561722237555, 323.95491263129134, 324.0536530388271, 324.1523934463629, 324.25113385389875, 324.34987426143454, 324.4486146689704, 324.54735507650616, 324.64609548404195, 324.7448358915778, 324.8435762991136, 324.94231670664936, 325.0410571141852, 325.139797521721, 325.2385379292568, 325.3372783367926, 325.4360187443284, 325.5347591518642, 325.63349955940004, 325.7322399669358, 325.8309803744716, 325.92972078200745, 326.02846118954324, 326.127201597079, 326.22594200461486, 326.32468241215065, 326.42342281968644, 326.5221632272223, 326.62090363475806, 326.7196440422939, 326.8183844498297, 326.9171248573655, 327.0158652649013, 327.1146056724371, 327.2133460799729, 327.31208648750874, 327.4108268950445, 327.5095673025803, 327.60830771011615, 327.70704811765194, 327.8057885251877, 327.90452893272357, 328.00326934025935, 328.10200974779514, 328.200750155331, 328.29949056286677, 328.39823097040255, 328.4969713779384, 328.5957117854742, 328.69445219301, 328.7931926005458, 328.8919330080816, 328.99067341561744, 329.0894138231532, 329.188154230689, 329.28689463822485, 329.38563504576064, 329.4843754532964, 329.58311586083227, 329.68185626836805, 329.78059667590384, 329.8793370834397, 329.97807749097547, 330.07681789851125, 330.1755583060471, 330.2742987135829, 330.37303912111867, 330.4717795286545, 330.5705199361903, 330.6692603437261, 330.7680007512619, 330.8667411587977, 330.96548156633355, 331.06422197386934, 331.1629623814051, 331.26170278894097, 331.36044319647675, 331.45918360401254, 331.5579240115484, 331.65666441908417, 331.75540482661995, 331.8541452341558, 331.9528856416916, 332.05162604922737, 332.1503664567632, 332.249106864299, 332.3478472718348, 332.4465876793706, 332.5453280869064, 332.6440684944422, 332.74280890197804, 332.8415493095138, 332.9402897170496, 333.03903012458545, 333.13777053212124, 333.2365109396571, 333.33525134719287, 333.43399175472865, 333.5327321622645, 333.6314725698003, 333.73021297733607, 333.8289533848719, 333.9276937924077, 334.0264341999435, 334.1251746074793, 334.2239150150151, 334.3226554225509, 334.42139583008674, 334.5201362376225, 334.6188766451583, 334.71761705269415, 334.81635746022994, 334.9150978677657, 335.01383827530157, 335.11257868283735, 335.21131909037314, 335.310059497909, 335.40879990544477, 335.5075403129806, 335.6062807205164, 335.7050211280522, 335.803761535588, 335.9025019431238, 336.0012423506596, 336.09998275819544, 336.1987231657312, 336.297463573267, 336.39620398080285, 336.49494438833864, 336.5936847958744, 336.69242520341027, 336.79116561094605, 336.88990601848184, 336.9886464260177, 337.08738683355347, 337.18612724108925, 337.2848676486251, 337.3836080561609, 337.4823484636967, 337.5810888712325, 337.6798292787683, 337.77856968630414, 337.8773100938399, 337.9760505013757, 338.07479090891155, 338.17353131644734, 338.2722717239831, 338.37101213151897, 338.46975253905475, 338.56849294659054, 338.6672333541264, 338.76597376166217, 338.86471416919795, 338.9634545767338, 339.0621949842696, 339.16093539180537, 339.2596757993412, 339.358416206877, 339.4571566144128, 339.5558970219486, 339.6546374294844, 339.75337783702025, 339.85211824455604, 339.9508586520918, 340.04959905962767, 340.14833946716345, 340.24707987469924, 340.3458202822351, 340.44456068977087, 340.54330109730665, 340.6420415048425, 340.7407819123783, 340.83952231991407, 340.9382627274499, 341.0370031349857, 341.1357435425215, 341.2344839500573, 341.3332243575931, 341.4319647651289, 341.53070517266474, 341.6294455802005, 341.7281859877363, 341.82692639527215, 341.92566680280794, 342.0244072103438, 342.12314761787957, 342.22188802541535, 342.3206284329512, 342.419368840487, 342.51810924802277, 342.6168496555586, 342.7155900630944, 342.8143304706302, 342.913070878166, 343.0118112857018, 343.1105516932376, 343.20929210077344, 343.3080325083092, 343.406772915845, 343.50551332338085, 343.60425373091664, 343.7029941384524, 343.80173454598827, 343.90047495352405, 343.9992153610599, 344.0979557685957, 344.19669617613147, 344.2954365836673, 344.3941769912031, 344.4929173987389, 344.5916578062747, 344.6903982138105, 344.7891386213463, 344.88787902888214, 344.9866194364179, 345.0853598439537, 345.18410025148955, 345.28284065902534, 345.3815810665611, 345.48032147409697, 345.57906188163275, 345.67780228916854, 345.7765426967044, 345.87528310424017, 345.97402351177595, 346.0727639193118, 346.1715043268476, 346.2702447343834, 346.3689851419192, 346.467725549455, 346.56646595699084, 346.6652063645266, 346.7639467720624, 346.86268717959825, 346.96142758713404, 347.0601679946698, 347.15890840220567, 347.25764880974145, 347.35638921727724, 347.4551296248131, 347.55387003234887, 347.65261043988465, 347.7513508474205, 347.8500912549563, 347.94883166249207, 348.0475720700279, 348.1463124775637, 348.2450528850995, 348.3437932926353, 348.4425337001711, 348.54127410770695, 348.64001451524274, 348.7387549227785, 348.83749533031437, 348.93623573785015, 349.03497614538594, 349.1337165529218, 349.23245696045757, 349.33119736799335, 349.4299377755292, 349.528678183065, 349.62741859060077, 349.7261589981366, 349.8248994056724, 349.9236398132082, 350.022380220744, 350.1211206282798, 350.2198610358156, 350.31860144335144, 350.4173418508872, 350.516082258423, 350.61482266595885, 350.71356307349464, 350.8123034810305, 350.91104388856627, 351.00978429610205, 351.1085247036379, 351.2072651111737, 351.30600551870947, 351.4047459262453, 351.5034863337811, 351.6022267413169, 351.7009671488527, 351.7997075563885, 351.8984479639243, 351.99718837146014, 352.0959287789959, 352.1946691865317, 352.29340959406755, 352.39215000160334, 352.4908904091391, 352.58963081667497, 352.68837122421075, 352.7871116317466, 352.8858520392824, 352.98459244681817, 353.083332854354, 353.1820732618898, 353.2808136694256, 353.3795540769614, 353.4782944844972, 353.577034892033, 353.67577529956884, 353.7745157071046, 353.8732561146404, 353.97199652217625, 354.07073692971204, 354.1694773372478, 354.26821774478367, 354.36695815231946, 354.46569855985524, 354.5644389673911, 354.66317937492687, 354.76191978246266, 354.8606601899985, 354.9594005975343, 355.0581410050701, 355.1568814126059, 355.2556218201417, 355.35436222767754, 355.4531026352133, 355.5518430427491, 355.65058345028496, 355.74932385782074, 355.8480642653565, 355.94680467289237, 356.04554508042816, 356.14428548796394, 356.2430258954998, 356.34176630303557, 356.44050671057136, 356.5392471181072, 356.637987525643, 356.73672793317877, 356.8354683407146, 356.9342087482504, 357.0329491557862, 357.131689563322, 357.2304299708578, 357.32917037839366, 357.42791078592944, 357.5266511934652, 357.62539160100107, 357.72413200853686, 357.82287241607264, 357.9216128236085, 358.02035323114427, 358.11909363868006, 358.2178340462159, 358.3165744537517, 358.41531486128747, 358.5140552688233, 358.6127956763591, 358.7115360838949, 358.8102764914307, 358.9090168989665, 359.0077573065023, 359.10649771403814, 359.2052381215739, 359.30397852910977, 359.40271893664556, 359.50145934418134, 359.6001997517172, 359.69894015925297, 359.79768056678876, 359.8964209743246, 359.9951613818604, 360.09390178939617, 360.192642196932, 360.2913826044678, 360.3901230120036, 360.4888634195394, 360.5876038270752, 360.686344234611, 360.78508464214684, 360.8838250496826, 360.9825654572184, 361.08130586475426, 361.18004627229004, 361.2787866798258, 361.37752708736167, 361.47626749489746, 361.5750079024333, 361.6737483099691, 361.77248871750487, 361.8712291250407, 361.9699695325765, 362.0687099401123, 362.1674503476481, 362.2661907551839, 362.3649311627197, 362.46367157025554, 362.5624119777913, 362.6611523853271, 362.75989279286296, 362.85863320039874, 362.9573736079345, 363.05611401547037, 363.15485442300616, 363.25359483054194, 363.3523352380778, 363.45107564561357, 363.54981605314936, 363.6485564606852, 363.747296868221, 363.8460372757568, 363.9447776832926, 364.0435180908284, 364.14225849836424, 364.2409989059, 364.3397393134358, 364.43847972097166, 364.53722012850744, 364.6359605360432, 364.73470094357907, 364.83344135111486, 364.93218175865064, 365.0309221661865, 365.12966257372227, 365.22840298125806, 365.3271433887939, 365.4258837963297, 365.52462420386547, 365.6233646114013, 365.7221050189371, 365.8208454264729, 365.9195858340087, 366.0183262415445, 366.11706664908036, 366.21580705661614, 366.31454746415193, 366.41328787168777, 366.51202827922356, 366.61076868675934, 366.7095090942952, 366.80824950183097, 366.90698990936676, 367.0057303169026, 367.1044707244384, 367.2032111319742, 367.30195153951, 367.4006919470458, 367.4994323545816, 367.59817276211743, 367.6969131696532, 367.795653577189, 367.89439398472484, 367.99313439226063, 368.09187479979647, 368.19061520733226, 368.28935561486804, 368.3880960224039, 368.4868364299397, 368.58557683747546, 368.6843172450113, 368.7830576525471, 368.8817980600829, 368.9805384676187, 369.0792788751545, 369.1780192826903, 369.27675969022613, 369.3755000977619, 369.4742405052977, 369.57298091283354, 369.67172132036933, 369.7704617279051, 369.86920213544096, 369.96794254297674, 370.06668295051253, 370.1654233580484, 370.26416376558416, 370.36290417312, 370.4616445806558, 370.5603849881916, 370.6591253957274, 370.7578658032632, 370.856606210799, 370.95534661833483, 371.0540870258706, 371.1528274334064, 371.25156784094224, 371.35030824847803, 371.4490486560138, 371.54778906354966, 371.64652947108544, 371.74526987862123, 371.8440102861571, 371.94275069369286, 372.04149110122864, 372.1402315087645, 372.2389719163003, 372.33771232383606, 372.4364527313719, 372.5351931389077, 372.63393354644353, 372.7326739539793, 372.8314143615151, 372.93015476905094, 373.02889517658673, 373.1276355841225, 373.22637599165836, 373.32511639919414, 373.42385680672993, 373.5225972142658, 373.62133762180156, 373.72007802933734, 373.8188184368732, 373.917558844409, 374.01629925194476, 374.1150396594806, 374.2137800670164, 374.3125204745522, 374.411260882088, 374.5100012896238, 374.60874169715964, 374.70748210469543, 374.8062225122312, 374.90496291976706, 375.00370332730284, 375.10244373483863, 375.2011841423745, 375.29992454991026, 375.39866495744604, 375.4974053649819, 375.5961457725177, 375.69488618005346, 375.7936265875893, 375.8923669951251, 375.9911074026609, 376.0898478101967, 376.1885882177325, 376.2873286252683, 376.38606903280413, 376.4848094403399, 376.5835498478757, 376.68229025541154, 376.78103066294733, 376.8797710704832, 376.97851147801896, 377.07725188555474, 377.1759922930906, 377.2747327006264, 377.37347310816216, 377.472213515698, 377.5709539232338, 377.6696943307696, 377.7684347383054, 377.8671751458412, 377.965915553377, 378.06465596091283, 378.1633963684486, 378.2621367759844, 378.36087718352024, 378.45961759105603, 378.5583579985918, 378.65709840612766, 378.75583881366344, 378.85457922119923, 378.9533196287351, 379.05206003627086, 379.1508004438067, 379.2495408513425, 379.3482812588783, 379.4470216664141, 379.5457620739499, 379.6445024814857, 379.74324288902153, 379.8419832965573, 379.9407237040931, 380.03946411162894, 380.13820451916473, 380.2369449267005, 380.33568533423636, 380.43442574177215, 380.53316614930793, 380.6319065568438, 380.73064696437956, 380.82938737191535, 380.9281277794512, 381.026868186987, 381.12560859452276, 381.2243490020586, 381.3230894095944, 381.42182981713023, 381.520570224666, 381.6193106322018, 381.71805103973765, 381.81679144727343, 381.9155318548092, 382.01427226234506, 382.11301266988085, 382.21175307741663, 382.3104934849525, 382.40923389248826, 382.50797430002405, 382.6067147075599, 382.7054551150957, 382.80419552263146, 382.9029359301673, 383.0016763377031, 383.1004167452389, 383.1991571527747, 383.2978975603105, 383.39663796784635, 383.49537837538213, 383.5941187829179, 383.69285919045376, 383.79159959798955, 383.89034000552533, 383.9890804130612, 384.08782082059696, 384.18656122813275, 384.2853016356686, 384.3840420432044, 384.48278245074016, 384.581522858276, 384.6802632658118, 384.7790036733476, 384.8777440808834, 384.9764844884192, 385.075224895955, 385.17396530349083, 385.2727057110266, 385.3714461185624, 385.47018652609825, 385.56892693363403, 385.6676673411699, 385.76640774870566, 385.86514815624145, 385.9638885637773, 386.0626289713131, 386.16136937884886, 386.2601097863847, 386.3588501939205, 386.4575906014563, 386.5563310089921, 386.6550714165279, 386.7538118240637, 386.85255223159953, 386.9512926391353, 387.0500330466711, 387.14877345420695, 387.24751386174273, 387.3462542692785, 387.44499467681436, 387.54373508435015, 387.64247549188593, 387.7412158994218, 387.83995630695756, 387.9386967144934, 388.0374371220292, 388.136177529565, 388.2349179371008, 388.3336583446366, 388.4323987521724, 388.53113915970823, 388.629879567244, 388.7286199747798, 388.82736038231565, 388.92610078985143, 389.0248411973872, 389.12358160492306, 389.22232201245885, 389.32106241999463, 389.4198028275305, 389.51854323506626, 389.61728364260205, 389.7160240501379, 389.8147644576737, 389.9135048652095, 390.0122452727453, 390.1109856802811, 390.20972608781693, 390.3084664953527, 390.4072069028885, 390.50594731042435, 390.60468771796013, 390.7034281254959, 390.80216853303176, 390.90090894056755, 390.99964934810333, 391.0983897556392, 391.19713016317496, 391.29587057071075, 391.3946109782466, 391.4933513857824, 391.59209179331816, 391.690832200854, 391.7895726083898, 391.8883130159256, 391.9870534234614, 392.0857938309972, 392.18453423853305, 392.28327464606883, 392.3820150536046, 392.48075546114046, 392.57949586867625, 392.67823627621203, 392.7769766837479, 392.87571709128366, 392.97445749881945, 393.0731979063553, 393.1719383138911, 393.27067872142686, 393.3694191289627, 393.4681595364985, 393.5668999440343, 393.6656403515701, 393.7643807591059, 393.8631211666417, 393.96186157417753, 394.0606019817133, 394.1593423892491, 394.25808279678495, 394.35682320432073, 394.4555636118566, 394.55430401939236, 394.65304442692815, 394.751784834464, 394.8505252419998, 394.94926564953556, 395.0480060570714, 395.1467464646072, 395.245486872143, 395.3442272796788, 395.4429676872146, 395.5417080947504, 395.64044850228623, 395.739188909822, 395.8379293173578, 395.93666972489365, 396.03541013242943, 396.1341505399652, 396.23289094750106, 396.33163135503685, 396.43037176257263, 396.5291121701085, 396.62785257764426, 396.7265929851801, 396.8253333927159, 396.9240738002517, 397.0228142077875, 397.1215546153233, 397.2202950228591, 397.31903543039493, 397.4177758379307, 397.5165162454665, 397.61525665300235, 397.71399706053813, 397.8127374680739, 397.91147787560976, 398.01021828314555, 398.10895869068133, 398.2076990982172, 398.30643950575296, 398.40517991328875, 398.5039203208246, 398.6026607283604, 398.7014011358962, 398.800141543432, 398.8988819509678, 398.99762235850363, 399.0963627660394, 399.1951031735752, 399.29384358111105, 399.39258398864683, 399.4913243961826, 399.59006480371846, 399.68880521125425, 399.78754561879003, 399.8862860263259, 399.98502643386166, 400.08376684139745, 400.1825072489333, 400.2812476564691, 400.37998806400486, 400.4787284715407, 400.5774688790765, 400.6762092866123, 400.7749496941481, 400.8736901016839, 400.97243050921975, 401.07117091675553, 401.1699113242913, 401.26865173182716, 401.36739213936295, 401.46613254689873, 401.5648729544346, 401.66361336197036, 401.76235376950615, 401.861094177042, 401.9598345845778, 402.05857499211356, 402.1573153996494, 402.2560558071852, 402.354796214721, 402.4535366222568, 402.5522770297926, 402.6510174373284, 402.74975784486423, 402.8484982524, 402.9472386599358, 403.04597906747165, 403.14471947500743, 403.2434598825433, 403.34220029007906, 403.44094069761485, 403.5396811051507, 403.6384215126865, 403.73716192022226, 403.8359023277581, 403.9346427352939, 404.0333831428297, 404.1321235503655, 404.2308639579013, 404.3296043654371, 404.42834477297293, 404.5270851805087, 404.6258255880445, 404.72456599558035, 404.82330640311613, 404.9220468106519, 405.02078721818776, 405.11952762572355, 405.2182680332594, 405.3170084407952, 405.41574884833096, 405.5144892558668, 405.6132296634026, 405.7119700709384, 405.8107104784742, 405.90945088601, 406.0081912935458, 406.10693170108163, 406.2056721086174, 406.3044125161532, 406.40315292368905, 406.50189333122484, 406.6006337387606, 406.69937414629646, 406.79811455383225, 406.89685496136804, 406.9955953689039, 407.09433577643966, 407.19307618397545, 407.2918165915113, 407.3905569990471, 407.4892974065829, 407.5880378141187, 407.6867782216545, 407.78551862919034, 407.8842590367261, 407.9829994442619, 408.08173985179775, 408.18048025933354, 408.2792206668693, 408.37796107440516, 408.47670148194095, 408.57544188947674, 408.6741822970126, 408.77292270454836, 408.87166311208415, 408.97040351962, 409.0691439271558, 409.16788433469156, 409.2666247422274, 409.3653651497632, 409.464105557299, 409.5628459648348, 409.6615863723706, 409.76032677990645, 409.85906718744224, 409.957807594978, 410.05654800251386, 410.15528841004965, 410.25402881758544, 410.3527692251213, 410.45150963265706, 410.55025004019285, 410.6489904477287, 410.7477308552645, 410.84647126280026, 410.9452116703361, 411.0439520778719, 411.1426924854077, 411.2414328929435, 411.3401733004793, 411.4389137080151, 411.53765411555094, 411.6363945230867, 411.7351349306225, 411.83387533815835, 411.93261574569414, 412.03135615323, 412.13009656076576, 412.22883696830155, 412.3275773758374, 412.4263177833732, 412.52505819090896, 412.6237985984448, 412.7225390059806, 412.8212794135164, 412.9200198210522, 413.018760228588, 413.1175006361238, 413.21624104365964, 413.3149814511954, 413.4137218587312, 413.51246226626705, 413.61120267380284, 413.7099430813386, 413.80868348887446, 413.90742389641025, 414.0061643039461, 414.1049047114819, 414.20364511901766, 414.3023855265535, 414.4011259340893, 414.4998663416251, 414.5986067491609, 414.6973471566967, 414.7960875642325, 414.89482797176834, 414.9935683793041, 415.0923087868399, 415.19104919437575, 415.28978960191154, 415.3885300094473, 415.48727041698316, 415.58601082451895, 415.68475123205474, 415.7834916395906, 415.88223204712637, 415.98097245466215, 416.079712862198, 416.1784532697338, 416.2771936772696, 416.3759340848054, 416.4746744923412, 416.57341489987704, 416.6721553074128, 416.7708957149486, 416.86963612248445, 416.96837653002024, 417.067116937556, 417.16585734509187, 417.26459775262765, 417.36333816016344, 417.4620785676993, 417.56081897523507, 417.65955938277085, 417.7582997903067, 417.8570401978425, 417.95578060537827, 418.0545210129141, 418.1532614204499, 418.2520018279857, 418.3507422355215, 418.4494826430573, 418.54822305059315, 418.64696345812894, 418.7457038656647, 418.84444427320057, 418.94318468073635, 419.04192508827214], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "tickangle": 45, "tickmode": "array", "ticktext": ["2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18", "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26", "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-08", "2020-08-09", "2020-08-10", "2020-08-11", "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-15", "2020-08-16", "2020-08-17", "2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27", "2020-08-28", "2020-08-29", "2020-08-30", "2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-05", "2020-09-06", "2020-09-07", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-12", "2020-09-13", "2020-09-14", "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-19", "2020-09-20", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-26", "2020-09-27", "2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-04", "2020-10-05", "2020-10-06", "2020-10-07", "2020-10-08", "2020-10-09", "2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16", "2020-10-17", "2020-10-18", "2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22", "2020-10-23", "2020-10-24", "2020-10-25", "2020-10-26", "2020-10-27", "2020-10-28", "2020-10-29", "2020-10-30", "2020-10-31", "2020-11-01", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05", "2020-11-06", "2020-11-07", "2020-11-08", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14", "2020-11-15", "2020-11-16", "2020-11-17", "2020-11-18", "2020-11-19", "2020-11-20", "2020-11-21", "2020-11-22", "2020-11-23", "2020-11-24", "2020-11-25", "2020-11-26", "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30", "2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-05", "2020-12-06", "2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14", "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-19", "2020-12-20", "2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25", "2020-12-26", "2020-12-27", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15", "2021-01-16", "2021-01-17", "2021-01-18", "2021-01-19", "2021-01-20", "2021-01-21", "2021-01-22", "2021-01-23", "2021-01-24", "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05", "2021-02-06", "2021-02-07", "2021-02-08", "2021-02-09", "2021-02-10", "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19", "2021-02-20", "2021-02-21", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25", "2021-02-26", "2021-02-27", "2021-02-28", "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05", "2021-03-06", "2021-03-07", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12", "2021-03-13", "2021-03-14", "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-20", "2021-03-21", "2021-03-22", "2021-03-23", "2021-03-24", "2021-03-25", "2021-03-26", "2021-03-27", "2021-03-28", "2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01", "2021-04-02", "2021-04-03", "2021-04-04", "2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08", "2021-04-09", "2021-04-10", "2021-04-11", "2021-04-12", "2021-04-13", "2021-04-14", "2021-04-15", "2021-04-16", "2021-04-17", "2021-04-18", "2021-04-19", "2021-04-20", "2021-04-21", "2021-04-22", "2021-04-23", "2021-04-24", "2021-04-25", "2021-04-26", "2021-04-27", "2021-04-28", "2021-04-29", "2021-04-30", "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08", "2021-05-09", "2021-05-10", "2021-05-11", "2021-05-12", "2021-05-13", "2021-05-14", "2021-05-15", "2021-05-16", "2021-05-17", "2021-05-18", "2021-05-19", "2021-05-20", "2021-05-21", "2021-05-22", "2021-05-23", "2021-05-24", "2021-05-25", "2021-05-26", "2021-05-27", "2021-05-28", "2021-05-29", "2021-05-30", "2021-05-31", "2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04", "2021-06-05", "2021-06-06", "2021-06-07", "2021-06-08", "2021-06-09", "2021-06-10", "2021-06-11", "2021-06-12", "2021-06-13", "2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18", "2021-06-19", "2021-06-20", "2021-06-21", "2021-06-22", "2021-06-23", "2021-06-24", "2021-06-25", "2021-06-26", "2021-06-27", "2021-06-28", "2021-06-29", "2021-06-30", "2021-07-01", "2021-07-02", "2021-07-03", "2021-07-04", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10", "2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18", "2021-07-19", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25", "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31", "2021-08-01", "2021-08-02", "2021-08-03", "2021-08-04", "2021-08-05", "2021-08-06", "2021-08-07", "2021-08-08", "2021-08-09", "2021-08-10", "2021-08-11", "2021-08-12", "2021-08-13", "2021-08-14", "2021-08-15", "2021-08-16", "2021-08-17", "2021-08-18", "2021-08-19", "2021-08-20", "2021-08-21", "2021-08-22", "2021-08-23", "2021-08-24", "2021-08-25", "2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29", "2021-08-30", "2021-08-31", "2021-09-01", "2021-09-02", "2021-09-03", "2021-09-04", "2021-09-05", "2021-09-06", "2021-09-07", "2021-09-08", "2021-09-09", "2021-09-10", "2021-09-11", "2021-09-12", "2021-09-13", "2021-09-14", "2021-09-15", "2021-09-16", "2021-09-17", "2021-09-18", "2021-09-19", "2021-09-20", "2021-09-21", "2021-09-22", "2021-09-23", "2021-09-24", "2021-09-25", "2021-09-26", "2021-09-27", "2021-09-28", "2021-09-29", "2021-09-30", "2021-10-01", "2021-10-02", "2021-10-03", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07", "2021-10-08", "2021-10-09", "2021-10-10", "2021-10-11", "2021-10-12", "2021-10-13", "2021-10-14", "2021-10-15", "2021-10-16", "2021-10-17", "2021-10-18", "2021-10-19", "2021-10-20", "2021-10-21", "2021-10-22", "2021-10-23", "2021-10-24", "2021-10-25", "2021-10-26", "2021-10-27", "2021-10-28", "2021-10-29", "2021-10-30", "2021-10-31", "2021-11-01", "2021-11-02", "2021-11-03", "2021-11-04", "2021-11-05", "2021-11-06", "2021-11-07", "2021-11-08", "2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12", "2021-11-13", "2021-11-14", "2021-11-15", "2021-11-16", "2021-11-17", "2021-11-18", "2021-11-19", "2021-11-20", "2021-11-21", "2021-11-22", "2021-11-23", "2021-11-24", "2021-11-25", "2021-11-26", "2021-11-27", "2021-11-28", "2021-11-29", "2021-11-30", "2021-12-01", "2021-12-02", "2021-12-03", "2021-12-04", "2021-12-05", "2021-12-06", "2021-12-07", "2021-12-08", "2021-12-09", "2021-12-10", "2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14", "2021-12-15", "2021-12-16", "2021-12-17", "2021-12-18", "2021-12-19", "2021-12-20", "2021-12-21", "2021-12-22", "2021-12-23", "2021-12-24", "2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28", "2021-12-29", "2021-12-30", "2021-12-31", "2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-08", "2022-01-09", "2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14", "2022-01-15", "2022-01-16", "2022-01-17", "2022-01-18", "2022-01-19", "2022-01-20", "2022-01-21", "2022-01-22", "2022-01-23", "2022-01-24", "2022-01-25", "2022-01-26", "2022-01-27", "2022-01-28", "2022-01-29", "2022-01-30", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04", "2022-02-05", "2022-02-06", "2022-02-07", "2022-02-08", "2022-02-09", "2022-02-10", "2022-02-11", "2022-02-12", "2022-02-13", "2022-02-14", "2022-02-15", "2022-02-16", "2022-02-17", "2022-02-18", "2022-02-19", "2022-02-20", "2022-02-21", "2022-02-22", "2022-02-23", "2022-02-24", "2022-02-25", "2022-02-26", "2022-02-27", "2022-02-28", "2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04", "2022-03-05", "2022-03-06", "2022-03-07", "2022-03-08", "2022-03-09", "2022-03-10", "2022-03-11", "2022-03-12", "2022-03-13", "2022-03-14", "2022-03-15", "2022-03-16", "2022-03-17", "2022-03-18", "2022-03-19", "2022-03-20", "2022-03-21", "2022-03-22", "2022-03-23", "2022-03-24", "2022-03-25", "2022-03-26", "2022-03-27", "2022-03-28", "2022-03-29", "2022-03-30", "2022-03-31", "2022-04-01", "2022-04-02", "2022-04-03", "2022-04-04", "2022-04-05", "2022-04-06", "2022-04-07", "2022-04-08", "2022-04-09", "2022-04-10", "2022-04-11", "2022-04-12", "2022-04-13", "2022-04-14", "2022-04-15", "2022-04-16", "2022-04-17", "2022-04-18", "2022-04-19", "2022-04-20", "2022-04-21", "2022-04-22", "2022-04-23", "2022-04-24", "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29", "2022-04-30", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", "2022-05-05", "2022-05-06", "2022-05-07", "2022-05-08", "2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12", "2022-05-13", "2022-05-14", "2022-05-15", "2022-05-16", "2022-05-17", "2022-05-18", "2022-05-19", "2022-05-20", "2022-05-21", "2022-05-22", "2022-05-23", "2022-05-24", "2022-05-25", "2022-05-26", "2022-05-27", "2022-05-28", "2022-05-29", "2022-05-30", "2022-05-31", "2022-06-01", "2022-06-02", "2022-06-03", "2022-06-04", "2022-06-05", "2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10", "2022-06-11", "2022-06-12", "2022-06-13", "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17", "2022-06-18", "2022-06-19", "2022-06-20", "2022-06-21", "2022-06-22", "2022-06-23", "2022-06-24", "2022-06-25", "2022-06-26", "2022-06-27", "2022-06-28", "2022-06-29", "2022-06-30", "2022-07-01", "2022-07-02", "2022-07-03", "2022-07-04", "2022-07-05", "2022-07-06", "2022-07-07", "2022-07-08", "2022-07-09", "2022-07-10", "2022-07-11", "2022-07-12", "2022-07-13", "2022-07-14", "2022-07-15", "2022-07-16", "2022-07-17", "2022-07-18", "2022-07-19", "2022-07-20", "2022-07-21", "2022-07-22", "2022-07-23", "2022-07-24", "2022-07-25", "2022-07-26", "2022-07-27", "2022-07-28", "2022-07-29", "2022-07-30", "2022-07-31", "2022-08-01", "2022-08-02", "2022-08-03", "2022-08-04", "2022-08-05", "2022-08-06", "2022-08-07", "2022-08-08", "2022-08-09", "2022-08-10", "2022-08-11", "2022-08-12", "2022-08-13", "2022-08-14", "2022-08-15", "2022-08-16", "2022-08-17", "2022-08-18", "2022-08-19", "2022-08-20", "2022-08-21", "2022-08-22", "2022-08-23", "2022-08-24", "2022-08-25", "2022-08-26", "2022-08-27", "2022-08-28", "2022-08-29", "2022-08-30", "2022-08-31", "2022-09-01", "2022-09-02", "2022-09-03", "2022-09-04", "2022-09-05", "2022-09-06", "2022-09-07", "2022-09-08", "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12", "2022-09-13", "2022-09-14", "2022-09-15", "2022-09-16", "2022-09-17", "2022-09-18", "2022-09-19", "2022-09-20", "2022-09-21", "2022-09-22", "2022-09-23", "2022-09-24", "2022-09-25", "2022-09-26", "2022-09-27", "2022-09-28", "2022-09-29", "2022-09-30", "2022-10-01", "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07", "2022-10-08", "2022-10-09", "2022-10-10", "2022-10-11", "2022-10-12", "2022-10-13", "2022-10-14", "2022-10-15", "2022-10-16", "2022-10-17", "2022-10-18", "2022-10-19", "2022-10-20", "2022-10-21", "2022-10-22", "2022-10-23", "2022-10-24", "2022-10-25", "2022-10-26", "2022-10-27", "2022-10-28", "2022-10-29", "2022-10-30", "2022-10-31", "2022-11-01", "2022-11-02", "2022-11-03", "2022-11-04", "2022-11-05", "2022-11-06", "2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-12", "2022-11-13", "2022-11-14", "2022-11-15", "2022-11-16", "2022-11-17", "2022-11-18", "2022-11-19", "2022-11-20", "2022-11-21", "2022-11-22", "2022-11-23", "2022-11-24", "2022-11-25", "2022-11-26", "2022-11-27", "2022-11-28", "2022-11-29", "2022-11-30", "2022-12-01", "2022-12-02", "2022-12-03", "2022-12-04", "2022-12-05", "2022-12-06", "2022-12-07", "2022-12-08", "2022-12-09", "2022-12-10", "2022-12-11", "2022-12-12", "2022-12-13", "2022-12-14", "2022-12-15", "2022-12-16", "2022-12-17", "2022-12-18", "2022-12-19", "2022-12-20", "2022-12-21", "2022-12-22", "2022-12-23", "2022-12-24", "2022-12-25", "2022-12-26", "2022-12-27", "2022-12-28", "2022-12-29", "2022-12-30", "2022-12-31", "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-14", "2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-28", "2023-01-29", "2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02", "2023-02-03", "2023-02-04", "2023-02-05", "2023-02-06", "2023-02-07", "2023-02-08", "2023-02-09", "2023-02-10", "2023-02-11", "2023-02-12", "2023-02-13", "2023-02-14", "2023-02-15", "2023-02-16", "2023-02-17", "2023-02-18", "2023-02-19", "2023-02-20", "2023-02-21", "2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25", "2023-02-26", "2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02", "2023-03-03", "2023-03-04", "2023-03-05", "2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09", "2023-03-10", "2023-03-11", "2023-03-12", "2023-03-13", "2023-03-14", "2023-03-15", "2023-03-16", "2023-03-17", "2023-03-18", "2023-03-19", "2023-03-20", "2023-03-21", "2023-03-22", "2023-03-23", "2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27", "2023-03-28", "2023-03-29", "2023-03-30", "2023-03-31", "2023-04-01", "2023-04-02", "2023-04-03", "2023-04-04", "2023-04-05", "2023-04-06", "2023-04-07", "2023-04-08", "2023-04-09", "2023-04-10", "2023-04-11", "2023-04-12", "2023-04-13", "2023-04-14", "2023-04-15", "2023-04-16", "2023-04-17", "2023-04-18", "2023-04-19", "2023-04-20", "2023-04-21", "2023-04-22", "2023-04-23", "2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27", "2023-04-28", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-06", "2023-05-07", "2023-05-08", "2023-05-09", "2023-05-10", "2023-05-11", "2023-05-12", "2023-05-13", "2023-05-14", "2023-05-15", "2023-05-16", "2023-05-17", "2023-05-18", "2023-05-19", "2023-05-20", "2023-05-21", "2023-05-22", "2023-05-23", "2023-05-24", "2023-05-25", "2023-05-26", "2023-05-27", "2023-05-28", "2023-05-29", "2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02", "2023-06-03", "2023-06-04", "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08", "2023-06-09", "2023-06-10", "2023-06-11", "2023-06-12", "2023-06-13", "2023-06-14", "2023-06-15", "2023-06-16", "2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20", "2023-06-21", "2023-06-22", "2023-06-23", "2023-06-24", "2023-06-25", "2023-06-26", "2023-06-27", "2023-06-28", "2023-06-29", "2023-06-30", "2023-07-01", "2023-07-02", "2023-07-03", "2023-07-04", "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-08", "2023-07-09", "2023-07-10", "2023-07-11", "2023-07-12", "2023-07-13", "2023-07-14", "2023-07-15", "2023-07-16", "2023-07-17", "2023-07-18", "2023-07-19", "2023-07-20", "2023-07-21", "2023-07-22", "2023-07-23", "2023-07-24", "2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-29", "2023-07-30", "2023-07-31", "2023-08-01", "2023-08-02", "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06", "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10", "2023-08-11", "2023-08-12", "2023-08-13", "2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17", "2023-08-18", "2023-08-19", "2023-08-20", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-25", "2023-08-26", "2023-08-27", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-02", "2023-09-03", "2023-09-04", "2023-09-05", "2023-09-06", "2023-09-07", "2023-09-08", "2023-09-09", "2023-09-10", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", "2023-09-15", "2023-09-16", "2023-09-17", "2023-09-18", "2023-09-19", "2023-09-20", "2023-09-21", "2023-09-22", "2023-09-23", "2023-09-24", "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-07", "2023-10-08", "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-14", "2023-10-15", "2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20", "2023-10-21", "2023-10-22", "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-28", "2023-10-29", "2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-04", "2023-11-05", "2023-11-06", "2023-11-07", "2023-11-08", "2023-11-09", "2023-11-10", "2023-11-11", "2023-11-12", "2023-11-13", "2023-11-14", "2023-11-15", "2023-11-16", "2023-11-17", "2023-11-18", "2023-11-19", "2023-11-20", "2023-11-21", "2023-11-22", "2023-11-23", "2023-11-24", "2023-11-25", "2023-11-26", "2023-11-27", "2023-11-28", "2023-11-29", "2023-11-30", "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07", "2023-12-08", "2023-12-09", "2023-12-10", "2023-12-11", "2023-12-12", "2023-12-13", "2023-12-14", "2023-12-15", "2023-12-16", "2023-12-17", "2023-12-18", "2023-12-19", "2023-12-20", "2023-12-21", "2023-12-22", "2023-12-23", "2023-12-24", "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29", "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20", "2024-01-21", "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-27", "2024-01-28", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17", "2024-02-18", "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22", "2024-02-23", "2024-02-24", "2024-02-25", "2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24", "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29", "2024-03-30", "2024-03-31", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05", "2024-04-06", "2024-04-07", "2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-13", "2024-04-14", "2024-04-15", "2024-04-16", "2024-04-17", "2024-04-18", "2024-04-19", "2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24", "2024-04-25", "2024-04-26", "2024-04-27", "2024-04-28", "2024-04-29", "2024-04-30", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09", "2024-05-10", "2024-05-11", "2024-05-12", "2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16", "2024-05-17", "2024-05-18", "2024-05-19", "2024-05-20", "2024-05-21", "2024-05-22", "2024-05-23", "2024-05-24", "2024-05-25", "2024-05-26", "2024-05-27", "2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31", "2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-08", "2024-06-09", "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14", "2024-06-15", "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20", "2024-06-21", "2024-06-22", "2024-06-23", "2024-06-24", "2024-06-25", "2024-06-26", "2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30", "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05", "2024-07-06", "2024-07-07", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12", "2024-07-13", "2024-07-14", "2024-07-15", "2024-07-16", "2024-07-17", "2024-07-18", "2024-07-19", "2024-07-20", "2024-07-21", "2024-07-22", "2024-07-23", "2024-07-24", "2024-07-25", "2024-07-26", "2024-07-27", "2024-07-28", "2024-07-29", "2024-07-30", "2024-07-31", "2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04", "2024-08-05", "2024-08-06", "2024-08-07", "2024-08-08", "2024-08-09", "2024-08-10", "2024-08-11", "2024-08-12", "2024-08-13", "2024-08-14", "2024-08-15", "2024-08-16", "2024-08-17", "2024-08-18", "2024-08-19", "2024-08-20", "2024-08-21", "2024-08-22", "2024-08-23", "2024-08-24", "2024-08-25", "2024-08-26", "2024-08-27", "2024-08-28", "2024-08-29", "2024-08-30", "2024-08-31", "2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05", "2024-09-06", "2024-09-07", "2024-09-08", "2024-09-09", "2024-09-10", "2024-09-11", "2024-09-12", "2024-09-13", "2024-09-14", "2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19", "2024-09-20", "2024-09-21", "2024-09-22", "2024-09-23", "2024-09-24", "2024-09-25", "2024-09-26", "2024-09-27", "2024-09-28", "2024-09-29", "2024-09-30", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07", "2024-10-08", "2024-10-09", "2024-10-10", "2024-10-11", "2024-10-12", "2024-10-13", "2024-10-14", "2024-10-15", "2024-10-16", "2024-10-17", "2024-10-18", "2024-10-19", "2024-10-20", "2024-10-21", "2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25", "2024-10-26", "2024-10-27", "2024-10-28", "2024-10-29", "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", "2024-11-09", "2024-11-10", "2024-11-11", "2024-11-12", "2024-11-13", "2024-11-14", "2024-11-15", "2024-11-16", "2024-11-17", "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21", "2024-11-22", "2024-11-23", "2024-11-24", "2024-11-25", "2024-11-26", "2024-11-27", "2024-11-28", "2024-11-29", "2024-11-30", "2024-12-01", "2024-12-02", "2024-12-03", "2024-12-04", "2024-12-05", "2024-12-06", "2024-12-07", "2024-12-08", "2024-12-09", "2024-12-10", "2024-12-11", "2024-12-12", "2024-12-13", "2024-12-14", "2024-12-15", "2024-12-16", "2024-12-17", "2024-12-18", "2024-12-19", "2024-12-20", "2024-12-21", "2024-12-22", "2024-12-23", "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-27", "2024-12-28", "2024-12-29", "2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17", "2025-01-18", "2025-01-19", "2025-01-20", "2025-01-21", "2025-01-22", "2025-01-23", "2025-01-24", "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07", "2025-02-08", "2025-02-09", "2025-02-10", "2025-02-11", "2025-02-12", "2025-02-13", "2025-02-14", "2025-02-15", "2025-02-16", "2025-02-17", "2025-02-18", "2025-02-19", "2025-02-20", "2025-02-21", "2025-02-22", "2025-02-23", "2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28", "2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07", "2025-03-08", "2025-03-09", "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14", "2025-03-15", "2025-03-16", "2025-03-17", "2025-03-18", "2025-03-19", "2025-03-20", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-24", "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06", "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11", "2025-04-12", "2025-04-13", "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18", "2025-04-19", "2025-04-20", "2025-04-21", "2025-04-22", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-26", "2025-04-27", "2025-04-28", "2025-04-29", "2025-04-30", "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06", "2025-05-07", "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11", "2025-05-12", "2025-05-13", "2025-05-14", "2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19", "2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-24", "2025-05-25", "2025-05-26", "2025-05-27", "2025-05-28", "2025-05-29", "2025-05-30", "2025-05-31", "2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15", "2025-06-16", "2025-06-17", "2025-06-18", "2025-06-19", "2025-06-20", "2025-06-21", "2025-06-22", "2025-06-23", "2025-06-24", "2025-06-25", "2025-06-26", "2025-06-27", "2025-06-28", "2025-06-29", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05", "2025-07-06", "2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11", "2025-07-12", "2025-07-13", "2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-19", "2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23", "2025-07-24", "2025-07-25", "2025-07-26", "2025-07-27", "2025-07-28", "2025-07-29", "2025-07-30", "2025-07-31", "2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-09", "2025-08-10", "2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14", "2025-08-15", "2025-08-16", "2025-08-17", "2025-08-18", "2025-08-19", "2025-08-20", "2025-08-21", "2025-08-22", "2025-08-23", "2025-08-24", "2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30", "2025-08-31", "2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05", "2025-09-06", "2025-09-07", "2025-09-08", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12", "2025-09-13", "2025-09-14", "2025-09-15", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19", "2025-09-20", "2025-09-21", "2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-26", "2025-09-27", "2025-09-28", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10", "2025-10-11", "2025-10-12", "2025-10-13", "2025-10-14", "2025-10-15", "2025-10-16", "2025-10-17", "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-21", "2025-10-22", "2025-10-23", "2025-10-24", "2025-10-25", "2025-10-26", "2025-10-27", "2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31", "2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04", "2025-11-05", "2025-11-06", "2025-11-07", "2025-11-08", "2025-11-09", "2025-11-10", "2025-11-11", "2025-11-12", "2025-11-13", "2025-11-14", "2025-11-15", "2025-11-16", "2025-11-17", "2025-11-18", "2025-11-19", "2025-11-20", "2025-11-21", "2025-11-22", "2025-11-23", "2025-11-24", "2025-11-25", "2025-11-26", "2025-11-27", "2025-11-28", "2025-11-29", "2025-11-30", "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05", "2025-12-06", "2025-12-07", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12", "2025-12-13", "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19", "2025-12-20", "2025-12-21", "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24", "2026-01-25", "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06", "2026-02-07", "2026-02-08", "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13", "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14", "2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26", "2026-03-27", "2026-03-28", "2026-03-29", "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17", "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24", "2026-04-25", "2026-04-26", "2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14", "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21", "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10", "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19", "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16", "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28", "2026-11-29", "2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04", "2026-12-05", "2026-12-06", "2026-12-07", "2026-12-08", "2026-12-09", "2026-12-10", "2026-12-11", "2026-12-12", "2026-12-13", "2026-12-14", "2026-12-15", "2026-12-16", "2026-12-17", "2026-12-18", "2026-12-19", "2026-12-20", "2026-12-21", "2026-12-22", "2026-12-23", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-27", "2026-12-28", "2026-12-29", "2026-12-30", "2026-12-31", "2027-01-01", "2027-01-02", "2027-01-03", "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07", "2027-01-08", "2027-01-09", "2027-01-10", "2027-01-11", "2027-01-12", "2027-01-13", "2027-01-14", "2027-01-15", "2027-01-16", "2027-01-17", "2027-01-18", "2027-01-19", "2027-01-20", "2027-01-21", "2027-01-22", "2027-01-23", "2027-01-24", "2027-01-25", "2027-01-26", "2027-01-27", "2027-01-28", "2027-01-29", "2027-01-30", "2027-01-31", "2027-02-01", "2027-02-02", "2027-02-03", "2027-02-04", "2027-02-05", "2027-02-06", "2027-02-07", "2027-02-08", "2027-02-09", "2027-02-10", "2027-02-11", "2027-02-12", "2027-02-13", "2027-02-14", "2027-02-15", "2027-02-16", "2027-02-17", "2027-02-18", "2027-02-19", "2027-02-20", "2027-02-21", "2027-02-22", "2027-02-23", "2027-02-24", "2027-02-25", "2027-02-26", "2027-02-27", "2027-02-28", "2027-03-01", "2027-03-02", "2027-03-03", "2027-03-04", "2027-03-05", "2027-03-06", "2027-03-07", "2027-03-08", "2027-03-09", "2027-03-10", "2027-03-11", "2027-03-12", "2027-03-13", "2027-03-14", "2027-03-15", "2027-03-16", "2027-03-17", "2027-03-18", "2027-03-19", "2027-03-20", "2027-03-21", "2027-03-22", "2027-03-23", "2027-03-24", "2027-03-25", "2027-03-26", "2027-03-27", "2027-03-28", "2027-03-29", "2027-03-30", "2027-03-31", "2027-04-01", "2027-04-02", "2027-04-03", "2027-04-04", "2027-04-05", "2027-04-06", "2027-04-07", "2027-04-08", "2027-04-09", "2027-04-10", "2027-04-11", "2027-04-12", "2027-04-13", "2027-04-14", "2027-04-15", "2027-04-16", "2027-04-17", "2027-04-18", "2027-04-19", "2027-04-20", "2027-04-21", "2027-04-22", "2027-04-23", "2027-04-24", "2027-04-25", "2027-04-26", "2027-04-27", "2027-04-28", "2027-04-29", "2027-04-30", "2027-05-01", "2027-05-02", "2027-05-03", "2027-05-04", "2027-05-05", "2027-05-06", "2027-05-07", "2027-05-08", "2027-05-09", "2027-05-10", "2027-05-11", "2027-05-12", "2027-05-13", "2027-05-14", "2027-05-15", "2027-05-16", "2027-05-17", "2027-05-18", "2027-05-19", "2027-05-20", "2027-05-21", "2027-05-22", "2027-05-23", "2027-05-24", "2027-05-25", "2027-05-26", "2027-05-27", "2027-05-28", "2027-05-29", "2027-05-30", "2027-05-31", "2027-06-01", "2027-06-02", "2027-06-03", "2027-06-04", "2027-06-05", "2027-06-06", "2027-06-07", "2027-06-08", "2027-06-09", "2027-06-10", "2027-06-11", "2027-06-12", "2027-06-13", "2027-06-14", "2027-06-15", "2027-06-16", "2027-06-17", "2027-06-18", "2027-06-19", "2027-06-20", "2027-06-21", "2027-06-22", "2027-06-23", "2027-06-24", "2027-06-25", "2027-06-26", "2027-06-27", "2027-06-28", "2027-06-29", "2027-06-30", "2027-07-01", "2027-07-02", "2027-07-03", "2027-07-04", "2027-07-05", "2027-07-06", "2027-07-07", "2027-07-08", "2027-07-09", "2027-07-10", "2027-07-11", "2027-07-12", "2027-07-13", "2027-07-14", "2027-07-15", "2027-07-16", "2027-07-17", "2027-07-18", "2027-07-19", "2027-07-20", "2027-07-21", "2027-07-22", "2027-07-23", "2027-07-24", "2027-07-25", "2027-07-26", "2027-07-27", "2027-07-28", "2027-07-29", "2027-07-30", "2027-07-31", "2027-08-01", "2027-08-02", "2027-08-03", "2027-08-04", "2027-08-05", "2027-08-06", "2027-08-07", "2027-08-08", "2027-08-09", "2027-08-10", "2027-08-11", "2027-08-12", "2027-08-13", "2027-08-14", "2027-08-15", "2027-08-16", "2027-08-17", "2027-08-18", "2027-08-19", "2027-08-20", "2027-08-21", "2027-08-22", "2027-08-23", "2027-08-24", "2027-08-25", "2027-08-26", "2027-08-27", "2027-08-28", "2027-08-29", "2027-08-30", "2027-08-31", "2027-09-01", "2027-09-02", "2027-09-03", "2027-09-04", "2027-09-05", "2027-09-06", "2027-09-07", "2027-09-08", "2027-09-09", "2027-09-10", "2027-09-11", "2027-09-12", "2027-09-13", "2027-09-14", "2027-09-15", "2027-09-16", "2027-09-17", "2027-09-18", "2027-09-19", "2027-09-20", "2027-09-21", "2027-09-22", "2027-09-23", "2027-09-24", "2027-09-25", "2027-09-26", "2027-09-27", "2027-09-28", "2027-09-29", "2027-09-30", "2027-10-01", "2027-10-02", "2027-10-03", "2027-10-04", "2027-10-05", "2027-10-06", "2027-10-07", "2027-10-08", "2027-10-09", "2027-10-10", "2027-10-11", "2027-10-12", "2027-10-13", "2027-10-14", "2027-10-15", "2027-10-16", "2027-10-17", "2027-10-18", "2027-10-19", "2027-10-20", "2027-10-21", "2027-10-22", "2027-10-23", "2027-10-24", "2027-10-25", "2027-10-26", "2027-10-27", "2027-10-28", "2027-10-29", "2027-10-30", "2027-10-31", "2027-11-01", "2027-11-02", "2027-11-03", "2027-11-04", "2027-11-05", "2027-11-06", "2027-11-07", "2027-11-08", "2027-11-09", "2027-11-10", "2027-11-11", "2027-11-12", "2027-11-13", "2027-11-14", "2027-11-15", "2027-11-16", "2027-11-17", "2027-11-18", "2027-11-19", "2027-11-20", "2027-11-21", "2027-11-22", "2027-11-23", "2027-11-24", "2027-11-25", "2027-11-26", "2027-11-27", "2027-11-28", "2027-11-29", "2027-11-30", "2027-12-01", "2027-12-02", "2027-12-03", "2027-12-04", "2027-12-05", "2027-12-06", "2027-12-07", "2027-12-08", "2027-12-09", "2027-12-10", "2027-12-11", "2027-12-12", "2027-12-13", "2027-12-14", "2027-12-15", "2027-12-16", "2027-12-17", "2027-12-18", "2027-12-19", "2027-12-20", "2027-12-21", "2027-12-22", "2027-12-23", "2027-12-24", "2027-12-25", "2027-12-26", "2027-12-27", "2027-12-28", "2027-12-29", "2027-12-30", "2027-12-31", "2028-01-01", "2028-01-02", "2028-01-03", "2028-01-04", "2028-01-05", "2028-01-06", "2028-01-07", "2028-01-08", "2028-01-09", "2028-01-10", "2028-01-11", "2028-01-12", "2028-01-13", "2028-01-14", "2028-01-15", "2028-01-16", "2028-01-17", "2028-01-18", "2028-01-19", "2028-01-20", "2028-01-21", "2028-01-22", "2028-01-23", "2028-01-24", "2028-01-25", "2028-01-26", "2028-01-27", "2028-01-28", "2028-01-29", "2028-01-30", "2028-01-31", "2028-02-01", "2028-02-02", "2028-02-03", "2028-02-04", "2028-02-05", "2028-02-06", "2028-02-07", "2028-02-08", "2028-02-09", "2028-02-10", "2028-02-11", "2028-02-12", "2028-02-13", "2028-02-14", "2028-02-15", "2028-02-16", "2028-02-17", "2028-02-18", "2028-02-19", "2028-02-20", "2028-02-21", "2028-02-22", "2028-02-23", "2028-02-24", "2028-02-25", "2028-02-26", "2028-02-27", "2028-02-28", "2028-02-29", "2028-03-01", "2028-03-02", "2028-03-03", "2028-03-04", "2028-03-05", "2028-03-06", "2028-03-07", "2028-03-08", "2028-03-09", "2028-03-10", "2028-03-11", "2028-03-12", "2028-03-13", "2028-03-14", "2028-03-15", "2028-03-16", "2028-03-17", "2028-03-18", "2028-03-19", "2028-03-20", "2028-03-21", "2028-03-22", "2028-03-23", "2028-03-24", "2028-03-25", "2028-03-26", "2028-03-27", "2028-03-28", "2028-03-29", "2028-03-30", "2028-03-31", "2028-04-01", "2028-04-02", "2028-04-03", "2028-04-04", "2028-04-05", "2028-04-06", "2028-04-07", "2028-04-08", "2028-04-09", "2028-04-10", "2028-04-11", "2028-04-12", "2028-04-13", "2028-04-14", "2028-04-15", "2028-04-16", "2028-04-17", "2028-04-18", "2028-04-19", "2028-04-20", "2028-04-21", "2028-04-22", "2028-04-23", "2028-04-24", "2028-04-25", "2028-04-26", "2028-04-27", "2028-04-28", "2028-04-29", "2028-04-30", "2028-05-01", "2028-05-02", "2028-05-03", "2028-05-04", "2028-05-05", "2028-05-06", "2028-05-07", "2028-05-08", "2028-05-09", "2028-05-10", "2028-05-11", "2028-05-12", "2028-05-13", "2028-05-14", "2028-05-15", "2028-05-16", "2028-05-17", "2028-05-18", "2028-05-19", "2028-05-20", "2028-05-21", "2028-05-22", "2028-05-23", "2028-05-24", "2028-05-25", "2028-05-26", "2028-05-27", "2028-05-28", "2028-05-29", "2028-05-30", "2028-05-31", "2028-06-01", "2028-06-02", "2028-06-03", "2028-06-04", "2028-06-05", "2028-06-06", "2028-06-07", "2028-06-08", "2028-06-09", "2028-06-10", "2028-06-11", "2028-06-12", "2028-06-13", "2028-06-14", "2028-06-15", "2028-06-16", "2028-06-17", "2028-06-18", "2028-06-19", "2028-06-20", "2028-06-21", "2028-06-22", "2028-06-23", "2028-06-24", "2028-06-25", "2028-06-26", "2028-06-27", "2028-06-28", "2028-06-29", "2028-06-30", "2028-07-01", "2028-07-02", "2028-07-03", "2028-07-04", "2028-07-05", "2028-07-06", "2028-07-07", "2028-07-08", "2028-07-09", "2028-07-10", "2028-07-11", "2028-07-12", "2028-07-13", "2028-07-14", "2028-07-15", "2028-07-16", "2028-07-17", "2028-07-18", "2028-07-19", "2028-07-20", "2028-07-21", "2028-07-22", "2028-07-23", "2028-07-24", "2028-07-25", "2028-07-26", "2028-07-27", "2028-07-28", "2028-07-29", "2028-07-30", "2028-07-31", "2028-08-01", "2028-08-02", "2028-08-03", "2028-08-04", "2028-08-05", "2028-08-06", "2028-08-07", "2028-08-08", "2028-08-09", "2028-08-10", "2028-08-11", "2028-08-12", "2028-08-13", "2028-08-14", "2028-08-15", "2028-08-16", "2028-08-17", "2028-08-18", "2028-08-19", "2028-08-20", "2028-08-21", "2028-08-22", "2028-08-23", "2028-08-24", "2028-08-25", "2028-08-26", "2028-08-27", "2028-08-28", "2028-08-29", "2028-08-30", "2028-08-31", "2028-09-01", "2028-09-02", "2028-09-03", "2028-09-04", "2028-09-05", "2028-09-06", "2028-09-07", "2028-09-08", "2028-09-09", "2028-09-10", "2028-09-11", "2028-09-12", "2028-09-13", "2028-09-14", "2028-09-15", "2028-09-16", "2028-09-17", "2028-09-18", "2028-09-19", "2028-09-20", "2028-09-21", "2028-09-22", "2028-09-23", "2028-09-24", "2028-09-25", "2028-09-26", "2028-09-27", "2028-09-28", "2028-09-29", "2028-09-30", "2028-10-01", "2028-10-02", "2028-10-03", "2028-10-04", "2028-10-05", "2028-10-06", "2028-10-07", "2028-10-08", "2028-10-09", "2028-10-10", "2028-10-11", "2028-10-12", "2028-10-13", "2028-10-14", "2028-10-15", "2028-10-16", "2028-10-17", "2028-10-18", "2028-10-19", "2028-10-20", "2028-10-21", "2028-10-22", "2028-10-23", "2028-10-24", "2028-10-25", "2028-10-26", "2028-10-27", "2028-10-28", "2028-10-29", "2028-10-30", "2028-10-31", "2028-11-01", "2028-11-02", "2028-11-03", "2028-11-04", "2028-11-05", "2028-11-06", "2028-11-07", "2028-11-08", "2028-11-09", "2028-11-10", "2028-11-11", "2028-11-12", "2028-11-13", "2028-11-14", "2028-11-15", "2028-11-16", "2028-11-17", "2028-11-18", "2028-11-19", "2028-11-20", "2028-11-21", "2028-11-22", "2028-11-23", "2028-11-24", "2028-11-25", "2028-11-26", "2028-11-27", "2028-11-28", "2028-11-29", "2028-11-30", "2028-12-01", "2028-12-02", "2028-12-03", "2028-12-04", "2028-12-05", "2028-12-06", "2028-12-07", "2028-12-08", "2028-12-09", "2028-12-10", "2028-12-11", "2028-12-12", "2028-12-13", "2028-12-14", "2028-12-15", "2028-12-16", "2028-12-17", "2028-12-18", "2028-12-19", "2028-12-20", "2028-12-21", "2028-12-22", "2028-12-23", "2028-12-24", "2028-12-25", "2028-12-26", "2028-12-27", "2028-12-28", "2028-12-29", "2028-12-30", "2028-12-31", "2029-01-01", "2029-01-02", "2029-01-03", "2029-01-04", "2029-01-05", "2029-01-06", "2029-01-07", "2029-01-08", "2029-01-09", "2029-01-10", "2029-01-11", "2029-01-12", "2029-01-13", "2029-01-14", "2029-01-15", "2029-01-16", "2029-01-17", "2029-01-18", "2029-01-19", "2029-01-20", "2029-01-21", "2029-01-22", "2029-01-23", "2029-01-24", "2029-01-25", "2029-01-26", "2029-01-27", "2029-01-28", "2029-01-29", "2029-01-30", "2029-01-31", "2029-02-01", "2029-02-02", "2029-02-03", "2029-02-04", "2029-02-05", "2029-02-06", "2029-02-07", "2029-02-08", "2029-02-09", "2029-02-10", "2029-02-11", "2029-02-12", "2029-02-13", "2029-02-14", "2029-02-15", "2029-02-16", "2029-02-17", "2029-02-18", "2029-02-19", "2029-02-20", "2029-02-21", "2029-02-22", "2029-02-23", "2029-02-24", "2029-02-25", "2029-02-26", "2029-02-27", "2029-02-28", "2029-03-01", "2029-03-02", "2029-03-03", "2029-03-04", "2029-03-05", "2029-03-06", "2029-03-07", "2029-03-08", "2029-03-09", "2029-03-10", "2029-03-11", "2029-03-12", "2029-03-13", "2029-03-14", "2029-03-15", "2029-03-16", "2029-03-17", "2029-03-18", "2029-03-19", "2029-03-20", "2029-03-21", "2029-03-22", "2029-03-23", "2029-03-24", "2029-03-25", "2029-03-26", "2029-03-27", "2029-03-28", "2029-03-29", "2029-03-30", "2029-03-31", "2029-04-01", "2029-04-02", "2029-04-03", "2029-04-04", "2029-04-05", "2029-04-06", "2029-04-07", "2029-04-08", "2029-04-09", "2029-04-10", "2029-04-11", "2029-04-12", "2029-04-13", "2029-04-14", "2029-04-15", "2029-04-16", "2029-04-17", "2029-04-18", "2029-04-19", "2029-04-20", "2029-04-21", "2029-04-22", "2029-04-23", "2029-04-24", "2029-04-25", "2029-04-26", "2029-04-27", "2029-04-28", "2029-04-29", "2029-04-30", "2029-05-01", "2029-05-02", "2029-05-03", "2029-05-04", "2029-05-05", "2029-05-06", "2029-05-07", "2029-05-08", "2029-05-09", "2029-05-10", "2029-05-11", "2029-05-12", "2029-05-13", "2029-05-14", "2029-05-15", "2029-05-16", "2029-05-17", "2029-05-18", "2029-05-19", "2029-05-20", "2029-05-21", "2029-05-22", "2029-05-23", "2029-05-24", "2029-05-25", "2029-05-26", "2029-05-27", "2029-05-28", "2029-05-29", "2029-05-30", "2029-05-31", "2029-06-01", "2029-06-02", "2029-06-03", "2029-06-04", "2029-06-05", "2029-06-06", "2029-06-07", "2029-06-08", "2029-06-09", "2029-06-10", "2029-06-11", "2029-06-12", "2029-06-13", "2029-06-14", "2029-06-15", "2029-06-16", "2029-06-17", "2029-06-18", "2029-06-19", "2029-06-20", "2029-06-21", "2029-06-22", "2029-06-23", "2029-06-24", "2029-06-25", "2029-06-26", "2029-06-27", "2029-06-28", "2029-06-29", "2029-06-30", "2029-07-01", "2029-07-02", "2029-07-03", "2029-07-04", "2029-07-05", "2029-07-06", "2029-07-07", "2029-07-08", "2029-07-09", "2029-07-10", "2029-07-11", "2029-07-12", "2029-07-13", "2029-07-14", "2029-07-15", "2029-07-16", "2029-07-17", "2029-07-18", "2029-07-19", "2029-07-20", "2029-07-21", "2029-07-22", "2029-07-23", "2029-07-24", "2029-07-25", "2029-07-26", "2029-07-27", "2029-07-28", "2029-07-29", "2029-07-30", "2029-07-31", "2029-08-01", "2029-08-02", "2029-08-03", "2029-08-04", "2029-08-05", "2029-08-06", "2029-08-07", "2029-08-08", "2029-08-09", "2029-08-10", "2029-08-11", "2029-08-12", "2029-08-13", "2029-08-14", "2029-08-15", "2029-08-16", "2029-08-17", "2029-08-18", "2029-08-19", "2029-08-20", "2029-08-21", "2029-08-22", "2029-08-23", "2029-08-24", "2029-08-25", "2029-08-26", "2029-08-27", "2029-08-28", "2029-08-29", "2029-08-30", "2029-08-31", "2029-09-01", "2029-09-02", "2029-09-03", "2029-09-04", "2029-09-05", "2029-09-06", "2029-09-07", "2029-09-08", "2029-09-09", "2029-09-10", "2029-09-11", "2029-09-12", "2029-09-13", "2029-09-14", "2029-09-15", "2029-09-16", "2029-09-17", "2029-09-18", "2029-09-19", "2029-09-20", "2029-09-21", "2029-09-22", "2029-09-23", "2029-09-24", "2029-09-25", "2029-09-26", "2029-09-27", "2029-09-28", "2029-09-29", "2029-09-30", "2029-10-01", "2029-10-02", "2029-10-03", "2029-10-04", "2029-10-05", "2029-10-06", "2029-10-07", "2029-10-08", "2029-10-09", "2029-10-10", "2029-10-11", "2029-10-12", "2029-10-13", "2029-10-14", "2029-10-15", "2029-10-16", "2029-10-17", "2029-10-18", "2029-10-19", "2029-10-20", "2029-10-21", "2029-10-22", "2029-10-23", "2029-10-24", "2029-10-25", "2029-10-26", "2029-10-27", "2029-10-28", "2029-10-29", "2029-10-30", "2029-10-31", "2029-11-01", "2029-11-02", "2029-11-03", "2029-11-04", "2029-11-05", "2029-11-06", "2029-11-07", "2029-11-08", "2029-11-09", "2029-11-10", "2029-11-11", "2029-11-12", "2029-11-13", "2029-11-14", "2029-11-15", "2029-11-16", "2029-11-17", "2029-11-18", "2029-11-19", "2029-11-20", "2029-11-21", "2029-11-22", "2029-11-23", "2029-11-24", "2029-11-25", "2029-11-26", "2029-11-27", "2029-11-28", "2029-11-29", "2029-11-30", "2029-12-01", "2029-12-02", "2029-12-03", "2029-12-04", "2029-12-05", "2029-12-06", "2029-12-07", "2029-12-08", "2029-12-09", "2029-12-10", "2029-12-11", "2029-12-12", "2029-12-13", "2029-12-14", "2029-12-15", "2029-12-16", "2029-12-17", "2029-12-18", "2029-12-19", "2029-12-20", "2029-12-21", "2029-12-22", "2029-12-23", "2029-12-24", "2029-12-25", "2029-12-26", "2029-12-27", "2029-12-28", "2029-12-29", "2029-12-30", "2029-12-31", "2030-01-01", "2030-01-02", "2030-01-03", "2030-01-04", "2030-01-05", "2030-01-06", "2030-01-07", "2030-01-08", "2030-01-09", "2030-01-10", "2030-01-11", "2030-01-12", "2030-01-13", "2030-01-14", "2030-01-15", "2030-01-16", "2030-01-17", "2030-01-18", "2030-01-19", "2030-01-20", "2030-01-21", "2030-01-22", "2030-01-23", "2030-01-24", "2030-01-25", "2030-01-26", "2030-01-27", "2030-01-28", "2030-01-29", "2030-01-30", "2030-01-31", "2030-02-01", "2030-02-02", "2030-02-03", "2030-02-04", "2030-02-05", "2030-02-06", "2030-02-07", "2030-02-08", "2030-02-09", "2030-02-10", "2030-02-11", "2030-02-12", "2030-02-13", "2030-02-14", "2030-02-15", "2030-02-16", "2030-02-17", "2030-02-18", "2030-02-19", "2030-02-20", "2030-02-21", "2030-02-22", "2030-02-23", "2030-02-24", "2030-02-25", "2030-02-26", "2030-02-27", "2030-02-28", "2030-03-01", "2030-03-02", "2030-03-03", "2030-03-04", "2030-03-05", "2030-03-06", "2030-03-07", "2030-03-08", "2030-03-09", "2030-03-10", "2030-03-11", "2030-03-12", "2030-03-13", "2030-03-14", "2030-03-15", "2030-03-16", "2030-03-17", "2030-03-18", "2030-03-19", "2030-03-20", "2030-03-21", "2030-03-22", "2030-03-23", "2030-03-24", "2030-03-25", "2030-03-26", "2030-03-27", "2030-03-28", "2030-03-29", "2030-03-30", "2030-03-31", "2030-04-01", "2030-04-02", "2030-04-03", "2030-04-04", "2030-04-05", "2030-04-06", "2030-04-07", "2030-04-08", "2030-04-09", "2030-04-10", "2030-04-11", "2030-04-12", "2030-04-13", "2030-04-14", "2030-04-15", "2030-04-16", "2030-04-17", "2030-04-18", "2030-04-19", "2030-04-20", "2030-04-21", "2030-04-22", "2030-04-23", "2030-04-24", "2030-04-25", "2030-04-26", "2030-04-27", "2030-04-28", "2030-04-29", "2030-04-30", "2030-05-01", "2030-05-02", "2030-05-03", "2030-05-04", "2030-05-05", "2030-05-06", "2030-05-07", "2030-05-08", "2030-05-09", "2030-05-10", "2030-05-11", "2030-05-12", "2030-05-13", "2030-05-14", "2030-05-15", "2030-05-16", "2030-05-17", "2030-05-18", "2030-05-19", "2030-05-20", "2030-05-21", "2030-05-22", "2030-05-23", "2030-05-24", "2030-05-25", "2030-05-26", "2030-05-27", "2030-05-28", "2030-05-29", "2030-05-30", "2030-05-31", "2030-06-01", "2030-06-02", "2030-06-03", "2030-06-04", "2030-06-05", "2030-06-06", "2030-06-07", "2030-06-08", "2030-06-09", "2030-06-10", "2030-06-11", "2030-06-12", "2030-06-13", "2030-06-14", "2030-06-15", "2030-06-16", "2030-06-17", "2030-06-18", "2030-06-19", "2030-06-20", "2030-06-21", "2030-06-22", "2030-06-23", "2030-06-24", "2030-06-25", "2030-06-26", "2030-06-27", "2030-06-28", "2030-06-29", "2030-06-30", "2030-07-01", "2030-07-02", "2030-07-03", "2030-07-04", "2030-07-05", "2030-07-06", "2030-07-07", "2030-07-08", "2030-07-09", "2030-07-10", "2030-07-11", "2030-07-12", "2030-07-13", "2030-07-14", "2030-07-15", "2030-07-16", "2030-07-17", "2030-07-18", "2030-07-19", "2030-07-20", "2030-07-21", "2030-07-22", "2030-07-23", "2030-07-24", "2030-07-25", "2030-07-26", "2030-07-27", "2030-07-28", "2030-07-29", "2030-07-30", "2030-07-31", "2030-08-01", "2030-08-02", "2030-08-03", "2030-08-04", "2030-08-05", "2030-08-06", "2030-08-07", "2030-08-08", "2030-08-09", "2030-08-10", "2030-08-11", "2030-08-12", "2030-08-13", "2030-08-14", "2030-08-15", "2030-08-16", "2030-08-17", "2030-08-18", "2030-08-19", "2030-08-20", "2030-08-21", "2030-08-22", "2030-08-23", "2030-08-24", "2030-08-25", "2030-08-26", "2030-08-27", "2030-08-28", "2030-08-29", "2030-08-30", "2030-08-31", "2030-09-01", "2030-09-02", "2030-09-03", "2030-09-04", "2030-09-05", "2030-09-06", "2030-09-07", "2030-09-08", "2030-09-09", "2030-09-10", "2030-09-11", "2030-09-12", "2030-09-13", "2030-09-14", "2030-09-15", "2030-09-16", "2030-09-17", "2030-09-18", "2030-09-19", "2030-09-20", "2030-09-21", "2030-09-22", "2030-09-23", "2030-09-24", "2030-09-25", "2030-09-26", "2030-09-27", "2030-09-28", "2030-09-29", "2030-09-30", "2030-10-01", "2030-10-02", "2030-10-03", "2030-10-04", "2030-10-05", "2030-10-06", "2030-10-07", "2030-10-08", "2030-10-09", "2030-10-10", "2030-10-11", "2030-10-12", "2030-10-13", "2030-10-14", "2030-10-15", "2030-10-16", "2030-10-17", "2030-10-18", "2030-10-19", "2030-10-20", "2030-10-21", "2030-10-22", "2030-10-23", "2030-10-24", "2030-10-25", "2030-10-26", "2030-10-27", "2030-10-28", "2030-10-29", "2030-10-30", "2030-10-31", "2030-11-01", "2030-11-02", "2030-11-03", "2030-11-04", "2030-11-05", "2030-11-06", "2030-11-07", "2030-11-08", "2030-11-09", "2030-11-10", "2030-11-11", "2030-11-12", "2030-11-13", "2030-11-14", "2030-11-15", "2030-11-16", "2030-11-17", "2030-11-18", "2030-11-19", "2030-11-20", "2030-11-21", "2030-11-22", "2030-11-23", "2030-11-24", "2030-11-25", "2030-11-26", "2030-11-27", "2030-11-28", "2030-11-29", "2030-11-30", "2030-12-01", "2030-12-02", "2030-12-03", "2030-12-04", "2030-12-05", "2030-12-06", "2030-12-07", "2030-12-08", "2030-12-09", "2030-12-10", "2030-12-11", "2030-12-12", "2030-12-13", "2030-12-14", "2030-12-15", "2030-12-16", "2030-12-17", "2030-12-18", "2030-12-19", "2030-12-20", "2030-12-21", "2030-12-22", "2030-12-23", "2030-12-24", "2030-12-25", "2030-12-26", "2030-12-27", "2030-12-28", "2030-12-29", "2030-12-30", "2030-12-31", "2031-01-01", "2031-01-02", "2031-01-03", "2031-01-04", "2031-01-05", "2031-01-06", "2031-01-07", "2031-01-08", "2031-01-09", "2031-01-10", "2031-01-11", "2031-01-12", "2031-01-13", "2031-01-14", "2031-01-15", "2031-01-16", "2031-01-17", "2031-01-18", "2031-01-19", "2031-01-20", "2031-01-21", "2031-01-22", "2031-01-23", "2031-01-24", "2031-01-25", "2031-01-26", "2031-01-27", "2031-01-28", "2031-01-29", "2031-01-30", "2031-01-31", "2031-02-01", "2031-02-02", "2031-02-03", "2031-02-04", "2031-02-05", "2031-02-06", "2031-02-07", "2031-02-08", "2031-02-09", "2031-02-10", "2031-02-11", "2031-02-12", "2031-02-13", "2031-02-14", "2031-02-15", "2031-02-16", "2031-02-17", "2031-02-18", "2031-02-19", "2031-02-20", "2031-02-21", "2031-02-22", "2031-02-23", "2031-02-24", "2031-02-25", "2031-02-26", "2031-02-27", "2031-02-28", "2031-03-01", "2031-03-02", "2031-03-03", "2031-03-04", "2031-03-05", "2031-03-06", "2031-03-07", "2031-03-08", "2031-03-09", "2031-03-10", "2031-03-11", "2031-03-12", "2031-03-13", "2031-03-14", "2031-03-15", "2031-03-16", "2031-03-17", "2031-03-18", "2031-03-19", "2031-03-20", "2031-03-21", "2031-03-22", "2031-03-23", "2031-03-24", "2031-03-25", "2031-03-26", "2031-03-27", "2031-03-28", "2031-03-29", "2031-03-30", "2031-03-31", "2031-04-01", "2031-04-02", "2031-04-03", "2031-04-04", "2031-04-05", "2031-04-06", "2031-04-07", "2031-04-08", "2031-04-09", "2031-04-10", "2031-04-11", "2031-04-12", "2031-04-13", "2031-04-14", "2031-04-15", "2031-04-16", "2031-04-17", "2031-04-18", "2031-04-19", "2031-04-20", "2031-04-21", "2031-04-22", "2031-04-23", "2031-04-24", "2031-04-25", "2031-04-26", "2031-04-27", "2031-04-28", "2031-04-29", "2031-04-30", "2031-05-01", "2031-05-02", "2031-05-03", "2031-05-04", "2031-05-05", "2031-05-06", "2031-05-07", "2031-05-08", "2031-05-09", "2031-05-10", "2031-05-11", "2031-05-12", "2031-05-13", "2031-05-14", "2031-05-15", "2031-05-16", "2031-05-17", "2031-05-18", "2031-05-19", "2031-05-20", "2031-05-21", "2031-05-22", "2031-05-23", "2031-05-24", "2031-05-25", "2031-05-26", "2031-05-27", "2031-05-28", "2031-05-29", "2031-05-30", "2031-05-31", "2031-06-01", "2031-06-02", "2031-06-03", "2031-06-04", "2031-06-05", "2031-06-06", "2031-06-07", "2031-06-08", "2031-06-09", "2031-06-10", "2031-06-11", "2031-06-12", "2031-06-13", "2031-06-14", "2031-06-15", "2031-06-16", "2031-06-17", "2031-06-18", "2031-06-19", "2031-06-20", "2031-06-21", "2031-06-22", "2031-06-23", "2031-06-24", "2031-06-25", "2031-06-26", "2031-06-27", "2031-06-28", "2031-06-29", "2031-06-30", "2031-07-01", "2031-07-02", "2031-07-03", "2031-07-04", "2031-07-05", "2031-07-06", "2031-07-07", "2031-07-08", "2031-07-09", "2031-07-10", "2031-07-11", "2031-07-12", "2031-07-13", "2031-07-14", "2031-07-15", "2031-07-16", "2031-07-17", "2031-07-18", "2031-07-19", "2031-07-20", "2031-07-21", "2031-07-22", "2031-07-23", "2031-07-24", "2031-07-25", "2031-07-26", "2031-07-27", "2031-07-28", "2031-07-29", "2031-07-30", "2031-07-31", "2031-08-01", "2031-08-02", "2031-08-03", "2031-08-04", "2031-08-05", "2031-08-06", "2031-08-07", "2031-08-08", "2031-08-09", "2031-08-10", "2031-08-11", "2031-08-12", "2031-08-13", "2031-08-14", "2031-08-15", "2031-08-16", "2031-08-17", "2031-08-18", "2031-08-19", "2031-08-20", "2031-08-21", "2031-08-22", "2031-08-23", "2031-08-24", "2031-08-25", "2031-08-26", "2031-08-27", "2031-08-28", "2031-08-29", "2031-08-30", "2031-08-31", "2031-09-01", "2031-09-02", "2031-09-03", "2031-09-04", "2031-09-05", "2031-09-06", "2031-09-07", "2031-09-08", "2031-09-09", "2031-09-10", "2031-09-11", "2031-09-12", "2031-09-13", "2031-09-14", "2031-09-15", "2031-09-16", "2031-09-17", "2031-09-18", "2031-09-19", "2031-09-20", "2031-09-21", "2031-09-22", "2031-09-23", "2031-09-24", "2031-09-25", "2031-09-26", "2031-09-27", "2031-09-28", "2031-09-29", "2031-09-30", "2031-10-01", "2031-10-02", "2031-10-03", "2031-10-04", "2031-10-05", "2031-10-06", "2031-10-07", "2031-10-08", "2031-10-09", "2031-10-10", "2031-10-11", "2031-10-12", "2031-10-13", "2031-10-14", "2031-10-15", "2031-10-16", "2031-10-17", "2031-10-18", "2031-10-19", "2031-10-20", "2031-10-21", "2031-10-22", "2031-10-23", "2031-10-24", "2031-10-25", "2031-10-26", "2031-10-27", "2031-10-28", "2031-10-29", "2031-10-30", "2031-10-31", "2031-11-01", "2031-11-02", "2031-11-03", "2031-11-04", "2031-11-05", "2031-11-06", "2031-11-07", "2031-11-08", "2031-11-09", "2031-11-10", "2031-11-11", "2031-11-12", "2031-11-13", "2031-11-14", "2031-11-15", "2031-11-16", "2031-11-17", "2031-11-18", "2031-11-19", "2031-11-20", "2031-11-21", "2031-11-22", "2031-11-23", "2031-11-24", "2031-11-25", "2031-11-26", "2031-11-27", "2031-11-28", "2031-11-29", "2031-11-30", "2031-12-01", "2031-12-02", "2031-12-03", "2031-12-04", "2031-12-05", "2031-12-06", "2031-12-07", "2031-12-08", "2031-12-09", "2031-12-10", "2031-12-11", "2031-12-12", "2031-12-13", "2031-12-14", "2031-12-15", "2031-12-16", "2031-12-17", "2031-12-18", "2031-12-19", "2031-12-20", "2031-12-21", "2031-12-22", "2031-12-23", "2031-12-24", "2031-12-25", "2031-12-26", "2031-12-27", "2031-12-28", "2031-12-29", "2031-12-30", "2031-12-31", "2032-01-01", "2032-01-02", "2032-01-03", "2032-01-04", "2032-01-05", "2032-01-06", "2032-01-07", "2032-01-08", "2032-01-09", "2032-01-10", "2032-01-11", "2032-01-12", "2032-01-13", "2032-01-14", "2032-01-15", "2032-01-16", "2032-01-17", "2032-01-18", "2032-01-19", "2032-01-20", "2032-01-21", "2032-01-22", "2032-01-23", "2032-01-24", "2032-01-25", "2032-01-26", "2032-01-27", "2032-01-28", "2032-01-29", "2032-01-30", "2032-01-31", "2032-02-01", "2032-02-02", "2032-02-03", "2032-02-04", "2032-02-05", "2032-02-06", "2032-02-07", "2032-02-08", "2032-02-09", "2032-02-10", "2032-02-11", "2032-02-12", "2032-02-13", "2032-02-14", "2032-02-15", "2032-02-16", "2032-02-17"], "tickvals": [0, 42, 87, 130, 171, 214, 256, 298, 346, 390, 434, 481, 523, 564, 605, 647, 690, 731, 773, 813, 853, 897, 942, 985, 1026, 1068, 1113, 1155, 1195, 1235, 1277, 1321, 1362, 1404, 1446, 1487, 1528, 1569, 1613, 1654, 1695, 1736, 1776, 1816, 1857, 1897, 1938, 1978, 2019, 2059, 2099, 2139, 2179, 2219, 2259, 2299, 2339, 2380, 2420, 2461, 2501, 2542, 2582, 2622, 2662, 2702, 2742, 2782, 2822, 2862, 2902, 2943, 2983, 3023, 3063, 3103, 3143, 3183, 3223, 3263, 3303, 3343, 3383, 3423, 3463, 3503, 3543, 3583, 3623, 3663, 3703, 3743, 3783, 3823, 3863, 3903, 3943, 3983, 4023, 4063, 4103, 4143, 4183, 4223, 4263, 4303, 4343], "title": {"text": "day_offset"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "workload"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('acc44288-4978-40f6-8686-1f23e606d105');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{x.observe(notebookContainer, {childList: true})}};

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{ x.observe(outputEl, {childList: true});}}

                        })
                };
                });
            </script>
        </div>


## Linear Regression in detail:

lets see what happens if we just treat the workload to a linear regression, with a random split:



```python
#lets see what happens if we just treat the workload to a linear regression 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train, test = train_test_split(df_dl, test_size=.25)     

X_train = train.day_offset.values.reshape(-1,1)
y_train = train.workload.values.reshape(-1,1)
X_test = test.day_offset.values.reshape(-1,1)
y_test = test.workload.values.reshape(-1,1)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.coef_, model.intercept_)      #same slope calculated by px above
print(model.score(y_pred, y_test), mean_absolute_error(y_pred,y_test))  #clearly regression leaves something to be desired.
```

    [[0.09824007]] [-11.4045169]
    -2.1516388983501664 43.90552281279854


## Linear Model Coeffs and Error:

The sklearn linear model yields a slope that is the same as that calculated by plotly express above: **0.098** Note that the intercept is slightly shifted- but we are using a random sample here vs the enter dataframe above. We use both the model's *.score()* method and the *mean_absolute_error* package to estimate how well the linear model fits the data. The negative **R<sup>2</sup>** value of **-2.09** is telling us that our model does no better than a mean baseline. MAE is 42.26893511094306 


df_dl.workload.mean(), df_dl.workload.std()  = (207.0688439849624, 134.49243473471824)

### Will a tree model perform better?

Linear regression isnt really working well for this data. Shall we see if a tree model can better handle the non-monotonic change in workload? Using the same train / test split from above, we have to shape the arrays differently before feeding them to the tree ensemble. 


```python
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, IntRangeSlider
import ipywidgets as widgets


def f(md, xlims):
    model = RandomForestRegressor(max_depth=md)

    X_train = train.day_offset.values.reshape(-1, 1) 
    y_train = train.workload.values
    X_test = test.day_offset.values.reshape(-1, 1) 
    y_test = test.workload.values

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Mean Abs Err : ', mean_absolute_error(y_pred,y_test)) 
    
    fig,ax = plt.subplots()
    rfpred= pd.DataFrame(data=[y_test,y_pred]).T
    
    ax= rfpred[0].plot( figsize=(30,6), color='red')           # true values
    ax= rfpred[1].plot(                color='blue', alpha=.7) # predicted values
    ax.legend (['actual', 'predicted'])
    
    ax.set_xlim(xlims)
    plt.title('RF Model Predicted vs Actual \n Daily Workload: (Max Queue length)')
    plt.show()

xlimr= int(X_train[-1])

xlims =  IntRangeSlider(description='x window',
                        value=[0,xlimr],
                        min=   0,
                        max=   xlimr,
                        step = 50,
                        disabled=False,
                       readout=True)

md =  IntSlider(description='max_depth',
                min=1,
                max=10,
                step=1, 
                value=3)

interact(f, md=md,xlims=xlims)

```


    interactive(children=(IntSlider(value=3, description='max_depth', max=10, min=1), IntRangeSlider(value=(0, 360…





    <function __main__.f(md, xlims)>



## Got trees?  
   It is evident that the tree-based regressor has no problem fitting the wild variance of the daily queue length (workload) ##TODO##  change to queue length? Using the slider to redraw the plot with increasing depth and watch the MAE drop. But are we done? Can this model really predict future workload? Since we did **random** sampling of our data we would expect the fitted model to predict well on the test sets, if we have enough data points in each set. It
   
### Time-Based Split
   To test our Random Forest model further, we can create a time based split. The datetime index is sorted, we will re-train the model using the first 80% of the observations along the time axis, with the remaining 20% for test. Use the *training set%* slider to observe the effect on the prediction. If you think *that's bogus, use the dropdown*, and watch what happens when you change the split as follows: 
   out of the original train set, create a ***random*** 80% sample and use the remainder for validation. prepare to be amazed...
    


```python
#use a time based split 
from ipywidgets import FloatSlider

def f(md,ts,tvt_split='OFF'):
    ## time axis split
    trainsize = int(df_dl.shape[0]*ts)  #initial split of 80/20 ffrom the slider default
    
    model = RandomForestRegressor(max_depth=md)
    
    train = df_dl.copy().iloc[0:trainsize]
    test =  df_dl.copy().iloc[trainsize:] 
    
    if tvt_split == 'ON' :                                           #random sample the first portion into train,val then we can test 
        train,val = train_test_split(train, test_size=.2)     
        X_val = val.day_offset.values.reshape(-1, 1) 
        y_val = val.workload.values
    
    X_train = train.day_offset.values.reshape(-1, 1) 
    y_train = train.workload.values
    X_test = test.day_offset.values.reshape(-1, 1) 
    y_test = test.workload.values
    
    model.fit(X_train, y_train)
    
    fig,ax = plt.subplots()
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    if tvt_split == 'ON':             
        y_val_pred = model.predict(X_val)
        plt.figtext(.145,.3,
                    f"val mae:{mean_absolute_error(y_val_pred,y_val):.2f}", 
                    ha='left',fontsize=15,color='green')   
     
        val['Val pred'] = y_val_pred     #add the predictions for val to df
        
        val.sort_index(inplace=True)
        val.plot( y= 'workload',  figsize=(30,6), color='red',ax=ax) #val actual
        val.plot( y='Val pred',  figsize=(30,6), color='green', ax=ax)#val predicted 
        ax.legend (['val actual', 'val predicted'])

    y_pred = model.predict(X_test)   
    
    plt.figtext(.5,.9,'RF model predicted values vs. actual,\n as function of split and max_depth',fontsize=20,ha='center')

    plt.figtext(.75,.3,f"test mae: {mean_absolute_error(y_pred,y_test):.2f}", fontsize=15, color='blue')
    
    test['Test predicted']= y_pred
    
    test.plot(y='workload', figsize=(30,6), color='maroon', ax=ax) #actual workload
    test.plot(y='Test predicted', figsize=(30,6), color='blue', ax=ax)#predicted on TEST

ts =  FloatSlider(description='training set%',min=.50, max=.9, step=.05 , value=.8)    
md =  IntSlider(description='max_depth',min=1,max=10, step=1,  value=3)

tvt_split = widgets.ToggleButtons(
    options=['ON', 'OFF'],
    value='OFF',
    description='Make Train-Val \n on time split',
    disabled=False,
    button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
    orientation = 'vertical',
    tooltips=['On', 'Off'],
)
tvt_split = widgets.Dropdown(
    options=['OFF','ON'],
    value='OFF',
    description='3-way SPLIT',
    disabled=False,
    orientation='horizontal'
)
interact(f, md=md, ts=ts, tvt_split=tvt_split)


```


    interactive(children=(IntSlider(value=3, description='max_depth', max=10, min=1), FloatSlider(value=0.8, descr…





    <function __main__.f(md, ts, tvt_split='OFF')>



##  Wow this tree model is very smart!

This is what happens when you train your model on a time series with time based split! It appears to have determined that its not worth the effort and just guess a constant! but this 


### Where do we go from here? Lets do some window-looking. 

now that we can see the deficiencies of regression models, we need to introduce some means to better interpret the data. If there appears to be a periodic fluctuation, we can use a **rolling average**, or **moving mean** to smooth out short term variance and visualize larger trends. Below I introduce another calculated feature of the data, the *time to resolution* or *ttr* for a specific case. This is very simple; it is the time difference between the open and close for that case. In the data its expressed as a *timedelta* object, and cast to various datatypes, such as *




```python
closed_ttr_day = wrangler.working.copy()


closed_ttr_day['ttr_sec'] = closed_ttr_day['ttr']  #keep seconds
closed_ttr_day['ttr'] = closed_ttr_day['ttr'].dt.total_seconds()/3600   #convert to hours

closed_ttr_day['date_close'] = pd.DatetimeIndex(closed_ttr_day.Closed).date

closed_ttr_day = closed_ttr_day.groupby('date_close')[['CaseID','workload','ttr']].agg({'CaseID': 'count', 'workload': np.mean, 'ttr': 'mean'})
closed_ttr_day = closed_ttr_day.groupby('date_close')[['CaseID','workload','ttr']].agg({'CaseID': 'count', 'workload': 'max', 'ttr': 'mean'})

closed_ttr_day.columns=['case_count','daily_workload', 'avg_ttr_hours']


years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')


ax= closed_ttr_day.daily_workload.plot(figsize=(25,8))

closed_ttr_day.daily_workload.rolling(window=10).mean().plot(ax=ax)
closed_ttr_day.daily_workload.rolling(window=30).mean().plot(ax=ax)
closed_ttr_day.daily_workload.rolling(window=360).mean().plot(ax=ax)

closed_ttr_day.avg_ttr_hours.rolling(window=120).mean().plot(ax=ax)      #


ax.set_title('daily workload w /rolling avg windows \n time to resolution(ttr) superimposed')

ax.xaxis.grid(True, which='major')

ax.set_ylim(0,600)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_formatter(years_fmt)


ax.legend(['Daily Workload', 'm=10', 'm=30', 'm=360', 'ttr_hrs'])
plt.show()
```


![png](output_36_0.png)



```python
closed_ttr_day
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>case_count</th>
      <th>daily_workload</th>
      <th>avg_ttr_hours</th>
    </tr>
    <tr>
      <th>date_close</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2008-07-01</th>
      <td>1</td>
      <td>5.000000</td>
      <td>6.716389</td>
    </tr>
    <tr>
      <th>2008-07-02</th>
      <td>1</td>
      <td>0.000000</td>
      <td>21.144167</td>
    </tr>
    <tr>
      <th>2008-07-05</th>
      <td>1</td>
      <td>2.000000</td>
      <td>89.510000</td>
    </tr>
    <tr>
      <th>2008-07-07</th>
      <td>1</td>
      <td>5.000000</td>
      <td>91.886667</td>
    </tr>
    <tr>
      <th>2008-07-08</th>
      <td>1</td>
      <td>12.000000</td>
      <td>2.459167</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-06-02</th>
      <td>1</td>
      <td>323.237805</td>
      <td>24416.341335</td>
    </tr>
    <tr>
      <th>2020-06-03</th>
      <td>1</td>
      <td>419.301887</td>
      <td>10817.836038</td>
    </tr>
    <tr>
      <th>2020-06-04</th>
      <td>1</td>
      <td>342.220000</td>
      <td>5662.015733</td>
    </tr>
    <tr>
      <th>2020-06-05</th>
      <td>1</td>
      <td>238.100000</td>
      <td>14.266972</td>
    </tr>
    <tr>
      <th>2020-06-06</th>
      <td>1</td>
      <td>222.166667</td>
      <td>3.414722</td>
    </tr>
  </tbody>
</table>
<p>4128 rows × 3 columns</p>
</div>



## Rolling Means for Daily Workload

To see through some of the daily variance, and try to look for seasonal effects we can calculate the rolling means with various window sizes. after doing so we can see that there is a slight upward linear trend in the daily workload but with the extreme smoothing using a 360 day window, the red line, there also appears to be a multi-year cycle, perhaps releated to staffing and or process changes, that reduce the workload for a time before the upward trend continues. 

In contrast, the *mean time to resolution (ttr) in hours* shows extreme fluctuations, but in perhaps without a clear upward trend. We'll look at this more later


```python


ax= closed_ttr_day.daily_workload.plot(figsize=(18,5))
ax.set_title('daily workload & Variance over rolling w')

closed_ttr_day.daily_workload.rolling(window=10).var().plot(ax=ax)
closed_ttr_day.daily_workload.rolling(window=30).var().plot(ax=ax)
closed_ttr_day.daily_workload.rolling(window=50).var().plot(ax=ax)
closed_ttr_day.avg_ttr_hours.rolling(window=50).var().plot(ax=ax)

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')


ax.legend(['Daily Workload', 'm=10', 'm=20', 'm=30','mean_ttr variation'])
ax.xaxis.grid(True, which='major')

ax.set_ylim(0,1500)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_formatter(years_fmt)

plt.show()
```


![png](output_39_0.png)



```python
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf

x = closed_ttr_day.avg_ttr_hours
plot_acf(np.array(x),lags=10, title='mean time to resolution hrs, ACF')

#plot_acf(np.array(closed_ttr_day.case_count),lags=20, title='case counts')  #check case count (open/close count) autocorrelation

```




    date_close
    2008-07-01        6.716389
    2008-07-02       21.144167
    2008-07-05       89.510000
    2008-07-07       91.886667
    2008-07-08        2.459167
                      ...     
    2020-06-02    24416.341335
    2020-06-03    10817.836038
    2020-06-04     5662.015733
    2020-06-05       14.266972
    2020-06-06        3.414722
    Name: avg_ttr_hours, Length: 4128, dtype: float64




![png](output_40_1.png)



```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(daily_load.iloc[:,0].values)
print(f'p value is {result[1]}')

```

    p value is 0.4110634655953654


Since the p value is small than the significance level of 0.05, we can reject the null hypothesis that the time series data is non-stationary. Thus, the time series data is stationary.


```python


```


```python
result =closed_ttr_day.iloc[:,2].values    #column is mean time to resolution in hours

print(f'p value is {result[1]}')  # this series is NOT stationary
```

    p value is 21.144166666666667



```python
mean_ttr_by_cat = wrangler.working.groupby(['Category','Request Type']).ttr.apply(lambda x : np.mean(x))
#mean_ttr_by_cat.sort_values(ascending=False).head(5).index.tolist()
mean_ttr_by_cat.head(90)


```




    Category                    Request Type                                      
    311 External Request        Damaged_Property                                      22 days 05:49:23.230769
                                Garbage                                              152 days 17:17:50.750000
                                Graffiti                                              59 days 17:47:51.275590
                                Human/Animal Waste                                            0 days 09:56:30
                                Illegal_Posting                                        6 days 16:38:17.500000
                                Other                                                  3 days 03:11:18.111111
                                Sewer_Water_Storm_Conditions                                  0 days 01:07:34
                                Sidewalk_or_Curb_Issues                               82 days 23:08:25.888888
                                Utility Lines/Wires                                  176 days 15:22:11.750000
                                Value5                                                        0 days 00:36:39
    Abandoned Vehicle           Abandoned Vehicle -                                    5 days 06:25:14.260869
                                Abandoned Vehicle - Car2door                           5 days 01:01:20.125000
                                Abandoned Vehicle - Car4door                           5 days 01:44:22.452380
                                Abandoned Vehicle - DeliveryTruck                             4 days 22:35:36
                                Abandoned Vehicle - Motorcycle                         4 days 23:16:52.512820
                                Abandoned Vehicle - Other                              5 days 00:27:46.231707
                                Abandoned Vehicle - PickupTruck                        5 days 15:03:24.440217
                                Abandoned Vehicle - SUV                                4 days 23:49:26.029411
                                Abandoned Vehicle - Trailer                            8 days 11:14:50.588235
                                Abandoned Vehicles                                     4 days 10:35:59.627692
                                SSP Abandoned Vehicles                                 3 days 19:19:46.500000
    Blocked Street or SideWalk  Blocked_Parking_Space_or_Strip                        41 days 06:06:05.009615
                                Blocked_Sidewalk                                      63 days 17:31:38.783919
    Damaged Property            Damaged Benches_on_Sidewalk                          210 days 00:42:46.352941
                                Damaged Bike_Rack                                     14 days 16:20:10.142857
                                Damaged Fire_Police_Callbox                           43 days 14:03:40.916666
                                Damaged Kiosk_Public_Toilet                          210 days 10:47:42.500000
                                Damaged News_Rack                                     63 days 19:27:11.296296
                                Damaged Parking_Meter                                 29 days 18:02:51.742778
                                Damaged Traffic_Signal                               179 days 11:13:51.232142
                                Damaged Transit_Shelter_Ad_Kiosk                       0 days 21:10:05.416666
                                Damaged Transit_Shelter_Platform                      26 days 09:16:23.854166
                                Damaged Transit_Shelter_Platform_Hazardous            17 days 14:17:47.700000
                                Damaged other                                         64 days 04:30:36.750000
    Encampments                 Encampment Reports                                     9 days 07:59:55.380156
                                Encampment items                                       4 days 01:35:38.826855
    Graffiti                    Graffiti on ATT_Property                              17 days 05:22:59.238805
                                Graffiti on Bike_rack                                        82 days 05:59:04
                                Graffiti on Bridge                                   103 days 01:44:55.250000
                                Graffiti on Building_commercial                       38 days 00:40:36.026284
                                Graffiti on Building_other                           232 days 19:09:23.131931
                                Graffiti on Building_residential                      31 days 23:26:40.134328
                                Graffiti on City_receptacle                           14 days 05:03:10.331125
                                Graffiti on Fire_Police_Callbox                       69 days 02:15:04.329268
                                Graffiti on Fire_call_box                             23 days 02:15:08.125000
                                Graffiti on Fire_hydrant                              41 days 11:13:08.983333
                                Graffiti on Fire_hydrant_puc                                 30 days 03:11:00
                                Graffiti on Mail_box                                 176 days 17:07:30.952662
                                Graffiti on News_rack                                 52 days 22:34:16.893442
                                Graffiti on Other                                      6 days 04:14:41.105263
                                Graffiti on Other_enter_additional_details_below      39 days 00:02:41.282692
                                Graffiti on Other_for_Parks_ONLY                       5 days 05:39:50.333333
                                Graffiti on Parking_meter                             34 days 23:13:01.350000
                                Graffiti on Pay_phone                                315 days 18:22:16.800000
                                Graffiti on Pole                                      21 days 17:27:47.627577
                                Graffiti on Private Property                          44 days 02:36:16.181818
                                Graffiti on Public Property                                   0 days 00:33:23
                                Graffiti on Sidewalk_in_front_of_property             34 days 18:33:18.188953
                                Graffiti on Sidewalk_structure                        25 days 16:45:32.025380
                                Graffiti on Sign                                     115 days 21:22:17.610169
                                Graffiti on Signal_box                                14 days 21:53:19.628787
                                Graffiti on Street                                    39 days 09:00:18.452631
                                Graffiti on Transit_Shelter_Platform                  34 days 23:05:53.637168
    Homeless Concerns           Individual Concerns                                    5 days 02:22:19.553797
    Illegal Postings            Illegal Posting - Affixed_Improperly                  70 days 23:24:26.727272
                                Illegal Posting - Multiple_Postings                          16 days 19:52:09
                                Illegal Posting - No_Posting_Date                      4 days 15:13:13.393442
                                Illegal Posting - Posted_Over_70_Days                  9 days 02:07:40.333333
                                Illegal Posting - Posted_on_Directional_Sign                  0 days 16:51:51
                                Illegal Posting - Posted_on_Historic_Street_Light           -4 days +19:36:10
                                Illegal Posting - Posted_on_Traffic_Light              5 days 19:48:31.939393
                                Illegal Posting - Posting_Too_High_on_Pole                    1 days 01:03:45
                                Illegal Postings - Affixed_Improperly                 33 days 02:13:44.957055
                                Illegal Postings - Multiple_Postings                  27 days 13:24:58.057142
                                Illegal Postings - No_Posting_Date                     3 days 22:31:31.500000
                                Illegal Postings - Posted_Over_70_Days                11 days 14:26:31.033333
                                Illegal Postings - Posted_on_Directional_Sign          1 days 18:05:32.800000
                                Illegal Postings - Posted_on_Historic_Street_Light            2 days 03:12:32
                                Illegal Postings - Posted_on_Traffic_Light            52 days 03:37:27.021739
                                Illegal Postings - Posting_Too_High_on_Pole                  28 days 02:48:01
                                Illegal Postings - Posting_Too_Large_in_Size           2 days 09:30:54.250000
    Litter Receptacles          Add_remove_garbage_can                                77 days 01:59:57.339622
                                Cans_Left_Out_24x7                                    46 days 14:27:30.111111
                                City_Can_Other                                        13 days 06:55:17.062500
                                City_Can_Removed                                              8 days 07:27:49
                                Damaged_City_Can                                      16 days 04:57:57.614285
                                Debris_Box                                                  597 days 08:14:22
                                Debris_box_maintenance_overflowing                     3 days 19:52:44.500000
                                Door_Lock_issues                                       1 days 18:43:43.750000
                                Door_lock_issue                                       16 days 01:29:28.437500
    Name: ttr, dtype: timedelta64[ns]




```python

def f(stepin):
    fig, ax = plt.subplots()
    needle = '(needle)|(syringe)|(sharp)'

    sharps = wrangler.working['Request Details'].str.contains(needle, case=False)
    med_waste = wrangler.working['Request Type'] == 'Medical Waste'
     
    poop2 = wrangler.working['Request Details'].str.contains('Human', case=False)
    camp  = wrangler.working['Request Details'].str.contains('Encampment', case=False)
    wrangler.working[sharps]
    wrangler.working[med_waste]

    zlt = wrangler.working.Latitude == 0
    zln = wrangler.working.Longitude == 0
    zl =  (zln | zlt)

  

    sdic = {'poop': (poop2 & ~zl), 'needles': med_waste, 'people': camp}  #selector label: boolfilter

    #ax1 = wrangler.working[poop2 & ~zl].plot(x='Longitude',y='Latitude', color='red', kind ='scatter',figsize=(10,10), title = 'SF poopmap', alpha=.5)
    #ax2 = wrangler.working[med_waste & ~zl].plot(x='Longitude',y='Latitude', color='orange', kind ='scatter',figsize=(10,10), title = 'SF needles', alpha=.5)


    ax = wrangler.working[sdic[stepin]].plot(x='Longitude',y='Latitude', color='orange', kind ='scatter',figsize=(10,10), title = 'Watch your Step.. ', alpha=.5)
    fig.show()
    
soptions= ['poop','needles','people']
stepin = widgets.SelectionSlider(
    options=soptions,
    value='poop',
    description='I like to step on...',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True
)

```


```python
## view some neighborhood stats
#wrangler.working.groupby(by=['case_year','Neighborhood','Request Type'])[['CaseID','ttr']].agg({'CaseID': 'count','ttr': lambda x: x.mean()})
ttr_by_neighborhood  = wrangler.working.groupby(by=['Neighborhood','Request Details'])[['CaseID','ttr']].agg({'CaseID': 'count','ttr': lambda x: x.mean()})
poop2 = ttr_by_neighborhood.index.get_level_values('Request Details').str.contains('Human', case=False)
med_waste = ttr_by_neighborhood.index.get_level_values('Request Details') == 'Medical Waste'

ttr_by_neighborhood[poop2]
ttr_by_neighborhood[med_waste]

```

## Explore class Distribution for ttr:



```python


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f07c48a28d0>




![png](output_49_1.png)



```python
#explore ttr class distribution


(wrangler.working.ttr.dt.total_seconds()/3600).describe()
fig, (ax1,ax2) = plt.subplots(1,2,figsize =(20,10))
(wrangler.working.ttr.dt.total_seconds()/3600).plot(ax=ax1)


min_value =0
max_value =100
l_m_bound = 8
m_h_bound = 80
xb = 20000      #low pass filter  

ttr = wrangler.working.ttr.dt.total_seconds()/3600  #ttr in   hours

a = widgets.FloatRangeSlider(min=ttr.min(),
                             max=1000,
                             value=[l_m_bound,m_h_bound],
                             description='define the class boundaries',
                             readout_format='.1f',
                             readout=False
                            )
b = widgets.FloatLogSlider(         #adjust extreme value cutoff
    value=xb,
    base=10,
    min=-10, # max exponent of base
    max=10, # min exponent of base
    step=0.2, # exponent step
    description='extreme cutoff'
)
    

ui = widgets.HBox([a,b])

def f(a,b):
    print(a)
    lb,ub = a   #bounds for middle class
     xb=b
    def mapttrs(ttr):
        low  = ttr < lb   
        med  = lb <= ttr <= ub
        high = ub < ttr < xb
        ext  = ttr >= xb
        if low:
            return 'low'
        elif med:
            return 'med'
        elif high:
            return 'high'
        else:
            return 'xb'
    
    print(ttr.map(mapttrs).value_counts(normalize=True))
    print(f"\nHigh cutoff= {b}")
    print(f"bounds for med class: {a}")
    
    valct = (ttr.map(mapttrs).value_counts(normalize=True))
    xbval = (f"\nExtreme Values cutoff= {b}")
    medval= (f"bounds for med class: {a}")
    ax2.text(.5,.5,f"{valct}\n{xbval}\n{medval}") 
    fig.show()
out = widgets.interactive_output(f, {'a': a, 'b': b})
display(ui, out)

```


![png](output_50_0.png)



    HBox(children=(FloatRangeSlider(value=(8.0, 80.0), description='define the class boundaries', max=1000.0, min=…



    Output()



```python
def mapttrs(ttr):
    low  = ttr < 8   
    med  = 8 <= ttr <= 80
    high = 80 < ttr < 20000
    ext  = ttr >= 20000
    if low:
        return 'low'
    elif med:
        return 'med'
    elif high:
        return 'high'
    else:
        return 'xb'
df= wrangler.working.copy()
df['ttr_class'] = (df['ttr'].dt.total_seconds()/3600).map(mapttrs)
df['opened_hour'] = df.Opened.dt.hour
df['opened_dow'] = df.Opened.dt.dayofweek

```


```python

features = df.columns.drop(['CaseID','Opened','Closed','Updated','Status','Status Notes','ttr','workload','ttr_class']) #drop time or status related columns
df[features].dtypes
```




    Responsible Agency           object
    Category                     object
    Request Type                 object
    Request Details              object
    Address                      object
    Street                       object
    Neighborhood                 object
    Police District              object
    Latitude                    float64
    Longitude                   float64
    Source                       object
    Media URL                    object
    Current Police Districts    float64
    Analysis Neighborhoods        int64
    Neighborhoods                 int64
    case_year                     int64
    case_month                    int64
    opened_hour                   int64
    opened_dow                    int64
    dtype: object




```python
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, recall_score

target='ttr_class'

train,test = train_test_split(df, train_size=.8, random_state=99)
train,val  = train_test_split(train, train_size=.8, random_state=99)

print(train.shape, val.shape)

X_train = train[features]
y_train = train[target]
X_val   = val[features]
y_val   = val[target]

X_test = test[features]
y_test = test[target]

logreg = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    StandardScaler(),
    LogisticRegression(max_iter=100)
)
logreg.fit(X_train,y_train);
y_val_pred= logreg.predict(X_val)
y_pred_proba=logreg.predict_proba(X_val); #need for ROC
y_test_pred= logreg.predict(X_test)
accuracy_score(y_val,y_val_pred) ,accuracy_score(y_test,y_test_pred) 

```

    (19432, 28) (4859, 28)





    (0.48075735748096315, 0.46039848509797465)




```python
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
import eli5
from eli5.sklearn  import PermutationImportance

baseline= df[target].value_counts(normalize=True).values[0] #baseline
k=6
models = [XGBClassifier() , RandomForestClassifier(),DecisionTreeClassifier() ]
    
for model in models:
    pipe = pipe = make_pipeline(ce.OrdinalEncoder(),SimpleImputer(),StandardScaler(),SelectKBest(k=k),model)
    pipe.fit(X_train,y_train)
    print(list(pipe.named_steps.keys())[-1])
    print('Training Accuracy:', pipe.score(X_train, y_train))
    print('Validation Accuracy:', pipe.score(X_val, y_val))
    print('Test Accuracy:', pipe.score(X_test, y_test))
    print('baseline Accuracy:', baseline,'\n')
    print(classification_report(y_test, pipe.predict(X_test)))

```

    xgbclassifier
    Training Accuracy: 0.7445965417867435
    Validation Accuracy: 0.668656102078617
    Test Accuracy: 0.6678741972665898
    baseline Accuracy: 0.3372085364247135 
    
                  precision    recall  f1-score   support
    
            high       0.67      0.62      0.64      1914
             low       0.78      0.77      0.78      2034
             med       0.57      0.62      0.59      2042
              xb       0.59      0.35      0.44        83
    
        accuracy                           0.67      6073
       macro avg       0.65      0.59      0.61      6073
    weighted avg       0.67      0.67      0.67      6073
    
    randomforestclassifier
    Training Accuracy: 0.8557533964594484
    Validation Accuracy: 0.6332578719901214
    Test Accuracy: 0.6441626873044624
    baseline Accuracy: 0.3372085364247135 
    
                  precision    recall  f1-score   support
    
            high       0.62      0.63      0.62      1914
             low       0.75      0.77      0.76      2034
             med       0.56      0.55      0.56      2042
              xb       0.60      0.30      0.40        83
    
        accuracy                           0.64      6073
       macro avg       0.63      0.56      0.58      6073
    weighted avg       0.64      0.64      0.64      6073
    
    decisiontreeclassifier
    Training Accuracy: 0.8557533964594484
    Validation Accuracy: 0.5999176785346779
    Test Accuracy: 0.6136999835336736
    baseline Accuracy: 0.3372085364247135 
    
                  precision    recall  f1-score   support
    
            high       0.58      0.60      0.59      1914
             low       0.71      0.76      0.74      2034
             med       0.54      0.49      0.51      2042
              xb       0.43      0.34      0.38        83
    
        accuracy                           0.61      6073
       macro avg       0.57      0.55      0.55      6073
    weighted avg       0.61      0.61      0.61      6073
    



```python

#perm = PermutationImportance(model, random_state=887).fit(X_val_enc,y_val)
#eli5.show_weights(perm, feature_names = X_val_enc.columns.tolist())

pipe.named_steps['ordinalencoder'].get_feature_names()

```




    ['Responsible Agency',
     'Category',
     'Request Type',
     'Request Details',
     'Address',
     'Street',
     'Neighborhood',
     'Police District',
     'Latitude',
     'Longitude',
     'Source',
     'Media URL',
     'Current Police Districts',
     'Analysis Neighborhoods',
     'Neighborhoods',
     'case_year',
     'case_month',
     'opened_hour',
     'opened_dow']




```python
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


y_val_pred = pipe.predict(X_val)

confusion = confusion_matrix(y_val, y_val_pred,  labels=None, sample_weight=None, normalize=None)
print(list(pipe.named_steps.keys())[-1])
print(confusion)

unique_labels(y_test)

def plot_confusion_matrix(y_true, y_pred):
    labels = unique_labels(y_true)
    columns = [f'Predicted {label}' for label in labels]
    index = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred), 
                         columns=columns, index=index)
    return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')

plot_confusion_matrix(y_val, y_val_pred);



```

    decisiontreeclassifier
    [[ 867  170  441   21]
     [ 148 1218  276    6]
     [ 491  335  810    9]
     [  27   10   10   20]]



![png](output_56_1.png)

