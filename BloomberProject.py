#!/usr/bin/env python
# coding: utf-8

# # BloombergProject

# Introduction:
# 
# Bloomberg ranking the 5,000 largest firms based on their market capitalization at current prices. The dataset included indicators describing the firms and the monthly opening stock prices from December 2019 to November 2020. This specific timeframe was selected to analyze the impact of the COVID-19 pandemic on the value of these companies.
# 
# The upcoming project is designed for both investors and the general public, aiming to shed light on the companies and economic sectors that thrived or suffered during the pandemic.The primary motivation behind this analysis is to communicate which sectors emerged as winners and losers during this period. Additionaly the project seeks to determine if there is a correlation between market capitalization and a firm's size throughout the pandemic.

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[65]:


df= pd.read_csv('Bloomberg-dataset.csv')


# In[64]:


df.shape


# In[61]:


df.head()


# In[66]:


df.tail()


# In[34]:


df.info()


# In[71]:


df.describe()


# In[79]:


#identifying names and columns contains mixed-types
for col in df.columns.tolist():
    if not df[col].apply(type).eq(type(df[col].iloc[0])).all():
        print(col)


# In[53]:


#finding missing values 
df.isnull().sum()


# The column 'Best Analyst Rating' contains 113 missing values. I currently do not intend to work with this column so I leave it.

# In[80]:


len(df[df["Best Analyst Rating"] == 0])


# In[81]:


#finding duplicates
dupl= df[df.duplicated()]
dupl


#  No duplicates find

# In[86]:


df.columns


# In[90]:


# I decided to start by creating a variable I predict will show some correlation with some of the variables in dataset.

# New variable: ratio between "the difference between 52Wk High price and 52Wk Low price" and "average_52wk_price"

df["ratio_yearvar_meanprice"] = (df["52Wk High"] - df["52Wk Low"])/(df[['2/12/2019','1/1/2020', '3/2/2020', '2/3/2020', 
        '1/4/2020', '1/5/2020', '1/6/2020', '1/7/2020', '3/8/2020', '1/9/2020', '1/10/2020','2/11/2020']].mean(axis=1))


# In[91]:


df["ratio_yearvar_meanprice"]


# In[92]:


df["ratio_yearvar_meanprice"].describe()


# In[97]:


# creating correlation matrix in pandas
cor_matrix = df.corr(numeric_only= True)
cor_matrix


# In[100]:


#creating correlation heatmap in seaborn
plt.figure(figsize=(20,15))
sns.heatmap(cor_matrix, annot=True)
plt.title('correlation Heatmap')
plt.show()


# The heatmap covering the entire dataframe contains redundant information. prices are likely to show correlation with other prices, which doesn't provide much valuable insight. The distribution of "1Y Tot Ret" and "Market Capitalisation" could provide insights into whether larger companies provided better returns over the past year. If larger companies tend to have higher returns, it might indicate that they were safer or more lucrative investments. 
# 
# To gain more detail, it would be helpful to focus on specific variables or pairs of interest.

# In[101]:


# creating a subset excluding share's prices
sub =  df[['Market Cap', 'Best Analyst Rating', '52Wk High', '52Wk Low', '1Y Tot Ret (%)' , 'ratio_yearvar_meanprice']]
sub


# In[99]:


#Creating a correlation matrix 
sub_corr= sub.corr()
sub_corr


# In[102]:


#creating correlation heatmap in seaborn
plt.figure(figsize=(5,5))
sns.heatmap(sub_corr, annot=True)
plt.title('correlation Heatmap')
plt.show()


# The only variables that show a moderate correlation are "1Y Tot Ret (%)" and "ratio_yearvar_meanprice".
# The hypothesis is: the companies that experienced the biggest drop in value because of the pandemic, are also likelier to have performed worse throughout the year.
# 
# Identifying and analyzing companies with exceptionally high market caps is essential. These companies, often referred to as 'blue-chip' companies. If their size correlates with performance, it can validate the belief in the stability and growth potential of such companies.
# 
# Let's look for any patterns in scatterplot.

# In[46]:


#Creating scatter plot for "Market cap" and "1Y Tot Ret (%)"
sns.lmplot(x='1Y Tot Ret (%)', y='Market Cap', data= df)
plt.show()


# There is no significant correlation between market capitalization and firm’s performance. Therefore, in this instance, size did not matter. still we need to dig dipper into data to compare categories with the variables distribution.

# In[103]:


# Creating a scatterplot for "1Y Tot Ret (%)" and "ratio_yearvar_meanprice" columns in seaborn

sns.lmplot(x = "1Y Tot Ret (%)", y = "ratio_yearvar_meanprice", data = df)


# This scatterplot  does not distinguish positive and negative values in "1Y Tot Ret (%)"
# It also contains a group of companies that are listed in the stockmarket for less than one year
# and that, for this reason, "1Y Tot Ret (%)" equals zero.

# In[104]:


# Creating a categorical variable that splits the "1Y Tot Ret (%)" into categories positive and negative values

df.loc[df["1Y Tot Ret (%)"] < 0, "performance category"] = "neg_performers"


# In[105]:


df.loc[df["1Y Tot Ret (%)"] > 0, "performance category"] = "pos_performers"


# In[106]:


df.loc[df["1Y Tot Ret (%)"] == 0, "performance category"] = "no_data"


# In[107]:


df["performance category"].value_counts(dropna = False)


# In[109]:


# Creating a dataframe that excludes companies that are listed in the stockmarket for less than one year and that, for this reason, "1Y Tot Ret (%)" equals zero. 

df_exc_no_data = df[df["1Y Tot Ret (%)"] != 0]


# In[110]:


df_exc_no_data.shape


# In[111]:


# Creating a scatterplot for "1Y Tot Ret (%) and "ratio_yearvar_meanprice" columns, distinguishing between negative and positive performers

sns.lmplot(x = "1Y Tot Ret (%)", y = "ratio_yearvar_meanprice", hue="performance category", data = df_exc_no_data , palette=["dodgerblue", "orange"])


# This scatterplot is better as it distinguishes between positive and negative performers.
# It also shows that the variance of positive performers is higher than the one from negative performers.
# The outliers in "1Y Tot Ret (%)" make it more difficult to see a clear pattern. I will exclude them in the next step.

# In[112]:


# defining ouliers

q1, q3= np.percentile(df["1Y Tot Ret (%)"],[25,75])
upper_bound = q3 +(1.5 * (q3-q1)) 
upper_bound


# In[113]:


# exclusing outliers

df_no_outliers = df_exc_no_data[df_exc_no_data["1Y Tot Ret (%)"] < 96.46]


# In[114]:


df_no_outliers.shape


# In[115]:


# recreating the scatterplot without outliers

sns.lmplot(x = "1Y Tot Ret (%)", y = "ratio_yearvar_meanprice", hue="performance category", data = df_no_outliers, palette=["dodgerblue", "orange"])


# By excluding the outliers, the scatterplot becomes clear.
# A relation can be observed in the group of companies with positive "1Y Tot Ret (%)". It suggests that companies that experienced higher drops in price tend to perform better throughout the year.
# But the opposite relation can be observed in the group of negative performers. Therefore, the correlation is not conclusive.
# From another angle, better understanding the outliers, i.e. the companies that increased their value disproportionally during the pandemic, is important and will be addressed as a research question.

# In[116]:


# selecting variables for pair plot. 

sub_2 = sub[['Market Cap', 'Best Analyst Rating', '52Wk High', '52Wk Low',
       '1Y Tot Ret (%)']]


# In[117]:


# Creating a pair plot 

g = sns.pairplot(sub_2)


# The distribution of "market capitalisation" and "1Y Tot Ret" could be an interesting relation to explore to see if most valuable companies performed in comparison with others.
# The distribution of "1Y Tot Ret" by sector and country is also going to be studied.

# In[118]:


# Using a histogram to visualize the distribution of Market capitalisation.

sns.displot(df['Market Cap'], bins = 100)


# In[119]:


df["Market Cap"].describe()


# In[120]:


df.loc[df['Market Cap'] < 4000000000, 'Value category'] = 'Low Value'


# In[121]:


df.loc[(df['Market Cap'] >= 4000000000) & (df['Market Cap'] < 15000000000), 'Value category'] = 'Middle Value'


# In[122]:


df.loc[df['Market Cap'] >= 15000000000, 'Value category'] = 'High Value'


# In[123]:


df['Value category'].value_counts(dropna = False)


# In[124]:


# Using a histogram to visualize the distribution of 1Y Tot Ret (%).

sns.displot(df["1Y Tot Ret (%)"], bins = 100)


# In[125]:


df.loc[df["1Y Tot Ret (%)"] < -20, "performance category_2"] = "high_negative_performance"


# In[126]:


df.loc[(df["1Y Tot Ret (%)"] >= -20) & (df["1Y Tot Ret (%)"] < 0), "performance category_2"] = "moderate_negative_performance"


# In[127]:


df.loc[(df["1Y Tot Ret (%)"] > 0) & (df["1Y Tot Ret (%)"] < 20), "performance category_2"] = "moderate_positive_performance"


# In[128]:


df.loc[(df["1Y Tot Ret (%)"] >= 20) & (df["1Y Tot Ret (%)"] < 100), "performance category_2"] = "high_positive_performance"


# In[129]:


df.loc[df["1Y Tot Ret (%)"] >= 100, "performance category_2"] = "Very_high_positive_performance"


# In[130]:


df.loc[df["1Y Tot Ret (%)"] == 0, "performance category_2"] = "no data"


# In[131]:


df["performance category_2"].value_counts(dropna = False)


# In[135]:


# Creating a dataframe that excludes companies with "1Y Tot Ret (%)" equal to zero. 

df_exc_no_data= df[(df["1Y Tot Ret (%)"] != 0)]


# In[136]:


df_exc_no_data.shape


# In[138]:


# Creating a categorical plot 

sns.set(style="ticks")
g = sns.catplot(x="Value category", y="1Y Tot Ret (%)", hue="performance category", data= df_exc_no_data)


# In[139]:


# Creating a dataframe that excludes companies with "1Y Tot Ret (%)" equal to zero and outliers. 

df_exc_no_data_outliers= df_exc_no_data2[(df_exc_no_data2["1Y Tot Ret (%)"] < 96.46)]


# In[140]:


df_exc_no_data_outliers.shape


# In[141]:


# reCreating a categorical plot 

sns.set(style="ticks")
g = sns.catplot(x="Value category", y="1Y Tot Ret (%)", hue="performance category_2", data= df_exc_no_data_outliers)


# The performance distribution of “1Y Tot Ret (%)” among the different value categories is relatively homogeneous, suggesting that market capitalization was not a determining factor for performance.
# Still, it is surprising to observe the amount of companies that performed "high" in the context of a global economic recession. Understand this paradox is especially important. This question will be addressed to the extend that is possible.
# Next, I will use this tool to study the distribution of "1Y Tot Ret" by sector.

# In[147]:


# Creating a categorical plot to study the distribution of "1Y Tot Ret" by sector.

sns.set(style="ticks")
g = sns.catplot(x="1Y Tot Ret (%)", y="Sector", data= df_exc_no_data_outliers, palette="muted")


# This categorical plot addresses an important question: How did different sectors performed during the COVID19 pandemic?
# Preliminary analysis suggests that some sectors performed worse overall: energy, utilities sectors performed worse than others.
# Still, some companies within these sectors performed very well. This raises new research questions.
# In sectors such as “financial and “consumer”, performance appears to depend more on other variables than the sector itself. In these sectors there are several cases of winners and losers due to the pandemic.
# The technology, industrial and basic material sectors tended to perform well. But again, there are exceptions that might prove interesting to research.
# The performance outliers are mostly in the consumer cyclical sector.
