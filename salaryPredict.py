# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:12:43 2019

@author: ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#%% Read DataSet Section

numberOfInstance = 199523
df = pd.read_csv('census-income.csv',names=["age", "class_of_worker", "industry_code", "occupation_code",
                                            "education","wage_per_hour","enrolled_in_edu_inst_last wk",
                                            "marital_status","major_industry_code","major_occupation code","mace","hispanic_Origin"
                                            ,"sex","member_of_a_labor_union","reason_for_unemployment","full_or_part_time_employment_stat",
                                            "capital gains","capital_losses","divdends_from_stocks","federal_income_tax_liability","tax_filer_status",
                                            "region_of_previous_residence","detailed_household_and_family_stat",
                                            "detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa",
                                            "migration_code_change_in_reg","migration_code-move_within_reg","live_in_this_house_1_year_ago",
                                            "migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18",
                                            "country_of_birth_father","country_of_birth_mother","country_of_birth_self",
                                            "citizenship","total_person_income","own_business_or_self_employed","taxable_income_amount",
                                            "fill_inc_questionnaire_for_veterans_admin","year","salary_temp"],nrows=numberOfInstance)
#%% Identify Data Section
df.info()
uniqueCounts = df.nunique()
print("Number of uniques in Data")
print(uniqueCounts)
#%% Deleting Irrelevant Columns Section
#df=df.drop('taxable_income_amount',1)                                          #
#df=df.drop('fill_inc_questionnaire_for_veterans_admin',1)                      #I don't know what they are...                                            #
#%% Fixing Columns Shit/Missing Values Section

#Trim all spaces from the data due to all of cateforical values starting with it 
df_obj = df.select_dtypes(['object'])
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

#Deleting '?' values from data in that case When one row has '?' most of the columns are non-deterministic
df = df[df.migration_code_change_in_msa != '?']

#if +50k salary 1, if -50k salary 0
df.loc[df.salary_temp=="- 50000.",'salary'] = '0'
df.loc[df.salary_temp=="50000+.",'salary'] = '1'
df=df.drop('salary_temp',1)

  
df_yeni = df[df.age >= 22]

plt.plot(df_yeni["age"], df_yeni.index, 'ro')
plt.axis([-10, 100, 0, 10000])
plt.show()

#%% Encoding non numerical columns & PCA Section

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
le = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
pca = PCA(n_components=3,whiten=True)