#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[ ]:


#classes
1.standing
2.sitting
3.walking
4.sleeping
5.running


# In[77]:


standing=pd.read_csv('3.csv')
standing=pd.concat([standing,pd.read_csv('4.csv'),pd.read_csv('51.csv'),pd.read_csv('81.csv'),pd.read_csv('3(2).csv'),pd.read_csv('4(2).csv'),pd.read_csv('4(1).csv'),pd.read_csv('5(1).csv')])
standing['code']=0


# In[78]:


sitting=pd.read_csv('5.csv')
sitting=pd.concat([sitting,pd.read_csv('21.csv'),pd.read_csv('1(2).csv'),pd.read_csv('1(1).csv')])
sitting['code']=1


# In[79]:


walking=pd.read_csv('61.csv')
walking=pd.concat([walking,pd.read_csv('71.csv'),pd.read_csv('7.csv'),pd.read_csv('8.csv'),pd.read_csv('5(2).csv'),pd.read_csv('6(2).csv'),pd.read_csv('3(1).csv'),pd.read_csv('6(1).csv')])
walking['code']=2


# In[80]:


sleeping=pd.read_csv('31.csv')
sleeping=pd.concat([sleeping,pd.read_csv('1.csv'),pd.read_csv('2(2).csv'),pd.read_csv('2(1).csv')])
sleeping['code']=3


# In[81]:


running=pd.read_csv('41.csv')
running=pd.concat([running,pd.read_csv('2.csv'),pd.read_csv('7(2).csv')])
running['code']=4


# In[82]:


data=pd.DataFrame


# In[83]:


data=pd.concat([standing,sitting,walking,sleeping,running])


# In[84]:


data.reset_index(drop=True,inplace=True)


# In[85]:


data=data.reindex(np.random.permutation(data.index))


# In[86]:


data['total_acc']=(data['accX']**2)+(data['accY']**2)+(data['accZ']**2)


# In[87]:


data['total_acc']=data['total_acc']**1/2


# In[88]:


data=data.drop(['time'],axis=1)


# In[89]:


data.iloc[:,1:24]=data.iloc[:,1:24].astype(np.float32)


# In[90]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,1:24],data['code'],test_size=0.25,random_state=125)


# In[38]:





# In[91]:


feature_column=[tf.contrib.layers.real_valued_column(k) for k in data.columns[1:24]]


# In[92]:


my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_column,hidden_units=[48,96,48],n_classes=6,optimizer=my_optimizer)


# In[93]:


i=120
input_fn=tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=i,num_epochs=1000,shuffle=True)
eval_fn=tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=i,num_epochs=1000,shuffle=True)
classifier.fit(input_fn=input_fn,steps=1500)


# In[ ]:




