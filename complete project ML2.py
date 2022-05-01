import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import openpyxl
sns.set()
from sklearn.linear_model import LinearRegression
## Create the file and the project:
filepath = "D:/MAIN/STUDY/The Data Science Course 2021 - All Resources/"
projectname = input("What is the Project Name? ")
#Read the Data
raw_data = pd.read_csv(filepath + "1.02.+Multiple+linear+regression.csv")
#descriptive statistics
data_description = raw_data.describe(include = "all")
data_description.to_csv(filepath + projectname + ".csv")
print(raw_data.columns)
#from the descriptive statistics you can define the unwanted variables
lst = []
n = int(input("No. of unwanted Variables : "))
if n != 0:
    for i in range(0, n):
        ele = input('Eliminate: ') #Write the column names
        lst.append(ele)
data = raw_data.drop(lst,axis=1)
#find variables with missing values:
MV = data.isnull().sum()
print(MV)
#create a new dataframe with no missing values
data_no_mv = data.dropna(axis=0)
SV = data_no_mv.isnull().sum()
print(SV)
print(data_no_mv.describe(include="all"))

lst_7 = []
n = int(input("No. of numerical Variables : "))
if n != 0:

    for i in range(0, n):
        ele3 = str(input('Numerical Variable: ')) #Write the column names
        lst_7.append(ele3)
        sns.distplot(data_no_mv[lst_7[i]])
        plt.show()

#lists containg the variables and the quantiles and values
lst1 = []
lst2 =[]
lst3 =[]
lst4 =[]
lst5 = []
Var = int(input("No. of variables to be processed: "))
if Var != 0:
    for i in range(0,Var):
        Vlst = input('Modify: ')
        lst1.append(Vlst)
        print("qH , qL , vH , vL")
        M = str(input("modify according to "))  # qH #qL #vH #vL
        if M == "qH":
            qH = input("quantile less than ")
            lst2.append(qH)
            qL = 0
            lst3.append(qL)
            vH = data[Vlst].max(axis=0)
            lst4.append(vH)
            vL = data[Vlst].min(axis=0)
            lst5.append(vL)
        elif M == "qL":
            qH = 1
            lst2.append(qH)
            qL = input("quantile more than ")
            lst3.append(qL)
            vH = data[Vlst].max(axis=0)
            lst4.append(vH)
            vL = data[Vlst].min(axis=0)
            lst5.append(vL)
        elif M == "vH":
            qH = 1
            lst2.append(qH)
            qL = 0
            lst3.append(qL)
            vH = input("Value Less than ")
            lst4.append(vH)
            vL = data[Vlst].min(axis=0)
            lst5.append(vL)
        elif M == "vL":
            qH = 1
            lst2.append(qH)
            qL = 0
            lst3.append(qL)
            vH = data[Vlst].max(axis=0)
            lst4.append(vH)
            vL = input("Value More than")
            lst5.append(vL)
        else: #you have to create all the lists above
            qH = 1
            lst2.append(qH)
            qL = 0
            lst3.append(qL)
            vH = data[Vlst].max(axis=0)
            lst4.append(vH)
            vL = data[Vlst].min(axis=0)
            lst5.append(vL)
print(lst, lst1, lst2, lst3, lst4, lst5)
print(data_no_mv.describe())

for i in range(0,Var):
    sns.distplot(data_no_mv[lst1[i]])
    plt.show()
    data_1 = data_no_mv[data_no_mv[lst1[i]] <= data_no_mv[lst1[i]].quantile(float(lst2[i]))]
    data_2 = data_1[data_1[lst1[i]] >= data_1[lst1[i]].quantile(float(lst3[i]))]
    data_3 = data_2[data_2[lst1[i]] <= float(lst4[i])]
    data_4 = data_3[data_3[lst1[i]] >= float(lst5[i])]
    data_no_mv = data_4
    sns.distplot(data_4[lst1[i]])
    plt.show()
data_4 = data_4.reset_index(drop=True)
print(data_4.describe())

#Checking for assumptions:
#checking for linearity:
dependent_variable = str(input("dependent Variable: "))
lst_6 = []
n = int(input("No. of Independent Variables : "))
if n != 0:

    for i in range(0, n):
        ele2 = str(input('Independent Variable: ')) #Write the column names
        lst_6.append(ele2)
print(lst_6)

for x in range(0,n):
    sns.scatterplot(data_4[lst_6[x]], data_4[dependent_variable], x=lst_6[x], y = dependent_variable)
    plt.show()
print("yes or no")
Question = input("Want to take the log? ")
if Question == "yes":
    log_Price = np.log(data_4[dependent_variable])
    data_4['log_' + dependent_variable] = log_Price
    for x in range(0,n):
        sns.scatterplot(data_4[lst_6[x]], data_4['log_' + dependent_variable], x=lst_6[x], y = dependent_variable)
        plt.show()
    data_4= data_4.drop([dependent_variable], axis=1)
    print(data_4.describe(include='all'))
print(data_4.describe(include='all'))



#check for multicolinearity:
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_4[lst_6]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
print(vif)
lst_8 = []
n = int(input("No. of multi-collinear variables: "))
if n != 0:

    for i in range(0, n):
        ele4 = str(input('Variable: ')) #Write the column names
        lst_8.append(ele4)
        data_no_multicollinearity = data_4.drop([ele4], axis=1)
else:
    data_no_multicollinearity = data_4

print(lst_8)
#add dummy variables:
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
print(data_with_dummies.columns.values)
#check for multicolinearity:
from statsmodels.stats.outliers_influence import variance_inflation_factor
target = input("What is the target: ")
TEST = data_with_dummies.drop([target], axis = 1)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(TEST.values, i) for i in range(TEST.shape[1])]
vif["features"] = TEST.columns
print(vif)
lst_9 = []
n = int(input("No. of multi-collinear variables: "))
if n != 0:
    for i in range(0, n):
        ele5 = str(input('Variable: ')) #Write the column names
        lst_8.append(ele5)
        data_no_multicollinearity = data_with_dummies.drop([ele5], axis=1)
else:
    data_no_multicollinearity = data_with_dummies

print(lst_9)

##Rearrange the dataframe by putting the dependent variable as the first column

# linear regression
target = input("What is the target: ")
targets = data_with_dummies[target] ##y
inputs = data_with_dummies.drop([target], axis = 1) ##x

#standardization by using the StandardScaler() from Sklearn
from sklearn.preprocessing import StandardScaler

#Create Empty standard scaler object
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
#data splitting
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(inputs_scaled,targets , test_size=0.2, random_state=365)
print(x_train, x_test, y_train , y_test)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)
plt.show()
#sns.distplot(y_train,y_hat)
#plt.title("residuals PDF", size=18)
R_squared = reg.score(x_train,y_train)
print("R_squared =" + str(R_squared))
print("your model explains " + str(R_squared) + " of the variability of th data ")
Intercept = reg.intercept_
print("intercept= " + str(Intercept))
Coeff = reg.coef_
reg_summery = pd.DataFrame(inputs.columns.values, columns= ['Features'])
reg_summery['weights'] = Coeff
print (reg_summery)
#testing
y_hat_test = reg.predict(x_test)
plt.scatter(y_test,y_hat_test, alpha= 0.4)
plt.xlabel('Targets (y_test)', size = 18)
plt.ylabel('predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
#How good is the model
df_pf = pd.DataFrame(np.exp(y_hat_test), columns= ['predictions'])
y_test = y_test.reset_index(drop=True)
df_pf['targets'] = np.exp(y_test)
df_pf['Residuals'] = df_pf['targets'] - df_pf['predictions']
df_pf['differences%'] = np.absolute(df_pf['Residuals']/df_pf['targets']*100)
print(df_pf.describe())
#Display the whole Differences
df_pf.sort_values(by=['differences%'])
print(df_pf)
#model eval with statmodels
y = targets
x1 = inputs
x= sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print (results.summary())
#prediction:
#Linear Regression:
#reg = LinearRegression()
#reg.fit(inputs,targets)
#new_data= pd.read_csv(filepath + "1.02.+Multiple+linear+regression_1.csv")
#new_data = new_data[inputs.columns.values]
#new_data = pd.DataFrame(data = [[225,2,0,1,0,0,0,0],[200,6,0,0,1,0,0,0]], columns=['Mileage', 'EngineV' , 'Brand_BMW', 'Brand_Mercedes-Benz' ,'Brand_Mitsubishi' ,'Brand_Renault' ,'Brand_Toyota' ,'Brand_Volkswagen'])
#predictions = np.exp(reg.predict(new_data))
#print(predictions)









