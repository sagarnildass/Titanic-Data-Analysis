import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_titanic = pd.read_csv('titanic-data.csv')
data_titanic = data_titanic.drop(["Ticket"], axis = 1)
data_titanic.columns

data_titanic.info()
data_titanic.describe()

mendata = data_titanic[data_titanic.Sex == 'male']
womendata = data_titanic[data_titanic.Sex == 'female']

plt.plot()

print("Males: ")
print(mendata.count()['Sex'])
print("")
print("Females: ")
print(womendata.count()['Sex'])



#basic
sns.lmplot('Age','Survived',data = data_titanic)
sns.lmplot('Age','Survived',data = mendata)

#Both on same plot
sns.lmplot('Age','Survived',data = data_titanic, hue = 'Sex')
sns.lmplot('SibSp','Survived',data = data_titanic, hue = 'Pclass')

#Subplot based
sns.lmplot('Age','Survived',data = data_titanic, col = 'Sex')

#Residual plot
sns.residplot('Age','Survived',data = data_titanic, color = 'indianred')

#UNIVARIATE DATA

#Strip plot

sns.stripplot(y = 'SibSp',data = data_titanic)
sns.stripplot(x = 'Pclass', y = 'SibSp',data = data_titanic, size = 2, jitter = True)


#Swarm plots
sns.swarmplot(x = 'Pclass', y = 'SibSp',data = data_titanic, hue = 'Sex')

#Merging two plots(Run them together)

sns.violinplot(x = 'Pclass', y = 'SibSp', data = data_titanic, inner = None, color = 'lightgray')
sns.stripplot(x = 'Pclass', y = 'SibSp',data = data_titanic, size = 2, jitter = True)

#BIVARIATE DATA

#1. Joint Plots

sns.jointplot(x = 'Age', y = 'SibSp', data = data_titanic)
#With kde
sns.jointplot(x = 'Age', y = 'SibSp', data = data_titanic, kind = 'kde')
#With regression line
sns.jointplot(x = 'Age', y = 'Survived', data = data_titanic, kind = 'reg')

#2. Pair plots
#Age = data_titanic['Age']
#sns.pairplot(Age) #Not a good example. Don't RUN THIS!

#age_sex = data_titanic.iloc[:,4:6]
#sns.pairplot(age_sex)

#sns.heatmap(age_sex)

#############################################################################################

# We have 3 dfs now. data_titanic, mendata, womendata

#Gender distribution

gender = sns.factorplot('Sex', data=data_titanic, kind='count')
gender.set_ylabels("count of passengers")

#Distribution by age
#age_data = data_titanic['Age'].hist(bins = 80)
#plt.set_ylabel("Age of Passengers")

age_data = data_titanic['Age']
plt.hist(age_data.dropna(), bins = 80)
plt.xlabel("Age of Passengers")
plt.ylabel("Frequency")
plt.title("Passenger's Age Distribution", fontsize = 30, color = 'black')
plt.show()

#Distribution by class
count_first = data_titanic.groupby('Pclass')['PassengerId'].count()
print(count_first)
class_data = sns.factorplot('Pclass', data=data_titanic, kind='count')
class_data.despine(left=True)
class_data.set_ylabels("count of passengers")
class_data.set_xlabels("Passenger Class")

#Distribution by Embarked

embarked_data = sns.factorplot('Embarked', data = data_titanic, kind = 'count')
plt.ylabel("Passsenger Count")

#Passenger description based on age(Oldest is 80 and youngest is 0.42. Mean is 29.69
data_titanic['Age'].describe()

#Age distribution by class

age_class_data = sns.FacetGrid(data_titanic, hue = 'Pclass', aspect = 3, hue_order = [1,2,3])
age_class_data.map(sns.kdeplot,'Age', shade=True)
age_class_data.set(xlim=(0,80))
age_class_data.add_legend()

#Gender Distribution by class


gender_class = sns.factorplot('Pclass', data = data_titanic, hue = 'Sex',kind = 'count', legend = False)
plt.legend(loc = "upper left")
plt.xlabel('Passenger Class')
plt.ylabel('Count of Passengers')
plt.show()

#Embarkment dist by class
embarked_class = sns.factorplot('Embarked', data = data_titanic, hue = 'Pclass',kind = 'count')

#Subdividing passengers into male, female, child
def childOrAdult(passenger):
    age, sex = passenger

    if age < 15:
        return 'child'
    else:
        return sex

#Adding the new column
data_titanic['AdultChild'] = data_titanic[['Age', 'Sex']].apply(childOrAdult, axis=1)
data_titanic.head()

#Counting the numbers
data_titanic.groupby('AdultChild')['PassengerId'].count()

#Male, Female and Child distribution
child_male_female = sns.FacetGrid(data_titanic, hue='AdultChild', aspect=3)
child_male_female.map(sns.kdeplot,'Age', shade=True)
child_male_female.set(xlim=(0,80))
child_male_female.add_legend()

#Child adult dist by class

child_adult_class = sns.factorplot('Pclass',data = data_titanic, hue = 'AdultChild', kind = 'count', legend = False)
plt.legend(loc = "upper left")

#Now we will divide the data into alone or family


data_titanic['Alone_or_family'] = data_titanic.Parch + data_titanic.SibSp
data_titanic['Alone_or_family'].loc[data_titanic['Alone_or_family'] > 0] = 'Family'
data_titanic['Alone_or_family'].loc[data_titanic['Alone_or_family'] == 0] = 'Alone'
data_titanic.head()
print(data_titanic.groupby('Alone_or_family')['PassengerId'].count())

#Dist of Alone vs family

alone_or_family = sns.factorplot('Alone_or_family', data = data_titanic, kind = 'count')
plt.xlabel("Alone vs Family")


###################################################
#############Survival Calculation##################
###################################################

#Basic survival dist

survival_dist = sns.factorplot('Survived',data=data_titanic,kind = 'count')


#Survival by age

plt.figure(figsize=(15,8))
plt.hist([data_titanic.dropna()[data_titanic.dropna()['Survived']==1]['Age'], data_titanic.dropna()[data_titanic.dropna()['Survived']==0]['Age']], stacked=True,
         color = ['g','r'],bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
#Survival by gender

survival_by_gender = sns.factorplot('Sex','Survived',data=data_titanic,kind = 'bar')
plt.ylabel("Survival Rate")

#Survival by class

survival_by_class = sns.factorplot('Pclass','Survived',data = data_titanic, kind = 'bar')
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")

survival_by_class = sns.factorplot('Pclass','Survived',data = data_titanic, kind = 'point')
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")

# Survival by Adult or child
survival_by_age = sns.factorplot('AdultChild','Survived',data = data_titanic, kind = 'bar')
plt.xlabel("Person status")
plt.ylabel("Survival Rate")

#Survival by alone or family

survival_by_family = sns.factorplot('Alone_or_family','Survived', data = data_titanic, kind = 'bar')

#Correlation between age and survival rate
age_corr = sns.regplot('Age','Survived',data = data_titanic)
age_corr_men = sns.regplot('Age','Survived',data = mendata)
age_corr_women = sns.regplot('Age','Survived',data = womendata)

#Get the mean values
age_corr.get_lines()[0].get_xdata().mean()
age_corr.get_lines()[0].get_ydata().mean()

age_corr_men.get_lines()[0].get_ydata().mean()

age_corr_women.get_lines()[0].get_ydata().mean()

#Correlation between age and survival rate for different sex and class
sns.lmplot('Age','Survived',data = data_titanic, hue = 'Sex')

sns.lmplot('Age','Survived',data = data_titanic, hue = 'Pclass')

#For males
sns.lmplot('Age','Survived',data = mendata, hue = 'Pclass')
#For females
sns.lmplot('Age','Survived',data = womendata, hue = 'Pclass')


#Correlation between Passenger class and survival rate

sns.lmplot('Pclass','Survived',data = data_titanic)
#Males
sns.lmplot('Pclass','Survived',data = mendata)
#Females
sns.lmplot('Pclass','Survived',data = womendata)

#Correlation between passenger class and survival for diff person status and sex

sns.lmplot('Pclass','Survived',data = data_titanic,hue = 'AdultChild')
sns.factorplot("Pclass", "Survived", hue='AdultChild', data=data_titanic, kind='point')
#mendata = data_titanic[data_titanic.Sex == 'male'] # Rerunning because new columns are not there
#womendata = data_titanic[data_titanic.Sex == 'female']

#sns.lmplot('Pclass','Survived',data = mendata,hue = 'PersonStatus') This doesnt make sense

sns.lmplot('Pclass','Survived',data = data_titanic,hue = 'Sex')

class_sex = sns.factorplot("Pclass", "Survived",order=[1,2,3],data=data_titanic,hue='Sex', kind='bar',legend = False)
plt.legend(loc = 'upper right')

class_sex = sns.factorplot("Pclass", "Survived",order=[1,2,3],data=data_titanic,hue='Sex', kind='point', legend = False)
plt.legend(loc = 'upper right')

#Correlattion between passenger class and survival for children
child_data = data_titanic[data_titanic.AdultChild == 'child']
child_data.head()
sns.factorplot('Pclass','Survived',data = child_data, kind = 'bar')




# Age distribution by survival

age_dist = sns.boxplot(data=data_titanic, x='Survived', y='Age')
age_dist.set(title='Age Distribution by Survival', xlabel = 'Survival', ylabel = 'Age Distribution', xticklabels = ['Died', 'Survived'])


#Age by gender and survival

age_gender_dist = sns.boxplot(data=data_titanic.dropna(subset = ['Age']), x= 'Sex', y = 'Age', hue='Survived')
age_gender_dist.set(title='Survival Distribution by Age and Gender')

#Age by class and survival

age_class_dist = sns.boxplot(data = data_titanic.dropna(subset = ['Age']).sort_values('Pclass'), x='Pclass', y='Age', hue='Survived')
age_class_dist.set(title='Survival distribution by Age and Class', xlabel='Pclass')

def get_possible_titles():
    global data_titanic

    # we extract the title from each name
    data_titanic['Title'] = data_titanic['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    return data_titanic['Title'].unique()




def get_titles():
    global data_titanic

    # we extract the title from each name
    data_titanic['Title'] = data_titanic['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }

    # we map each title
    data_titanic['Title'] = data_titanic.Title.map(Title_Dictionary)


get_possible_titles()

data_titanic.Title.unique()

