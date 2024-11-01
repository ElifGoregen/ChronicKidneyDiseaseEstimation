#Classification:ckd:Kronik Böbrek Hastalığı , notckd:Normal

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

#load dataset
df=pd.read_csv("kidney_disease.csv")
df.drop("id",axis=1,inplace=True)

df.columns = ["age","blood_pressure" ,"spesific_gravity","albumin","sugar",
              "red_blood_cell","pus_cell","pus_cell_clumbs",
              "bacteria","blood_glucose_random","blood_urea",
              "serum_creatinine","sodium","potassium","hemoglobin",
              "packed_cell_volume","white_blood_cell_count","red_blood_cell_count","hypertension",
              "diabetes_melitus","coronary_artery_disease",
              "appetite","peda_edema","aanemia","class"]        
df.info()
# 16  whit_blood_cell_count    295 non-null    object 
# 17  red_blood_cell_count     270 non-null 
describe = df.describe()
#"packed_cell_volume","whit_blood_cell_count","red_blood_cell_count" string değerler nümeriğe çevirelim.
df["packed_cell_volume"]=pd.to_numeric(df["packed_cell_volume"],errors="coerce")
df["white_blood_cell_count"]=pd.to_numeric(df["white_blood_cell_count"],errors="coerce")
df["red_blood_cell_count"]=pd.to_numeric(df["red_blood_cell_count"],errors="coerce")
#"100" --> 100 yapar ancak
#"asd" --> 100 yapamayız bunu düzeltmek için errorsu kullandık. Nan değer yapar.

df.info() #kontrol ettik float olmuş.
#EDA:KDE
#Veri setini nümerik ve kategorik feature olarak ikiye ayıralım.
cat_cols = [col for col in df.columns if df[col].dtype == "object"] #categoric veri
num_cols = [col for col in df.columns if df[col].dtype != "object"] #float,int ->numeric
#kategorik değerleri kullanarak kaç adet unique değer var?
 
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
#Problem :
"""
diabetes_melitus: ['yes' 'no' ' yes' '\tno' '\tyes' nan]
#\tyes =yes ,\tno = no olmalı.
coronary_artery_disease: ['no' 'yes' '\tno' nan]
appetite: ['good' 'poor' nan]
class: ['ckd' 'ckd\t' 'notckd']
"""
df["diabetes_melitus"].replace(to_replace = {'\tno':"no",'\tyes':"yes",'yes':"yes"},inplace =True)
df["coronary_artery_disease"].replace(to_replace = {'\tno':"no"},inplace =True)
df["class"].replace(to_replace = {'ckd\t':"ckd"},inplace =True)
#Mapping İşlemi

df["class"] = df["class"].map({"ckd":0 ,"notckd":1})

plt.figure()
plotnumber = 1

for col in num_cols:
    if plotnumber <=14:
        ax = plt.subplot(3,5,plotnumber)
        sns.distplot(df[col])
        plt.xlabel(col)
        
    plotnumber +=1
plt.tight_layout()
plt.show()
#Ax 3 satır ve 5 sütundan oluşsun.
#displot dağılım plotu
plt.figure()
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, linecolor="white", linewidths=2)
plt.show()
#class en yüksek korelasyon.

def kde(col):
    grid = sns.FacetGrid(df,hue="class",height =6,aspect =2)
    grid.map(sns.kdeplot,col)
    grid.add_legend()
    
kde("hemoglobin")
kde("white_blood_cell_count")
kde("packed_cell_volume")
kde("red_blood_cell_count")
kde("albumin")
kde("specific_gravity")


#Preprocessing:Missing value problem
df.isna().sum().sort_values(ascending=False)
"""
red_blood_cell             152
red_blood_cell_count       131
white_blood_cell_count     106
potassium                   88
sodium                      87
packed_cell_volume          71
pus_cell                    65
hemoglobin                  52
sugar                       49
spesific_gravity            47
albumin                     46
blood_glucose_random        44
blood_urea                  19
serum_creatinine            17
blood_pressure              12
age                          9
bacteria                     4
pus_cell_clumbs              4
hypertension                 2
diabetes_melitus             2
coronary_artery_disease      2
appetite                     1
peda_edema                   1
aanemia                      1
class                        0
"""
def solve_missingvalue_random_value(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(),feature] = random_sample
    
for col in num_cols:
    solve_missingvalue_random_value(col)
df[num_cols].isnull().sum()

#en çok hangi kategorik değerden var ? :
def solve_missingvalue_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

solve_missingvalue_random_value("red_blood_cell")
solve_missingvalue_random_value("pus_cell")


for col in cat_cols :
    solve_missingvalue_mode(col)
df[cat_cols].isnull().sum()
#Herhangi bir missing value kalmadı.

#Preprocessing:Feature encoding

for col in cat_cols:
    print(f"{col}:{df[col].nunique()}")
    
#kategorik değerleri nümerik değerlere çevirdik. 
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])
"""
for col in cat_cols:
    print(f"{col}:{df[col].nunique()}")
red_blood_cell:2
pus_cell:2
pus_cell_clumbs:2
bacteria:2
hypertension:2
diabetes_melitus:3
coronary_artery_disease:2
appetite:2
peda_edema:2
aanemia:2
class:2
"""
#Model training and testing (DT)
independent_col = [col for col in df.columns if col !="class"] #X değişkeni
dependent_col = "class" # Y değişkeni.

X = df[independent_col]
y = df[dependent_col]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)


dtc =DecisionTreeClassifier()
#dtc =DecisionTreeClassifier(max_depth=2) iki derinlikli ağaç
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, y_pred)
#0.95

cm = confusion_matrix(y_test,y_pred)

cr = classification_report(y_test, y_pred)

print("Confusion Matrix \n",cm)
print("Classification Report",cr)

"""
Confusion Matrix 
 [[71  5]
 [ 1 43]]
Classification Report               precision    recall  f1-score   support

           0       0.99      0.93      0.96        76
           1       0.90      0.98      0.93        44

    accuracy                           0.95       120
   macro avg       0.94      0.96      0.95       120
weighted avg       0.95      0.95      0.95       120
"""

#DT visualization - feature importance
class_names = ["ckd","notckd"]
plt.figure (figsize = (20,10))
plot_tree(dtc,feature_names=independent_col,filled =True,rounded = True,fontsize=8)
plt.show()
#Hemoglobin most important feature

feature_importance = pd.DataFrame({"Feature":independent_col,"Importance":dtc.feature_importances_})
#Önemsiz feature 0 değerleri var feature_importantsta
print("Most important feature",feature_importance.sort_values(by="Importance",ascending=False).iloc[0])

plt.figure()
sns.barplot (x ="Importance",y="Feature",data=feature_importance)
plt.title("Feature Importance")
plt.show()




