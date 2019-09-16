from pyspark.mllib import random
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import  RFormula
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#below mentioned code will prediction regarding survival
def main():

    spork = SparkSession.builder.appName("titanic").getOrCreate()

    #Gathering data
    df= spork.read.format("csv").option("inferschema","true").option("header","true").load("titanic.csv")
   # df.show()
    df.printSchema()
    df= df.na.drop("any") #has to that if any null value in row otherwise it will show error while feature engineering

    #feature Engineering
    #Change the formula and check the result
    supervised = RFormula(formula="Survived ~ Sex:Age + Pclass : Cabin + SibSp+Embarked ")
    fittedRF= supervised.fit(df)
    preparedDF= fittedRF.transform(df)
    preparedDF.show()
    #spliting data in train and validation data
    train, test = preparedDF.randomSplit([0.7,0.3])
    #classification
    #configure classifier
    lr= LogisticRegression(featuresCol="features",labelCol="label")
    #train classifier
    fittedLR=lr.fit(train)

    #check result
    result= fittedLR.transform(test)
    print("Coefficients:" + str(fittedLR.coefficients))
    result.show(100)
    truePositive = float(result.filter("prediction =1.0 and label =1.0").count())
    falsePositive = float(result.filter("prediction =1.0 and  label = 0.0").count())
    falseNegative = float(result.filter("prediction =0.0 and label = 1.0").count())
    trueNegative = float(result.filter("prediction=0.0 and label =0.0 ").count())
    print("True Positive :" + str(truePositive))
    print("True Negative :" + str(trueNegative))
    print("False Positive :" + str(falsePositive))
    print("False Negative :" + str(falseNegative))
    sensitivityOrRecall = truePositive / (truePositive + falseNegative)
    specificity = truePositive / (truePositive + falsePositive)
    precision = truePositive / (truePositive + falsePositive)
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    print("sensitivityOrRecall :" + str(sensitivityOrRecall))
    print("specificity :" + str(specificity))
    print("precision :" + str(precision))
    print("accuracy :" + str(accuracy))



    spork.stop()
if __name__ == '__main__':
    main()




