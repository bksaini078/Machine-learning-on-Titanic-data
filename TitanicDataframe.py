from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import pandas as pd

def main():
    spark= SparkSession.builder.appName("Titanic").getOrCreate()
    df= spark.read.format("csv").option("inferschema","true").option("header","true")\
        .load("/Users/teetu/Documents/Master_in_Web_science/Summer_2019/Big_Data/Assignments/TutorialsM/Titanic/titanic.csv")
    df.printSchema()
    #How many male and female survived
    df_Survived=df.select("sex","survived").groupBy("sex").agg(f.expr("count(survived) as survivedNum"))\
        .orderBy("survivedNum", ascending=False)
    df_Survived.show()
    # df_Survived.printSchema()
    # df_Survived = df_Survived.toPandas()
    # plt.scatter(df_Survived['sex'], df_Survived['survivedNum'])
    # plt.show()
    # df_Survived.plot(x="sex", y="survivedNum", kind="bar")
    # +------+-----------+
    # | sex | survivedNum |
    # +------+-----------+
    # | female | 233 |
    # | male | 109 |
    # +------+-----------+
    #dead as per sex
    df_dead = df.select("sex", "survived").filter(f.expr("survived==0")).groupBy("sex")\
        .agg(f.expr("count(survived) as deadNum")) \
        .orderBy("deadNum", ascending=False)
    df_dead.show()
    #How many survived as per pclass'
    df_Survived_Pclass= df.select("Pclass","survived").groupBy("Pclass").agg(f.count("survived").alias("survived"))\
        .orderBy("survived", ascending=False)
    df_Survived_Pclass.show()
    # +------+--------+
    # | Pclass | survived |
    # +------+--------+
    # | 1 | 136 |
    # | 3 | 119 |
    # | 2 | 87 |
    # +------+--------+
    #age group as per survived
    df_survived_age= df.select("Age","survived").groupBy("Age").agg(f.count("survived").alias("survivedNum"))\
        .orderBy("survivedNum", ascending= False).na.drop()
    df_survived_age.show()
    # df_survived_age = df_survived_age.toPandas()
    # df_survived_age.plot(x="Age", y="survivedNum", kind="bar")

    spark.stop()
if __name__ =="__main__":
    main()
