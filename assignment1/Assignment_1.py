import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

class Boston:

    def __init__ (self):
        self.columns_1 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.boston = pd.read_csv('housing.csv', delim_whitespace=True, names=self.columns_1, header=None)
        print("Original Dataset:\n",self.boston)
        print("\n----------Linear Regression----------\n")
        print("Dataset:\n",self.boston.iloc[:,[12,13]])
    
    def correlation_matrix (self):
        plt.figure(figsize = (9, 9))
        sns.heatmap(self.boston.corr(), annot = True)
        plt.title("Correlation Matrix\n")
        plt.show()
    
    def set_columns_lr (self):
        self.x_1 = self.boston.iloc[:,12:13]
        self.y_1 = self.boston.iloc[:,-1:]
        self.x_1_train, self.x_1_test, self.y_1_train, self.y_1_test = train_test_split(self.x_1, self.y_1, test_size = 0.3, random_state = 20)
        return self.x_1_train, self.x_1_test, self.y_1_train, self.y_1_test
        
    def Linear_Regression (self):
        self.regressor = LinearRegression() 
        self.regressor.fit(self.x_1_train,self.y_1_train)
        self.y_1_pred = self.regressor.predict(self.x_1_test)
        return self.y_1_pred

    def scatter_plot_lr (self):
        plt.scatter(self.x_1_test,self.y_1_test, color = 'b')
        plt.xlabel("Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Linear Regression\n")
        plt.plot(self.x_1_test,self.y_1_pred, color = 'r')
        plt.show(block=True)
        
    def r2_score (self,test,pred):
        self.r2 = r2_score(test, pred)
        print("\nR2 Score: ", self.r2)

    def rmse_score (self,test,pred):
        self.rmse = sqrt(mean_squared_error(test, pred))
        print("\nRMSE: ", self.rmse)
        
    #POLYNOMIAL REGRESSION
    def set_columns_poly(self):
        self.x_2 = self.boston['MEDV']
        self.y_2 = self.boston['LSTAT']
        print("\n----------Polynomial Regression----------\n")
        print("Dataset:\n",self.boston.iloc[:,[12,13]])
    
    def split(self):
        self.x_2_train, self.x_2_test, self.y_2_train, self.y_2_test = train_test_split(self.x_2, self.y_2, random_state = 20)
        self.x_train_df = pd.DataFrame(self.x_2_train)
        self.x_test_df = pd.DataFrame(self.x_2_test)
        return  self.x_2_train, self.x_2_test, self.y_2_train, self.y_2_test


    def Polynomial_Regression_2 (self):
        self.poly = PolynomialFeatures(degree = 2)
        self.x_train_poly = self.poly.fit_transform(self.x_train_df)
        self.x_test_poly = self.poly.fit_transform(self.x_test_df)
        self.lr = LinearRegression()
        self.lr = self.lr.fit(self.x_train_poly,self.y_2_train)
        self.w = self.lr.coef_
        self.C = self.lr.intercept_
        self.X = np.arange(5,50,0.1)
        self.Y = self.C + self.w[1] * self.X + self.w[2] * self.X**2
        self.y_2_pred = self.lr.predict(self.x_test_poly)
        plt.scatter(self.x_2, self.y_2, color = 'b')
        plt.xlabel("MEDV")
        plt.ylabel("LSTAT")
        plt.title("Polynomial Regression Degree: 2\n")
        plt.plot(self.X, self.Y, color = 'r')
        plt.show()
        return self.y_2_pred

    def Polynomial_Regression_20 (self):

        self.poly = PolynomialFeatures(degree = 20)
        self.x_train_poly = self.poly.fit_transform(self.x_train_df)
        self.x_test_poly = self.poly.fit_transform(self.x_test_df)
        self.lr = LinearRegression()
        self.lr = self.lr.fit(self.x_train_poly,self.y_2_train)
        self.w = self.lr.coef_
        self.C = self.lr.intercept_
        self.X = np.arange(5,50,0.1)
        self.Y = self.C
        for i in range(1,21):
            self.Y += self.w[i]*self.X**i
        self.y_2_pred = self.lr.predict(self.x_test_poly)
        plt.scatter(self.x_2, self.y_2, color = 'b')
        plt.xlabel("MEDV")
        plt.ylabel("LSTAT")
        plt.title("Polynomial Regression Degree: 20\n")
        plt.plot(self.X, self.Y, color = 'r')
        plt.show()
        
    #MULTIPLE REGRESSION
    def set_columns_mul(self):
        self.boston_2 = self.boston.copy()
        self.x_3 = self.boston_2.iloc[:,[5,10,12]]
        self.y_3 = self.boston_2.iloc[:,-1:]
        print("\n----------Multiple Regression----------\n")
        print("Dataset:\n",self.boston_2.iloc[:,[5,10,12,13]])

    def Multiple_Regression (self):
        self.x_3_train, self.x_3_test, self.y_3_train, self.y_3_test = train_test_split(self.x_3, self.y_3, test_size = 0.3, random_state = 1)
        self.regressor_1 = LinearRegression()
        self.regressor_1.fit(self.x_3_train,self.y_3_train)
        self.y_3_pred = self.regressor_1.predict(self.x_3_test)
        return self.x_3_train, self.x_3_test, self.y_3_train, self.y_3_test, self.y_3_pred
        
    def Adjusted_r2 (self):
        self.r2 = r2_score(self.y_3_test, self.y_3_pred)
        self.numerator = (1-self.r2)*(len(self.x_3_test)-1)
        self.denominator = (len(self.x_3_test)-len(self.x_3_test.values[0])-1)
        self.adjusted_r2 = 1-(self.numerator/self.denominator)
        print("\nAdjusted R2 Score: ",self.adjusted_r2)
        

def test():
    sol = Boston()
    #POLYNOMIAL REGRESSION
    sol.correlation_matrix()
    x_1_train, x_1_test, y_1_train, y_1_test = sol.set_columns_lr()
    y_1_pred = sol.Linear_Regression()
    sol.scatter_plot_lr()
    sol.r2_score(y_1_test, y_1_pred)
    sol.rmse_score(y_1_test, y_1_pred)
    
    
    #POLYNOMIAL REGRESSION
    sol.set_columns_poly()
    x_2_train, x_2_test, y_2_train, y_2_test = sol.split()
    y_2_pred = sol.Polynomial_Regression_2()
    sol.Polynomial_Regression_20()
    sol.r2_score(y_2_pred,y_2_test)
    sol.rmse_score(y_2_pred,y_2_test)
    
    
    #MULTIPLE REGRESSION
    sol.set_columns_mul()
    x_3_train, x_3_test, y_3_train, y_3_test, y_3_pred = sol.Multiple_Regression()
    sol.r2_score(y_3_pred,y_3_test)
    sol.rmse_score(y_3_pred,y_3_test)
    sol.Adjusted_r2()
    
if __name__ == "__main__":
    test()
