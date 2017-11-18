
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import random,math
mydata = pd.read_csv("/Users/lizhengxing/Desktop/data.csv", header = None,names =['Date','Num_of_bike_trip','rain','snow','temp_max','temp_min','temp_avg','wind_speed_avg','holiday'])

num_bike = mydata['Num_of_bike_trip'].values

def calculate_avg(data):
    total=0;
    count=0;
    for i in range(len(data)):
        if data[i] != 0:
            total =total+data[i]
            count = count+1
    return int(total/count);

def replace_zero(data):
    avg =calculate_avg(data)
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = avg
    return data;

def quantile(x, p):
	p_index = int(p * len(x))
	return sorted(x)[p_index]

def interquartile_range(x):
 	return quantile(x, 0.75) - quantile(x, 0.25)
#
# def replace_nan(data):
#     for i in range(len(data)):
#         if data[i] is None or data[i] == "":
#             data[i] = 0
#     return data;

num_bike_no_zero = replace_zero(num_bike)
avg_temp = mydata['temp_avg'].values
rain = mydata['rain'].values
snow = mydata['snow'].values
wind_speed_avg = mydata['wind_speed_avg'].values
holiday = mydata['holiday'].values


# rain_no_nan = replace_nan(rain)
# snow_no_nan = replace_nan(snow)
# print(num_bike_no_zero)
# print(len(avg_temp))
# print(len(wind_speed_avg))
# print(len(holiday))

# x = [[1,avg_temp[i],wind_speed_avg[i]] for i in range(len(num_bike_no_zero))]
# x = [i for i in avg_temp if i<quantile(avg_temp,0.75)+1.5*interquartile_range(avg_temp) and i>quantile(avg_temp,0.25)-1.5*interquartile_range(avg_temp)]
# y = [i for i in num_bike_no_zero if i<quantile(num_bike_no_zero,0.75)+1.5*interquartile_range(num_bike_no_zero) or i>quantile(num_bike_no_zero,0.25)-1.5*interquartile_range(num_bike_no_zero)]
#
# x_outliers = [avg_temp.index(i) for i in avg_temp if i>quantile(avg_temp,0.75)+1.5*interquartile_range(avg_temp) or i<quantile(avg_temp,0.25)-1.5*interquartile_range(avg_temp)]

d = [[j,k] for j,k in zip(avg_temp,num_bike_no_zero) if
j<quantile(avg_temp,0.75)+1.5*interquartile_range(avg_temp) and
j>quantile(avg_temp,0.25)-1.5*interquartile_range(avg_temp) and
k<quantile(num_bike_no_zero,0.75)+1.5*interquartile_range(num_bike_no_zero) and
 k>quantile(num_bike_no_zero,0.25)-1.5*interquartile_range(num_bike_no_zero)]

x = [d[i][0] for i in range(len(d))]
y = [d[i][1] for i in range(len(d))]

print(d,len(d))
def mean(x):
  return sum(x)/len(x)
#
# def de_mean(x):
# 	x_bar=mean(x)
# 	return[x_i-x_bar for x_i in x]
#
# def sum_of_squares(x):
# 	return sum([x_i**2 for x_i in x])
#
# def variance(x):
# 	n=len(x)
# 	deviations=de_mean(x)
# 	return sum_of_squares(deviations)/(n-1)
#
# def standard_deviation(x):
# 	return math.sqrt(variance(x))
#
# def covariance(x,y):
# 	n=len(x)
# 	return dot(de_mean(x),de_mean(y))/(n-1)
#
# def correlation(x,y):
# 	stdev_x=standard_deviation(x)
# 	stdev_y=standard_deviation(y)
# 	if stdev_x>0 and stdev_y>0:
# 		return covariance(x,y)/stdev_x/stdev_y
# 	else:
# 		return 0
#
# def rsq(r):
# 	return r**2
#
# def partial_difference_quotient(f,v,i,x,y,h):
#     w = [v_j +(h if j == i else 0) for j,v_j in enumerate(v)]
#     return ( f(w,x,y) - f(v,x,y) )/h
#
# def estimate_gradient_stochastic(f,v,x,y,h=0.01):
#     return [partial_difference_quotient(f,v,i,x,y,h) for i,_ in enumerate(v)]
#
# def step_stochastic(v, direction, step_size):
#     return [v_i-step_size * direction_i for v_i, direction_i in zip(v,direction)]
#
# def predict_y(v,x):
# 	return numpy.dot(v, x)
#     #return v[0]+v[1]*x
#     #return sum(v_i*x_i for v_i,x_i in zip(v,x))
#
# def error(v,x,y):
# 	error = y-predict_y(v,x)
# 	return error
#
# def squared_errors(v,x,y):
# 	return error(v,x,y)**2
#
# #pick a random starting point
# v = [random.randint(-10,10) for i in range (2)];
# print("Initial point: ",v)
#
# #Initialization
# step_0 = 0.001
# iterations_with_no_improvement = 0
# min_v = v;#or none
# min_value = float("inf")
#
# #main loop
# while iterations_with_no_improvement < 100:
#     value = sum(squared_errors(v, x_i, y_i) for x_i, y_i in zip(x,y))
#     if value < min_value: #found a new min
#         min_v, min_value = v,value
#         iterations_with_no_improvement = 0
#         step_size = step_0
#     else:
#         iterations_with_no_improvement += 1
#         step_size *= 0.9
#
#     print("min_v: ",min_v,"  sum_of_squared_errors: ",value);
#     indexes = numpy.random.permutation(len(x))
#     for i in indexes:
#         x_i = x[i]
#         y_i = y[i]
#         gradient_i=estimate_gradient_stochastic(squared_errors,v,x_i, y_i)
#         v = step_stochastic(v, gradient_i, step_size)
#
# print("\nMinimum occurs at ",min_v,".\nIntercept: ",min_v[0]," Slope: ",min_v[1],"\nLeast Regression Line: y = ",min_v[0],"+",min_v[1],"x");
#
# print("R^2: ",multiple_r_squared(min_v,x,y))
def b1(x,y):
    x_mean = mean(x)
    y_mean = mean(y)
    n =len(x)
    num = sum([j*k for j,k in zip(x,y)])-n*x_mean*y_mean
    den = sum([i*i for i in x])-n*x_mean*x_mean

    #return (standard_deviation(y)/standard_deviation(x))*pearson_correlation(x,y)
    return num/den

def b0(x,y):
	return mean(y) - b1(x,y)*mean(x)

print("Intercept: ",b0(x,y))
print("Slope: ",b1(x,y))
