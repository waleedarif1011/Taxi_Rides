<h1>Taxi Rides data Analysis</h1> <br/>

# Introduction
In this project, I analyzed taxi ride data using NumPy, a powerful library for numerical computing in Python. The dataset contains information about various taxi rides, including the taxi ID, passenger count, and fare amount. By leveraging NumPy's efficient data structures and functions, I performed various analyses to gain insights into the taxi ride patterns.
# Data Preprocessing
Before conducting the analysis, I preprocessed the data to ensure its integrity and suitability for the intended analyses. This included:
Loading the data into a NumPy array using np.genfromtxt().
Converting data types to ensure compatibility with NumPy's operations.

## Data Analysis

## Average speed of Taxi's
To calculate the Average speed of Taxi's, I used the distance and time data

This provides a quick and efficient way to determine the average speed taxi rides in the dataset.

## Rides in Feburary 
To calculate the rides in specific month, I used the np.count_nonzero() function  and boolean array on month column

By selecting the second column (index 1) of the NumPy array, which represents the months, and applying the np.count_nonzero() function, I obtained the average fare per ride.

## Rides with tip more than $50
To find the Rides with tip more than $50, I used the tip count and used boolean array,

## Number of drops ar NYC Airport
Used drop data column and get id of NYC Airport and count them.

# Conclusion
In this project, I demonstrated how to use NumPy for analyzing taxi ride data efficiently. By leveraging NumPy's powerful functions and data structures, I was able to perform various analyses, such as calculating the Average speed of Taxi's, Rides in Feburary , Rides with tip more than $50, and Number of drops ar NYC Airport.
