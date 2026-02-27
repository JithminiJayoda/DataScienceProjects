# **Waze Project**
**Course 1 - Foundations of Data Science**

Welcome to the Waze Project!

Your Waze data analytics team is still in the early stages of their user churn project. You have received notice that your team has been given access to Waze's user data. To get clear insights, the user data must be inspected and prepared for the upcoming process of exploratory data analysis (EDA).

A Python notebook has been prepared to guide you through this project. Answer the questions and create an executive summary for the Waze data team.

# **Course 1 End-of-course project: Inspect and analyze data**

In this activity, you will examine data provided and prepare it for analysis. This activity will help ensure the information is,

1.   Ready to answer questions and yield insights

2.   Ready for visualizations

3.   Ready for future hypothesis testing and statistical methods
<br/>

**The purpose** of this project is to investigate and understand the data provided.

**The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings.
<br/>

*This activity has three parts:*

**Part 1:** Understand the situation
* How can you best prepare to understand and organize the provided information?

**Part 2:** Understand the data

* Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities

* Compile summary information about the data to inform next steps

**Part 3:** Understand the variables

* Use insights from your examination of the summary data to guide deeper investigation into variables


<br/>

Follow the instructions and answer the following questions to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.

Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.



# **Identify data types and compile summary information**


<img src="images/Pace.png" width="100" height="100" align=left>

# **PACE stages**

Throughout these project notebooks, you'll see references to the problem-solving framework, PACE. The following notebook components are labeled with the respective PACE stages: Plan, Analyze, Construct, and Execute.

<img src="images/Plan.png" width="100" height="100" align=left>


## **PACE: Plan**

Consider the questions in your PACE Strategy Document and those below to craft your response:

### **Task 1. Understand the situation**

*   How can you best prepare to understand and organize the provided driver data?


*Begin by exploring your dataset and consider reviewing the Data Dictionary.*

==> ENTER YOUR RESPONSE HERE

<img src="images/Analyze.png" width="100" height="100" align=left>

## **PACE: Analyze**

Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

### **Task 2a. Imports and data loading**

Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:

*   `import pandas as pd`

*   `import numpy as np`



```python
# Import packages for data manipulation
### YOUR CODE HERE ###

```

Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.

**Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.


```python
# Load dataset into dataframe
df = pd.read_csv('waze_dataset.csv')
```

### **Task 2b. Summary information**

View and inspect summary information about the dataframe by **coding the following:**

1.   df.head(10)
2.   df.info()

*Consider the following questions:*

1. When reviewing the `df.head()` output, are there any variables that have missing values?

2. When reviewing the `df.info()` output, what are the data types? How many rows and columns do you have?

3. Does the dataset have any missing values?


```python
### YOUR CODE HERE ###

```


```python
### YOUR CODE HERE ###

```

==> ENTER YOUR RESPONSES TO QUESTIONS 1-3 HERE

### **Task 2c. Null values and summary statistics**

Compare the summary statistics of the 700 rows that are missing labels with summary statistics of the rows that are not missing any values.

**Question:** Is there a discernible difference between the two populations?



```python
# Isolate rows with null values
### YOUR CODE HERE ###

# Display summary stats of rows with null values
### YOUR CODE HERE ###

```


```python
# Isolate rows without null values
### YOUR CODE HERE ###

# Display summary stats of rows without null values
### YOUR CODE HERE ###
```

==> ENTER YOUR RESPONSE HERE

### **Task 2d. Null values - device counts**

Next, check the two populations with respect to the `device` variable.

**Question:** How many iPhone users had null values and how many Android users had null values?


```python
# Get count of null values by device
### YOUR CODE HERE ###
```

==> ENTER YOUR RESPONSE HERE

Now, of the rows with null values, calculate the percentage with each device&mdash;Android and iPhone. You can do this directly with the [`value_counts()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) function.


```python
# Calculate % of iPhone nulls and Android nulls
### YOUR CODE HERE ###

```

How does this compare to the device ratio in the full dataset?


```python
# Calculate % of iPhone users and Android users in full dataset
### YOUR CODE HERE ###

```

The percentage of missing values by each device is consistent with their representation in the data overall.

There is nothing to suggest a non-random cause of the missing data.

Examine the counts and percentages of users who churned vs. those who were retained. How many of each group are represented in the data?


```python
# Calculate counts of churned vs. retained
### YOUR CODE HERE ###


```

This dataset contains 82% retained users and 18% churned users.

Next, compare the medians of each variable for churned and retained users. The reason for calculating the median and not the mean is that you don't want outliers to unduly affect the portrayal of a typical user. Notice, for example, that the maximum value in the `driven_km_drives` column is 21,183 km. That's more than half the circumference of the earth!


```python
# Calculate median values of all columns for churned and retained users
### YOUR CODE HERE ###

```

This offers an interesting snapshot of the two groups, churned vs. retained:

Users who churned averaged ~3 more drives in the last month than retained users, but retained users used the app on over twice as many days as churned users in the same time period.

The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median retained user.

It seems that churned users had more drives in fewer days, and their trips were farther and longer in duration. Perhaps this is suggestive of a user profile. Continue exploring!

Calculate the median kilometers per drive in the last month for both retained and churned users.

Begin by dividing the `driven_km_drives` column by the `drives` column. Then, group the results by churned/retained and calculate the median km/drive of each group.


```python
# Add a column to df called `km_per_drive`
### YOUR CODE HERE ###

# Group by `label`, calculate the median, and isolate for km per drive
### YOUR CODE HERE ###

```

The median retained user drove about one more kilometer per drive than the median churned user. How many kilometers per driving day was this?

To calculate this statistic, repeat the steps above using `driving_days` instead of `drives`.


```python
# Add a column to df called `km_per_driving_day`
### YOUR CODE HERE ###

# Group by `label`, calculate the median, and isolate for km per driving day
### YOUR CODE HERE ###
```

Now, calculate the median number of drives per driving day for each group.


```python
# Add a column to df called `drives_per_driving_day`
### YOUR CODE HERE ###

# Group by `label`, calculate the median, and isolate for drives per driving day
### YOUR CODE HERE ###
```

The median user who churned drove 698 kilometers each day they drove last month, which is almost ~240% the per-drive-day distance of retained users. The median churned user had a similarly disproporionate number of drives per drive day compared to retained users.

It is clear from these figures that, regardless of whether a user churned or not, the users represented in this data are serious drivers! It would probably be safe to assume that this data does not represent typical drivers at large. Perhaps the data&mdash;and in particular the sample of churned users&mdash;contains a high proportion of long-haul truckers.

In consideration of how much these users drive, it would be worthwhile to recommend to Waze that they gather more data on these super-drivers. It's possible that the reason for their driving so much is also the reason why the Waze app does not meet their specific set of needs, which may differ from the needs of a more typical driver, such as a commuter.

Finally, examine whether there is an imbalance in how many users churned by device type.

Begin by getting the overall counts of each device type for each group, churned and retained.


```python
# For each label, calculate the number of Android users and iPhone users
### YOUR CODE HERE ###

```

Now, within each group, churned and retained, calculate what percent was Android and what percent was iPhone.


```python
# For each label, calculate the percentage of Android users and iPhone users
### YOUR CODE HERE ###

```

The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and those ratios are both consistent with the ratio found in the overall dataset.

<img src="images/Construct.png" width="100" height="100" align=left>

## **PACE: Construct**

**Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.



<img src="images/Execute.png" width="100" height="100" align=left>

## **PACE: Execute**

Consider the questions in your PACE Strategy Document and those below to craft your response:

### **Task 3. Conclusion**

Recall that your supervisor, May Santer, asked you to share your findings with the data team in an executive summary. Consider the following questions as you prepare to write your summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.

**Questions:**

1. Did the data contain any missing values? How many, and which variables were affected? Was there a pattern to the missing data?

2. What is a benefit of using the median value of a sample instead of the mean?

3. Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?

4. What percentage of the users in the dataset were Android users and what percentage were iPhone users?

5. What were some distinguishing characteristics of users who churned vs. users who were retained?

6. Was there an appreciable difference in churn rate between iPhone users vs. Android users?





==> ENTER YOUR RESPONSES TO QUESTIONS 1-6 HERE

**Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
