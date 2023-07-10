
**SECTION 1: Exploratory Data Analysis (EDA)**


<div align="justify">
Customer churn is defined as when customers or subscribers discontinue doing business
with a firm or service. Customers in the telecom industry can choose from a variety of service
providers and actively switch from one to the next.
</div>
<br>
<div align="justify">
Individualized customer retention is tough because most firms have a large number of
customers and can't afford to devote much time to each of them. The costs would be too great,
outweighing the additional revenue. However, if a corporation could forecast which customers
are likely to leave ahead of time, it could focus customer retention efforts only on these "high
risk" clients. The ultimate goal is to expand its coverage area and retrieve more customer’s
loyalty.
</div>
<br>

<div align="justify">
Customer churn is a critical metric because it is much less expensive to retain existing
customers than it is to acquire new customers. To reduce customer churn, we need to predict
which customers are at high risk of churn.
</div>
<br>

<div align="justify">
Before performing the unsupervised learning approaches, we had to know the data well
in order to label the observations correctly. as well as applying data visualization techniques to
observe breakpoints and helps us to internalize the data.
</div>
<br>

<div align="justify">
The dataset C-TelcoChurn consists of 7043 rows and 21. While each row represents a
customer, each column contains customer’s attributes.
</div>
<br>


The data set includes information about:

1. Customers who left within the last month: the column is called Churn
2. Services that each customer has signed up for: phone, multiple lines, internet, online
    security, online backup, device protection, tech support, and streaming TV and movies
3. Customer account information: how long they’ve been a customer, contract, payment
    method, paperless billing, monthly charges, and total charges
4. Demographic info about customers: gender, age range, and if they have partners and
    dependents
<br>

<div align="justify">

Some initial visualization of the dataset includes the proportions of churn in the dataset,
counts of customers by gender who churned and counts of customers by partner who churned.
They are visualized respectively in Figure 1, Figure 2 and Figure 3. Figure 1 illustrates that
there are more “No” churns in the dataset, Figure 2 shows that gender does not influence churn
in any way and an almost identical nature is visible in Figure 3 for Partners.

</div>
<br>

<div align="center">
  <img src="https://github.com/ahvshim/Customer-Segmentation/assets/126220185/dc32d589-9b4f-4b00-8f29-474d28f4db98">
</div>
<br>
<div align="center">
  <img src="https://github.com/ahvshim/Customer-Segmentation/assets/126220185/e1bbbe3a-b7a8-4b78-9e75-91862d7807f4">
</div>
<br>
<div align="center">
  <img src="https://github.com/ahvshim/Customer-Segmentation/assets/126220185/d6eac879-b060-4a02-99eb-603f54b47aa6">
</div>

<br>

**SECTION 2: Preparation of Dataset**

<div align="justify">
The process of making raw data ready for further processing and analysis is known as
data preparation. The gathering, cleaning, and labelling of raw data to prepare it for machine
learning (ML) algorithms, as well as data exploration and visualization, are crucial processes.
For this dataset analysis, we apply unsupervised machine learning approaches, which are one
of two methods available. Data is provided, which supports business executives in their next
decision. An analyst's analysis will be more accurate and useful if the data have been prepared
carefully and thoroughly.
</div>
<br>

<div align="justify">
The overall statistics of the continuous variables in the dataset are described in Figure
4 for the data processing process of this dataset.
</div>
<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 4 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/ba42024c-d810-4cb3-8ac5-84c0b7a6cbdf)

<div align="justify">
We present statistics on count, mean, standard deviation, mean, and the proportion of
multiples of 25%. Senior citizen, tenure, monthly charges, and total charges are the pertinent
facts. Following that, we looked at the general traits of the categories in the data set, as shown
in Figure 5.
</div>
<br>

<div align="center">
  <img src="https://github.com/ahvshim/Customer-Segmentation/assets/126220185/026000ab-b8d2-475b-a29d-1865c5c74f13">
</div>
<br>

<div align="justify">
The categories in the data set we are displaying have the following features: count,
unique, top data, and frequency as describe function. The data in the entire column is
considered. Subsequently, we are getting a countable value in a particular column, then
labelling the value and turning the data's coding into a binary value, will make the next
analytical process easier. Results are displayed below and include the following columns:
gender, Partners, Dependents, PhoneService, PaperlessBilling, and Churn.
</div>

<br>
<div align="center">
  <img src="https://github.com/ahvshim/Customer-Segmentation/assets/126220185/5e93275c-0d1c-492b-a27b-aa1454a402ac">
</div>
<br>

<div align="justify">
Next One-hot coding techniques were employed to transform the remaining categorical
data into numerical values. Ordinal labels and coding would not be suitable because they would
bring rank into the data.
</div>
<br>

<div align="justify">
The relevant columns are MultipleLines, InternetService, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract and
PaymentMethod. The result is displayed in Figure 6 below.
</div>

<br>
<div align="center">
  <img src="https://github.com/ahvshim/Customer-Segmentation/assets/126220185/9bfaf133-bedf-4b5f-a351-f2f861f241af">
</div>
<br>

**SECTION 3: Methodology**

The methodology for the research for this project is explained in Figure 7 below.

![Screenshot 2023-07-11 024938](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/ba8d6e05-0ec0-4371-9046-96c96875ba8b)
<br>

<div align="justify">
According to our overview of the dataset, the information is related to the business
industries as churn services industry. The service industry, which economists more technically
refer to as the "tertiary sector of industry," entails the delivery of services to both enterprises
and end customers. These include accounting, skilled trades (such mechanic or plumbing
services), computer services, dining establishments, and tourism. The services offered for this
dataset include internet and telecommunications services. Additionally, this company offers
each of its clients an online rental payment option. Direct internet streaming of TV shows and
films is one of the amenities offered.
</div>
<br>

<div align="justify">
The data set was downloaded from the Kaggle website. The following URL serves as a
reference: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn].
Data analysis utilizing visual methods is called exploratory data analysis (EDA). With
the use of statistical summaries and graphical representations, it is used to identify trends,
patterns, or to verify assumptions and is a method for improving visual comprehension of data
analysis on this dataset studies.
</div>
<br>

<div align="justify">
Data preprocessing, a part of data preparation, refers to any type of processing done on
raw data to get it ready for another data processing technique. It has historically been a
significant first step in the data mining process. The first step in preprocessing this dataset is to
delete any extraneous columns, such as the customerID column that won't be used in the later
study. Next, we convert continuous variables, such the TotalCharges column from object to
numeric, to_numeric data types. We next check the dataset's null count, and find that
TotalCharges has 11 counts, so we replace the null to a zero value, and after that we are
checking the null count once more to verify.
</div>

<br>

**K-Means Algorithm**

<div align="justify">
Partition-based clustering, also known as centroid-based clustering, is a method of
grouping similar data points (or "observations") together into clusters. The basic idea is to
partition the data into a fixed number of clusters, with each cluster represented by its centroid
(i.e., the mean of the points in the cluster). One of the most common partition-based clustering
algorithms is k-means, which works by iteratively reassigning each point to the cluster with
the closest centroid, and then recomputing the centroids based on the new assignments. Other
examples of partition-based clustering algorithms include k-medoids and the hierarchical
clustering algorithm.
</div>
<br>

<div align="justify">
The goal of K-means clustering, a vector quantization technique that originated in
signal processing, is to divide n observations into k clusters, each of which has a prototype
consisting of the observation that belongs to the cluster with the closest mean. (also known as
the cluster centroid or cluster centers), acting as a prototype for the cluster. The process of
allocating each data point to the groups iteratively results in a gradual clustering of data points
based on shared characteristics. The goal is to determine which appropriate group each data
point should belong to by minimizing the sum of the distances between each data point and the
cluster centroid.
</div>
<br>

<div align="justify">
Since we lacked a specific outcome variable to try to predict, K-means clustering was
applied. It is employed because we have a set of features that we wish to use to locate groups
of observations with a common set of features. For example, elliptical clusters. K Means
clustering necessitates prior K information, cluster one does not want to split our data.
</div>
<br>

<div align="justify">
K-means is a popular choice for partition-based clustering because it is relatively simple
to understand and implement, and it can be applied to a wide variety of data types. Additionally,
the k-means algorithm is computationally efficient, making it well-suited for large datasets.
Another advantage of the k-means algorithm is that it can be used to identify clusters of any
shape, as long as they are roughly spherical. This makes it a versatile choice for exploring the
structure of a dataset, as it can be used to identify clusters of points that are tightly grouped
together, even if they are not perfectly spherical.
</div>
<br>

<div align="justify">
In addition to its computational efficiency and ability to handle a wide range of data
types, k-means is also relatively easy to interpret. Each cluster is represented by its centroid,
which can be thought of as a "prototype" of the points in the cluster, and the cluster assignments
for each point in the dataset can be easily visualized. This makes it a useful tool for exploratory
data analysis, as it can help to identify patterns and structure in the data that may not be
immediately apparent. Overall, k-means algorithm is a simple, efficient and easy to interpret
algorithm that is well-suited for partition-based clustering of large and complex datasets.
</div>
<br>


**DBScan Algorithm**

<div align="justify">
Density-based clustering is a method of grouping similar data points together into
clusters based on the density of points in a particular region of the feature space. The basic idea
is that clusters are formed by high-density regions of the data, separated by low-density regions.
</div>
<br>

<div align="justify">
One of the most common density-based clustering algorithms is DBSCAN (Density-
Based Spatial Clustering of Applications with Noise). DBSCAN works by identifying "core
points" (points that have a sufficient number of nearby points) and expanding clusters from
these core points. Points that are not part of a core point are considered as "noise". DBSCAN
has two important parameters: minPts and eps. minPts represents the minimum number of
points needed to form a dense region, and eps represents the maximum radius of the
neighborhood around each point.
</div>
<br>

<div align="justify">
Another example of density-based clustering is OPTICS (Ordering Points to Identify
the Clustering Structure) which is an extension of DBSCAN. It builds the reachability-distance
ordering of the data points, and the density-connected sets are extracted from the reachability-
distance plot. Density-based clustering has several advantages over other types of clustering
methods. It is able to find clusters of arbitrary shape and can discover clusters with varying
densities. It is also robust to noise and can find clusters of different sizes. However, density-
based clustering can be sensitive to the choice of parameters and may not be as efficient for
large datasets.
</div>
<br>

<div align="justify">
Overall, density-based clustering is a useful method for identifying clusters in the data
based on the density of points in a particular region of the feature space. The DBSCAN and
OPTICS are two of the popular density-based clustering algorithms. DBSCAN (Density-Based
Spatial Clustering of Applications with Noise) is a popular choice for density-based clustering
because it has several advantages over other methods:
</div>
<br>
<br>

1. It can find clusters of arbitrary shape: Unlike partition-based methods such as k-means,
    DBSCAN does not rely on spherical clusters, so it can identify clusters of any shape,
    such as elongated or spiral shapes.
2. It can discover clusters with varying densities: DBSCAN can identify clusters with
    varying densities, which means that it can find clusters that have a different number of
    points and different densities.
3. It is robust to noise: DBSCAN can identify clusters of different sizes and can separate
    noise points from the clusters.
4. It does not require the number of clusters to be specified in advance: DBSCAN does
    not require the number of clusters to be specified, which can be useful when the number
    of clusters is unknown or the data is not clearly separated into clusters.
5. It is relatively efficient: DBSCAN is a linear-time algorithm, which means that it can
    be applied to large datasets.
<br>

<div align="justify">   
In summary, DBSCAN algorithm is particularly useful when the data has clusters of
arbitrary shape, clusters with varying densities, and the number of clusters is unknown, or the
data is not clearly separated into clusters. It is also robust to noise and relatively efficient,
making it a good choice for density-based clustering of large and complex datasets.
</div>

<br>

**SECTION 4: Result and Interpretation**
<br>
_Note: interpret the results like: how similar are the members in the clusters? What are the similarities within
the group? Etc._
<br>

## K-Means Clustering

<div align="justify"> 
1. Figure 8 shows the graph of the distortion score elbow for K-Means clustering. In order
    to find an appropriate number of clusters, the elbow method is used. In this method for
    this case, the inertia for a number of clusters between 2 and 10 will be calculated. The
    rule is to choose the number of clusters where you see a kink or “an elbow” in the graph.
    The graph also shows that the reduction of a distortion score as the number of clusters
    increases. However, there is no clear “elbow” visible. The underlying algorithm
    suggests 4 clusters. Hence, the choice of four or five clusters seems to be fair.
</div>
<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 11 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/bc1eccec-e966-4c4a-b1ae-ba04efb15a69)
<br>

<div align="justify"> 
2. There is another way to choose the best number of clusters which is by plotting the
    silhouette score in a function of a number of clusters. Figure 9 shows the implementing
    silhouette coefficient in the elbow chart, and this is the first step that need to be taken.
</div>
<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 12 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/a275b60f-8b7e-4da2-9155-162c768ba7f8)
<br>

3. The algorithm is fit using the two clusters as suggested by assigning the K-means label
    in the dataframe. Figure 10 shows the coding used for this step.

<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 12 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/b002a0db-ea37-4e2b-a6e4-d56f179133fa)
<br>


4. A new dataframe is created to get the relationship between the K-mean labels of the
    data. Figure 11 shows the coding and the new dataframe.
<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 13 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/e40a5dc3-952a-41d4-b167-d6d4db29ce88)
<br>

<div align="justify"> 
5. The clusters in Monthly Charges vs Tenure are visualized by using the coding in Figure
    12 and Figure 13 shows the scatter plot of Monthly Charges vs Tenure. The clusters of
    monthly charges and tenure are shown in Figure 6. However, the clusters did not give
    any clear relationship between each other as the data are closer to each other.
</div>
<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 13 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/a7fadbc1-c761-47cc-8866-d0bf0e68844e)
![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 13 image 3](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/698af11b-2c4e-4ac8-8fde-0a53421026cf)
<br>

<div align="justify"> 
6. Figure 14 shows the coding to visualize a 3D plot of Tenure, Monthly Charges and
    Churn and Figure 15 shows the 3D plot. The 3D plot does not actually show the actual
    3D plot as the churn data is a binary data which the value will be either 0 or 1 only.
    Hence, the plotting only shows a flat data on 0 and 1 axis of churn. Thus, this plot also
    does not give any clear relationship between each cluster.
</div>
<br>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 14 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/6e7869bc-60af-454b-8911-d3037cd2899c)
![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 14 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/a03f762f-c461-4b0c-8ffe-e0c52c941774)
<br>

<div align="justify"> 
7. The quality of each cluster is checked by plotting silhouette of K-means clustering for
    7043 samples in two centers as shown in Figure 16. A silhouette score ranges from - 1
    to 1, with higher values indicating that the objects well matched to its own cluster and
    are further apart from neighboring clusters. For cluster 0, the silhouette score is around
    0.85 while the silhouette score for cluster 1 is around 0.72. Hence, these clusters are
    having a good silhouette score which indicates that both of them are in a good quality.
</div>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 15 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/d70ae398-04d7-40db-92fd-69ecb7589d7b)
<br>

## DBSCAN

1. It is difficult arbitrarily to say what values of epsilon and min_samples will work the
    best. Therefore, a matrix of investigated combinations is created first. (Matrix of Epsilon and min-samples)

   ![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 15 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/d612f19d-21ff-4634-a980-59b3ba3ec524)
<br>

2. Because DBSCAN creates clusters itself based on those two parameters, the number of
    generated clusters based on the parameters from step 1 is collected.
   
![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 16 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/63003904-ce12-4e61-988a-0cdfd5ee34c5)

<br>

3. A heatplot shows how many clusters were generated by the DBSCAN algorithm for the
    respective parameters combinations.

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 16 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/b5eb0815-0f15-4bd8-814d-98ed6fb6b301)

<br>

<div align="justify"> 
4. Heatplot from step 3 shows the number of clusters varies greatly with the minimum
    number of clusters of 43 and the maximum at about 520. A silhouette score is plotted
    as a heatmap to decide which combination of epsilon and minimum density threshold
    to choose.
</div>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 16 image 3](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/58c5365c-dbfa-4faa-a3ac-88f3ab2aa775)
<br>

<div align="justify"> 
5. According to silhouette value conventions, values near +1 indicates that the samples
    are far away from the neighboring clusters while a value of 0 indicates that the sample
    is on or very close to the decision boundary between 2 neighboring clusters. Negative
    values on the other hand indicates that samples might have been assigned to the wrong
    cluster. From step 4, resulting silhouette values are all in negative which indicates that
    something is either wrong with the data or the algorithm chosen itself. The model is
    trained again using only continuous variables to check whether there is something
    wrong with the data. All binary data and one-hot encoded data are omitted. The
    processes for DBSCAN from step 1 to 4 are repeated.
</div>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 17 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/61a5bb1f-42bd-44b6-89c1-f4efd23965bd)
<br>

<div align="justify"> 
6. From the Figure 18, there seems to be no major difference in the number of clusters that
    was derived earlier in step 3. The heatmap of silhouette scores for the updated criteria
    are visualized.
</div>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 17 image 2](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/28bbf21d-ea8a-4c1c-a3c6-a9eed4e8760c)
<br>

<div align="justify"> 
7. As was in step 4, step 6 too produced negative values which are indicative that the
    dataset is not suitable for DBSCAN as no apparent number or clusters can be generated.
    In essence, silhouette scores rewards clustering where points are very close to their
    assigned centroids and far from other centroids (good cohesion and good separation).
    Negative scores here could be taken to mean that the algorithm could not distinguish
    the presence of clear and obvious clusters in the data. A scatterplot of Monthly Charges
    vs Tenure is visualized to show that there are no apparent clusters that can be derived.
</div>

![MCSD2123-ASSIGNMENT 3_Dinesh_Amira_Ahmed_Azni(1) pdf Page 18 image 1](https://github.com/ahvshim/Customer-Segmentation/assets/126220185/f2845aa7-c29d-4033-a243-666dd6a16810)
<br>

**Performance Comparison (K-Means vs DBSCAN)**

<div align="justify"> 
As was explored in the 2 sections above, K-Means model resulted in 2 clusters with a
silhouette score of 0.703 while DBSCAN model did not result in any meaningful number of
clusters and silhouette scores. This could be explained by the fact that K-Means require the
number of clusters as input from the user which generally means that more than 1 cluster is
possible to be derived. DBSCAN fundamentally on the hand does not need number of clusters
to be specified and it locates regions of high density that are separated from one another by
regions of low density. Since the majority of the data is densely populated as a whole, DBSCAN
is unable to detect any apparent clusters in the dataset and is an unsuitable choice of clustering
algorithm for this dataset. K-Means on the other hand proves to be a good clustering algorithm
for this dataset with a silhouette score of 0.703 (close to +1). But 2 key points has to be
addressed here which are:
</div>
<br>

<div align="justify"> 
1. Although 2 clusters are derived, no clear and apparent relationship or interesting
    insights could be derived from them. Plotting most of the features on a scatterplot
    against each other would be of minimal use as most data is either binary or
    numerical label which would not transmit any visual information on the presence
    of clusters.
</div>

<div align="justify"> 
2. The number of clusters are user defined, and for this case, additional number of
    clusters with a hit on the silhouette score is possible but there is no guarantee of
    generating useful insights as the data is tightly packed together as a whole.
</div>
<br>

**SECTION 5 : Conclusion**

<div align="justify"> 
The dataset was analyzed with two clustering methods, K-means for Partition-Based
and DBSCAN for Density-Based clustering in order to identify patterns and relationships
within the data that may not be immediately obvious.
</div>
<br>
<div align="justify"> 
On one-way, K-means method was used to group the customers based on their
demographics and usage patterns. After clustering the customers into 2 clusters as suggested,
the mean of the data was grouped by KM-cluster. Eventually, visualizations technique such as
scatter plot and silhouette visualizer were obtained to find any interesting patterns in the data
and check the quality of each cluster respectively.
</div>
<br>

<div align="justify"> 
On the other way, DBSCAN (Density-Based Spatial Clustering of Applications with
Noise) was used to identify clusters of arbitrary shapes using Epsilon and min-samples. Based
on those parameters DBSCAN algorithm created the number of clusters and sill score and thus,
a Heat plot visualized it as 43-520.
</div>
<br>

<div align="justify"> 
K-Means proved to be a good clustering algorithm for this dataset with a silhouette
score of 0.703 (close to +1) while DBSCAN was unable to detect any apparent clusters in the
dataset and is an unsuitable choice of clustering algorithm for this dataset.
</div>


