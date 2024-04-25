# music-recommendation
Item based music recommender with KNN algorithm .

**Content-Based Filtering System Description**
- Finding all pairs of content that have been evaluated by the same user.
- A measurement of the evaluations of two people who watched the same content according to the similarities of the reviews.
- Sort by content first, then by similarity.

**Content Based Suggestion System Introduction**
- It is basically classified as a growth and continuation of information filtering research. In this system, objects are mainly defined by their associated properties. A contextual suggestive learns a profile of the new user's interests based on the properties found in the objects the user has rated. It's basically a keyword specific recommendation system where keywords are used to describe items. Thus, the algorithms used in a content-based recommendation system are such that they suggest similar items to users that they have liked in the past or are currently viewing.

**KNN**
- K-Nearest Neighbors is a machine learning technique and algorithm that can be used for both regression and classification tasks. K-Nearest Neighbors examine the labels of a selected number of data points surrounding a target data point to make a guess about the class to which the data point belongs. K-Nearest Neighbors (KNN) is conceptually a simple but very powerful algorithm and for these reasons it is one of the most popular learning algorithms.

![Neumann_NDH20_01-OQ9sM5Leg5fu_WRAyyB6lRfIUzyAzwvX](https://user-images.githubusercontent.com/57988026/104448871-c86a1d00-55ae-11eb-8898-f0d417988f61.jpg)

**KNN FOR RECOMMENDATION**
- To give a new recommendation to a user, the idea of ​​the content-based referral method is to find items similar to those the user has already interacted with "positively". The two items are considered to be similar if most of the users interacting with the two did it similarly. This method is said to be "content-centered" as it represents items based on users' interactions with them and evaluates the distances between these items.

- Suppose we want to make a suggestion for a specific user. First, we take the content that this user likes best and represent it with each user interaction vector (the "column in the interaction matrix). Then, we can calculate the similarities between the" best content "and all the other products. We can also keep the selected "best content" new to and recommend this content.


