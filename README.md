# BigBasket-Data-Analysis
## Group Members: Beka Kikutadze, Papuna Mamageishvili, Mariami Shotniashvili, Elene Baiashvili
Final project at Kutaisi International University's computer science faculty subject "Introduction to Data Science with Python" 

### Beka Kikutadze
In this project, my primary responsibility was to train an AI model using a subset of the data and predict ratings for the remaining portion. To achieve this, I divided the dataset into two parts and applied machine learning algorithms to generate predictions. I conducted research to determine the optimal data split ratios and identify the most suitable algorithms for our supermarket product data.

Based on my analysis, I selected two algorithms and developed methods to prepare the data by splitting it into four parts. The first part consisted of training data, which included key columns along with their corresponding target values in the second part, the rating_category column. The third part comprised testing data with critical columns, which was used to evaluate how accurately the machine learning algorithms could predict the rating_categories.

I then analyzed the prediction results and generated a confusion matrix and a classification report to present the error metrics in a clear and comprehensive format.

### Elene Baiashvili
In our project, my primary responsibility was designing and implementing the DataProcessor class, which plays a crucial role in preparing our dataset for analysis. I focused on ensuring that the data was clean, consistent, and ready for any downstream tasks, such as visualization or modeling.
1. Data Loading and Cleaning (loadAndCleanData method)
• Objective: Load raw data, clean it, and prepare it for analysis.
• What I did:
• Removed rows with missing critical values (index, product, category, sale_price, market_price).
• Filled missing rating values with the mean rating of their respective category.
• Calculated discount percentages and categorized products into rating categories (Excellent, Good, Average, Disappearing).
• Why: To ensure data consistency, handle missing values, and derive meaningful insights for analysis.
2. Outlier Handling (_handleOutliers method)
• Objective: Remove extreme values from pricing columns (sale_price, market_price) using the IQR (Interquartile Range) method.
Why: Outliers can distort results, and removing them ensures reliable analysis

### Mariami Shotniashvili 
My primary contribution to this project was developing the DataAnalyzer class, which focuses on exploring and visualizing the cleaned dataset to extract meaningful insights. 

I designed this class to generate key visualizations that I thought would be most meaningful, including histogram for rating distributions, scatterplot for price-rating relationships, and pie chart for rating categories. These visualizations were chosen to highlight critical aspects of the dataset, such as customer preferences, product performance across categories, and potential pricing strategies. 

Additionally, I implemented statistical summaries to provide a comprehensive overview of the data. I highlighted the following statistics: 

General statistics: Metrics like average ratings, average discounts, and price range provide a high-level snapshot of the dataset, helping stakeholders understand the overall state of the products.

Category-wise statistics: Detailed insights into each category’s performance allow for targeted strategies to improve specific product groups.

### Papuna Mamageishvili



