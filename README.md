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


 
### Overview
This project performs comprehensive data analysis on BigBasket's product dataset, including data cleaning,
visualization, machine learning model training, and sales projections.
The analysis includes product ratings categorization, price analysis, and projected profit calculations based on popularity scores.
 
### Features
- **Data Processing & Cleaning**
  - Handles missing values in critical columns
  - Fills missing ratings with category means
  - Removes outliers in price columns using IQR method
  - Calculates discount percentages
 
- **Data Analysis**
  - Generates multiple visualizations for product insights
  - Computes comprehensive statistics per category
  - Creates rating categories (Excellent, Good, Average, Disappearing)
  - Analyzes price-rating relationships
 
- **Machine Learning Models**
  - Implements Random Forest and Logistic Regression classifiers
  - Features proper train/test split with standardization
  - Includes model evaluation and comparison
  - Uses encoded categorical variables for prediction
 
- **Sales Projection**
  - Projects 6-month sales based on product popularity
  - Calculates potential profits using sale and market prices
  - Considers product ratings and category sizes for popularity scoring
  - Generates visualization of projected profits
 
### Project Structure
├── BigBasket Products.csv     # Input dataset
├── requirements.txt           # Project dependencies
└── main.py                   # Main script containing all classes
    ├── DataProcessor         # Handles data loading and cleaning
    │   ├── __init__         # Initializes with file path
    │   ├── loadAndCleanData # Loads and cleans the dataset
    │   └── _handleOutliers  # Removes price outliers using IQR
    │
    ├── DataAnalyzer         # Performs analysis and visualization
    │   ├── __init__         # Initializes with cleaned data
    │   ├── generateVisualizations  # Creates data visualizations
    │   └── generateStatistics      # Computes dataset statistics
    │
    ├── ModelTrainer         # Implements ML models
    │   ├── __init__         # Initializes with cleaned data
    │   ├── prepareData      # Prepares data for ML models
    │   └── trainAndEvaluate # Trains and evaluates models
    │
    └── SalesProjector       # Projects future sales and profits
        ├── __init__         # Initializes with data and projection period
        ├── _calculate_popularity_scores  # Calculates product popularity
        ├── project_sales    # Projects sales and profits
        ├── generate_projection_visualizations  # Creates projection charts
        └── generate_summary_report  # Generates projection summary
 
#### Class Functions Documentation
 
##### DataProcessor
- `__init__(filePath)`: Initializes with path to CSV file
  - Parameters: filePath (str) - Path to the dataset
  - Attributes: data, processedData (pandas DataFrames)
 
- `loadAndCleanData()`: Main data cleaning function
  - Drops rows with missing critical values
  - Fills missing ratings with category means
  - Calculates discount percentages
  - Creates rating categories
  - Returns cleaned DataFrame
 
- `_handleOutliers()`: Internal function for outlier removal
  - Uses IQR method for sale_price and market_price
  - Returns DataFrame with outliers removed
 
##### DataAnalyzer
- `__init__(data)`: Initializes with cleaned dataset
  - Parameters: data (pandas DataFrame)
 
- `generateVisualizations()`: Creates data visualizations
  - Rating distribution histogram
  - Category-wise rating boxplot
  - Price vs Rating scatterplot
  - Discount distribution histogram
  - Rating categories pie chart
 
- `generateStatistics()`: Computes dataset statistics
  - Returns two items:
    1. General statistics dictionary
    2. Category-wise statistics DataFrame
 
##### ModelTrainer
- `__init__(data)`: Initializes with cleaned dataset
  - Parameters: data (pandas DataFrame)
  - Attributes: XTrain, XTest, yTrain, yTest for model training
 
- `prepareData()`: Prepares data for ML models
  - Encodes categorical variables
  - Splits data into train/test sets
  - Standardizes features
 
- `trainAndEvaluate()`: Trains and evaluates models
  - Implements Random Forest and Logistic Regression
  - Returns dictionary with classification reports and confusion matrices
 
##### SalesProjector
- `__init__(data, projection_months=6)`: Initializes projector
  - Parameters:
    - data (pandas DataFrame)
    - projection_months (int, default=6)
 
- `_calculate_popularity_scores()`: Internal scoring function
  - Considers ratings, category size, and rating categories
  - Returns normalized popularity scores
 
- `project_sales(base_daily_sales=10000)`: Projects sales
  - Parameters: base_daily_sales (int) - Base daily sales
  - Returns DataFrame with projected sales and profits
 
- `generate_projection_visualizations(projections)`: Creates charts
  - Top 10 products by profit
  - Category-wise profits
  - Popularity vs profit correlation
 
- `generate_summary_report(projections)`: Creates summary
  - Returns dictionary with key projection metrics
 
### Installation
1. Clone the repository
2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  
#### On Windows: venv\Scripts\activate
3. Install required packages:
   pip install -r Requirements.txt 
 
### Usage
1. Ensure your dataset "BigBasket Products.csv" is in the project directory
   Dataset link: https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints
2. Run the main script:
   python main.py
 
### Output
The script will generate:
- Visualizations for product analysis
- Statistical reports
- Machine learning model evaluation results
- Sales projections and profit analysis
- Summary reports with key metrics
 
### Data Requirements
The input CSV file should contain the following columns:
- index
- product
- category
- sub_category
- brand
- sale_price
- market_price
- type
- rating

