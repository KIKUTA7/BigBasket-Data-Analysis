import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class DataProcessor:
    def __init__(self, filePath):
        """
        Initializes the DataProcessor class with the file path of the dataset.
        Attributes:
        - filePath: Path to the CSV file.
        - data: The original dataset loaded from the file.
        - processedData: A cleaned and processed version of the dataset.
        """
        self.filePath = filePath
        self.data = None
        self.processedData = None

    def loadAndCleanData(self):
        """
        Loads the dataset from the given file path and performs cleaning operations:
        - Drops rows with missing critical column values.
        - Fills missing ratings with the mean rating of the respective category. Except rating, sub_category,
          brand and type columns all columns are critical in the data.
        - Computes discount percentages for products.
        - Categorizes products into rating categories based on their ratings. Disappearing items are
          products that result in financial losses for the company and should therefore
          be withdrawn from the market.
        - Handles outliers in price columns with _handleOutliers method.

        Returns:
        - A cleaned and processed dataset.
        """
        self.data = pd.read_csv(self.filePath)
        self.processedData = self.data.copy()

        criticalColumns = ['index', 'product', 'category', 'sale_price', 'market_price']
        mask = self.processedData[criticalColumns].notna().all(axis=1)
        self.processedData = self.processedData[mask].copy()

        category_means = self.processedData.groupby('category')['rating'].transform('mean')
        self.processedData.loc[:, 'rating'] = self.processedData['rating'].fillna(category_means)

        self.processedData.loc[:, 'discount'] = (
                (self.processedData['market_price'] - self.processedData['sale_price']) /
                self.processedData['market_price'] * 100
        )

        conditions = [
            (self.processedData['rating'] >= 4.5),
            (self.processedData['rating'] >= 3.5) & (self.processedData['rating'] < 4.5),
            (self.processedData['rating'] >= 2.8) & (self.processedData['rating'] < 3.5),
            (self.processedData['rating'] < 2.8)
        ]
        choices = ['Excellent', 'Good', 'Average', 'Disappearing']
        self.processedData.loc[:, 'rating_category'] = np.select(conditions, choices, default='Unknown')

        self.processedData = self._handleOutliers()

        return self.processedData

    def _handleOutliers(self):
        """
        Method uses the Interquartile Range (IQR) method to detect and
        remove outliers in the dataset. The IQR method is applied to clean the dataset by
        filtering out extreme values (outliers) from two specific columns: 'sale_price' and
        'market_price'. Outliers are defined as values that fall below or above a certain
        range determined by the IQR.

        Returns:
        - A dataset with outliers removed.
        """
        clean_data = self.processedData.copy()

        for column in ['sale_price', 'market_price']:
            Q1 = clean_data[column].quantile(0.25)
            Q3 = clean_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lowerBound = Q1 - 1.5 * IQR
            upperBound = Q3 + 1.5 * IQR
            clean_data = clean_data[
                (clean_data[column] >= lowerBound) &
                (clean_data[column] <= upperBound)
                ]

        return clean_data


class DataAnalyzer:
    def __init__(self, data):
        """
        Initializes the DataAnalyzer class with the cleaned dataset.
        Attributes:
        - data: The dataset to analyze.
        """
        self.data = data

    def generateVisualizations(self):
        """
        Creates visualizations to explore and analyze the dataset:
        - Histogram for the distribution of product ratings.
        - Boxplot for rating distribution across different categories.
        - Scatterplot to observe the relationship between price and rating.
        - Histogram for discount distribution.
        - Pie chart for the distribution of rating categories.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='rating', bins=30)
        plt.title('Distribution of Product Ratings')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x='category', y='rating')
        plt.xticks(rotation=45)
        plt.title('Rating Distribution by Category')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='sale_price', y='rating',
                        hue='category', alpha=0.6)
        plt.title('Price vs Rating Relationship')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='discount', bins=50)
        plt.title('Distribution of Discounts')
        plt.show()

        plt.figure(figsize=(10, 6))
        rating_dist = self.data['rating_category'].value_counts()
        plt.pie(rating_dist.values, labels=rating_dist.index, autopct='%1.1f%%')
        plt.title('Distribution of Rating Categories')
        plt.show()

    def generateStatistics(self):
        """
        Computes and returns statistics about the dataset:
        - General statistics: Total products, unique categories, unique brands,
          average rating, average discount, and price range.
        - Category-wise statistics: Mean rating, discount, and sale price for each category.

        Returns:
        - stats: General statistics as a dictionary.
        - categoryStats: Category-wise statistics as a DataFrame.
        """
        stats = {
            'Total Products': len(self.data),
            'Categories': len(self.data['category'].unique()),
            'Brands': len(self.data['brand'].unique()),
            'Average Rating': self.data['rating'].mean(),
            'Average Discount': self.data['discount'].mean(),
            'Price Range': f"{self.data['sale_price'].min():.2f} - {self.data['sale_price'].max():.2f}"
        }

        categoryStats = self.data.groupby('category').agg({
            'rating': 'mean',
            'discount': 'mean',
            'sale_price': 'mean'
        }).round(2)

        return stats, categoryStats


class ModelTrainer:
    def __init__(self, data):
        """
        Initializes the ModelTrainer class with the dataset to be used for training and testing models.
        Attributes:
        - data: The cleaned dataset.
        - XTrain, XTest, yTrain, yTest: Data splits for training and testing.
        """
        self.data = data
        self.XTrain = None
        self.XTest = None
        self.yTrain = None
        self.yTest = None

    def prepareData(self):
        """
        Prepares the data for model training:
        - Encodes categorical variables ('category' and 'brand') into numerical values, because
          machine learning models typically cannot work with string data directly.
        - Defines features and target variable. features includes numerical and encoded
          categorical columns relevant to prediction.
        - Splits data into training and testing sets.
        - Standardizes the feature variables.
        """
        labelEncoder = LabelEncoder()
        self.data.loc[:, 'encoded_category'] = labelEncoder.fit_transform(self.data['category'])
        self.data.loc[:, 'encoded_brand'] = labelEncoder.fit_transform(self.data['brand'])

        features = ['encoded_category', 'encoded_brand', 'sale_price',
                    'market_price', 'discount']
        X = self.data[features]
        y = self.data['rating_category']

        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        self.XTrain = scaler.fit_transform(self.XTrain)
        self.XTest = scaler.transform(self.XTest)

    def trainAndEvaluate(self):
        """
        Trains and evaluates models on the prepared data:
        - Uses Random Forest and Logistic Regression models.
        - Fits models to the training data.
        - Evaluates model performance using classification reports and confusion matrices.

        Returns:
        - results: A dictionary containing evaluation metrics for each model.
        """
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100,
                                                    class_weight='balanced',
                                                    random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000,
                                                      class_weight='balanced',
                                                      random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(self.XTrain, self.yTrain)
            predictions = model.predict(self.XTest)
            results[name] = {
                'Classification Report': classification_report(self.yTest, predictions,
                                                               zero_division=0),
                'Confusion Matrix': confusion_matrix(self.yTest, predictions)
            }

        return results


class SalesProjector:
    def __init__(self, data, projection_months=6):
        """
        Initializes the SalesProjector class with cleaned dataset and projection period.

        Attributes:
        - data: The cleaned dataset
        - projection_months: Number of months to project sales for
        - popularity_scores: Calculated popularity scores for each product
        """
        self.data = data
        self.projection_months = projection_months
        self.popularity_scores = None

    def _calculate_popularity_scores(self):
        """
        Calculates popularity scores based on:
        1. Product rating (normalized)
        2. Category size (number of products in category)
        3. Rating category weights

        Returns normalized popularity scores for each product
        """
        category_sizes = self.data['category'].value_counts()
        self.data['category_size'] = self.data['category'].map(category_sizes)

        normalized_ratings = (self.data['rating'] - self.data['rating'].min()) / \
                             (self.data['rating'].max() - self.data['rating'].min())
        normalized_category_sizes = (self.data['category_size'] - self.data['category_size'].min()) / \
                                    (self.data['category_size'].max() - self.data['category_size'].min())

        rating_weights = {
            'Excellent': 1.0,
            'Good': 0.8,
            'Average': 0.5,
            'Disappearing': 0.2,
            'Unknown': 0.1
        }
        rating_category_weights = self.data['rating_category'].map(rating_weights)

        self.popularity_scores = (0.4 * normalized_ratings +
                                  0.3 * normalized_category_sizes +
                                  0.3 * rating_category_weights)

        return self.popularity_scores

    def project_sales(self, base_daily_sales=1000):
        """
        Projects sales and profits for the specified period.

        Parameters:
        - base_daily_sales: Base number of total items sold per day

        Returns:
        - DataFrame with projected sales and profits
        """
        if self.popularity_scores is None:
            self._calculate_popularity_scores()

        sales_distribution = self.popularity_scores / self.popularity_scores.sum()

        monthly_sales = base_daily_sales * sales_distribution

        unit_profit = self.data['market_price'] - self.data['sale_price']

        monthly_profit = monthly_sales * unit_profit
        total_profit = monthly_profit * (self.projection_months * 30) # Convert months to Days

        projections = pd.DataFrame({
            'product': self.data['product'],
            'category': self.data['category'],
            'popularity_score': self.popularity_scores,
            'projected_monthly_sales': monthly_sales.round(0),
            'unit_profit': unit_profit.round(2),
            'projected_monthly_profit': monthly_profit.round(2),
            'projected_6month_profit': total_profit.round(2)
        })

        return projections

    def generate_projection_visualizations(self, projections):
        """
        Generates visualizations for sales projections:
        1. Top 10 products by projected profit
        2. Category-wise projected profits
        3. Correlation between popularity scores and projected profits
        """
        plt.figure(figsize=(12, 6))
        top_10_products = projections.nlargest(10, 'projected_6month_profit')
        sns.barplot(data=top_10_products, x='projected_6month_profit', y='product')
        plt.title('Top 10 Products by Projected 6-Month Profit')
        plt.xlabel('Projected Profit')
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(12, 6))
        category_profits = projections.groupby('category')['projected_6month_profit'].sum()
        category_profits.sort_values(ascending=True).plot(kind='barh')
        plt.title('Projected 6-Month Profit by Category')
        plt.xlabel('Projected Profit')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=projections, x='popularity_score', y='projected_6month_profit',
                        alpha=0.6, hue='category')
        plt.title('Popularity Score vs Projected Profit')
        plt.tight_layout()
        plt.show()

    def generate_summary_report(self, projections):
        """
        Generates a summary report of the projections
        """
        summary = {
            'Total Projected 6-Month Profit': f"₹{projections['projected_6month_profit'].sum():,.2f}",
            'Average Monthly Profit': f"₹{(projections['projected_6month_profit'].sum() / self.projection_months):,.2f}",
            'Top Performing Category': projections.groupby('category')['projected_6month_profit'].sum().idxmax(),
            'Number of Products Analyzed': len(projections),
            'Average Projected Monthly Sales per Product': f"{projections['projected_monthly_sales'].mean():.0f} units"
        }

        return summary


processor = DataProcessor('BigBasketProducts.csv')
cleanedData = processor.loadAndCleanData()

analyzer = DataAnalyzer(cleanedData)
analyzer.generateVisualizations()
stats, categoryStats = analyzer.generateStatistics()

print("\nGeneral Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")

print("\nCategory-wise Statistics:")
print(categoryStats)

modelTrainer = ModelTrainer(cleanedData)
modelTrainer.prepareData()
modelResults = modelTrainer.trainAndEvaluate()

print("\nModel Evaluation Results:")
for model, results in modelResults.items():
    print(f"\n{model} Results:")
    print(results['Classification Report'])

projector = SalesProjector(cleanedData)
projections = projector.project_sales()
projector.generate_projection_visualizations(projections)
summary = projector.generate_summary_report(projections)

print("\nSales Projection Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nTop 10 Products by Projected Profit:")
print(projections.nlargest(10, 'projected_6month_profit')[
          ['product', 'category', 'projected_monthly_sales', 'projected_6month_profit']
      ])
