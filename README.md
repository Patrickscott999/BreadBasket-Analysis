# Bread Basket Analysis - Association Rule Mining

A Streamlit application for analyzing retail transaction data using the Apriori algorithm for association rule mining.

## Features

- **Data Overview**: View and analyze transaction data
- **Item Analysis**: Explore item frequencies and distributions
- **Association Rules**: Discover patterns in customer purchasing behavior
- **Interactive Network Graph**: Visualize relationships between items
- **Dark/Light Mode**: Toggle between dark and light themes
- **Download Functionality**: Export analysis results as CSV files
- **Insights and Recommendations**: Get actionable insights based on the analysis

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bread-basket-analysis.git
   cd bread-basket-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload your transaction data in CSV format or use the default "bread basket.csv" dataset
2. Navigate through the tabs to explore different aspects of the data
3. Adjust the Apriori algorithm parameters in the sidebar
4. Run the analysis to discover association rules
5. Explore the network graph to visualize item relationships
6. Review insights and recommendations for actionable strategies

## Data Format

The application expects a CSV file with at least the following columns:
- `Transaction`: Transaction ID
- `Item`: Item name

## Technologies Used

- Python
- Streamlit
- Pandas
- MLxtend (Apriori algorithm)
- Plotly
- NetworkX
- Pyvis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Bread Basket dataset is used for demonstration purposes
- MLxtend library for implementing the Apriori algorithm 