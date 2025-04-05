import streamlit as st
import pandas as pd
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import tempfile
import base64
import random

# Set page configuration
st.set_page_config(
    page_title="Bread Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #1976D2;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1976D2;
    }
    .dark-mode {
        background-color: #121212;
        color: #ffffff;
    }
    .dark-mode .metric-card {
        background-color: #1E1E1E;
        color: #ffffff;
    }
    .dark-mode .section-header {
        color: #64B5F6;
    }
    .dark-mode .subsection-header {
        color: #90CAF9;
    }
    .dark-mode .main-header {
        color: #90CAF9;
    }
    .dark-mode .stButton>button {
        background-color: #1976D2;
    }
    .dark-mode .stButton>button:hover {
        background-color: #0D47A1;
    }
    .summary-panel {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .dark-mode .summary-panel {
        background-color: #1E1E1E;
        color: #ffffff;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-decoration: none;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .dark-mode .download-button {
        background-color: #388E3C;
    }
</style>
""", unsafe_allow_html=True)

def clean_and_preprocess_data(df):
    # Drop rows with missing values in Transaction and Item columns
    df_cleaned = df.dropna(subset=['Transaction', 'Item'])
    
    # Group items by transaction ID
    grouped_transactions = df_cleaned.groupby('Transaction')['Item'].apply(list).reset_index()
    
    return df_cleaned, grouped_transactions

def create_one_hot_encoding(df):
    # Create a pivot table with transactions as index and items as columns
    # Fill with 1 for presence and 0 for absence
    one_hot = pd.crosstab(df['Transaction'], df['Item'])
    
    # Convert to binary (0 or 1)
    one_hot = one_hot.astype(bool).astype(int)
    
    return one_hot

def run_apriori_analysis(one_hot_data, min_support, min_confidence, min_lift):
    # Generate frequent itemsets
    frequent_itemsets = apriori(one_hot_data, 
                              min_support=min_support,
                              use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, 
                            metric="lift",
                            min_threshold=min_lift)
    
    # Filter rules based on confidence
    rules = rules[rules['confidence'] >= min_confidence]
    
    return frequent_itemsets, rules

def create_top_items_chart(df):
    # Count item frequencies
    item_counts = df['Item'].value_counts().head(10)
    
    # Create bar chart
    fig = px.bar(x=item_counts.index, 
                 y=item_counts.values,
                 title='Top 10 Most Frequent Items',
                 labels={'x': 'Item', 'y': 'Frequency'},
                 color_discrete_sequence=['#2196F3'])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_network_graph(rules, dark_mode=False):
    # Create a new network
    net = Network(height='600px', width='100%', 
                 bgcolor='#ffffff' if not dark_mode else '#121212', 
                 font_color='black' if not dark_mode else 'white')
    
    # Set physics options for better visualization
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.005,
                "springLength": 200,
                "springConstant": 0.18
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # Create a color map for different confidence levels
    def get_color(confidence):
        if confidence >= 0.8:
            return '#FF5252'  # Red for high confidence
        elif confidence >= 0.6:
            return '#FFA726'  # Orange for medium-high confidence
        elif confidence >= 0.4:
            return '#FFEB3B'  # Yellow for medium confidence
        else:
            return '#81C784'  # Green for low confidence
    
    # Create a size map for different lift values
    def get_size(lift):
        return min(30, max(5, 5 + lift * 2))
    
    # Track nodes to avoid duplicates
    nodes_added = set()
    
    # Add nodes and edges
    for idx, row in rules.iterrows():
        # Convert frozenset to list for antecedents and consequents
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # Add nodes for antecedents
        for item in antecedents:
            if item not in nodes_added:
                # Get item frequency for node size
                item_freq = row['antecedent support'] * len(rules)
                node_size = min(30, max(10, 10 + item_freq * 5))
                
                # Add node with custom styling
                net.add_node(str(item), 
                            label=str(item), 
                            title=f"Item: {item}<br>Support: {row['antecedent support']:.2f}",
                            size=node_size,
                            color='#2196F3' if not dark_mode else '#64B5F6')
                nodes_added.add(item)
        
        # Add nodes for consequents
        for item in consequents:
            if item not in nodes_added:
                # Get item frequency for node size
                item_freq = row['consequent support'] * len(rules)
                node_size = min(30, max(10, 10 + item_freq * 5))
                
                # Add node with custom styling
                net.add_node(str(item), 
                            label=str(item), 
                            title=f"Item: {item}<br>Support: {row['consequent support']:.2f}",
                            size=node_size,
                            color='#4CAF50' if not dark_mode else '#81C784')
                nodes_added.add(item)
        
        # Add edge between antecedent and consequent
        for ant in antecedents:
            for cons in consequents:
                # Calculate edge properties
                edge_color = get_color(row['confidence'])
                edge_width = get_size(row['lift'])
                
                # Add edge with custom styling
                net.add_edge(str(ant), str(cons), 
                           title=f"Confidence: {row['confidence']:.2f}<br>Lift: {row['lift']:.2f}<br>Support: {row['support']:.2f}",
                           color=edge_color,
                           width=edge_width,
                           arrows='to')
    
    # Save the network to a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

def main():
    # Initialize session state for dark mode
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Main header
    st.markdown("<h1 class='main-header'>🛒 Bread Basket Analysis</h1>", unsafe_allow_html=True)
    
    # Dark mode toggle in sidebar
    st.sidebar.markdown("<h2 class='section-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Dark mode toggle
    dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.experimental_rerun()
    
    # Apply dark mode if enabled
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #121212;
                color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # File uploader in sidebar
    st.sidebar.markdown("<h3 class='subsection-header'>Data Input</h3>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", 
                                           help="Upload your dataset or use the default bread basket dataset")

    try:
        if uploaded_file is not None:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
        else:
            # Use default file if no file is uploaded
            default_file = "bread basket.csv"
            if os.path.exists(default_file):
                df = pd.read_csv(default_file)
            else:
                st.error("Default file 'bread basket.csv' not found!")
                return

        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Data Overview", "Item Analysis", "Association Rules"])
        
        with tab1:
            st.markdown("<h2 class='section-header'>Data Overview</h2>", unsafe_allow_html=True)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Rows", len(df))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Columns", len(df.columns))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Unique Items", df['Item'].nunique())
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Data preview in expander
            with st.expander("View Dataset Preview", expanded=False):
                st.dataframe(df.head())
            
            # Data cleaning section
            st.markdown("<h3 class='subsection-header'>Data Cleaning</h3>", unsafe_allow_html=True)
            
            # Clean and preprocess the data
            df_cleaned, grouped_transactions = clean_and_preprocess_data(df)
            
            # Display cleaning results in columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Rows Dropped", len(df) - len(df_cleaned))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Cleaned Rows", len(df_cleaned))
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Sample transactions in expander
            with st.expander("View Sample Transactions", expanded=False):
                st.write("First 5 transactions with their items:")
                sample_transactions = grouped_transactions.head()
                for _, row in sample_transactions.iterrows():
                    st.write(f"Transaction {row['Transaction']}: {row['Item']}")
        
        with tab2:
            st.markdown("<h2 class='section-header'>Item Analysis</h2>", unsafe_allow_html=True)
            
            # Create and display top items chart
            top_items_chart = create_top_items_chart(df_cleaned)
            st.plotly_chart(top_items_chart, use_container_width=True)
            
            # Item distribution in expander
            with st.expander("View Item Distribution", expanded=False):
                item_counts = df_cleaned['Item'].value_counts()
                st.dataframe(item_counts)
                
                # Download button for item distribution
                st.markdown(get_download_link(
                    pd.DataFrame({'Item': item_counts.index, 'Count': item_counts.values}),
                    'item_distribution.csv',
                    'Download Item Distribution CSV'
                ), unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<h2 class='section-header'>Association Rules Analysis</h2>", unsafe_allow_html=True)
            
            # Apriori parameters in sidebar
            st.sidebar.markdown("<h3 class='subsection-header'>Apriori Parameters</h3>", unsafe_allow_html=True)
            
            min_support = st.sidebar.slider(
                "Minimum Support", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.01, 
                step=0.01,
                help="Minimum support threshold for itemsets. Lower values will generate more itemsets."
            )
            
            min_confidence = st.sidebar.slider(
                "Minimum Confidence", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.5, 
                step=0.1,
                help="Minimum confidence threshold for rules. Higher values indicate stronger rules."
            )
            
            min_lift = st.sidebar.slider(
                "Minimum Lift", 
                min_value=1.0, 
                max_value=10.0, 
                value=1.0, 
                step=0.1,
                help="Minimum lift threshold for rules. Higher values indicate more interesting rules."
            )
            
            # Network graph options
            st.sidebar.markdown("<h3 class='subsection-header'>Network Graph Options</h3>", unsafe_allow_html=True)
            
            max_rules = st.sidebar.slider(
                "Maximum Rules to Display", 
                min_value=10, 
                max_value=100, 
                value=50, 
                step=10,
                help="Limit the number of rules to display in the network graph for better performance."
            )
            
            # Run Analysis button
            if st.sidebar.button("Run Apriori Analysis", key="run_analysis"):
                with st.spinner("Running Apriori analysis..."):
                    # Create one-hot encoding
                    one_hot_data = create_one_hot_encoding(df_cleaned)
                    
                    # Run Apriori analysis
                    frequent_itemsets, rules = run_apriori_analysis(
                        one_hot_data, min_support, min_confidence, min_lift
                    )
                    
                    # Limit rules for network graph if needed
                    if len(rules) > max_rules:
                        rules_for_graph = rules.nlargest(max_rules, 'lift')
                    else:
                        rules_for_graph = rules
                    
                    # Store results in session state for download
                    st.session_state.frequent_itemsets = frequent_itemsets
                    st.session_state.rules = rules
                    
                    # Display summary panel
                    st.markdown("<div class='summary-panel'>", unsafe_allow_html=True)
                    st.markdown("<h3 class='subsection-header'>Analysis Summary</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(df_cleaned['Transaction'].unique()))
                    
                    with col2:
                        st.metric("Unique Items", df_cleaned['Item'].nunique())
                    
                    with col3:
                        st.metric("Frequent Itemsets", len(frequent_itemsets))
                    
                    with col4:
                        st.metric("Association Rules", len(rules))
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h3 class='subsection-header'>Frequent Itemsets</h3>", unsafe_allow_html=True)
                        frequent_itemsets_sorted = frequent_itemsets.sort_values('support', ascending=False)
                        st.dataframe(frequent_itemsets_sorted)
                        
                        # Download button for frequent itemsets
                        st.markdown(get_download_link(
                            frequent_itemsets_sorted,
                            'frequent_itemsets.csv',
                            'Download Frequent Itemsets CSV'
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<h3 class='subsection-header'>Association Rules</h3>", unsafe_allow_html=True)
                        rules_sorted = rules.sort_values('confidence', ascending=False)
                        st.dataframe(rules_sorted)
                        
                        # Download button for association rules
                        st.markdown(get_download_link(
                            rules_sorted,
                            'association_rules.csv',
                            'Download Association Rules CSV'
                        ), unsafe_allow_html=True)
                    
                    # Display summary statistics in columns
                    st.markdown("<h3 class='subsection-header'>Rule Metrics</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Frequent Itemsets", len(frequent_itemsets))
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Association Rules", len(rules))
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Average Lift", f"{rules['lift'].mean():.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Average Confidence", f"{rules['confidence'].mean():.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Network graph in expander
                    with st.expander("View Association Rules Network Graph", expanded=True):
                        st.markdown("<h3 class='subsection-header'>Association Rules Network</h3>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style='margin-bottom: 1rem;'>
                            <p><strong>Network Graph Legend:</strong></p>
                            <ul>
                                <li><span style='color: #2196F3;'>■</span> Blue nodes: Antecedent items (if X then Y)</li>
                                <li><span style='color: #4CAF50;'>■</span> Green nodes: Consequent items (if X then Y)</li>
                                <li>Edge colors: Confidence level (Red > Orange > Yellow > Green)</li>
                                <li>Edge width: Lift value (thicker = higher lift)</li>
                                <li>Node size: Item frequency (larger = more frequent)</li>
                            </ul>
                            <p><em>Hover over nodes and edges for detailed information.</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        network_file = create_network_graph(rules_for_graph, st.session_state.dark_mode)
                        with open(network_file, 'r') as f:
                            st.components.v1.html(f.read(), height=700)
                        os.unlink(network_file)  # Clean up temporary file
                        
                        # Add Insights and Recommendations section
                        st.markdown("<h3 class='subsection-header'>Insights and Recommendations</h3>", unsafe_allow_html=True)
                        
                        # Calculate key metrics for insights
                        top_items = df_cleaned['Item'].value_counts().head(5)
                        avg_transaction_size = df_cleaned.groupby('Transaction')['Item'].count().mean()
                        strong_rules = rules[rules['confidence'] > 0.7]
                        
                        # Create insights container
                        insights_container = st.container()
                        with insights_container:
                            st.markdown("""
                            <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                                <h4 style='color: #1976D2; margin-bottom: 15px;'>📊 Key Insights</h4>
                            """, unsafe_allow_html=True)
                            
                            # Top Items Insight
                            st.markdown(f"""
                            <div style='margin-bottom: 15px;'>
                                <strong>Top Selling Items:</strong><br>
                                {', '.join([f"{item} ({count} sales)" for item, count in top_items.items()])}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Transaction Size Insight
                            st.markdown(f"""
                            <div style='margin-bottom: 15px;'>
                                <strong>Average Transaction Size:</strong><br>
                                {avg_transaction_size:.1f} items per transaction
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Strong Rules Insight
                            if not strong_rules.empty:
                                st.markdown("""
                                <div style='margin-bottom: 15px;'>
                                    <strong>Strong Product Associations:</strong><br>
                                """, unsafe_allow_html=True)
                                
                                for _, rule in strong_rules.head(3).iterrows():
                                    antecedents = list(rule['antecedents'])
                                    consequents = list(rule['consequents'])
                                    st.markdown(f"""
                                    • When customers buy {', '.join(antecedents)}, 
                                    they often also buy {', '.join(consequents)} 
                                    (Confidence: {rule['confidence']:.2%})
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Recommendations section
                            st.markdown("""
                            <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px;'>
                                <h4 style='color: #2E7D32; margin-bottom: 15px;'>💡 Strategic Recommendations</h4>
                            """, unsafe_allow_html=True)
                            
                            # Product Placement Recommendations
                            st.markdown("""
                            <div style='margin-bottom: 15px;'>
                                <strong>📦 Product Placement Strategy:</strong><br>
                                • Place frequently co-purchased items near each other<br>
                                • Position high-margin items next to popular items<br>
                                • Create dedicated sections for complementary products
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Bundling Recommendations
                            st.markdown("""
                            <div style='margin-bottom: 15px;'>
                                <strong>🎁 Product Bundling Opportunities:</strong><br>
                                • Create combo deals for strongly associated items<br>
                                • Offer discounts on complementary products<br>
                                • Develop meal/snack bundles based on common combinations
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Marketing Recommendations
                            st.markdown("""
                            <div style='margin-bottom: 15px;'>
                                <strong>📢 Marketing and Promotions:</strong><br>
                                • Target promotions based on item associations<br>
                                • Create cross-promotional campaigns<br>
                                • Implement personalized recommendations
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Inventory Management
                            st.markdown("""
                            <div style='margin-bottom: 15px;'>
                                <strong>📊 Inventory Management:</strong><br>
                                • Stock higher quantities of frequently co-purchased items<br>
                                • Adjust reorder points based on association patterns<br>
                                • Monitor seasonal trends in product associations
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Action Items
                            st.markdown("""
                            <div style='margin-top: 20px; background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
                                <h4 style='color: #ffffff; margin-bottom: 15px; font-size: 1.2rem; font-weight: bold;'>⚡ Immediate Action Items</h4>
                                <ol style='color: #ffffff; font-size: 1.1rem; line-height: 1.6;'>
                                    <li style='margin-bottom: 10px;'>Review and optimize product placement based on association patterns</li>
                                    <li style='margin-bottom: 10px;'>Implement targeted promotions for high-confidence item pairs</li>
                                    <li style='margin-bottom: 10px;'>Adjust inventory levels for frequently co-purchased items</li>
                                    <li style='margin-bottom: 10px;'>Develop cross-selling strategies for complementary products</li>
                                    <li style='margin-bottom: 10px;'>Monitor and analyze the impact of implemented changes</li>
                                </ol>
                            </div>
                            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please make sure you've uploaded a valid CSV file.")

if __name__ == "__main__":
    main() 