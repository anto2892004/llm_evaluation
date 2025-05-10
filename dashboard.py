import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .model-selector {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 LLM Evaluation Dashboard")
st.markdown("""
Compare the performance of different LLM models across various evaluation metrics.
""")

# Load evaluation results
@st.cache_data
def load_results():
    results = {}
    metrics = ['faithfulness', 'context_relevancy', 'context_recall']  # Add more metrics if available
    
    for metric in metrics:
        csv_path = f"./result_ragas/{metric}_scores.csv"
        if os.path.exists(csv_path):
            results[metric] = pd.read_csv(csv_path)
    
    return results

results = load_results()

# Sidebar - Model selection
st.sidebar.title("Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to compare:",
    options=["gpt4_1_nano", "llama-3-8b", "nova-micro", "qwen-3-4b"],
    default=["gpt4_1_nano", "llama-3-8b"]
)

# Main content
if not results:
    st.warning("No evaluation results found. Please run the evaluation scripts first.")
else:
    # Display metrics in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Faithfulness", "🔍 Context Relevancy", "🎯 Context Recall"])
    
    with tab1:
        if 'faithfulness' in results:
            df = results['faithfulness']
            filtered_df = df[df['Model'].isin(selected_models)]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Faithfulness Scores")
                st.dataframe(filtered_df.style.format({"Faithfulness": "{:.3f}"}))
                
                # Calculate and display average
                avg_score = filtered_df['Faithfulness'].mean()
                st.metric("Average Faithfulness", f"{avg_score:.3f}")
            
            with col2:
                st.subheader("Visualization")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(filtered_df['Model'], filtered_df['Faithfulness'], 
                       color=['blue', 'green', 'orange', 'red'])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                ax.set_title("Faithfulness Comparison")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
        else:
            st.warning("Faithfulness results not available")
    
    with tab2:
        if 'context_relevancy' in results:
            df = results['context_relevancy']
            filtered_df = df[df['Model'].isin(selected_models)]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Context Relevancy Scores")
                st.dataframe(filtered_df.style.format({"Context_Relevancy": "{:.3f}"}))
                
                # Calculate and display average
                avg_score = filtered_df['Context_Relevancy'].mean()
                st.metric("Average Context Relevancy", f"{avg_score:.3f}")
            
            with col2:
                st.subheader("Visualization")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(filtered_df['Model'], filtered_df['Context_Relevancy'], 
                       color=['blue', 'green', 'orange', 'red'])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                ax.set_title("Context Relevancy Comparison")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
        else:
            st.warning("Context Relevancy results not available")
    
    with tab3:
        if 'context_recall' in results:
            df = results['context_recall']
            filtered_df = df[df['Model'].isin(selected_models)]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Context Recall Scores")
                st.dataframe(filtered_df.style.format({"Context_Recall": "{:.3f}"}))
                
                # Calculate and display average
                avg_score = filtered_df['Context_Recall'].mean()
                st.metric("Average Context Recall", f"{avg_score:.3f}")
            
            with col2:
                st.subheader("Visualization")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(filtered_df['Model'], filtered_df['Context_Recall'], 
                       color=['blue', 'green', 'orange', 'red'])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                ax.set_title("Context Recall Comparison")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
        else:
            st.warning("Context Recall results not available")

    # Model comparison summary
    st.markdown("---")
    st.subheader("📋 Model Comparison Summary")
    
    # Create a combined dataframe for all metrics
    combined_data = []
    for metric, df in results.items():
        temp_df = df.copy()
        temp_df['Metric'] = metric.replace('_', ' ').title()
        temp_df = temp_df.rename(columns={
            'Faithfulness': 'Score',
            'Context_Relevancy': 'Score',
            'Context_Recall': 'Score'
        })
        combined_data.append(temp_df)
    
    if combined_data:
        combined_df = pd.concat(combined_data)
        filtered_combined = combined_df[combined_df['Model'].isin(selected_models)]
        
        # Pivot for better visualization
        pivot_df = filtered_combined.pivot(index='Model', columns='Metric', values='Score')
        
        st.dataframe(
            pivot_df.style.format("{:.3f}")
            .background_gradient(cmap='Blues', axis=0)
            .set_properties(**{'text-align': 'center'})
        )
    else:
        st.warning("No data available for summary")

# Footer
st.markdown("---")
st.caption("LLM Evaluation Dashboard - Created with Streamlit")