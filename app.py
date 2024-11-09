
import streamlit as st

st.markdown("<h1 style='text-align: center; color: #002967;'>🎈Business Intelligence (BI)</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: #002967;'>Agriculture by Countries</h1>", unsafe_allow_html=True)
def main():
    # Register pages
    pages = {
        "Commodity": Commodity,
        # "Statistics": Statistics,
        # "AgroBusiness": AgroBusiness,
    }
    st.sidebar.title("Statistics")
    page = st.sidebar.selectbox("Select Menu", tuple(pages.keys()))
    pages[page]()

def Commodity():
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from scipy import stats
    import matplotlib.pyplot as plt
    # Title of the app
    st.title("Global Commodity Trade Statistics")

    # Create a sidebar for navigation
    st.sidebar.header("Menu")

    # Define options for the selectbox
    options = ["Commodity: Statistics", "Visualization", "Predictions"]

    # Create a selectbox in the sidebar
    selected_option = st.sidebar.selectbox("Choose an option:", options)

    # Display content based on user selection
    if selected_option == "Commodity: Statistics":
        st.subheader("Dataset Info", divider=True)
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        st.write(supply.head())
        supply = pd.DataFrame(supply)

        country_nan = supply[supply['weight_kg'].isnull()].groupby('country_or_area').size().sort_values(ascending=False)
        category_nan = supply[supply['weight_kg'].isnull()].groupby('category').size().sort_values(ascending=False)
        percentage_miss = supply.isnull().sum() * 100/len(supply)
        supply.nunique(axis=0)
        supply=supply.dropna()

         # Create a scatter plot
        st.subheader("Statistical Summary after Clean Data", divider=True)
        st.write(supply.describe())
        
        # Convert to numeric, coercing errors
        # Function to clean and convert
        df = pd.DataFrame(supply)
        
        #Frequency counts
        frequency_counts = df.groupby('flow')['year'].sum().reset_index()
        st.write("Frequency Counts:")
        st.dataframe(frequency_counts)

        # Perform Chi-square test
        chi2_stat, p_value = stats.chisquare(frequency_counts['year'])
        st.write(f"Chi-square Statistic: {chi2_stat:.2f}")
        st.write(f"P-value: {p_value:.4f}")

        # Create a bar chart for frequency counts
        plt.figure(figsize=(8, 5))
        plt.bar(frequency_counts['flow'], frequency_counts['year'], color='skyblue')
        plt.title("Frequency Counts by Flow")
        plt.xlabel("flow")
        plt.ylabel("year")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif selected_option == "Visualization":
        st.write("Select different variables to analyze: Categories")
        st.subheader("Dataset Visualization", divider=True)
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        supply = pd.DataFrame(supply)
        
        grouped = supply.groupby(['country_or_area', 'year', 'comm_code',
                            'trade_usd', 'weight_kg', 'quantity_name', 'category']).agg({'flow': 'sum'}).reset_index()

        # Add a dropdown to select the x-axis column
        x_axis_column = st.selectbox('Select 1 option', grouped.columns)

        # Add a dropdown to select the y-axis column
        y_axis_column = st.selectbox('Select 2 option', grouped.columns)

        # Create the Plotly figure
        fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')

        # Customize the figure (optional)
        fig.update_layout(
            xaxis_title=x_axis_column,
            yaxis_title=y_axis_column
        )
        # Display the figure in Streamlit
        st.plotly_chart(fig)

    # elif selected_option == "Visualization":
    #     st.write("Visualize your data here.")
    #     # Additional code for visualization features can go here
    

    elif selected_option == "Predictions":
        import streamlit as st
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import matplotlib.pyplot as plt
        
        st.subheader("Dataset Info", divider=True)
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        df = pd.DataFrame(supply)
        
        df['country_or_area'] = df['country_or_area'].astype(str)
        df['weight_kg'].fillna('weight_kgn', inplace=True)
        
        # Define target and features
        target = 'flow'
        features = ['country_or_area', 'year', 'comm_code', 
                    'trade_usd', 'weight_kg', 'quantity_name', 'category']

        # Check if all features are in the DataFrame
        if not all(col in df.columns for col in features + [target]):
            st.error("Some feature columns or target column are missing from the dataset.")
        else:
            # Prepare features and target variable
            X = df[features]
            y = df[target]

            # Convert categorical columns to string if they contain mixed types
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype(str)

            # Handle NaN values by filling with a placeholder or dropping them
            X.fillna('Unknown', inplace=True)

            # Create a ColumnTransformer with OneHotEncoder for categorical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), X.select_dtypes(include=['object']).columns.tolist())
                ],
                remainder='passthrough'  # Keep other columns unchanged
            )

            # Create a pipeline for preprocessing and model fitting
            pipeline = Pipeline([
                ('preprocessor', preprocessor),  # Preprocessing step
                ('gnb', GaussianNB())             # Gaussian Naive Bayes model
            ])

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # Fit the model
            pipeline.fit(X_train, y_train)

            # Make predictions on test data
            test_predicted = pipeline.predict(X_test)

            # Evaluate the model using confusion matrix
            cm = confusion_matrix(y_test, test_predicted)
            
            # Display confusion matrix using ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=pipeline.classes_)

            # Plotting the confusion matrix in Streamlit
            st.subheader("Confusion Matrix")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.title("Confusion Matrix")
            
            st.pyplot(fig)

            # Display classification report (optional)
            st.subheader("Classification Report")
            report = classification_report(y_test, test_predicted)
            st.text(report)
        
        
        

if __name__ == "__main__":
     main()