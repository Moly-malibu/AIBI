
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
    options = ["Commodity: Statistics", "Visualization", "Category", "Commodity","Predictions"]

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
        
        # Optional: Display additional insights based on p-value
        if p_value < 0.05:
            st.write("The result is statistically significant (p < 0.05).")
        else:
            st.write("The result is not statistically significant (p >= 0.05).")

    elif selected_option == "Visualization":
        st.subheader("Dataset Visualization", divider=True)
        st.write("Select different variables to analyze: Categories")
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

    elif selected_option == "Commodity":
        import streamlit as st
        import pandas as pd
        import plotly.express as px

        # Sample dataset creation
        st.subheader("Export and Import by Commodity", divider=True)
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        df = pd.DataFrame(supply)

        # Selectbox for grouping columns
        group_by_country = st.selectbox("Select Flow", options=df['flow'].unique())
        group_by_product = st.selectbox("Select category", options=df['category'].unique())

        # Grouping the data based on user selection
        grouped_data = df.groupby(['flow']).sum().reset_index()

        # Filter based on user selection
        filtered_data = grouped_data[
            (grouped_data['flow'] == group_by_country) & 
            (grouped_data['category'] == group_by_product)
        ]

        # Display the filtered data
        st.write("Grouped Data:")
        st.dataframe(filtered_data)

        # Create a bar chart using Plotly
        fig = px.bar(filtered_data, x='flow', y='category',
                    title=f'category for {group_by_product} in {group_by_country}',
                    labels={'category': 'flow', 'category': 'category'},
                    color='category')

        # Show the plot in Streamlit
        st.plotly_chart(fig)
    
    
    
    elif selected_option == "Category":
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        st.title("Commodity Trade by Country")
        # Load the dataset
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        
        # Selectbox for grouping by country or area
        selected_country = st.selectbox("Select Country or Area", options=supply['country_or_area'].unique())
        selected_commodity = st.selectbox("Select Commodity", options=supply['commodity'].unique())

        # Filter the data based on user selection
        filtered_data = supply[(supply['country_or_area'] == selected_country) & 
                            (supply['commodity'] == selected_commodity)]

        # Grouping the data by year and summing up trade values
        grouped_data = filtered_data.groupby('year').agg({'trade_usd': 'sum'}).reset_index()

        # Display the grouped data
        st.write("Grouped Data:")
        st.dataframe(grouped_data)

        # Create a bar chart using Plotly
        fig = px.bar(grouped_data, x='year', y='trade_usd',
                    title=f'Trade Value for {selected_commodity} in {selected_country}',
                    labels={'trade_usd': 'Total Trade Value (USD)', 'year': 'Year'})

        # Show the plot in Streamlit
        st.plotly_chart(fig)
    

    elif selected_option == "Predictions":
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from tensorflow import keras
        from tensorflow.keras import layers
        import streamlit as st
        import plotly.express as px
        
        st.subheader("Dataset Info", divider=True)
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        
        # Step 2: Prepare Dataset
        Feature1 = ['year', 'comm_code', 'trade_usd', 'weight_kg', 'quantity']
        Feature2 = ['country_or_area', 'commodity', 'quantity_name', 'category']
        Target = ['flow']
        
        # Check if all features exist in the DataFrame
        missing_features = set(Feature1 + Feature2) - set(supply.columns)
        if missing_features:
            st.error(f"Missing features in dataset: {missing_features}")
        else:
        
            # Combine features for modeling
            X = supply[Feature1 + Feature2]
            y = supply[Target].values

            # One-hot encode categorical variables
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = encoder.fit_transform(X[Feature2])

            # Combine numerical features with encoded categorical features
            X_final = np.hstack((X[Feature1].values, X_encoded))

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
            
            #Step 3: Build and Train the Deep Learning Model
            # Build the neural network model
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
                layers.Dense(32, activation='relu'),                                   # Hidden layer
                layers.Dense(1, activation='sigmoid')                                 # Output layer for binary classification (adjust if needed)
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model
            model.fit(X_train, y_train, epochs=100, batch_size=32)
            
            
            #Step 4: Evaluate Model Accuracy
            # Evaluate the model on test data
            loss, accuracy = model.evaluate(X_test, y_test)
            st.write(f'Test Accuracy: {accuracy:.4f}')
            
            #Step 5: Visualize Results
            # Streamlit app layout
            st.title("Commodity Trade Statistics Deep Learning Model")

            # Display accuracy
            st.write(f"Model Accuracy: {accuracy:.4f}")

            # Create a DataFrame for predictions and actual values for comparison
            predictions = model.predict(X_test).flatten()
            predicted_classes = (predictions > 0.5).astype(int)

            results_df = pd.DataFrame({
                'Actual': y_test.flatten(),
                'Predicted': predicted_classes,
            })

            # Display results DataFrame
            st.write("Predictions vs Actual:")
            st.dataframe(results_df)

            # Create a bar chart for actual vs predicted values using Plotly
            fig = px.bar(results_df, x=results_df.index,
                        y=['Actual', 'Predicted'],
                        title="Actual vs Predicted Values",
                        labels={'value': 'Flow', 'index': 'Index'},
                        barmode='group')

            st.plotly_chart(fig)
        
        

if __name__ == "__main__":
     main()