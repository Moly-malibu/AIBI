
import streamlit as st

#https://media.istockphoto.com/id/1154350410/es/foto/abstracto-de-forma-lisa-para-la-idea-arquitect%C3%B3nica-l%C3%ADnea-curva-fondo-blanco-con-forma-libre.jpg?s=612x612&w=0&k=20&c=odwLLTQeQMzPQsLusqS1mONxZiUIq8DW_0s0J8qNFz0=
st.markdown("<h1 style='text-align: center; color: #002967;'>🎈Business Intelligence (BI)</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: #002967;'>Agriculture by Countries</h1>", unsafe_allow_html=True)
def main():
    pages = {
            "Commodity": Commodity,
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
    options = ["Commodity: Statistics", "Visualization", "Category"]

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

        country_nan = supply[supply['weight_kg'].isnull()].groupby('country_or_area').size().sort_values(ascending=False)
        category_nan = supply[supply['weight_kg'].isnull()].groupby('category').size().sort_values(ascending=False)
        percentage_miss = supply.isnull().sum() * 100/len(supply)
        supply.nunique(axis=0)
        supply=supply.dropna()
        
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
    
    elif selected_option == "Category":
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        st.title("Commodity Trade by Country")
        # Load the dataset
        supply = pd.read_csv('commodity_trade_statistics_data.csv')
        supply = pd.DataFrame(supply)

        country_nan = supply[supply['year'].isnull()].groupby('trade_usd').size().sort_values(ascending=False)
        category_nan = supply[supply['country_or_area'].isnull()].groupby('commodity').size().sort_values(ascending=False)
        percentage_miss = supply.isnull().sum() * 100/len(supply)
        supply.nunique(axis=0)
        supply=supply.dropna()
        supply = pd.DataFrame(supply)
        
        
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
    

    # elif selected_option == "Predictions":
    #     import streamlit as st
    #     import pandas as pd
    #     import numpy as np
    #     from sklearn.model_selection import train_test_split
    #     from sklearn.preprocessing import OneHotEncoder
    #     from sklearn.model_selection import train_test_split, cross_val_score
    #     from sklearn.preprocessing import MinMaxScaler
    #     from sklearn.compose import ColumnTransformer
    #     from sklearn.ensemble import RandomForestClassifier
    #     from sklearn.ensemble import RandomForestRegressor
    #     from sklearn.metrics import mean_squared_error
    #     from sklearn.metrics import r2_score
    #     from sklearn.model_selection import cross_val_score
    #     from sklearn.metrics import precision_score
    #     from sklearn.metrics import f1_score
    #     from sklearn.metrics import plot_confusion_matrix
    #     import matplotlib.pyplot as plt


    #     # Load the dataset
    #     supply = pd.read_csv('commodity_trade_statistics_data.csv')
    #     country_nan = supply[supply['year'].isnull()].groupby('trade_usd').size().sort_values(ascending=False)
    #     category_nan = supply[supply['country_or_area'].isnull()].groupby('commodity').size().sort_values(ascending=False)
    #     percentage_miss = supply.isnull().sum() * 100/len(supply)
    #     supply.nunique(axis=0)
    #     supply=supply.dropna()
    #     df = pd.DataFrame(supply)
        
    #     #Moodel
    #     cat_encoder = OneHotEncoder()
    #     # one-hot encode text/categorical attributes
    #     country_cat_1hot = cat_encoder.fit_transform(df[['country_or_area']])
    #     flow_cat_1hot = cat_encoder.fit_transform(df[['flow']])
    #     category_cat_1hot = cat_encoder.fit_transform(df[['category']])
        
    #     scaler = MinMaxScaler()
    #     data = df[['trade_usd', 'weight_kg', 'quantity']]
    #     scaled = scaler.fit_transform(data)
        
    #     def transform_data(num_at, cat_at, dataframe):
    #         pipeline = ColumnTransformer([
    #             ('num', MinMaxScaler(), num_at),
    #             ('cat', OneHotEncoder(handle_unknown='ignore'), cat_at), #ignore errors because dataset is huge and might encounter new categories
    #         ])
    #         return pipeline.fit_transform(dataframe), pipeline
    
    #     # utility functions to improve prints
    #     def display_scores(scores):
    #         st.write(f"Scores: {np.round(scores/1000000, decimals=2)}")
    #         st.write(f"RMSE: {to_millions(scores.mean()):.2f}")
    #         st.write(f"Standard deviation: {scores.std()/1000000:.2f}")
    #     def to_millions(usd):
    #         return round(usd/1000000, 2)
        
    #     train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
        
    #     def prepare_data(dataset, chosen_column, df_num_attribs, df_cat_attribs, test=False):
    #         df_input = dataset.drop(chosen_column, axis=1)
    #         df_output = dataset[chosen_column].copy()
    #         if not test:
    #             df_prepared, pipeline = transform_data(df_num_attribs, df_cat_attribs, df_input)
    #             return df_prepared, df_output, pipeline
    #         else:
    #             return df_input, df_output
            
    #     df_prepared, df_output, pipeline = prepare_data(train_set,
    #                       'trade_usd',
    #                       ['weight_kg', 'quantity'],
    #                       ['country_or_area', 'flow', 'category'])
        
    #     # Mean of 'trade_usd' in millions
    #     st.write(f"Mean of 'Trade Usd': {round(df['trade_usd'].mean()/1000000, 2)} millions")

    #     # Initialize the Random Forest Regressor
    #     rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    #     # Fit the model
    #     rf_reg.fit(df_prepared, df_output)

    #     # Predictions and evaluation
    #     rf_predictions = rf_reg.predict(df_prepared)
    #     rf_rmse = np.sqrt(mean_squared_error(df_output, rf_predictions))
    #     st.write(f'Random Forest RMSE: {to_millions(rf_rmse)}')

    #     # Cross-validation scores
    #     rf_scores = cross_val_score(rf_reg, df_prepared, df_output, scoring="neg_mean_squared_error", cv=10)
    #     rf_rmse_scores = np.sqrt(-rf_scores)
    #     display_scores(rf_rmse_scores)
        
    #     rf_reg = RandomForestRegressor(C=100, class_weight='balanced')
    #     rf_reg.fit(df_prepared, df_output)

    #     # predict train and test
    #     X_test_prepared = pipeline.transform(X_test)
    #     test_predicted = lr_clf.predict(X_test_prepared)

    #     precision = precision_score(y_test, test_predicted, average='weighted')
    #     accuracy = rf_reg.score(X_test_prepared, y_test)
    #     f1_score_ = f1_score(y_test, test_predicted, average='weighted')

    #     st.write(f'Accuracy: {round(accuracy, 2)}')
    #     st.write(f'Precision: {round(precision, 2)}')
    #     st.write(f'F1: {round(f1_score_, 2)}')
        
    #     supply = supply.dropna(subset=['trade_usd'])  # Drop rows with NaN in target variable
    #     X = supply.drop(columns=['trade_usd'])  # Features
    #     y = supply['trade_usd']  # Target variable

    #     # Split the dataset into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        
    #     #Model Random Forest Classifier
    #     rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Use RandomForestRegressor for regression tasks

    #     # Fit the model
    #     rf_clf.fit(X_train, y_train)

    #     # Predictions on test data
    #     y_pred = rf_clf.predict(X_test)

    #     # Evaluate performance metrics
    #     accuracy = accuracy_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Adjust for classification tasks
    #     f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    #     st.write(f'Accuracy: {round(accuracy * 100, 2)}%')
    #     st.write(f'Precision: {round(precision * 100, 2)}%')
    #     st.write(f'F1 Score: {round(f1 * 100, 2)}%')

    #     # Plot confusion matrix using ConfusionMatrixDisplay
    #     fig, ax = plt.subplots()
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #     disp.plot(ax=ax)
    #     plt.title("Confusion Matrix")
    #     st.pyplot(fig)

    #     # Optional: Feature importance visualization
    #     importance = rf_clf.feature_importances_
    #     feature_names = X.columns

    #     # Create a DataFrame for feature importances and sort them
    #     feature_importance_df = pd.DataFrame({'flow': feature_names, 'Importance': importance})
    #     feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    #     # Plot feature importance
    #     st.bar_chart(feature_importance_df.set_index('flow'))
        
       
        
       
        
        

if __name__ == "__main__":
     main()