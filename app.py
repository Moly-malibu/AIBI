
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
    

    # elif selected_option == "Predictions":
        # import pandas as pd
        # # pipeline
        # from sklearn.preprocessing import OneHotEncoder
        # from sklearn.preprocessing import MinMaxScaler
        # from sklearn.compose import ColumnTransformer
        # st.subheader("Dataset Info", divider=True)
        # supply = pd.read_csv('commodity_trade_statistics_data.csv')
        # supply = pd.DataFrame(supply)
        # country_nan = supply[supply['weight_kg'].isnull()].groupby('country_or_area').size().sort_values(ascending=False)
        # category_nan = supply[supply['weight_kg'].isnull()].groupby('category').size().sort_values(ascending=False)
        # percentage_miss = supply.isnull().sum() * 100/len(supply)
        # supply.nunique(axis=0)
        # df_clean=supply.dropna()
        # encoder=OnehotEncoder()
        # # one-hot encode text/categorical attributes
        # country_cat_1hot = cat_encoder.fit_transform(df[['country_or_area']])
        # flow_cat_1hot = cat_encoder.fit_transform(df[['flow']])
        # category_cat_1hot = cat_encoder.fit_transform(df[['category']])
        # category_cat_1hot
        # #Feature
        # scaler = MinMaxScaler()
        # data = df[['trade_usd', 'weight_kg', 'quantity']]
        # scaled = scaler.fit_transform(data)
        # #Transformation Pipeline
        # def transform_data(num_at, cat_at, dataframe):
        #     pipeline = ColumnTransformer([
        #         ('num', MinMaxScaler(), num_at),
        #         ('cat', OneHotEncoder(handle_unknown='ignore'), cat_at), #ignore errors because dataset is huge and might encounter new categories
        #     ])
        # return pipeline.fit_transform(dataframe), pipeline
        # #Train and Test set
        # train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
        # def prepare_data(dataset, chosen_column, df_num_attribs, df_cat_attribs, test=False):
        #     df_input = dataset.drop(chosen_column, axis=1)
        #     df_output = dataset[chosen_column].copy()

            
        #     if not test:
        #         df_prepared, pipeline = transform_data(df_num_attribs, df_cat_attribs, df_input)
        #         return df_prepared, df_output, pipeline
        #     else:
        #         return df_input, df_output
        
        

if __name__ == "__main__":
     main()