import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import streamlit_authenticator as stauth
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import yaml
from yaml.loader import SafeLoader

# ------------------------
# 1. PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="D2C Brand Dashboard",
    layout="wide"
)

st.title("📊 D2C Brand Performance Dashboard")
st.markdown("Built with ❤️ by sujith  |  WTLB & WALB")

# ------------------------Auth



# Sample user credentials
usernames = ['Founder']
names = ['D2C Founder']
passwords = ['powerpass123']

# Hash passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# YAML-style config (inline)
config = {
    'credentials': {
        'usernames': {
            usernames[0]: {
                'name': names[0],
                'password': hashed_passwords[0]
            }
        }
    },
    'cookie': {
        'name': 'd2c_dashboard_cookie',
        'key': 'd2c_secret_key',
        'expiry_days': 7
    },
    'preauthorized': {
        'emails': []
    }
}

# Create Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Welcome {name} 👋')
    # ✨ Your full dashboard code starts below this point
    # ------------------------
    # 2. LOAD DATA
    # ------------------------
    @st.cache_data
    def load_data():
        df = pd.read_csv("d2c_data.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    df = load_data()


    # ------------------------
    # 3. SIDEBAR FILTERS
    # ------------------------
    with st.sidebar:
        st.header("🔍 Filters")
        
        start_date = st.date_input("Start Date", df["Date"].min().date())
        end_date = st.date_input("End Date", df["Date"].max().date())

        products = st.multiselect("Select Product(s)", options=df["Top Product"].unique(), default=df["Top Product"].unique())

    # Apply filters
    filtered_df = df[
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date)) &
        (df["Top Product"].isin(products))
    ]


    # ------------------------
    # 4. KPI METRICS
    # ------------------------
    total_revenue = filtered_df["Revenue"].sum()
    total_orders = filtered_df["Orders"].sum()
    avg_cac = round(filtered_df["CAC"].mean(), 2)

    st.markdown("## 📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"₹ {total_revenue:,.0f}")
    col2.metric("📦 Total Orders", total_orders)
    col3.metric("🎯 Average CAC", f"₹ {avg_cac:,.2f}")

    st.divider()

    # ------------------------
    # 5. VISUALIZATIONS
    # ------------------------

    # Revenue Trend Line Chart
    st.markdown("### 📊 Daily Revenue Trend")
    revenue_chart = alt.Chart(filtered_df).mark_line(point=True).encode(
        x='Date:T',
        y='Revenue:Q',
        tooltip=['Date:T', 'Revenue']
    ).properties(height=300)
    st.altair_chart(revenue_chart, use_container_width=True)

    # CAC Over Time Area Chart
    st.markdown("### 💸 CAC Over Time")
    cac_chart = alt.Chart(filtered_df).mark_area(opacity=0.6).encode(
        x='Date:T',
        y='CAC:Q',
        tooltip=['Date:T', 'CAC']
    ).properties(height=300)
    st.altair_chart(cac_chart, use_container_width=True)

    # Top 5 Products by Revenue Bar Chart
    st.markdown("### 🏆 Top Products by Revenue")
    top_products = (
        filtered_df.groupby("Top Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    bar_chart = alt.Chart(top_products).mark_bar().encode(
        x='Top Product:N',
        y='Revenue:Q',
        tooltip=['Top Product', 'Revenue']
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # Pie Chart: Product Share by Orders
    st.markdown("### 🥧 Product Share by Orders")
    product_share = (
        filtered_df.groupby("Top Product")["Orders"]
        .sum()
        .reset_index()
    )

    st.dataframe(product_share, use_container_width=True)

    # ------------------------
    # 6. RAW DATA
    # ------------------------
    with st.expander("📄 View Raw Data"):
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

    # ------------------------
    # 7. DOWNLOAD OPTION
    # ------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data", data=csv, file_name="filtered_d2c_data.csv", mime="text/csv")

    ################################################################################################
    st.header("📈 Revenue Forecasting")

    # Prepare data for Prophet
    df_forecast = df[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'})

    # Initialize and fit the model
    model = Prophet(daily_seasonality=True)
    model.fit(df_forecast)

    # Make future dataframe
    future = model.make_future_dataframe(periods=15)

    # Forecast
    forecast = model.predict(future)

    # Plot forecast using Plotly
    st.subheader("🔮 Forecast for Next 15 Days")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast)

    # Show forecast data
    st.subheader("📅 Forecast Data Preview")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15))
    ################################################################################################

elif authentication_status is False:
    st.error('Username or password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')


# ------------------------





################################################################################################
# import streamlit as st
# import pandas as pd
# import altair as alt
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Page config
# st.set_page_config(page_title="D2C Performance Dashboard", layout="wide")

# # Load your Google Sheet export (convert to .csv first and save locally as 'd2c_data.csv')
# df = pd.read_csv("d2c_data.csv")  # Make sure the CSV is in the same folder

# # Convert 'Date' to datetime format
# df['Date'] = pd.to_datetime(df['Date'])

# # Title
# st.title("📊 D2C Brand Performance Dashboard")

# #######################################
# st.subheader("🗂️ Filter the Data")

# # Convert 'Date' to datetime just in case
# df['Date'] = pd.to_datetime(df['Date'])

# # Date Range Filter
# start_date = st.date_input("Start Date", df['Date'].min())
# end_date = st.date_input("End Date", df['Date'].max())

# # Product Filter
# selected_products = st.multiselect(
#     "Choose Products", df['Top Product'].unique(), default=df['Top Product'].unique()
# )

# # Apply Filters
# filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) &
#                  (df['Date'] <= pd.to_datetime(end_date)) &
#                  (df['Top Product'].isin(selected_products))]

# st.dataframe(filtered_df)
############################################


# KPIs
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Orders", f"{filtered_df['Orders'].sum():,}")
# col2.metric("Total Revenue", f"₹{filtered_df['Revenue'].sum():,.0f}")

# # Calculate Ad Spend Efficiency (Revenue per ₹1 Ad Spend)
# efficiency = (filtered_df['Revenue'].sum() / filtered_df['Ad Spend'].sum())
# col3.metric("Ad Spend Efficiency", f"₹{efficiency:.2f} per ₹1")
################################################################################################
# Charts
# st.subheader("📈 Revenue Over Time")
# line_chart = alt.Chart(df).mark_line().encode(
#     x='Date:T',
#     y='Revenue:Q',
#     tooltip=['Date:T', 'Revenue:Q']
# ).properties(height=300)
# st.altair_chart(line_chart, use_container_width=True)

# # DAILY REVENUE TREND
# st.subheader("📈 Daily Revenue Trend")
# fig1, ax1 = plt.subplots()
# sns.lineplot(data=df, x='Date', y='Revenue', ax=ax1)
# ax1.set_title("Revenue Over Time")
# ax1.set_xlabel("Date")
# ax1.set_ylabel("Revenue (₹)")
# st.pyplot(fig1)

# # TOP PRODUCTS SOLD
# st.subheader("📊 Top Products Sold")
# top_products = df['Top Product'].value_counts()
# fig2, ax2 = plt.subplots()
# sns.barplot(x=top_products.values, y=top_products.index, ax=ax2)
# ax2.set_title("Top Products Sold")
# ax2.set_xlabel("No. of Days as Top Seller")
# st.pyplot(fig2)

# # DAILY CAC TREND
# st.subheader("💸 Daily CAC Trend")
# fig3, ax3 = plt.subplots()
# sns.lineplot(data=df, x='Date', y='CAC', ax=ax3, color='orange')
# ax3.set_title("Customer Acquisition Cost Over Time")
# ax3.set_xlabel("Date")
# ax3.set_ylabel("CAC (₹)")
# st.pyplot(fig3)
###############################################################################################
# KPIs
# st.subheader("💎 Top Products by Total Revenue")
# product_revenue = df.groupby('Top Product')['Revenue'].sum().sort_values(ascending=False)
# st.bar_chart(product_revenue)


# col31, col32, col33, col34 = st.columns(4)
# col31.metric("MIN CAC", f"{df['CAC'].min():,}")
# col32.metric("MAX CAC", f"₹{df['CAC'].max():,.0f}")
# col33.metric("MIN Revenue", f"{df['Revenue'].min():,}")
# col34.metric("MAX Revenue", f"₹{df['Revenue'].max():,.0f}")


# st.subheader("📅 Best Performing Days (Lowest CAC & Highest Revenue)")
# best_cac_days = df.sort_values('CAC').head(5)
# st.dataframe(best_cac_days[['Date', 'CAC', 'Revenue', 'Orders', 'Top Product']])


# st.subheader("📅 Worst Performing Days (Highest CAC & Lowest Revenue)")
# worst_cac_days = df.sort_values('CAC',ascending=False).head(5)
# st.dataframe(worst_cac_days[['Date', 'CAC', 'Revenue', 'Orders', 'Top Product']])
 
################################################################################################

# st.subheader("📈 Product-Wise Revenue vs CAC")

# product_metrics = filtered_df.groupby('Top Product').agg({
#     'Revenue': 'sum',
#     'CAC': 'mean',
#     'Orders': 'sum'
# }).sort_values('Revenue', ascending=False)

# st.dataframe(product_metrics)

# st.bar_chart(product_metrics[['Revenue', 'CAC']])


# st.subheader("⚙️ Conversion Efficiency (Revenue per Order)")

# filtered_df['Revenue per Order'] = filtered_df['Revenue'] / filtered_df['Orders']
# avg_conversion_efficiency = filtered_df['Revenue per Order'].mean()

# st.metric("Average Revenue per Order", f"₹{avg_conversion_efficiency:,.2f}")

# st.line_chart(filtered_df[['Date', 'Revenue per Order']].set_index('Date'))

################################################################################################