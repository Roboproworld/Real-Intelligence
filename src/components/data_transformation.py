import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


class DataTransformation:
    def __init__(self, filepath="dataset/data.csv", encoding="ISO-8859-1"):
        self.df = pd.read_csv(filepath, encoding=encoding)
    
    def drop_missing_customer(self):
        """Drop rows where CustomerID is missing."""
        self.df = self.df.dropna(subset=["CustomerID"])
        return self.df

    def fill_missing_description(self):
        """Fill missing product descriptions with 'Unknown'."""
        self.df["Description"].fillna("Unknown", inplace=True)
        return self.df

    def drop_duplicates(self):
        """Remove exact duplicate rows."""
        self.df = self.df.drop_duplicates()
        return self.df

    @staticmethod
    def safe_convert_date(date_str):
        """Convert date string to datetime using multiple possible formats."""
        formats = ["%m/%d/%Y %H:%M", "%d-%m-%Y %H:%M", "%Y/%m/%d %H:%M", "%m-%d-%Y %H:%M"]
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        # If none work, return the original value.
        return date_str

    def convert_invoice_date(self):
        """Convert InvoiceDate column to datetime and report any unconverted values."""
        self.df["InvoiceDate"] = self.df["InvoiceDate"].astype(str).apply(self.safe_convert_date)
        invalid = self.df[~self.df["InvoiceDate"].apply(lambda x: isinstance(x, pd.Timestamp))]
        if not invalid.empty:
            print("Unconverted values:", invalid["InvoiceDate"].unique())
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        return self.df

    def remove_returns(self):
        """Extract rows where Quantity is negative (returns)."""
        self.df_returns = self.df[self.df["Quantity"] < 0]
        return self.df_returns

    def filter_valid_transactions(self):
        """Keep only transactions with positive Quantity and UnitPrice."""
        self.df = self.df[self.df["Quantity"] > 0]
        self.df = self.df[self.df["UnitPrice"] > 0]
        return self.df

    def add_time_features(self):
        """Add additional time-based columns."""
        self.df["Year"] = self.df["InvoiceDate"].dt.year
        self.df["Month"] = self.df["InvoiceDate"].dt.month
        self.df["Day"] = self.df["InvoiceDate"].dt.day
        self.df["Weekday"] = self.df["InvoiceDate"].dt.weekday
        self.df["Hour"] = self.df["InvoiceDate"].dt.hour
        return self.df

    def add_total_price(self):
        """Create a TotalPrice column as Quantity * UnitPrice."""
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]
        return self.df

    def get_daily_sales(self):
        """Aggregate sales by day."""
        df_daily_sales = self.df.groupby(self.df["InvoiceDate"].dt.floor("D"))["TotalPrice"].sum().reset_index()
        df_daily_sales.columns = ["Date", "TotalSales"]
        df_daily_sales["Date"] = pd.to_datetime(df_daily_sales["Date"])
        df_daily_sales = df_daily_sales.sort_values(by="Date")
        return df_daily_sales

    def get_monthly_sales(self, daily_sales_df=None):
        """Aggregate daily sales into monthly sales."""
        if daily_sales_df is None:
            daily_sales_df = self.get_daily_sales()
        df_monthly_sales = daily_sales_df.groupby(daily_sales_df["Date"].dt.to_period("M"))["TotalSales"].sum().reset_index()
        df_monthly_sales["Date"] = df_monthly_sales["Date"].dt.to_timestamp()
        return df_monthly_sales

    def plot_daily_sales(self, daily_sales_df=None):
        """Plot daily sales over time."""
        if daily_sales_df is None:
            daily_sales_df = self.get_daily_sales()
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sales_df["Date"], daily_sales_df["TotalSales"], label="Daily Sales", color="blue")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.title("Daily Sales Over Time")
        plt.legend()
        plt.show()


class Prediction(DataTransformation):
    def __init__(self, filepath="dataset/data.csv", encoding="ISO-8859-1"):
        super().__init__(filepath, encoding)
        # Run the core data transformation steps
        self.drop_missing_customer()
        self.fill_missing_description()
        self.drop_duplicates()
        self.convert_invoice_date()
        self.filter_valid_transactions()
        self.add_time_features()
        self.add_total_price()
        self.remove_returns()
        # Prepare sales aggregations
        self.daily_sales = self.get_daily_sales()
        self.monthly_sales = self.get_monthly_sales(self.daily_sales)
        # Placeholders for model and forecasts
        self.model_fit = None
        self.forecast_df = None
        self.forecast_monthly_df = None

    def train_arima_model(self, seasonal=False, trace=True):
        """Train ARIMA model using auto_arima to find the best parameters."""
        self.model_auto = auto_arima(self.daily_sales["TotalSales"], seasonal=seasonal, trace=trace)
        p, d, q = self.model_auto.order
        self.model = ARIMA(self.daily_sales["TotalSales"], order=(p, d, q))
        self.model_fit = self.model.fit()
        return self.model_fit

    def forecast_daily(self, steps=30):
        """Forecast daily sales for a specified number of days."""
        if self.model_fit is None:
            raise ValueError("Train the ARIMA model first using train_arima_model().")
        forecast = self.model_fit.forecast(steps=steps)
        future_dates = pd.date_range(start=self.daily_sales["Date"].max(), periods=steps+1, freq='D')[1:]
        self.forecast_df = pd.DataFrame({"Date": future_dates, "PredictedSales": forecast})
        return self.forecast_df

    def forecast_monthly(self, steps=12):
        """Forecast monthly sales for a specified number of months."""
        if self.model_fit is None:
            raise ValueError("Train the ARIMA model first using train_arima_model().")
        forecast = self.model_fit.forecast(steps=steps)
        future_dates = pd.date_range(start=self.daily_sales["Date"].max(), periods=steps+1, freq='M')[1:]
        self.forecast_monthly_df = pd.DataFrame({"Date": future_dates, "PredictedSales": forecast})
        return self.forecast_monthly_df

    def plot_sales_forecast(self, forecast_df=None):
        """Plot daily sales forecast."""
        if forecast_df is None:
            if self.forecast_df is None:
                raise ValueError("Daily forecast not available. Run forecast_daily() first.")
            forecast_df = self.forecast_df
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df["Date"], forecast_df["PredictedSales"], label="Forecasted Sales",
                 color="red", linestyle="dashed")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.title("Sales Forecast for Next 30 Days (ARIMA)")
        plt.legend()
        plt.show()

    def plot_monthly_sales_forecast(self, forecast_monthly_df=None):
        """Plot monthly actual and forecasted sales."""
        if forecast_monthly_df is None:
            if self.forecast_monthly_df is None:
                raise ValueError("Monthly forecast not available. Run forecast_monthly() first.")
            forecast_monthly_df = self.forecast_monthly_df
        plt.figure(figsize=(12, 6))
        plt.plot(self.monthly_sales["Date"], self.monthly_sales["TotalSales"], label="Actual Sales",
                 color="blue")
        plt.plot(forecast_monthly_df["Date"], forecast_monthly_df["PredictedSales"], label="Forecasted Sales",
                 color="red", linestyle="dashed")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.title("Monthly Sales Forecast")
        plt.legend()
        plt.show()

    def calculate_conversion_rates(self):
        """
        Calculate conversion rates based on:
          1. Basic conversion rate (orders per unique customer)
          2. Corrected conversion rate (first-time buyers vs. estimated visitors)
        """
        total_orders = self.df["InvoiceNo"].nunique()
        total_customers = self.df["CustomerID"].nunique()
        basic_conversion_rate = (total_orders / total_customers) * 100

        # Calculate first-time purchase conversion
        first_orders = self.df.groupby("CustomerID")["InvoiceDate"].min()
        first_time_buyers = len(first_orders)
        estimated_visitors = first_time_buyers * 3  # Assuming 1 in 3 visitors buys
        corrected_conversion_rate = (first_time_buyers / estimated_visitors) * 100

        print(f"Basic Conversion Rate: {basic_conversion_rate:.2f}%")
        print(f"Corrected Conversion Rate: {corrected_conversion_rate:.2f}%")
        return {"basic": basic_conversion_rate, "corrected": corrected_conversion_rate}

    def calculate_customer_lifetime_value(self):
        """
        Calculate key customer metrics and the Customer Lifetime Value (CLV):
          - Total Revenue, Average Order Value (AOV), Purchase Frequency,
            Average Customer Lifespan (in months), and CLV.
        """
        total_revenue = self.df["TotalPrice"].sum()
        total_orders = self.df["InvoiceNo"].nunique()
        total_customers = self.df["CustomerID"].nunique()
        average_order_value = total_revenue / total_orders
        purchase_frequency = total_orders / total_customers

        # Calculate average customer lifespan (in months)
        customer_lifespan = self.df.groupby("CustomerID")["InvoiceDate"].agg(["min", "max"])
        customer_lifespan["lifespan_days"] = (customer_lifespan["max"] - customer_lifespan["min"]).dt.days
        average_customer_lifespan = customer_lifespan["lifespan_days"].mean() / 30

        clv = average_order_value * purchase_frequency * average_customer_lifespan

        print(f"Average Order Value (AOV): {average_order_value:.2f}")
        print(f"Purchase Frequency: {purchase_frequency:.2f}")
        print(f"Average Customer Lifespan (months): {average_customer_lifespan:.2f}")
        print(f"Customer Lifetime Value (CLV): {clv:.2f}")
        return {"AOV": average_order_value, "PurchaseFrequency": purchase_frequency,
                "AverageLifespan": average_customer_lifespan, "CLV": clv}

    def calculate_customer_return_rate(self):
        """
        Calculate the customer return rate based on:
          - Unique customers who returned items vs. total unique customers.
        """
        customers_who_returned = self.df_returns["CustomerID"].nunique()
        total_customers = self.df["CustomerID"].nunique()
        customer_return_rate = (customers_who_returned / total_customers) * 100
        print(f"Customer Return Rate: {customer_return_rate:.2f}%")
        return customer_return_rate

    def plot_top_returned_products(self, top_n=10):
        """Plot the top N most returned products."""
        top_returns = self.df_returns['Description'].value_counts().head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_returns.values, y=top_returns.index, palette="magma")
        plt.xlabel("Number of Returns", fontsize=12)
        plt.ylabel("Product Description", fontsize=12)
        plt.title("Top {} Most Returned Products".format(top_n), fontsize=14)
        plt.show()

    def plot_top_selling_products(self, top_n=10):
        """
        Plot the top N best-selling products based on:
          - Total quantity sold and
          - Number of times sold (invoices count).
        """
        total_quantity_sold = self.df.groupby("Description")["Quantity"].sum()
        times_sold = self.df.groupby("Description")["InvoiceNo"].nunique()
        product_sales = pd.DataFrame({
            "Total Quantity Sold": total_quantity_sold,
            "Times Sold": times_sold
        })
        top_products = product_sales.sort_values(by="Total Quantity Sold", ascending=False).head(top_n)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_products.index, y=top_products["Total Quantity Sold"],
                    ax=ax1, color="dodgerblue", label="Total Quantity Sold")
        ax1.set_xlabel("Product Description", fontsize=12)
        ax1.set_ylabel("Total Quantity Sold", fontsize=12, color="dodgerblue")
        ax1.set_xticklabels(top_products.index, rotation=45, ha="right")

        ax2 = ax1.twinx()
        sns.lineplot(x=top_products.index, y=top_products["Times Sold"],
                     ax=ax2, color="red", marker="o", label="Times Sold")
        ax2.set_ylabel("Times Sold", fontsize=12, color="red")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title(f"Top {top_n} Best-Selling Products (Quantity Sold & Times Sold)", fontsize=14)
        plt.show()


