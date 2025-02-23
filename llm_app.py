import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()

# Import our modules for ingestion and transformation
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Prediction

# LangChain imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# Set OpenAI API key (make sure your .env file has OPENAI_API_KEY set)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def build_context(prediction_obj):
    """
    Build a summary context of the sales data from key metrics.
    """
    # Get daily sales summary
    daily_sales_df = prediction_obj.get_daily_sales()
    num_days = daily_sales_df.shape[0]
    max_sale = daily_sales_df["TotalSales"].max()
    min_sale = daily_sales_df["TotalSales"].min()

    # Conversion rates and customer metrics
    conv_rates = prediction_obj.calculate_conversion_rates()
    clv_metrics = prediction_obj.calculate_customer_lifetime_value()

    # Top returned products (top 3)
    top_returns = prediction_obj.df_returns['Description'].value_counts().head(3).index.tolist()

    # Top selling products (top 3)
    total_quantity_sold = prediction_obj.df.groupby("Description")["Quantity"].sum()
    top_products = total_quantity_sold.sort_values(ascending=False).head(3).index.tolist()

    context = (
        f"The sales dataset spans {num_days} days, with daily sales ranging from {min_sale:.2f} "
        f"to {max_sale:.2f}. The basic conversion rate is {conv_rates['basic']:.2f}% and the "
        f"corrected conversion rate is {conv_rates['corrected']:.2f}%. The average order value is "
        f"{clv_metrics['AOV']:.2f}, purchase frequency is {clv_metrics['PurchaseFrequency']:.2f}, "
        f"and the customer lifetime value is {clv_metrics['CLV']:.2f}. Top returned products include: "
        f"{', '.join(top_returns)}. Top selling products include: {', '.join(top_products)}."
    )
    return context


def get_llm_response(question, data_context):
    """
    Uses LangChain's OpenAI LLM to answer the question given the data context.
    """
    prompt_template = PromptTemplate(
        input_variables=["data_context", "question"],
        template=(
            "You are a sales data expert. Below is a summary of the sales data:\n"
            "{data_context}\n\n"
            "Answer the following question based on this context:\n"
            "{question}\n"
        )
    )
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"data_context": data_context, "question": question})
    return response


def main():
    st.title("Sales Analytics Dashboard with Chatbot")
    st.write(
        "This dashboard integrates data ingestion, transformation, forecasting, and interactive plots. "
        "You can also ask questions about the data and get answers powered by OpenAI."
    )

    # --- Data Ingestion ---
    
    ingestion = DataIngestion()
    try:
        result = ingestion.initiate_data_ingestion()
        raw_data_path = result[0] if isinstance(result, tuple) else result
        
    except Exception as e:
        st.error(f"Data ingestion failed: {e}")
        return

    # --- Data Transformation & Prediction ---
    
    with st.spinner("Processing data and training model..."):
        prediction_obj = Prediction(filepath=raw_data_path)
        prediction_obj.train_arima_model()
        prediction_obj.forecast_daily()
        prediction_obj.forecast_monthly()
    st.success("Data processed and forecasts generated.")

    # --- Create and Display Plots ---
    st.header("Sales Analytics Plots")
    
    # Daily Sales Plot
    daily_sales_df = prediction_obj.get_daily_sales()
    fig_daily, ax_daily = plt.subplots()
    ax_daily.plot(daily_sales_df["Date"], daily_sales_df["TotalSales"], label="Daily Sales", color="blue")
    ax_daily.set_title("Daily Sales Over Time")
    ax_daily.set_xlabel("Date")
    ax_daily.set_ylabel("Total Sales")
    ax_daily.legend()
    st.subheader("Daily Sales")
    st.pyplot(fig_daily)

    # Daily Sales Forecast Plot (Next 30 Days)
    fig_forecast, ax_forecast = plt.subplots()
    forecast_df = prediction_obj.forecast_df
    ax_forecast.plot(forecast_df["Date"], forecast_df["PredictedSales"],
                     label="Forecasted Sales", color="red", linestyle="dashed")
    ax_forecast.set_title("Daily Sales Forecast (Next 30 Days)")
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("Sales")
    ax_forecast.legend()
    st.subheader("Daily Sales Forecast")
    st.pyplot(fig_forecast)

    # Monthly Sales Forecast Plot (Next 12 Months)
    fig_monthly, ax_monthly = plt.subplots()
    monthly_sales_df = prediction_obj.monthly_sales
    ax_monthly.plot(monthly_sales_df["Date"], monthly_sales_df["TotalSales"], label="Actual Sales", color="blue")
    forecast_monthly_df = prediction_obj.forecast_monthly_df
    ax_monthly.plot(forecast_monthly_df["Date"], forecast_monthly_df["PredictedSales"],
                    label="Forecasted Sales", color="red", linestyle="dashed")
    ax_monthly.set_title("Monthly Sales Forecast (Next 12 Months)")
    ax_monthly.set_xlabel("Date")
    ax_monthly.set_ylabel("Sales")
    ax_monthly.legend()
    st.subheader("Monthly Sales Forecast")
    st.pyplot(fig_monthly)

    # Top 10 Best-Selling Products Plot (Quantity Sold & Times Sold)
    fig_top_selling, ax_top_selling = plt.subplots()
    total_quantity_sold = prediction_obj.df.groupby("Description")["Quantity"].sum()
    times_sold = prediction_obj.df.groupby("Description")["InvoiceNo"].nunique()
    product_sales = pd.DataFrame({
        "Total Quantity Sold": total_quantity_sold,
        "Times Sold": times_sold
    })
    top_products = product_sales.sort_values(by="Total Quantity Sold", ascending=False).head(10)
    sns.barplot(x=top_products.index, y=top_products["Total Quantity Sold"],
                ax=ax_top_selling, color="dodgerblue", label="Total Quantity Sold")
    ax_top_selling.set_xlabel("Product Description")
    ax_top_selling.set_ylabel("Total Quantity Sold", color="dodgerblue")
    ax_top_selling.set_xticklabels(top_products.index, rotation=45, ha="right")
    
    # Second axis for Times Sold
    ax2 = ax_top_selling.twinx()
    sns.lineplot(x=top_products.index, y=top_products["Times Sold"],
                 ax=ax2, color="red", marker="o", label="Times Sold")
    ax2.set_ylabel("Times Sold", color="red")
    ax_top_selling.set_title("Top 10 Best-Selling Products")
    st.subheader("Best-Selling Products")
    st.pyplot(fig_top_selling)

    # --- Chatbot Section with LangChain LLM ---
    st.header("Chat with DataBot")
    
    # Build the context summary for the LLM
    data_context = build_context(prediction_obj)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Ask a question about the data:")
    if st.button("Send") and user_input:
        with st.spinner("Generating answer..."):
            response = get_llm_response(user_input, data_context)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("DataBot", response))
        st.write("DataBot:", response)
        

    if st.session_state.chat_history:
        for sender, message in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**DataBot:** {message}")


if __name__ == "__main__":
    main()
