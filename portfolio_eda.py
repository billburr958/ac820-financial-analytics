import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import financedatabase as fd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Portfolio Analysis')

@st.cache_data()
def load_data():
    tickers = pd.concat([fd.ETFs().select().reset_index()[['symbol', 'name']],
                         fd.Equities().select().reset_index()[['symbol', 'name']]])
    ticker_symbols = tickers[tickers['symbol'].notna()]
    ticker_symbols.symbol = ticker_symbols.symbol.apply(lambda x: x.replace('^', ''))
    ticker_symbols["ticker_symbols_text"] = ticker_symbols['symbol'] + ' - ' + ticker_symbols['name']
    return ticker_symbols

ticker_symbols = load_data()

def get_company_logo(ticker):
    stock = yf.Ticker(ticker)
    website = stock.info.get("website", None)
    if website:
        return website.replace('https://www.', '')
    return None

with st.sidebar:
    selected_tickers = st.multiselect('Select Tickers', placeholder="Select Tickers", options=ticker_symbols['ticker_symbols_text'].tolist())
    selected_tickers_list = ticker_symbols[ticker_symbols['ticker_symbols_text'].isin(selected_tickers)]['symbol'].tolist()

    cols = st.columns(5)
    for index, ticker in enumerate(selected_tickers_list):
        try:
            logo_url = f"https://logo.clearbit.com/{get_company_logo(ticker)}"
            cols[index % 4].image(logo_url, use_container_width=True)
        except:
            cols[index % 4].image("https://img.icons8.com/?size=50&id=j1UxMbqzPi7n&format=png", use_container_width=True)

    cols = st.columns(2)
    date_start = cols[0].date_input('Start Date', value=dt.datetime.now() - dt.timedelta(days=365))
    date_end = cols[1].date_input('End Date', value=dt.datetime.now())

    if date_start >= date_end:
        st.error('Error: Start date must be before end date.')
    elif selected_tickers_list:
        data = yf.download(selected_tickers_list, start=date_start, end=date_end)['Close'].reset_index()

        data = data.melt(id_vars='Date', var_name='Ticker', value_name='Adjusted Closing Price')

        data_grouped = data.copy()
        data_grouped['Price_First'] = data.groupby('Ticker')['Adjusted Closing Price'].transform('first')
        data_grouped['Price_End'] = data.groupby('Ticker')['Adjusted Closing Price'].transform('last')
        data_grouped['Price_Percentage_Daily'] = data.groupby('Ticker')['Adjusted Closing Price'].pct_change()
        data_grouped['Price_Percentage'] = (data_grouped['Price_End'] - data_grouped['Price_First']) / data_grouped['Price_First']

portfolio_tab, performance_tab = st.tabs(['Portfolio', 'Performance Calculator'])

if not selected_tickers_list:
    st.info('Please select at least one ticker to begin.')
else:
    with portfolio_tab:
        st.subheader('Portfolio Overview')

        fig = px.line(data, 
                      x='Date', 
                      y='Adjusted Closing Price', 
                      color='Ticker',  # Different colors for each ticker
                      title='Portfolio Composition')

        fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
        fig.update_yaxes(tickprefix='$')

        st.plotly_chart(fig, use_container_width=True)
