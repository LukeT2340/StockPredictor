import yfinance as yf


# Class to load data from Yahoo Finance
class DataLoader:
    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = yf.download(ticker, start=start, end=end)


    def get_data(self):
        return self.data