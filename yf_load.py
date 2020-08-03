# Functions to interact with yf data

def load_data(yf_list, start, end, interval):
    ''' Calls the yf api to download market data '''

    import yfinance as yf

    return yf.download(yf_list, start=start, end=end,interval=interval)


def get_close_data(df):

    return df['Adj Close']


def invert_fx(fx_list, names_dict, df):
    ''' yf defines FX as number of currency per 1 USD
    We want number of USD per 1 currency
    fx_list contains the FX we want to modify
    names_dict provides the mapping between fx_list and yf names '''

    # Adjust for FX convention

    for x in fx_list:
        df[names_dict[x]] = 1/df[names_dict[x]]
    return df

def rename_columns(df, inst_list, names_dict):
    ''' Rename the columns of dataframe from yf names to names
    Using mapping in names_dict '''
    dic = {}

    for x in inst_list:
        dic[names_dict[x]]=x
    df = df.rename(columns=dic)

    return df
