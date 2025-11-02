import datetime as dt
from flask import Flask, render_template, request
import numpy_financial as npf
import yfinance as yf
import pandas as pd
from datetime import datetime
import traceback

app = Flask(__name__)

def Average(lst):
    return sum(lst) / len(lst) if len(lst) > 0 else 0.0

def complex_to_float(z, method='real', tol=1e-9, default=0.0):
    """
    Convert z to float.
    method: 'real'  -> use real part (allowed if imag small or ignored)
            'abs'   -> use magnitude abs(z)
            'strict'-> require imag approximately 0, else return default
    tol: tolerance for imaginary part when using 'real' or 'strict'
    """
    try:
        if z is None:
            return float(default)
        if isinstance(z, complex):
            if method == 'abs':
                return float(abs(z))
            imag = abs(z.imag)
            if method == 'real':
                # if imaginary part is tiny accept real part
                if imag <= tol:
                    return float(z.real)
                return float(z.real)  # or choose to fallback to abs(z)
            if method == 'strict':
                if imag <= tol:
                    return float(z.real)
                return float(default)
        # already real-ish (int/float/numpy scalar)
        return float(z)
    except Exception:
        return float(default)
# ------------------------------
# Financial Data Fetching Functions
# ------------------------------

# EPS History
def get_eps_history(ticker):
    t = yf.Ticker(ticker)
    eps_list = []
    try:
        income_stmt = t.income_stmt
        if income_stmt is None or 'Diluted EPS' not in income_stmt.index:
            return []
        eps_series = income_stmt.loc['Diluted EPS']
        for date, eps in eps_series.items():
            if pd.notna(eps):
                eps_list.append({'year': str(date.year), 'eps': f"{eps:.2f}"})
        eps_list.sort(key=lambda x: int(x['year']), reverse=True)
    except Exception as e:
        print(f"EPS Error: {e}")
    return eps_list

# Dividend History
def get_dividend_history(ticker):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    # Exclude current year, get previous 10 years
    years = range(current_year - 1, current_year - 11, -1)
    try:
        dividends = t.dividends
        if dividends.empty:
            return [{'year': str(y), 'dividend': '0.00'} for y in years]
        if dividends.index.tz is not None:
            dividends.index = dividends.index.tz_localize(None)
        dividends_by_year = dividends.groupby(dividends.index.year).sum()
        return [
            {'year': str(year), 'dividend': f"{dividends_by_year.get(year, 0.0):.2f}"}
            for year in years
        ]
    except Exception as e:
        print(f"Dividend Error: {e}")
        return [{'year': str(y), 'dividend': '0.00'} for y in years]
    
# Net Income History
def get_net_income_history(ticker):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    net_income_data = {}
    try:
        income_stmt = t.income_stmt
        if income_stmt is None or 'Net Income' not in income_stmt.index:
            return []
        ni_series = income_stmt.loc['Net Income']
        for date, val in ni_series.items():
            if pd.notna(val):
                net_income_data[date.year] = str(int(round(val, 0)))
    except Exception as e:
        print(f"Net Income Error: {e}")
    # Exclude current year, get previous 4 years
    start_year = current_year - 4
    return [
        {'year': str(y), 'net_income': net_income_data.get(y, '0.00')}
        for y in range(current_year - 1, start_year - 1, -1)
    ]

# Current Liabilities History
def get_current_liabilities_history(ticker):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    data = {}
    try:
        bs = t.balance_sheet
        if bs is None or 'Current Liabilities' not in bs.index:
            return []
        row = bs.loc['Current Liabilities']
        for date, val in row.items():
            if pd.notna(val):
                data[date.year] = str(int(round(val, 0)))
    except Exception as e:
        print(f"Current Liabilities Error: {e}")
    start_year = min(data.keys()) if data else current_year - 9
    # Start from last year (current_year - 1)
    return [
        {'year': str(y), 'current_liabilities': data.get(y, '0.00')}
        for y in range(current_year - 1, start_year - 1, -1)
    ]

# Cash and Cash Equivalents History
def get_cash_and_cash_equivalents_history(ticker):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    data = {}
    try:
        bs = t.balance_sheet
        if bs is None or 'Cash Cash Equivalents And Short Term Investments' not in bs.index:
            return []
        row = bs.loc['Cash Cash Equivalents And Short Term Investments']
        for date, val in row.items():
            if pd.notna(val):
                data[date.year] = str(int(round(val, 0)))
    except Exception as e:
        print(f"Cash Error: {e}")
    start_year = min(data.keys()) if data else current_year - 9
    # Start from last year (current_year - 1)
    return [
        {'year': str(y), 'cash_and_cash_equivalents': data.get(y, '0.00')}
        for y in range(current_year - 1, start_year - 1, -1)
    ]

# Total Liabilities History
def get_total_liabilities_history(ticker):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    data = {}
    try:
        bs = t.balance_sheet
        if bs is None or 'Total Liabilities Net Minority Interest' not in bs.index:
            return []
        row = bs.loc['Total Liabilities Net Minority Interest']
        for date, val in row.items():
            if pd.notna(val):
                data[date.year] = str(int(round(val, 0)))
    except Exception as e:
        print(f"Total Liabilities Error: {e}")
    start_year = min(data.keys()) if data else current_year - 9
    # Start from last year (current_year - 1)
    return [
        {'year': str(y), 'total_liabilities': data.get(y, '0.00')}
        for y in range(current_year - 1, start_year - 1, -1)
    ]

# Total Assets History
def get_total_assets_history(ticker):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    data = {}
    try:
        bs = t.balance_sheet
        if bs is None or 'Total Assets' not in bs.index:
            return []
        row = bs.loc['Total Assets']
        for date, val in row.items():
            if pd.notna(val):
                data[date.year] = str(int(round(val, 0)))
    except Exception as e:
        print(f"Total Assets Error: {e}")
    start_year = min(data.keys()) if data else current_year - 9
    # Start from last year (current_year - 1)
    return [
        {'year': str(y), 'total_assets': data.get(y, '0.00')}
        for y in range(current_year - 1, start_year - 1, -1)
    ]

# Price History
def get_price_history(ticker, years_back=5):
    t = yf.Ticker(ticker)
    current_year = datetime.now().year
    price_list = []
    try:
        start_date = f"{current_year - years_back}-01-01"
        end_date = f"{current_year + 1}-01-01"
        hist = t.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError("No price data")
        # For previous years, get last close in December
        for year in range(current_year - years_back + 1, current_year):
            dec = hist[(hist.index.year == year) & (hist.index.month == 12)]
            price = dec['Close'].iloc[-1] if not dec.empty else 0.0
            price_list.append({'year': str(year), 'price': f"{price:.2f}"})
        # For current year, get the last available close
        current_year_data = hist[hist.index.year == current_year]
        price = current_year_data['Close'].iloc[-1] if not current_year_data.empty else 0.0
        price_list.append({'year': str(current_year), 'price': f"{price:.2f}"})
        price_list.sort(key=lambda x: int(x['year']), reverse=True)
    except Exception as e:
        print(f"Price Error: {e}")
        for y in range(current_year, current_year - years_back, -1):
            price_list.append({'year': str(y), 'price': '0.00'})
    return price_list

# ------------------------------
# Core Fundamentals Calculator
# ------------------------------

def fundamentals(symbol, nb_shares, RealDiscountRate, AverageInflation):
    try:
        symbol = symbol.strip().upper()
        eps_data = get_eps_history(symbol)
        dividends_data = get_dividend_history(symbol)
        net_income_data = get_net_income_history(symbol)
        price_data = get_price_history(symbol, years_back=5)
        current_liab = get_current_liabilities_history(symbol)
        cash_equiv = get_cash_and_cash_equivalents_history(symbol)
        total_liab = get_total_liabilities_history(symbol)
        total_assets = get_total_assets_history(symbol)
        if len(eps_data) < 4 or len(dividends_data) < 4 or len(net_income_data) < 4 or len(price_data) < 1:
            raise ValueError("Insufficient data")

        # RORE 3 Years
        total_eps_3 = sum(float(eps_data[i]['eps']) for i in range(3))
        total_div_3 = sum(float(dividends_data[i]['dividend']) for i in range(3))
        retained_3 = total_eps_3 - total_div_3
        eps_lastyear = float(eps_data[0]['eps'])      # last year
        eps_3yago = float(eps_data[2]['eps'])        # 3 years ago (start of 3-year period)
        RORE_3years = ((eps_lastyear - eps_3yago) / retained_3) * 100 if retained_3 != 0 else 0.0

        # Net Income Growth (3 years)
        NetIncomeGrowth_3years = ((float(net_income_data[0]['net_income']) - float(net_income_data[2]['net_income'])) / float(net_income_data[2]['net_income']) * 100 if float(net_income_data[2]['net_income']) != 0 else 0.0)
        NetIncomeGrowth_Average = Average([
            ((float(net_income_data[i]['net_income']) - float(net_income_data[i+1]['net_income'])) / float(net_income_data[i+1]['net_income'])) * 100
            for i in range(2)
            ])
        
        # CAGR for 3 years (if you want 5 years, you need 6 data points)
        cagr3years = ((((float(net_income_data[0]['net_income']) / float(net_income_data[2]['net_income'])) ** (1/3)) - 1) * 100 if float(net_income_data[2]['net_income']) != 0 else 0.0)
        cagr3years = complex_to_float(cagr3years, method='real', tol=1e-9, default=0.0)
        
        # EPS Growth (3 years)
        
        EPSGrowth_3years = ((float(eps_data[0]['eps']) - float(eps_data[2]['eps'])) / float(eps_data[2]['eps'])) * 100 if float(eps_data[2]['eps']) != 0 else 0.0
        EPSGrowth_Average = Average([
            ((float(eps_data[i]['eps']) - float(eps_data[i+1]['eps'])) / float(eps_data[i+1]['eps'])) * 100
            for i in range(2)
            ])

        PaidDividends_3years = round(sum(float(dividends_data[i]['dividend']) for i in range(3)), 2)

        # Liquidity & Leverage
        latest_current_liab = float(current_liab[0]['current_liabilities']) if current_liab else 1
        latest_cash = float(cash_equiv[0]['cash_and_cash_equivalents']) if cash_equiv else 1
        current_liabilities_to_cash_factor = (latest_current_liab / latest_cash) * 100 if latest_cash != 0 else 0.0

        latest_total_liab = float(total_liab[0]['total_liabilities']) if total_liab else 1
        latest_total_assets = float(total_assets[0]['total_assets']) if total_assets else 1
        total_liabilities_to_assets_factor = (latest_total_liab / latest_total_assets) * 100 if latest_total_assets != 0 else 0.0
        
        # P/E Ratio
        # Last year P/E Ratio
        last_year_latest_price = float(price_data[1]['price'])
        last_year_latest_eps = float(eps_data[0]['eps'])
        last_year_pe_ratio = last_year_latest_price / last_year_latest_eps if last_year_latest_eps != 0 else 0.0

    ################################################
        # Overpriced Calculation
        latest_price = float(price_data[0]['price'])
        price_3_years_ago = float(price_data[3]['price']) if len(price_data) > 3 else 0
        investment = nb_shares * price_3_years_ago
        RetainedErnings = []
        for i in range(3):
            if i < len(eps_data) and i < len(dividends_data):
                retained = nb_shares * (float(eps_data[i]['eps']) - float(dividends_data[i]['dividend']))
                RetainedErnings.append(retained)

        NominalDiscountRate = RealDiscountRate + AverageInflation
        NetPresentValue = npf.npv(NominalDiscountRate, RetainedErnings) if RetainedErnings else 0

        fv_npv = npf.fv(-AverageInflation, 3, 0, -NetPresentValue)
        fv_investment = npf.fv(-AverageInflation, 3, 0, -investment)
        TotalValueYears = fv_npv + fv_investment
        EstimatedPrice = TotalValueYears / nb_shares if nb_shares > 0 else 0
        overpriced_float = round(100 * (latest_price / EstimatedPrice - 1), 2) if EstimatedPrice != 0 else 0.0
        overpriced_float = 0.0 if overpriced_float < -100 else overpriced_float

        return (
            round(latest_price, 2),
            round(last_year_latest_price, 2), 
            round(last_year_pe_ratio, 2),
            round(RORE_3years, 2),
            round(cagr3years, 2), 
            round(NetIncomeGrowth_3years, 2), 
            round(NetIncomeGrowth_Average, 2),  
            round(EPSGrowth_3years, 2),
            round(EPSGrowth_Average, 2),
            round(current_liabilities_to_cash_factor, 2),
            round(total_liabilities_to_assets_factor, 2),
            round(PaidDividends_3years, 2),
            round(overpriced_float, 2)
        )
    except Exception as e:
        print(f"Fundamentals error for {symbol}: {e}")
        traceback.print_exc()
        return (0,)*13


# ------------------------------
# Flask Routes
# ------------------------------

@app.route('/')
def student():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        nb_shares = int(request.form['Number_Shares'])
        RealDiscountRate = float(request.form['RealDiscountRate'])
        AverageInflation = float(request.form['AverageInflation'])

        # Table Header
        t = [['Stock', 'Curent price', 'Last year Price', 'Last year P/E', 'RORE 3y', 'CAGR 3y', 'NetIncome Growth 3y', 'NetIncome Avg', 
              'EPS Growth 3y', 'EPS Growth Average', 'Curr Liab/Cash %', 'Tot Liab/Assets %', 'Div 3y', 'Overpriced %']]
            
        now = dt.datetime.now()
        graphdata = [['Year']] + [[now.year - i] for i in range(6)]  # Initialize once

        for i in range(6):
            symbol_key = f'Stock_symbol{i+1}'
            #price_key = f'price_5_years_ago{i+1}'
            if request.form.get(symbol_key):
                symbol = request.form[symbol_key].strip().upper()
                try:
                        r = fundamentals(symbol, nb_shares, RealDiscountRate, AverageInflation)

                        # helper: safely get numeric and format
                        def safe_num(rtuple, idx, decimals=2, percent=False, default=0.0):
                            try:
                                val = float(rtuple[idx])
                            except Exception:
                                val = default
                            s = f"{val:.{decimals}f}"
                            return f"{s}%" if percent else s

                        # helper: safe float for numeric logic (graphing)
                        def safe_float(rtuple, idx, default=0.0):
                            try:
                                return float(rtuple[idx])
                            except Exception:
                                return default

                        stockname = symbol
                        table_data = [
                            stockname,
                            safe_num(r, 0, 2, False, 0.0),   # Current Price
                            safe_num(r, 1, 2, False, 0.0),   # Last year Price
                            safe_num(r, 2, 2, False, 0.0),   # Last year P/E
                            safe_num(r, 3, 2, True, 0.0),    # RORE 3y
                            safe_num(r, 4, 2, True, 0.0),    # CAGR 3y
                            safe_num(r, 5, 2, True, 0.0),    # NetIncome Growth 3y
                            safe_num(r, 6, 2, True, 0.0),    # NetIncome Avg
                            safe_num(r, 7, 2, True, 0.0),    # EPS Growth 3y
                            safe_num(r, 8, 2, True, 0.0),    # EPS Growth Average
                            safe_num(r, 9, 2, True, 0.0),   # Curr Liab/Cash
                            safe_num(r, 10, 2, True, 0.0),  # Tot Liab/Assets
                            safe_num(r, 11, 2, False, 0.0),  # Div 3y
                            safe_num(r, 12, 2, True, 0.0)    # Overpriced %
                        ]
                        t.append(table_data)

                        # graphdata expects numeric r[11] (Div 3y) — use safe_float
                        div_val = safe_float(r, 11, 0.0)
                        if div_val > 0:
                            graphdata[0].append(stockname)
                            for j in range(6):
                                graphdata[j+1].append(div_val)

                except Exception as e:
                        print(f"Error processing {symbol}: {e}")

        # Clean table: remove NaN/inf rows and ensure every cell is a string
        t = [row for row in t if all(str(cell) != 'nan' and 'inf' not in str(cell) for cell in row)]
        t = [[str(cell) for cell in row] for row in t]

        columnNames = t[0]
        rows = t[1:]

        return render_template("result2.html", columnNames=columnNames, rows=rows, graphdata=graphdata)


if __name__ == '__main__':
    app.run(debug=True)
