import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from sklearn.model_selection import train_test_split
from prophet.serialize import model_to_json, model_from_json
from datetime import datetime, timedelta

def get_from_api_historical_prices(symbol, client, year, currency="EUR"):
    """
    Function which gets historical prices for given symbol
    """
    symbol += currency
    historical_prices = pd.DataFrame()
    
    interval = '1d'  # Intervalle de 24 heures pour l'API

    # Obtenir le temps de fin (temps actuel)
    end_time = datetime.now()
    # Calculer le temps de début (3 ans)
    start_time = end_time - timedelta(days=year*12*31)  # Environ 3 ans

    # Convertir start_time et end_time en millisecondes pour l'API
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # Fetch historical klines data
    klines = client.get_historical_klines(symbol, interval, start_time_str, end_time_str)

    # Define the columns for the DataFrame
    spot_kline_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    # Convert the klines data to a DataFrame
    df = pd.DataFrame(klines, columns=spot_kline_cols)
    df['symbol'] = symbol
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)  # Optionnel: Utiliser 'timestamp' comme index
    historical_prices = pd.concat([historical_prices, df])
    
    historical_prices['close'] = historical_prices['close'].astype(float)
    historical_prices['timestamp'] = pd.to_datetime(historical_prices['open_time'], unit='ms')

    return historical_prices


def prepare_money_to_model(symbol_prices):
    df_money = symbol_prices[['timestamp','close']]
    df_money = df_money.rename(columns={'timestamp': 'ds', 'close': 'y'})
    return df_money

def apply_prophet_model(money_train):
    model_prophet = Prophet()
    model_prophet.fit(money_train)
    return model_prophet

def save_model(model_prophet, symbol):
    path_save = 'home/serialized_model_for_'+str(symbol)+'.json'
    with open(path_save, 'w') as fout:
        fout.write(model_to_json(model_prophet))  # Save model
    return path_save

def load_model(symbol):
    with open('home/serialized_model_for_'+str(symbol)+'.json', 'r') as fin:
        m = model_from_json(fin.read())  # Load model
    print(f"load model done for {symbol}")
    return m
    
def run_training_model(symbol_prices):
    df_money= prepare_money_to_model(symbol_prices)
    money_train, money_test = train_test_split(df_money,
                                            test_size=0.02,
                                            shuffle=False,
                                            random_state = 42)
    model_prophet = apply_prophet_model(money_train)
    money_future = model_prophet.make_future_dataframe(periods=len(money_test), freq='B')
    money_predit = model_prophet.predict(money_future)
    return model_prophet


def get_predict_price(symbol):
    m = load_model(symbol)
    future_df = pd.DataFrame({'ds': [datetime.now() + timedelta(days=7)]})
    predicted_value = float(m.predict(future_df).yhat.iloc[0])
    print(f"get predict price done for {symbol}")
    return predicted_value


def get_choice(symbol, current_price):
    print(f"Prix prédit par le modéle prophet du {symbol} dans 7 jours est de: {get_predict_price(symbol):.2f}") 
    print(f"Prix actuel est de: {current_price:.2f}")
    if get_predict_price(symbol) > current_price: return 'Buy'
    elif get_predict_price(symbol) < current_price: return 'Sell'


def calculate_total_actif(wallet_row, client, currency='EUR'):
    try:
        # Initialize total_actif with the value of fonds_disponibles
        total_actif = wallet_row['fonds_disponibles']
        
        # Loop through the columns to calculate the total value of cryptos
        for column in wallet_row.index:
            if column not in ['timestamp', 'fonds_disponibles', 'fonds_investis', 'total_actif']:
                # Get the amount of crypto
                amount = wallet_row[column]
                
                # Get the current price of the crypto
                ticker = client.get_symbol_ticker(symbol=(column + currency))
                current_price = float(ticker['price'])
                
                # Update total_actif with the current value of the crypto
                total_actif += amount * current_price
        
        # Update the total_actif value
        wallet_row['total_actif'] = total_actif
        
        return wallet_row
    except Exception as e:
        raise Exception(f"Error calculating total_actif: {e}")



def get_action(wallet, symbol, unite, client, currency='EUR'):
    try: 
        # Obtenir le dernier index des données du portefeuille
        last_index = wallet.index[-1]
    
        # Obtenir le timestamp actuel
        current_time = datetime.now()
    
        # Obtenir les prix actuels
        ticker = client.get_symbol_ticker(symbol=(symbol+currency))
        current_price = float(ticker['price'])
    
        # Obtenir le choix de l'action
        action_choice = get_choice(symbol, current_price)

        # Copie de la dernière ligne et mise à jour du timestamp
        new_row = wallet.iloc[last_index].copy()
        new_row['timestamp'] = current_time

        # Mise à jour du DataFrame selon l'action choisie
        if action_choice == 'Buy':
            if new_row['fonds_disponibles'] >= unite:
                # Calcul de la quantité achetée
                amount_to_buy = unite / current_price
                new_row['fonds_disponibles'] -= unite
                new_row[symbol] += amount_to_buy
                new_row['fonds_investis'] += unite
                print(f"Action : Acheter {amount_to_buy:.6f} {symbol} au prix de {current_price:.2f} EUR/{symbol}.")
            else:
                raise ValueError("Action : Acheter impossible. Fonds insuffisants.")
    
        elif action_choice == 'Sell':
            if new_row[symbol] >= unite:
                # Calcul de la quantité vendue et des fonds reçus
                amount_to_sell = unite
                amount_received = amount_to_sell * current_price
                new_row['fonds_disponibles'] += amount_received
                new_row[symbol] -= amount_to_sell
                new_row['fonds_investis'] -= amount_received
                print(f"Action : Vendre {amount_to_sell:.6f} {symbol} au prix de {current_price:.2f} EUR/{symbol}.")
            else:
                raise ValueError(f"Action : Vendre impossible. {symbol} insuffisants.")
    
        else:
            raise ValueError("Action non reconnue. Veuillez choisir 'Buy' ou 'Sell'.")

        # Recalculate the total asset value after the action
        new_row = calculate_total_actif(new_row, client)
    
        # Ajouter la nouvelle ligne au DataFrame
        wallet = pd.concat([wallet, new_row.to_frame().T], ignore_index=True)
        return wallet
    except Exception as e:
        raise e


def get_updated_wallet(wallet, client):
    """Function of which update total_actif based on current value of each symbol
    """
    # Obtenir le dernier index des données du portefeuille
    last_index = wallet.index[-1]
    new_row = wallet.iloc[last_index].copy()
    
    # Obtenir le timestamp actuel
    current_time = datetime.now()

    # update timestamp
    new_row['timestamp'] = current_time

    # update total_actif
    new_row = calculate_total_actif(new_row, client)
    
    # Ajouter la nouvelle ligne au DataFrame
    wallet = pd.concat([wallet, new_row.to_frame().T], ignore_index=True)
    return wallet