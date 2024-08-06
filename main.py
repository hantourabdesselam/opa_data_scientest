from sqlalchemy import create_engine, text
from sqlalchemy.schema import Column, Table
from sqlalchemy.exc import SQLAlchemyError
from binance.client import Client
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import random
import os
import logging
from models import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# serveur fastapi
api = FastAPI(title='Projet OPA - Data scientest')

#infos mysql
db_user = os.getenv('MYSQL_USER') 
db_password = os.getenv('MYSQL_PASSWORD')
db_host = "10.43.0.42"
db_port = "3307"
db_database = os.getenv('MYSQL_DATABASE') 

#infos binance
key = os.getenv('BINANCE_kEY')
secret = os.getenv('BINANCE_SECRET')
client = Client(key,secret)

#connexion mysql
conn_string = f"mysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
mysql_engine = create_engine(conn_string)

# check mysql enging
with mysql_engine.connect() as connection:
        results = connection.execute(text('SHOW TABLES;'))
        print(f"connexion successed, dispo tables in {db_database} are: {results}")

class WalletInit(BaseModel):
    fonds_disponibles: int = Field(..., description="Fonds disponible pour investir dans la portefeuille", example=10000)
    cryptos: Dict[str, float] = Field(..., description="Nom du crpyto et son unite dans la portefeuille", example={"BTC": 0.5, "ETH": 2.0, "DOGE": 1000.0})

class Actions(BaseModel):
    cryptos: Dict[str, int] = Field(..., description="Nom du crypto et l'unite que vous voulez acheter ", example={"BTC": 100, "ETH": 20, "DOGE": 500})


# endpoint which checks api status
@api.get('/status', name="Check API status")
async def get_status():
    """Returns 1 if the API is up
    """
    return {
        'SUCCESS'
    }

@api.post("/init_wallet/")
async def init_wallet(wallet: WalletInit):
    """Initiation de portefeuille
    """
    res = {}
    try:
        # Get the current timestamp
        current_timestamp = datetime.now()

        # Create the base wallet data
        wallet_data = {
            'timestamp': [current_timestamp],
            'fonds_disponibles': [wallet.fonds_disponibles],
            'fonds_investis': [0]
        }
        total_actif = wallet.fonds_disponibles
        # Add the provided cryptocurrencies to the wallet data
        for crypto, amount in wallet.cryptos.items():
            wallet_data[crypto] = [amount]
            ticker = client.get_symbol_ticker(symbol=(crypto+'EUR'))
            current_price = float(ticker['price'])
            total_actif += current_price * amount
        
        wallet_data['total_actif'] = [total_actif]
        # Create a DataFrame from the wallet data
        wallet_df = pd.DataFrame(wallet_data)
        
        # Write the DataFrame to the database
        wallet_df.to_sql('wallet', mysql_engine, if_exists='replace', index=False)

        res['message'] = "Success generating wallet"
        res['wallet'] = wallet_df.to_dict('records')
        
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing wallet: {e}")


# endpoint which gets actions for give symbol
@api.post('/update_wallet_with_action', name="Update Wallet with action")
def update_wallet_with_action(action:Actions):
    """Update wallet by given action and prediction
    """
    res = {}
    for symbol, unite in action.cryptos.items():
        try:
            # read wallet from mysql engine
            wallet = pd.read_sql('SELECT * FROM wallet', mysql_engine)
            logger.info(f"read wallet done")

            # get historical price for given symbol for the last one year
            symbol_price = get_from_api_historical_prices(symbol, client, 1)
            logger.info(f"fetch historical price done for {symbol}")
            
            # save historical price into bdd mysql
            symbol_price_table_name = f"historical_prices_{symbol}"
            symbol_price.to_sql(symbol_price_table_name, mysql_engine, if_exists='append', index=False)
            logger.info(f"saving historical prices done for {symbol}")

            # read historical price from mysql engine
            symbol_price = pd.read_sql(f"SELECT * FROM {symbol_price_table_name}", mysql_engine)
            logger.info(f"read historical prices done for {symbol}")

            # train model
            symbol_model = run_training_model(symbol_price)
            logger.info(f"running model done for {symbol}")

            # save model
            model_save_path = save_model(symbol_model, symbol)
            logger.info(f"saving model done for {symbol}")

            # get action based on historical price and add it into wallet
            new_wallet = get_action(wallet, symbol, unite, client)

            # repush new wallet to replace the table wallet in mysql
            new_wallet.to_sql('wallet', mysql_engine, if_exists='replace', index=False)

        except Exception as e:
            error_message = f"Error updating wallet for {symbol}: {str(e)}"
            logger.error(error_message)
            res['error'] = str(e)
        
        # Read the updated wallet to return in the response
        updated_wallet = pd.read_sql('SELECT * FROM wallet', mysql_engine)
        res['message'] = "Success updateing wallet for given actions"
        res['wallet'] = updated_wallet.to_dict('records')
    return res



@api.get('/update_wallet', name="Update total_actif in wallet")
async def update_wallet():
    """Update Total actif in wallet based on current price of each symbol
    """
    res = {}
    try:
        wallet = pd.read_sql('SELECT * FROM wallet', mysql_engine)
        updated_wallet = get_updated_wallet(wallet, client)
        # updated_wallet = updated_wallet.applymap(lambda x: str(x) if isinstance(x, float) else x)

        res['message'] = "Success updateing total_actif in wallet by getting current prices"
        res['wallet'] = updated_wallet.to_dict('records')

        # repush updated wallet to replace the table wallet in mysql
        updated_wallet.to_sql('wallet', mysql_engine, if_exists='replace', index=False)
    except Exception as e:
        res['error'] = str(e)
    return res