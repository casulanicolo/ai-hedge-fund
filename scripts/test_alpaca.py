from src.execution.alpaca_adapter import AlpacaBrokerAdapter
a = AlpacaBrokerAdapter()
acc = a.get_account()
print('VPS Alpaca OK:', acc.account_id, 'cash=', acc.cash, 'status=', acc.status)
