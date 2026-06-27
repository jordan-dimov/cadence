"""cadence: a small automated power-trading bot you can read and run.

This is the companion code for the guide "Algorithmic Trading in Power
Markets". It is built one piece at a time, and each piece is one section of
the guide. Together they make two simple trading bots:

  - Alice's bot: she owns a wind farm and must sell its power, so her bot
    forecasts the wind, decides how much to sell, and places the orders.
  - Dan's bot: he owns nothing and just hunts for a price pattern between two
    countries, trading it when it appears and sitting still when it does not.

Everything runs on invented (but realistic) data, so you can run every
example on your own machine with nothing else installed. The pieces:

  data       - invents realistic market data to practise on
  forecast   - guesses future numbers as a range, not a single value
  sizing     - decides how much to sell, erring on the safe side
  execution  - places orders in pieces so as not to move the price
  signals    - spots a tradeable price gap between two countries
  backtest   - checks honestly whether a strategy would have made money
  risk       - the safety gate every order must pass before it is sent
  pipeline   - the finale: every piece working together as one bot
"""

__version__ = "0.1.0"
