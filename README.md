# cadence

A small, readable, automated power-trading bot you can run on your own
laptop. It is the companion code for the guide **Algorithmic Trading in
Power Markets** (Practitioner Track, Energy Trading & ETRM Skills). The code
is built one piece at a time, and each piece matches one section of the
guide.

The name says what the job really is. Automated trading is mostly about
*timing*: feeding orders into a market that never stops, one beat at a time,
always racing a cut-off.

## What it builds

Two simple bots, on purpose, because they are two different kinds of trading:

- **Alice's bot.** Alice owns a wind farm and has to sell its power. Her bot
  guesses how much the wind will make, decides how much to sell, and places
  the orders.
- **Dan's bot.** Dan owns nothing. He just watches the price gap between two
  countries and bets when it looks unusual, sitting still the rest of the
  time.

## The pieces

| File | Guide section | What it does, in one line |
|---|---|---|
| `data` | 2 | Invents realistic market data so the examples run anywhere |
| `forecast` | 3 | Guesses a future number as a *range*, not a single value |
| `sizing` | 4 | Decides how much to sell, leaning to the safe side |
| `execution` | 5 | Places orders in small pieces so as not to move the price |
| `signals` | 6 | Spots a tradeable price gap between two countries |
| `backtest` | 7 | Honestly checks whether a strategy would have made money |
| `risk` | 8 | The safety gate every order must pass before it is sent |
| `pipeline` | 9 | The finale: every piece working together as one bot |

## Running it

You need [uv](https://docs.astral.sh/uv/) and Python 3.14 or newer.

```bash
uv sync --dev
uv run pytest -q
```

That is it. Everything runs on invented data, so there is no exchange to
connect to and no database to set up.

## The safety gate, and Morpholog

`risk.py` is the gate every order passes before it is sent. It stops the bot
building too big a position, lets you switch a strategy off instantly, lets
you halt a whole book, and makes sure every order says which strategy created
it (something regulators increasingly require).

It comes in two interchangeable versions, and comparing them is the real
lesson of that section:

- **The plain-Python version** holds everything in memory. It needs nothing
  extra and is what the examples use by default.
- **The Morpholog version** hands the rules to
  [Morpholog](https://github.com/jordan-dimov/morpholog), a separate engine
  that stores the record properly and simply refuses to record an order that
  breaks a rule, no matter how that order arrives.

On a small single-program demo like this, the plain version is all you need,
and Morpholog can look like more than the job requires. Its value shows up
when the demo grows up:

- the program restarts and must not forget what it already owns;
- several bots trade the same book at once and must share one honest view of
  the position;
- an auditor needs a record that provably has not been altered and can be
  wound back to any moment in the past.

Those are the things that turn a few dozen lines of Python into thousands.
The guide builds the plain version first, then deliberately breaks it in
those three ways to show where Morpholog earns its keep. The rules
themselves live in plain sight in [`cadence.morph`](./cadence.morph), readable
without knowing any Python.

### Running the Morpholog governor locally

The Morpholog version is the production shape, and it runs locally (and in
continuous integration, the way Glasshouse already does) in three light
steps. For installation details and a fuller tour, see the official
[Morpholog repo](https://github.com/jordan-dimov/morpholog) and its developer
guide; the cadence-specific parts are:

1. A throwaway PostgreSQL database:

   ```bash
   createdb cadence_demo
   export DATABASE_URL=postgres:///cadence_demo
   ```

2. The `morpholog` command-line tool, installed from a Morpholog checkout
   (see its developer guide):

   ```bash
   cargo install --path crates/morpholog-cli
   ```

3. The typed client, generated from this project's rules:

   ```bash
   morpholog generate python-client cadence.morph --out src/cadence/_morph_client
   ```

Then select the Morpholog-backed gate with `CADENCE_GOVERNOR=morpholog`.
Wiring the governor's methods onto the generated client is the exercise of
the guide's Section 8; the in-process gate stays the default, so the rest of
the project always runs with nothing extra installed.

## Status

This is a scaffold. Every piece has a small working core so the whole thing
runs today; the guide fills in the depth section by section.
