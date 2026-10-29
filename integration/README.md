# Adding the neural engine to the web front end

`model7.py` is a drop-in for the models the Flask application already knows
about.  It plays the neural engine: the same C++ search, the same network
that was distilled from the MCTS teacher's self-play data.

## What has to be in place

| file | where | what it is |
|---|---|---|
| `model7.py` | `backend/` | the model itself |
| `libjungle.dylib` (or `.so`) | `cpp_engine/` | the engine, built from this repository |
| `net.nnue` | `net/` | the network |

The model looks for the library and the network in the project root
(`cpp_engine/` and `net/`), and also in `jungle-nnue/engine/build/` and
`jungle-nnue/net/`, so a checkout of this repository next to the front end
works without copying anything.  `JUNGLE_NNUE_LIB` and `JUNGLE_NNUE_NET`
override the search.

## Installing it

```bash
cd jungle-nnue/engine && make NATIVE=1          # builds the library
cp build/libjungle.dylib ../cpp_engine/          # macOS
cp build/libjungle.so    ../cpp_engine/          # Linux

cp ../jungle-nnue/net/net.nnue ../net/
cp ../jungle-nnue/integration/model7.py ../backend/
```

Then register it in `backend/app.py`:

```python
from model7 import Model7AI
...
    'model7': Model7AI()
```

and add it to the two selectors in `frontend/index.html`:

```html
<option value="model7">Model 7 (Neural NNUE)</option>
```

## Checking it

```bash
python3 integration/test_model7.py
```

The test plays a few moves against the front end's own rules and fails if
the engine produces a move the rules reject.

## The rules the front end applies that the engine cannot

The C++ search knows nothing about the history of the game, so the model
works out which moves would repeat a position and passes them to the engine
as moves it must not play.  The front end's repetition rule is the one the
Python engine applies: a position that has already occurred twice may not
occur a third time.

The engine's own repetition detection inside the search is separate: it
scores a repeated position as a draw, which is how the engine prefers a
repetition to a loss.

## Notes

* The model falls back to Model 5 if the library or the network is missing,
  so the front end keeps working.
* The per-move time is the one the front end asks for.  The neural engine
  is not uniformly better at every time control: see `docs/DESIGN.md` for
  the measurements.
