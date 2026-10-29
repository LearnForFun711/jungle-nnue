# filepath: backend/model7.py
"""Model 7: the neural engine.

This is the drop-in replacement for the handcrafted engine.  It loads the
shared library built from the `jungle-nnue` project and the network that was
distilled from the MCTS teacher's self-play data, and plays the same search
the C++ engine plays.

The board encoding is the one the C++ side expects: 63 bytes, row-major,
0 = empty, 1..8 = a Red piece of that type, 9..16 = a Black piece.

The C++ search does not know the history of the game, so the moves that would
repeat a position are ruled out here and passed to the engine as a list of
moves it must not play.  The engine repeats a position when it has occurred
twice before, which is the same three-fold rule the Python engine applies.

If the library or the network cannot be found, the model falls back to
Model 5 rather than failing, so the front end keeps working.
"""
import ctypes
import os
import platform

from model5 import Model5AI  # the fallback, and the search-time conventions


class Model7AI(Model5AI):
    def __init__(self):
        super().__init__()
        self.lib = None
        self.net_path = None
        self.load_engine()

    # -- locating the library and the network ------------------------------
    def _project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _find_library(self):
        names = (["libjungle.dylib", "libjungle.so"] if platform.system() == "Darwin"
                 else ["libjungle.so", "libjungle.dylib"])
        root = self._project_root()

        # An explicit path wins, and it is a file rather than a directory.
        explicit = os.environ.get("JUNGLE_NNUE_LIB")
        if explicit and os.path.isfile(explicit):
            return explicit

        places = [
            os.path.join(root, "cpp_engine"),
            os.path.join(root, "jungle-nnue", "engine", "build"),
            os.path.join(root, "engine", "build"),
        ]
        for place in places:
            for name in names:
                path = os.path.join(place, name)
                if os.path.exists(path):
                    return path
        return None

    def _find_net(self):
        root = self._project_root()
        candidates = [
            os.environ.get("JUNGLE_NNUE_NET"),
            os.path.join(root, "net", "net.nnue"),
            os.path.join(root, "jungle-nnue", "net", "net.nnue"),
            os.path.join(root, "cpp_engine", "net.nnue"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.nnue"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def load_engine(self):
        lib_path = self._find_library()
        net_path = self._find_net()
        if not lib_path or not net_path:
            print("Model 7: neural engine not found "
                  f"(library={lib_path}, network={net_path})")
            return
        try:
            self.lib = ctypes.CDLL(lib_path)
            self.lib.jungle_get_best_move_nnue_ex.argtypes = [
                ctypes.POINTER(ctypes.c_char),   # board
                ctypes.c_int,                    # turn
                ctypes.c_int,                    # move count
                ctypes.c_int,                    # time in ms
                ctypes.c_char_p,                 # network path
                ctypes.POINTER(ctypes.c_int),    # banned moves (from, to) pairs
                ctypes.c_int,                    # number of banned pairs
                ctypes.POINTER(ctypes.c_int),    # out: from, to, score, nodes
            ]
            self.lib.jungle_get_best_move_nnue_ex.restype = ctypes.c_int
            self.net_path = net_path
            print(f"Model 7: neural engine loaded ({lib_path}, {net_path})")
        except OSError as exc:
            print(f"Model 7: cannot load {lib_path}: {exc}")
            self.lib = None

    # -- the game ----------------------------------------------------------
    def get_move(self, engine, max_time=3.0):
        if not self.lib or not self.net_path:
            print("Model 7: falling back to Model 5")
            return super().get_move(engine, max_time)

        board_flat = self._serialize_board(engine.board)
        banned = self._repeating_moves(engine)

        turn_int = 0 if engine.turn == 'r' else 1
        move_count = len(engine.move_history)
        time_ms = max(1, int(max_time * 1000))

        c_board = (ctypes.c_char * 63)(*board_flat)
        out_move = (ctypes.c_int * 4)()
        if banned:
            c_banned = (ctypes.c_int * len(banned))(*banned)
            banned_cnt = len(banned) // 4
        else:
            c_banned = None
            banned_cnt = 0

        res = self.lib.jungle_get_best_move_nnue_ex(
            c_board, turn_int, move_count, time_ms,
            self.net_path.encode(), c_banned, banned_cnt, out_move)

        fr, fc, tr, tc = out_move[0] // 7, out_move[0] % 7, out_move[1] // 7, out_move[1] % 7
        if res != 1 or (fr == 0 and fc == 0 and tr == 0 and tc == 0):
            print("Model 7: engine returned no move, falling back to Model 5")
            return super().get_move(engine, max_time)

        move = {'from': [fr, fc], 'to': [tr, tc]}
        # Safety net: the front end's rules have the last word.  If the
        # engine's move is not one of them, take a legal one instead --
        # a move the rules reject would leave the game stuck.
        legal = {(tuple(m['from']), tuple(m['to'])) for m in engine.get_valid_moves()}
        if ((fr, fc), (tr, tc)) not in legal:
            print(f"Model 7: the engine's move {move} is not legal here, "
                  "picking a legal one instead")
            for candidate in engine.get_valid_moves():
                move = candidate
                break
            if move is None:
                return None
        engine.last_eval = out_move[2]
        print(f"Model 7 (neural): {time_ms}ms  move {move}  score {out_move[2]}  "
              f"nodes {out_move[3]}  banned {banned_cnt}")
        return move

    # -- helpers -----------------------------------------------------------
    def _repeating_moves(self, engine):
        """Moves whose result would repeat a position already seen twice.

        The engine's C API counts squares, not rows and columns, so the
        pairs are converted: a move from (row, column) is the square
        ``row * 7 + column``.
        """
        banned = []
        try:
            moves = engine.get_valid_moves()
        except Exception:
            return banned
        board = engine.board
        # The engine hashes the position *after* the move but with the turn
        # of the player who moved, so the hash of a candidate is built the
        # same way.
        turn = engine.turn
        for mv in moves:
            (fr, fc), (tr, tc) = mv['from'], mv['to']
            piece, target = board[fr][fc], board[tr][tc]
            board[tr][tc] = piece
            board[fr][fc] = '.'
            h = str(board) + turn
            board[fr][fc] = piece
            board[tr][tc] = target
            if engine.history.count(h) >= 2:
                banned.extend([fr * 7 + fc, tr * 7 + tc])
        return banned

    def _serialize_board(self, board_2d):
        # 9x7 -> 63 bytes: 0 empty, 1..8 Red, 9..16 Black.
        data = []
        for r in range(9):
            for c in range(7):
                p = board_2d[r][c]
                val = 0
                if p != '.':
                    rank = int(p[1])
                    val = rank if p[0] == 'r' else rank + 8
                data.append(val)
        return bytes(data)
