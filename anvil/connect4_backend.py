import argparse
import numpy as np
import tensorflow as tf

try:
    import anvil.server
except Exception:
    anvil = None

ROWS, COLS = 6, 7


def valid_moves(board: np.ndarray):
    return [c for c in range(COLS) if board[0, c] == 0]


def encode_board_optionB(board: np.ndarray):
    p0 = (board == +1).astype(np.int8)
    p1 = (board == -1).astype(np.int8)
    return np.stack([p0, p1], axis=-1)


def flip_perspective_optionB(board_2ch: np.ndarray):
    return board_2ch[..., ::-1].astype(np.int8)


class ModelServer:
    def __init__(self, model_path="transformer_v2.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def predict_move(self, board, player=+1, encoding="B"):
        board = np.array(board, dtype=np.int8)
        if encoding == "A":
            board_in = board if player == +1 else (-board).astype(np.int8)
            x = board_in[np.newaxis, ..., np.newaxis].astype(np.float32)
        else:
            boardB = encode_board_optionB(board)
            board_in = boardB if player == +1 else flip_perspective_optionB(boardB)
            x = board_in[np.newaxis, ...].astype(np.float32)

        probs = self.model.predict(x, verbose=0)[0]
        moves = valid_moves(board)
        if not moves:
            return None
        masked = np.full_like(probs, -1e9)
        masked[moves] = probs[moves]
        return int(np.argmax(masked))


server = None


def get_server(model_path="transformer_v2.keras"):
    global server
    if server is None:
        server = ModelServer(model_path=model_path)
    return server


def get_move(board, player=+1, model_path="transformer_v2.keras", encoding="B"):
    srv = get_server(model_path=model_path)
    return srv.predict_move(board, player=player, encoding=encoding)


# Anvil callable (optional)
if "anvil" in globals() and anvil is not None:
    @anvil.server.callable
    def anvil_get_move(board, player=+1, model_path="transformer_v2.keras", encoding="B"):
        return get_move(board, player=player, model_path=model_path, encoding=encoding)


def main():
    parser = argparse.ArgumentParser(description="Connect4 model backend")
    parser.add_argument("--test", action="store_true", help="Run a quick CLI test")
    parser.add_argument("--model-path", default="transformer_v2.keras")
    args = parser.parse_args()

    if args.test:
        model = tf.keras.models.load_model(args.model_path)
        empty_board = np.zeros((ROWS, COLS), dtype=np.int8)
        move = get_move(empty_board, player=+1, model_path=args.model_path, encoding="B")
        print(f"CLI test move (empty board): {move}")
        return

    if "anvil" in globals() and anvil is not None:
        # Replace with your Anvil uplink key
        anvil.server.connect("YOUR_ANVIL_UPLINK_KEY")
        anvil.server.wait_forever()
    else:
        print("Anvil not available. Use get_move(board, player) locally.")


if __name__ == "__main__":
    main()
