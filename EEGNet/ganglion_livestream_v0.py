# wrote a simple script to show workflow of streaming data from the ganglion board with python - this will integrate with CNN workflow

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

def main():

    params = BrainFlowInputParams()
    params.serial_port = "COM6"

    board = BoardShim(BoardIds.GANGLION_BOARD, params)
    board.prepare_session()
    board.start_stream ()
    time.sleep(10)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    print(data)


if __name__ == "__main__":
    main()