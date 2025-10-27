# wrote a simple script to show workflow of streaming data from the ganglion board with python - this will integrate with CNN workflow

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import sys

def main():
    try:
        params = BrainFlowInputParams()
        params.serial_port = "COM6"

        board = BoardShim(BoardIds.GANGLION_BOARD, params)
        board.prepare_session()

    except:
        print("Error: could not connect to Ganglion.")
        board.release_session()
        sys.exit()

    finally:
        time.sleep(1)



    try:
        board.start_stream()

        while True:
            time.sleep(5)
            # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
            data = board.get_board_data()  # get all data and remove it from internal buffer
            print(data)

    except:
        print("Stream Interrupt")
        sys.exit()

    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()