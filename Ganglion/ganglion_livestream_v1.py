# wrote a simple script to show workflow of streaming data from the ganglion board with python - this will integrate with CNN workflow

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import sys
import csv
import os
from tkinter import filedialog
import scipy
import mne

def main():
    try:
        dir = filedialog.askdirectory()
        file_dir = os.path.join(dir, "ganglion_data.csv")

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
            time.sleep(1)
            data = board.get_board_data()  # get all data and remove it from internal buffer
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)

            info = mne.create_info(ch_names=['Fp1','Fp2','TP7','TP8'], sfreq=200, ch_types='eeg')
            raw = mne.io.RawArray(data[eeg_channels, :], info)
            raw_resampled = raw.resample(128)
            write_data = raw_resampled.get_data()  # numpy array: shape (4, N)
            write_data = write_data.T  # transpose so each row = one time sample, columns = channels

            #convert from uV into mV
            write_data = write_data / 1000.0

            with open(file_dir, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow(['Fp1', 'Fp2', 'TP7', 'TP8'])
                spamwriter.writerows(write_data)

            print(write_data)

    except Exception as e:
        print(f"Stream Interrupt: {e}")
        # traceback.print_exc()
        # print("Stream Interrupt")
        sys.exit()

    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()