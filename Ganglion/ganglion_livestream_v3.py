# wrote a simple script to show workflow of streaming data from the ganglion board with python - this will integrate with CNN workflow

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import sys
import csv
import os
from tkinter import filedialog
import tkinter as tk
import threading
import scipy
import mne
import queue
import numpy as np

from tensorflow.keras.models import load_model

pred_queue = queue.Queue()

def main():
    try:
        # dir = filedialog.askdirectory()
        # file_dir = os.path.join(dir, "ganglion_data.csv")

        params = BrainFlowInputParams()
        params.serial_port = "COM6"

        board = BoardShim(BoardIds.GANGLION_BOARD, params)
        board.prepare_session()

        model = load_model(r"C:\Users\Will\Documents\Boston University\Classes\EC601\Code\eegnet_4ch_best_model.h5")
        # model.summary()

    except:
        print("Error: could not connect to Ganglion.")
        board.release_session()
        sys.exit()

    finally:
        time.sleep(1)



    try:
        board.start_stream()
        window_size = 129
        buffer = np.zeros((4, window_size))

        while True:
            
            data = board.get_board_data()  # get all data and remove it from internal buffer
 
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)

            info = mne.create_info(ch_names=['Fp1','Fp2','T7','T8'], sfreq=200, ch_types='eeg')

            raw = mne.io.RawArray(data[eeg_channels, :], info)
            eeg = data[eeg_channels, :]
            n = eeg.shape[1]

            if n >= window_size:
                buffer = eeg[:, -window_size:]
            else:
                buffer[:, :-n] = buffer[:, n:]
                buffer[:, -n:] = eeg

            X_input = buffer[np.newaxis, :, :, np.newaxis]  # (1, 4, 129, 1)

            y_pred = model.predict(X_input, verbose=0)
            cls = int(np.argmax(y_pred, axis=1)[0])
            probs = y_pred[0]

            # print(f"Prediction: class {cls}, probs {probs}")

            pred_queue.put(cls)

            time.sleep(0.01)


    except Exception as e:
        print(f"Stream Interrupt: {e}")
        # traceback.print_exc()
        # print("Stream Interrupt")
        sys.exit()

    finally:
        board.stop_stream()
        board.release_session()

class EEGGui:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Prediction Display")
        self.root.geometry("400x200")

        # Title
        self.label_title = tk.Label(
            root,
            text="Current Prediction",
            font=("Helvetica", 20)
        )
        self.label_title.pack(pady=10)

        # Prediction label (big)
        self.label_pred = tk.Label(
            root,
            text="Waiting…",
            font=("Helvetica", 48),
            fg="#0044AA"
        )
        self.label_pred.pack(pady=20)

        # Start periodic check
        self.check_queue()

    def check_queue(self):
        """Check prediction updates from worker thread."""
        try:
            pred = pred_queue.get_nowait()
            self.label_pred.config(text=str(pred))
        except queue.Empty:
            pass

        # Schedule next check (every 50 ms)
        self.root.after(50, self.check_queue)

if __name__ == "__main__":

    t = threading.Thread(target=main, daemon=True)
    t.start()

    # Start GUI
    root = tk.Tk()
    gui = EEGGui(root)
    root.mainloop()