import time
import socket
from accuracy import scores
import sys
import os

import numpy as np

from micro_watch import micro_watch
from cusum import cusum_detector
from bocpd import bocpd_detector
from bocpdms_detector import bocpdms_detector
from pelt import pelt_detector


def get_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    return data


class TCPDeviceController:

    def __init__(self, host="0.0.0.0", port=12345, DEBUG=False):
        self.host = host
        self.port = port
        self.model_name = ""
        self.model_set = False
        self.dataset_path = ""
        self.data = []
        self.data_set = False
        self.raise_error = False
        self.detector = None
        self.params_path = ""
        self.detector = None
        self.version = -1
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.DEBUG = DEBUG
        self.root_path = "/home/firepond/development/microCPD"
        self.client_socket = None  # do not need multiple clients

    def start_server(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

    def send_message(self, message):
        self.client_socket.send(message.encode("utf-8") + b"\n")
        if self.DEBUG:
            print(f"DEBUG: Sent response: {message}")

    def handle_client(self):
        try:
            while True:
                # client_socket.send(b">\n")
                command = self.client_socket.recv(1024).decode("utf-8").strip()
                if not command:
                    continue
                if self.DEBUG:
                    print(f"DEBUG: Received command: {command}")
                if command == "Q":
                    # got Q command, close the connection
                    self.client_socket.close()
                    self.client_socket = None
                    break
                self.process_command(command)
                # self.client_socket.send(response.encode('utf-8') + b"\n")
                # self.send_message(response)

        except Exception as e:
            print(f"Client connection error: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()

    def process_command(self, command):
        if self.raise_error:
            self.send_message("Error occured")
            self.raise_error = False
            return
        if command == "H":
            message = "hello\nboard: Python TCP Server\nlanguage: Python"
            self.send_message(message)
        elif command == "C":
            self.send_message("H A M D S")
        elif command[0] == "M":
            self.handle_model_command(command)
        elif command[0] == "D":
            self.handle_data_command(command)
        elif command[0] == "S":
            self.handle_start_command(command)
        elif command == "R":
            self.reset_device()
            self.send_message("Device reset")
        elif command == "A":
            self.send_message(">")
        else:
            self.send_message("Error: Invalid command")

    def handle_model_command(self, command):

        if len(command) > 2 and command[1] == "+":
            model_name = command[2:]
            if "+v" in model_name:
                try:
                    self.version = int(model_name.split("+v")[-1])
                    model_name = model_name.split("+v")[0]
                except:
                    self.raise_error = True
                    self.send_message("Error: Invalid version")
                    return
            else:
                self.version = -1

            if model_name == "WATCH":
                self.detector = micro_watch(self.version)
                self.model_set = True
                self.params_path = os.path.join(
                    self.root_path, "params/params_watch_best.csv"
                )
                self.model_name = model_name
            elif model_name == "CUSUM":
                self.detector = cusum_detector(self.version)
                self.model_set = True
                self.params_path = os.path.join(
                    self.root_path, "params/params_cusum_best.csv"
                )
                self.model_name = model_name
            elif model_name == "BOCPD":
                self.detector = bocpd_detector(self.version)
                self.model_set = True
                self.params_path = os.path.join(
                    self.root_path, "params/params_bocpd_best.csv"
                )
                self.model_name = model_name
            elif model_name == "BOCPDMS":
                self.detector = bocpdms_detector(self.version)
                self.model_set = True
                self.params_path = os.path.join(
                    self.root_path, "params/params_bocpdms_best.csv"
                )
                self.model_name = model_name
            elif model_name == "PELT":
                self.detector = pelt_detector(self.version)
                self.model_set = True
                self.params_path = os.path.join(
                    self.root_path, "params/params_pelt_best.csv"
                )
                self.model_name = model_name
            else:
                self.raise_error = True
                self.send_message("Errror: Algorithm not implemented yet")
                return
            print(self.params_path)

            message = (
                f"M:{self.model_name}, version: {self.version}"
                if self.version != -1
                else f"M:{self.model_name}"
            )
            self.send_message(message)
        elif self.model_set:
            message = (
                f"M:{self.model_name}, version: {self.version}"
                if self.version != -1
                else f"M:{self.model_name}"
            )
            self.send_message(message)
        else:
            self.raise_error = True
            self.send_message("Error: No such model")

    def handle_data_command(self, command):
        if len(command) > 2 and command[1] == "+":
            self.datafile_name = command[2:]
            # print(self.root_path)
            dataset_dir = os.path.join(self.root_path, "datasets/csv/")
            # print(dataset_dir)
            self.dataset_path = os.path.join(dataset_dir, self.datafile_name)
            print(self.dataset_path)
            try:
                self.data = get_data(self.dataset_path)
            except:
                self.raise_error = True
                self.send_message("Load data error")
            self.data_set = True
            if self.DEBUG:
                print(f"DEBUG: Loaded data from {self.dataset_path}")
            self.send_message("D:" + self.datafile_name)
        elif self.data_set:
            self.send_message("D:" + self.datafile_name)
        else:
            self.raise_error = True
            self.send_message("Error: No data loaded")

    def handle_start_command(self, command):
        print("Start command")
        if not self.model_set or not self.data_set:
            self.raise_error = True
            print("Model or data not set")
            self.send_message("Model or data not set")

        dataset_name = self.datafile_name.split(".")[0]
        self.detector.set_params(self.params_path, dataset_name)
        print("Params set")
        self.detector.reinit()

        if len(command) > 2 and command[1] == "+":
            mode = command[2]
            if mode == "0":
                self.detect_time_mode()
            elif mode == "1":
                self.detect_accuracy_mode(dataset_name)
            elif mode == "2":
                self.detect_raw_mode()
            else:
                self.raise_error = True
                self.send_message("Invalid mode")
        else:
            self.detect_time_mode()

    def detect_time_mode(self):
        self.send_message("S")
        start_time = time.time()
        i = 0
        while True:
            self.detector.detect(self.data)
            self.detector.reinit()
            i += 1
            elapsed_time = time.time() - start_time
            if i >= 10 and elapsed_time > 20:
                break
        elapsed_time *= 1000
        self.send_message(f"F, count: {i}, time: {elapsed_time:.2f} ms")

    def detect_accuracy_mode(self, dataset_name):
        if self.DEBUG:
            print("DEBUG: Detecting change points in accuracy mode")
        self.send_message("S")
        locations = self.detector.detect(self.data)
        if self.DEBUG:
            print(f"DEBUG: Detected change points: {locations}")
        annotations_file = os.path.join(self.root_path, "annotations.json")
        f1, cover = scores(
            locations,
            dataset_name,
            self.data.shape[0],
            annotations_file=annotations_file,
        )
        self.send_message(f"F, f1: {f1}, cover: {cover}")

    def detect_raw_mode(self):
        # send "S" to indicate start of detection
        self.send_message("S")

        locations = self.detector.detect(self.data)
        print(locations)
        self.send_message(f"F, locations: {locations}")

    def reset_device(self):
        self.__init__(self.host, self.port)

    def run(self):
        self.start_server()
        while True:
            client_socket, addr = self.server_socket.accept()
            self.client_socket = client_socket

            print(f"Connection from {addr}")
            self.handle_client()

    def __del__(self):
        # kill every connection on the socket
        if self.server_socket:
            self.server_socket.shutdown(socket.SHUT_RDWR)
            self.server_socket.close()
        print("Server closed")


if __name__ == "__main__":

    controller = TCPDeviceController(DEBUG=True)
    controller.run()
