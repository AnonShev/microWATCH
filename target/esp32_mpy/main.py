# this runs on device with serial port, parse message from the host and respond to it

# protocol specification:
# H\n : hello message, send back hello and board name and programming language

# C\n : ask for available commands, send back available commands

# M+ModelName+v0\n : set model name, send back model name
# M\n : ask for model name, send back model name

# D+DataName\n : set data name, send back data name
# D\n : ask for data name, send back data name

# S\n : start model, send back S when ready to start
# S+1 for accuracy mode, S+0 or S for time mode, S+2 for raw mode that returns the locations

# F\n : for device to send this when model is done only, not for host to send

# E\n Error, for device to send this when there is an error
# ">": device ready to receive command

# "A": if device is ready to receive command, send back ">"

# "R": soft reset, clear all data and model

import gc
import time
from accuracy import scores
import sys
from machine import UART  # type: ignore


try:
    from ulab import numpy as np  # type: ignore
except ImportError:
    import numpy as np

from micro_watch import micro_watch
from cusum import cusum_detector
from bocpd import bocpd_detector
from pelt import pelt_detector


def get_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    gc.collect()
    return data


class DeviceController:
    def __init__(self):
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

    def process_command(self, command):
        command = command.strip()
        if command == "":
            return

        if command[0] == "H":
            print("hello, board: ESP32C6 (RISC-V), language: micropython")
        elif command[0] == "C":
            print("H A M D S")
        elif command[0] == "M":
            self.handle_model_command(command)
        elif command[0] == "D":
            self.handle_data_command(command)
        elif command[0] == "S":
            self.handle_start_command(command)
        elif command[0] == "R":
            sys.exit()
        elif command[0] == "A":
            return
        else:
            print("Invalid command")
            return

    def handle_model_command(self, command):
        # model operations
        if len(command) > 2 and command[1] == "+":
            # set model
            model_name = command[2:]
            # check if there is a version number
            if "+v" in model_name:
                try:
                    version = int(model_name.split("+v")[-1])
                    self.version = version
                except:
                    print("Invalid version number")
                    self.raise_error = True
                    return
                model_name = model_name.split("+v")[0]
            else:
                self.version = -1
            if model_name == "WATCH":
                self.detector = micro_watch(version)
                self.model_set = True
                self.params_path = "/params/params_watch_best.csv"
                self.model_name = model_name
            elif model_name == "CUSUM":
                self.detector = cusum_detector(version)
                self.model_set = True
                self.params_path = "/params/params_cusum_best.csv"
                self.model_name = model_name
            elif model_name == "BOCPD":
                self.detector = bocpd_detector(version)
                self.model_set = True
                self.params_path = "/params/params_bocpd_best.csv"
                self.model_name = model_name
            elif model_name == "PELT":
                self.detector = pelt_detector(version)
                self.model_set = True
                self.params_path = "/params/params_pelt_best.csv"
                self.model_name = model_name
            else:
                # not implemented yet
                print("Algorithm not implemented yet")
                self.raise_error = True
                self.model_name = ""

            # print model name for confirmation
            if self.model_set is True:
                if self.version == -1:
                    print(f"M:{self.model_name}")
                else:
                    print(f"M:{self.model_name}, version: {self.version}")

        elif self.model_set is True:
            if self.version == -1:
                print(f"M:{self.model_name}")
            else:
                print(f"M:{self.model_name}, version: {self.version}")
        else:
            print("No model")
            self.raise_error = True

    def handle_data_command(self, command):
        if len(command) > 2 and command[1] == "+":
            self.datafile_name = command[2:]
            dataset_path = "/csv/" + self.datafile_name
            self.dataset_path = dataset_path
            # load data
            try:
                self.data = get_data(self.dataset_path)
            except:
                print("load data error")
                self.raise_error = True
                return
            self.data_set = True
            print("D:" + self.datafile_name)
        elif self.data_set is True:
            print("D:" + self.datafile_name)
        else:
            print("No data")
            self.raise_error = True

    def handle_start_command(self, command):
        if self.model_set is False or self.data_set is False:
            print("Model or data not set")
            self.raise_error = True
            return

        # set params
        dataset_name = self.datafile_name.split(".")[0]
        self.detector.set_params(self.params_path, dataset_name)

        self.detector.reinit()

        # parse for mode
        if len(command) > 2 and command[1] == "+":
            mode = command[2]
            if mode == "0":
                self.detect_time_mode()
            elif mode == "1":
                self.detect_accuracy_mode(dataset_name)
            elif mode == "2":
                self.detect_raw_mode()
            else:
                print("Invalid mode")
                self.raise_error = True
        else:
            self.detect_time_mode()

    def detect_time_mode(self):
        # time.sleep(3)
        # time mode, 10 time average
        # "warm" the processor
        self.detector.reinit()
        self.detector.detect(self.data)

        print("S")
        start_time = time.ticks_ms()
        # run at least 10 times and 10 seconds (achieve both)
        i = 0
        time_diff = 0
        while True:
            # for i in range(10):
            # time.sleep(0.2)
            self.detector.reinit()
            self.detector.detect(self.data)
            # time.sleep(0.2)
            end_time = time.ticks_ms()
            time_diff = time.ticks_diff(end_time, start_time)
            i += 1
            if i >= 10 and time_diff > 30000:
                break
        print(f"F, count: {i}, time: {time_diff} ms")
        # print(f"Time: {time_diff/i} ms")

    def detect_accuracy_mode(self, dataset_name):
        # run and get locations, then calculate f1 and cover
        print("S")
        locations = self.detector.detect(self.data)
        f1, cover = scores(locations, dataset_name, self.data.shape[0])
        # locations to str for sending, separate by blank space
        print(f"F, f1: {f1}, cover: {cover}")

    def detect_raw_mode(self):
        # run and get locations, then calculate f1 and cover
        print("S")
        locations = self.detector.detect(self.data)
        print(f"F, locations: {locations}")

    def run(self):
        while True:
            if self.raise_error:
                print("Error")
                self.raise_error = False
            print(">")
            command = input()
            self.process_command(command)


def echo():
    # do nothing, test the latency of the natural echo
    while True:
        command = input()
        #
        print(len(command))


if __name__ == "__main__":
    # echo()
    gc.collect()
    controller = DeviceController()

    # run, if any error occurs, go back to main loop
    while True:
        gc.collect()
        try:
            controller.run()
        except Exception as e:
            print(f"Device error: exception {e}")
        continue
