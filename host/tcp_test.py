import copy
import os
import socket
import threading
import time
from time import sleep
from collections import deque
import usb_logger
import numpy as np

# rewrite a tcp version of the serial_test.py


def get_all_files(csv_dir):
    file_names = os.listdir(csv_dir)
    # sort by file name
    file_names.sort()
    # join path
    file_paths = [os.path.join(csv_dir, file) for file in file_names]
    return file_paths


def get_all_univariate_files(csv_dir):
    files = get_all_files(csv_dir)
    univariate_files = []
    for file in files:
        data = np.loadtxt(file, delimiter=",")
        if len(data.shape) == 1:
            univariate_files.append(file)
    return univariate_files


class TcpClient:
    def __init__(
        self,
        ip="192.168.50.4",
        port=12345,
        buffer_size=1024,
        DEBUG=False,
        timeout=5,
        usb_power_measuring=False,
    ):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.client = None
        self.connected = False
        self.DEBUG = DEBUG
        self.timeout = timeout

        if usb_power_measuring:
            self.open_usb_logger()
            self.usb_power_measuring = True
        else:
            self.usb_deque = None
            self.usb_stop_wrapper = None
            self.usb_power_measuring = False

    def open_usb_logger(self):
        # start usb_logger in another thread
        sps = 100
        max_size = sps * 86400  # 1 day
        usb_deque = deque(maxlen=max_size)
        self.usb_deque = usb_deque

        self.usb_stop_wrapper = [False]
        usb_logger_thread = threading.Thread(
            target=usb_logger.power_log, args=(usb_deque, self.usb_stop_wrapper)
        )
        usb_logger_thread.daemon = True
        usb_logger_thread.start()
        self.usb_thread = usb_logger_thread

    def get_usb_data(self):
        # format: [timestamp, voltage, current, energy]
        if self.usb_deque:  # Check if the queue is not empty
            return list(
                self.usb_deque
            )  # Return a snapshot of the current data in the queue
        return []

    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.ip, self.port))
        self.connected = True

    def send_command(self, command):
        self.client.send(command.encode() + b"\n")

    def get_response(self):
        response = self.client.recv(self.buffer_size).decode()
        response = response.strip()
        if self.DEBUG:
            if response != "":
                print(f"[INFO]  Host: {response}")
        return response

    def wait_for_response(self, indicator="F", timeout=5):
        start_time = time.time()
        while True:
            response = self.get_response()
            if indicator in response:
                return response
            if time.time() - start_time > timeout:
                break

    def reset(self):
        self.send_command("R\n")
        self.wait_for_response(indicator="Device reset")

    def set_data(self, datafile_name):

        self.send_command("D+" + datafile_name)
        # expect response: D:DataName
        response = self.wait_for_response(indicator="D:")
        print(f"[INFO]  Host: {response}")
        print("D:" + datafile_name)

        if response == "D:" + datafile_name:
            print(f"[INFO]  Host: Data {datafile_name} set")
        else:
            raise Exception("Error: Data not set")
        return

    def set_model(self, model_name, model_version=-1):
        if model_version >= 0:
            message = "M+" + model_name + "+v" + str(model_version)
        else:
            message = "M+" + model_name

        self.send_command(message)
        # expect response: M:ModelName
        response = self.wait_for_response(indicator="M:")
        if model_version >= 0:
            expected_response = "M:" + model_name + "+v" + str(model_version)
        else:
            expected_response = "M:" + model_name

        if response == expected_response:
            print(f"[INFO]  Host: Model {model_name} set")
        return

    def start_detect(self, mode=1):
        # send S+mode
        message = "S+" + str(mode)
        self.send_command(message)
        response = self.wait_for_response(indicator="S")
        # expect response: S
        if "S" in response:
            print("[INFO]  Host: Detection started in mode " + str(mode))
        else:
            raise Exception("Error: Detection not started")

        # wait and parse the response
        response = self.wait_for_response(indicator="F", timeout=3600)
        # parse
        if mode == 0:
            # time mode, response format: f"F, count: {i}, time: {time_diff} ms"
            count = int(response.split("count: ")[-1].split(",")[0])
            time_diff = int(response.split("time: ")[-1].split("ms")[0])
            print(
                f"[INFO]  Host: Detection done, count: {count}, total time: {time_diff} ms, average time: {time_diff/count} ms"
            )
            return time_diff / count
        if mode == 1:
            # accuracy mode, response format: f"F, f1: {f1}, cover: {cover}"
            f1 = response.split("f1: ")[-1].split(",")[0]
            cover = response.split("cover: ")[-1]
            print(f"[INFO]  Host: Detection done, f1: {f1}, cover: {cover}")
            return f1, cover
        if mode == 2:
            # raw mode, response format: f"F, locations: {locations}"
            locations_str = response.split("locations: ")[-1]
            print(f"[INFO]  Host: Detection done, locations: {locations_str}")
            # parse locations into a list, locations are in format: "[locs1, locs2 ..., locsn]", might be an empty list
            locations = []
            if len(locations_str) > 2:
                locations = [
                    int(loc) for loc in locations_str[1:-1].split(",") if loc != ""
                ]
            return locations

    def start_detect_power(self):
        """_summary_
        Returns:
        average_time: int, average time in ms
        average_energy_consumed: float, average energy consumed in J
        filtered_data: list, list of data points in format [timestamp, voltage, current, energy]
        """
        offset_time = 0.027
        # sleep(3)
        mode = 0
        # send S+mode
        sleep(1)
        message = "S+" + str(mode)
        # start_data = self.get_usb_data()[-1]
        # sleep(0.3)
        self.send_command(message)
        response = self.wait_for_response(indicator="S", timeout=20)
        time.sleep(offset_time)
        start_data = self.get_usb_data()[-1]
        print("[INFO]  Host: Detection started in energy mode ")

        # wait for F
        response = self.wait_for_response(indicator="F", timeout=7200)
        # parse
        # sleep(0.02)
        time.sleep(offset_time)
        power_data = self.get_usb_data()
        end_data = power_data[-1]

        # time mode, response format: f"F, count: {i}, time: {time_diff} ms"
        count = int(response.split("count: ")[-1].split(",")[0])
        time_diff = float(response.split("time: ")[-1].split("ms")[0])
        start_stamp = start_data[0]
        end_stamp = end_data[0]
        filtered_data = [
            data
            for data in power_data
            if data[0] >= start_stamp and data[0] <= end_stamp
        ]
        average_energy_consumed = (filtered_data[-1][3] - filtered_data[0][3]) / count
        average_time = time_diff / count
        print(
            f"[INFO]  Host: Detection done, count: {count}, average time: {average_time} ms, average energy: {average_energy_consumed}J"
        )
        return average_time, average_energy_consumed, filtered_data

    def stop_connection(self):
        # stop connection from then host side
        self.send_command("Q")

    def close(self):
        if self.usb_power_measuring not in [None, False]:
            self.usb_stop_wrapper[0] = True
            # close the usb_logger thread
            print("Closing usb_logger thread, join")
            self.usb_thread.join()
            print("Closed")
            self.usb_power_measuring = False

        if self.connected:
            self.stop_connection()
        self.client.close()
        self.connected = False
        print("[INFO]  Host: Connection closed")

    def __del__(self):
        self.close()


def main():
    device = TcpClient(DEBUG=True, usb_power_measuring=True)
    device.connect()

    multi_mode = True
    datasets_dir = "../datasets/csv"

    if multi_mode:
        files = get_all_files(datasets_dir)
        print(f"Total files: {len(files)}")
    else:
        files = get_all_univariate_files(datasets_dir)

    # measure accuracy for all watch versions and all files. For each version, write the accuracy to a different file
    result_path = "../results/pi_energy/"
    # result_path = "../results/pi_accuracy/"
    # versions for watch model: [0,38]
    versions_range = range(0, 39)
    for version in versions_range:
        result_file = result_path + f"watch_v{version}.csv"
        with open(result_file, "w") as f:
            model_name = "WATCH"
            device.set_model(model_name, model_version=version)

            f.write("dataset, time(ms), energy(J)\n")
            # f.write("dataset f1,cover\n")

            for file_path in files:
                # e.g. file_name = "bank.csv"
                file_name = file_path.split("/")[-1]
                device.set_data(file_name)

                # f1, cover = device.start_detect(mode=1)
                time, energy, raw_data = device.start_detect_power()

                dataset_name = file_name.split(".")[0]

                # f.write(f"{dataset_name},{f1},{cover}\n")
                f.write(f"{dataset_name},{time},{energy}\n")

    device.close()


if __name__ == "__main__":
    main()
