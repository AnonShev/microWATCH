from collections import deque
import os
import threading
import time
import numpy as np
import serial
from time import sleep
import usb_logger
from tqdm import tqdm

# S\n : start model, send back S when ready to start
# S+1 for accuracy mode, S+0 or S for time mode, S+2 for raw mode that returns the locations


def get_all_files(csv_dir):
    file_names = os.listdir(csv_dir)
    # join path
    file_paths = [os.path.join(csv_dir, file) for file in file_names]
    # sort by elements count
    file_paths.sort(key=lambda x: len(np.loadtxt(x, delimiter=",")))
    return file_paths


def get_all_univariate_files(csv_dir):
    files = get_all_files(csv_dir)
    univariate_files = []
    for file in files:
        data = np.loadtxt(file, delimiter=",")
        if len(data.shape) == 1:
            univariate_files.append(file)
    return univariate_files


class SerialDevice:
    def __init__(
        self,
        port,
        baudrate=115200,
        timeout=3,
        DEBUG=False,
        reboot=False,
        usb_power_measuring=False,
    ):

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.DEBUG = DEBUG
        self.last_response = ""
        self.timeout = timeout
        if reboot:
            self.bootloader()
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

    def bootloader(self):
        print("[INFO]  Host: Bootloader started")
        # check for the device status, single ">" for ready to receive command, ">>> " for python prompt
        self.send_command(" ")
        rebooot_indicator = "MPY: soft reboot"
        try:
            response = self.wait_for_response(indicator=">")
        except:
            print("[INFO]  Host: Device not ready, try to reboot")
            self.reboot()

        if ">>>" in response:
            # send ctrl D to soft reset
            self.ser.write(b"\x04")
            response = self.wait_for_response(indicator=rebooot_indicator)
            print("[INFO]  Host: Micropython soft rebooted")
        elif response == ">":
            self.send_command("R")
            response = self.wait_for_response(indicator=rebooot_indicator)
            print("[INFO]  Host: Micropython soft rebooted")
        else:
            self.reboot()

    def reboot(self):
        rebooot_indicator = "MPY: soft reboot"
        print("[INFO]  Host: Trying to reboot")
        try:
            # send ctrl c to quit to REPL
            self.ser.write(b"\x03")
            # wait unitl the ">>>" prompt
            self.wait_for_response(indicator=">>>", timeout=10)

            # send ctrl D to soft reset
            self.ser.write(b"\x04")
            self.wait_for_response(indicator=rebooot_indicator, timeout=10)
        except:
            print("[ERROR] Host: Reboot failed")
            raise Exception("Error: Reboot failed")

    def check_ready(self):
        # send "A\r" until get a ">" response of ready to receive command, timeout =10s
        self.send_command("A")
        response = self.wait_for_response(indicator=">")
        if response == ">":
            print("[INFO]  Host: Device is ready to receive command")
            return True
        return False

    def set_model(self, model_name, model_version=-1):
        # send M+ModelName, add version if provided and not negative
        if model_version >= 0:
            message = "M+" + model_name + "+v" + str(model_version)
        else:
            message = "M+" + model_name
        self.send_command(message)
        response = self.wait_for_response(indicator="M:")
        # check is the response if valid
        if model_version >= 0:
            if model_name + ", version: " + str(model_version) in response:
                print(f"[INFO]  Host: Model {model_name}v{model_version} set")
            else:
                raise Exception("Error: Model name not set")
        elif model_name in response:
            print(f"[INFO]  Host: Model {model_name} set")
        else:
            raise Exception("Error: Model name not set")

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
        message = "S+" + str(mode)
        # start_data = self.get_usb_data()[-1]
        # sleep(0.3)
        self.send_command(message)
        response = self.wait_for_response(indicator="S", timeout=20)
        time.sleep(offset_time)
        start_data = self.get_usb_data()[-1]
        print("[INFO]  Host: Detection started in energy mode ")

        # wait for F
        response = self.wait_for_response(indicator="F", timeout=3600)
        # parse
        # sleep(0.02)
        time.sleep(offset_time)
        power_data = self.get_usb_data()
        end_data = power_data[-1]

        # time mode, response format: f"F, count: {i}, time: {time_diff} ms"
        count = int(response.split("count: ")[-1].split(",")[0])
        time_diff = int(response.split("time: ")[-1].split("ms")[0])
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

    def set_data(self, datafile_name):
        # send D+DataName
        self.send_command("D+" + datafile_name)
        # expect response: D:DataName
        response = self.wait_for_response(indicator="D:")
        if response == "D:" + datafile_name:
            print(f"[INFO]  Host: Data {datafile_name} set")
        else:
            raise Exception("Error: Data not set")

    def send_command(self, command):
        commands_bytes = (command + "\r\n").encode()
        self.ser.write(commands_bytes)
        self.ser.flush()
        self.ser.read_until(commands_bytes)

    def get_response(self):
        response = self.ser.readline().decode().strip()
        if response != "" and self.DEBUG:
            print('[DEBUG] Device response: "' + response + '"')
        return response

    def close(self):
        # if anything is not closed, close it
        if self.ser.is_open:
            print("Closing serial port")
            self.ser.close()

        if self.usb_power_measuring not in [None, False]:
            self.usb_stop_wrapper[0] = True
            # close the usb_logger thread
            print("Closing usb_logger thread, join")
            self.usb_thread.join()
            print("Closed")
            self.usb_power_measuring = False

    def wait_for_response(self, indicator="F", error_hint="Device error", timeout=None):
        if timeout is None:
            timeout = self.timeout
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for response")
            response = self.get_response()
            if indicator in response:
                return response
            if error_hint in response:
                raise Exception("Error: " + response)

    def __del__(self):
        self.close()


def main():

    # test the SerialDevice class
    device = SerialDevice(
        "/dev/ttyACM0",
        baudrate=115200,
        DEBUG=False,
        reboot=True,
        timeout=3,
        usb_power_measuring=False,
    )

    multi_mode = False
    datasets_dir = "../datasets/csv"

    if multi_mode:
        files = get_all_files(datasets_dir)
        print(f"Total files: {len(files)}")
    else:
        files = get_all_univariate_files(datasets_dir)

    # measure energy consumption for all watch versions and all files. For each version, write the accuracy to a different file
    result_path = "../results/esp32_accuracy/"
    # result_path = "../results/esp32_energy/"
    # versions for watch model: [0,38]
    versions_range = range(0, 1)
    for version in versions_range:
        result_file = result_path + f"bocpd.csv"
        with open(result_file, "w") as f:
            device.set_model("BOCPD", model_version=version)
            # f.write("dataset, time(ms), energy(J)\n")
            f.write("dataset, f1, cover\n")
            for file_path in tqdm(files):
                # e.g. file_name = "bank.csv"
                file_name = file_path.split("/")[-1]
                device.set_data(file_name)
                # time, energy, raw_data = device.start_detect_power()
                f1, cover = device.start_detect(mode=1)
                dataset_name = file_name.split(".")[0]
                # f.write(f"{dataset_name}, {time}, {energy}\n")
                f.write(f"{dataset_name}, {f1}, {cover}\n")

    device.close()


if __name__ == "__main__":
    main()
