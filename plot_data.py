import pandas as pd
import matplotlib.pyplot as plt

DataKriaCPU = pd.read_csv("Data_KriaCPU.lvm", sep="\t", header=None, names=["timestamp", "Voltage", "Current"])
DataKriaFPGA = pd.read_csv("Data_KriaFPGA.lvm", sep="\t", header=None, names=["timestamp", "Voltage", "Current"])
DataKriaCPU2 = pd.read_csv("Data_KriaCPU_2.lvm", sep="\t", header=None, names=["timestamp", "Voltage", "Current"])
DataKriaFPGA2 = pd.read_csv("Data_KriaFPGA_2.lvm", sep="\t", header=None, names=["timestamp", "Voltage", "Current"])
DataNoLoad = pd.read_csv("Data_BrakObciazenia.lvm", sep="\t", header=None, names=["timestamp", "Voltage", "Current"])

for data in [DataKriaCPU, DataKriaCPU2, DataKriaFPGA, DataKriaFPGA2, DataNoLoad]:
    data["Power"] = data.Voltage * data.Current

StandardPower = DataNoLoad.Power.mean()
SpanCPU = (591.68, 1694.4)
SpanCPU2 = (574.28, 1662.65)
SpanFPGA = (549.33, 810.78)
SpanFPGA2 = (549.15, 810.64)

plt.figure(1)
plt.subplot(3, 1, 1)
plt.suptitle("Experiment 1 - calculations on CPU")
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Voltage, label="Voltage")
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Voltage.ewm(span=50).mean(), label="Voltage (moving average)")
plt.ylabel("[V]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Current, label="Current")
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Current.ewm(span=50).mean(), label="Current (moving average)")
plt.ylabel("[A]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Power, label="Power")
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Power.ewm(span=50).mean(), label="Power (moving average)")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()

plt.figure(2)
plt.subplot(3, 1, 1)
plt.suptitle("Experiment 2 - calculations on FPGA")
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Voltage, label="Voltage")
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Voltage.ewm(span=50).mean(), label="Voltage (moving average)")
plt.ylabel("[V]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Current, label="Current")
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Current.ewm(span=50).mean(), label="Current (moving average)")
plt.ylabel("[A]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Power, label="Power")
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Power.ewm(span=50).mean(), label="Power (moving average)")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()

plt.figure(3)
plt.subplot(3, 1, 1)
plt.suptitle("Experiment 3 - calculations on CPU")
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Voltage, label="Voltage")
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Voltage.ewm(span=50).mean(), label="Voltage (moving average)")
plt.ylabel("[V]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Current, label="Current")
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Current.ewm(span=50).mean(), label="Current (moving average)")
plt.ylabel("[A]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Power, label="Power")
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Power.ewm(span=50).mean(), label="Power (moving average)")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()

plt.figure(4)
plt.subplot(3, 1, 1)
plt.suptitle("Experiment 4 - calculations on FPGA")
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Voltage, label="Voltage")
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Voltage.ewm(span=50).mean(), label="Voltage (moving average)")
plt.ylabel("[V]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Current, label="Current")
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Current.ewm(span=50).mean(), label="Current (moving average)")
plt.ylabel("[A]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Power, label="Power")
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Power.ewm(span=50).mean(), label="Power (moving average)")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()

plt.figure(5)
plt.subplot(3, 1, 1)
plt.suptitle("Measurement flat system power usage")
plt.plot(DataNoLoad.timestamp, DataNoLoad.Voltage, label="Voltage")
plt.plot(DataNoLoad.timestamp, DataNoLoad.Voltage.ewm(span=20).mean(), label="Voltage (moving average)")
plt.ylabel("[V]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(DataNoLoad.timestamp, DataNoLoad.Current, label="Current")
plt.plot(DataNoLoad.timestamp, DataNoLoad.Current.ewm(span=20).mean(), label="Current (moving average)")
plt.ylabel("[A]")
plt.xlabel("time")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(DataNoLoad.timestamp, DataNoLoad.Power, label="Power")
plt.plot(DataNoLoad.timestamp, DataNoLoad.Power.ewm(span=20).mean(), label="Power (moving average)")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()

plt.figure(6)
plt.subplot(2, 2, 1)
plt.suptitle("Comparison of Power for experiments")
plt.plot(DataKriaCPU.timestamp, DataKriaCPU.Power - StandardPower, label="Power Ex1 CPU")
plt.plot(DataKriaCPU.timestamp, (DataKriaCPU.Power - StandardPower).ewm(span=50).mean(), label="moving average")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(DataKriaFPGA.timestamp, DataKriaFPGA.Power - StandardPower, label="Power Ex2 FPGA")
plt.plot(DataKriaFPGA.timestamp, (DataKriaFPGA.Power - StandardPower).ewm(span=50).mean(), label="moving average")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(DataKriaCPU2.timestamp, DataKriaCPU2.Power - StandardPower, label="Power Ex3 CPU")
plt.plot(DataKriaCPU2.timestamp, (DataKriaCPU2.Power - StandardPower).ewm(span=50).mean(), label="moving average")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(DataKriaFPGA2.timestamp, DataKriaFPGA2.Power - StandardPower, label="Power Ex4 FPGA")
plt.plot(DataKriaFPGA2.timestamp, (DataKriaFPGA2.Power - StandardPower).ewm(span=50).mean(), label="moving average")
plt.ylabel("[W]")
plt.xlabel("time")
plt.legend()

plt.figure(7)
plt.subplot(2, 2, 1)
plt.plot(DataKriaCPU[(SpanCPU[0] < DataKriaCPU.timestamp) & (DataKriaCPU.timestamp < SpanCPU[1])].timestamp,
         DataKriaCPU[(SpanCPU[0] < DataKriaCPU.timestamp) &
                     (DataKriaCPU.timestamp < SpanCPU[1])].Power - StandardPower)
plt.plot(DataKriaCPU[(SpanCPU[0] < DataKriaCPU.timestamp) & (DataKriaCPU.timestamp < SpanCPU[1])].timestamp,
         (DataKriaCPU[(SpanCPU[0] < DataKriaCPU.timestamp) &
                      (DataKriaCPU.timestamp < SpanCPU[1])].Power - StandardPower).ewm(span=50).mean())
plt.subplot(2, 2, 2)
plt.plot(DataKriaFPGA[(SpanFPGA[0] < DataKriaFPGA.timestamp) & (DataKriaFPGA.timestamp < SpanFPGA[1])].timestamp,
         DataKriaFPGA[(SpanFPGA[0] < DataKriaFPGA.timestamp) &
                      (DataKriaFPGA.timestamp < SpanFPGA[1])].Power - StandardPower)
plt.plot(DataKriaFPGA[(SpanFPGA[0] < DataKriaFPGA.timestamp) & (DataKriaFPGA.timestamp < SpanFPGA[1])].timestamp,
         (DataKriaFPGA[(SpanFPGA[0] < DataKriaFPGA.timestamp) &
                       (DataKriaFPGA.timestamp < SpanFPGA[1])].Power - StandardPower).ewm(span=50).mean())
plt.subplot(2, 2, 3)
plt.plot(DataKriaCPU2[(SpanCPU2[0] < DataKriaCPU2.timestamp) & (DataKriaCPU2.timestamp < SpanCPU2[1])].timestamp,
         DataKriaCPU2[(SpanCPU2[0] < DataKriaCPU2.timestamp) &
                      (DataKriaCPU2.timestamp < SpanCPU2[1])].Power - StandardPower)
plt.plot(DataKriaCPU2[(SpanCPU2[0] < DataKriaCPU2.timestamp) & (DataKriaCPU2.timestamp < SpanCPU2[1])].timestamp,
         (DataKriaCPU2[(SpanCPU2[0] < DataKriaCPU2.timestamp) &
                       (DataKriaCPU2.timestamp < SpanCPU2[1])].Power - StandardPower).ewm(span=50).mean())
plt.subplot(2, 2, 4)
plt.plot(DataKriaFPGA2[(SpanFPGA2[0] < DataKriaFPGA2.timestamp) & (DataKriaFPGA2.timestamp < SpanFPGA[1])].timestamp,
         DataKriaFPGA2[(SpanFPGA2[0] < DataKriaFPGA2.timestamp) &
                       (DataKriaFPGA2.timestamp < SpanFPGA[1])].Power - StandardPower)
plt.plot(DataKriaFPGA2[(SpanFPGA2[0] < DataKriaFPGA2.timestamp) & (DataKriaFPGA2.timestamp < SpanFPGA[1])].timestamp,
         (DataKriaFPGA2[(SpanFPGA2[0] < DataKriaFPGA2.timestamp) &
                        (DataKriaFPGA2.timestamp < SpanFPGA[1])].Power - StandardPower).ewm(span=50).mean())

# 100Hz sampling * 100Hz sampling means i have to divide by 10000
PowerUsageNNCPU = sum(DataKriaCPU[(SpanCPU[0] < DataKriaCPU.timestamp) &
                                  (DataKriaCPU.timestamp < SpanCPU[1])].Power - StandardPower) / 10000
PowerUsageNNCPU2 = sum(DataKriaCPU2[(SpanCPU2[0] < DataKriaCPU2.timestamp) &
                                    (DataKriaCPU2.timestamp < SpanCPU2[1])].Power - StandardPower) / 10000
PowerUsageNNFPGA = sum(DataKriaFPGA[(SpanFPGA[0] < DataKriaFPGA.timestamp) &
                                    (DataKriaFPGA.timestamp < SpanFPGA[1])].Power - StandardPower) / 10000
PowerUsageNNFPGA2 = sum(DataKriaFPGA2[(SpanFPGA2[0] < DataKriaFPGA2.timestamp) &
                                      (DataKriaFPGA2.timestamp < SpanFPGA2[1])].Power - StandardPower) / 10000

print(f"Used Power (just NN) for CPU (mean from 2 runs): {(PowerUsageNNCPU + PowerUsageNNCPU2) / 2}")
print(f"Used Power (just NN) for FPGA (mean from 2 runs): {(PowerUsageNNFPGA + PowerUsageNNFPGA2) / 2}")

plt.show()
