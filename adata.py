from read_data import BSData
import matplotlib.pyplot as plt
import pandas as pd

Data2 = BSData('PWM500_BAR_1CML.txt')
Data1 = BSData('PWM500_BAR_1CMR.txt')
Data0 = BSData('PWM500_BAR_CENTERED.txt')
Data0.read_data()
Data1.read_data()
Data2.read_data()

dataa = pd.DataFrame(data={"Current": list(Data2.data["Current"])}, dtype=float)
timea = pd.DataFrame(data={"Time": list(Data2.data["Time"])})
directiona = pd.DataFrame(data={"Dir": list(Data2.data["Dir"])}, dtype=bool)
dataa_dir1 = dataa["Current"][directiona['Dir']]
dataa_dir0 = dataa["Current"][directiona['Dir'] == False]
timea_dir1 = timea["Time"][directiona['Dir']]
timea_dir0 = timea["Time"][directiona['Dir'] == False]
plt.scatter(timea_dir1[:5000], dataa_dir1[:5000])
plt.scatter(timea_dir0[:5000], dataa_dir0[:5000])

datab = pd.DataFrame(data={"Current": list(Data1.data["Current"])}, dtype=float)
timeb = pd.DataFrame(data={"Time": list(Data1.data["Time"])})
directionb = pd.DataFrame(data={"Dir": list(Data1.data["Dir"])}, dtype=bool)
datab_dir1 = datab["Current"][directionb['Dir']]
datab_dir0 = datab["Current"][directionb['Dir'] == False]
timeb_dir1 = timeb["Time"][directionb['Dir']]
timeb_dir0 = timeb["Time"][directionb['Dir'] == False]
plt.scatter(timeb_dir1[:5000], datab_dir1[:5000])
plt.scatter(timeb_dir0[:5000], datab_dir0[:5000])

datac = pd.DataFrame(data={"Current": list(Data0.data["Current"])}, dtype=float)
timec = pd.DataFrame(data={"Time": list(Data0.data["Time"])})
directionc = pd.DataFrame(data={"Dir": list(Data0.data["Dir"])}, dtype=bool)
datac_dir1 = datac["Current"][directionc['Dir']]
datac_dir0 = datac["Current"][directionc['Dir'] == False]
timec_dir1 = timec["Time"][directionc['Dir']]
timec_dir0 = timec["Time"][directionc['Dir'] == False]
plt.scatter(timec_dir1[:5000], datac_dir1[:5000])
plt.scatter(timec_dir0[:5000], datac_dir0[:5000])

plt.show()
