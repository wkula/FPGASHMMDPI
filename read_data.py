class BSData:

    def __init__(self, filename: str, separator: str = '\t', numheaderlines: int = 8):
        from array import array
        self.filename = filename
        self.sep = separator
        self.numheaderlines = numheaderlines
        self.data = {'Time': array('f'), 'Dir': array('B'),
                     'AX': array('f'), 'AY': array('f'), 'AZ': array('f'),
                     'GX': array('f'), 'GY': array('f'), 'GZ': array('f'),
                     'Current': array('f'), 'TimeDiff': array('f')}

    def read_data(self) -> None:
        with open(self.filename, 'r') as file:
            for i in range(self.numheaderlines):
                file.readline()
            line = file.readline()
            while line:
                temp_data = line.strip().split(sep=self.sep)
                self.data['Time'].append(float(temp_data[1]))
                self.data['AX'].append(float(temp_data[2]))
                self.data['AY'].append(float(temp_data[3]))
                self.data['AZ'].append(float(temp_data[4]))
                self.data['GX'].append(float(temp_data[5]))
                self.data['GY'].append(float(temp_data[6]))
                self.data['GZ'].append(float(temp_data[7]))
                self.data['Current'].append(float(temp_data[8]))
                self.data['Dir'].append(int(temp_data[9]))
                if len(self.data['Time']) > 1:
                    self.data['TimeDiff'].append(self.data['Time'][-1] - self.data['Time'][-2])
                else:
                    self.data['TimeDiff'].append(0)
                line = file.readline()

    def scatter_single(self, variable_name: str = "Current"):
        from pandas import DataFrame as DF
        import matplotlib.pyplot as plt

        direction = DF(data={"Dir": list(self.data["Dir"])}, dtype=bool)
        data = DF(data={variable_name: list(self.data[variable_name])}, dtype=float)
        time = DF(data={"Time": list(self.data["Time"])})
        data_dir1 = data[variable_name][direction['Dir']]
        data_dir0 = data[variable_name][direction['Dir'] == False]
        time_dir1 = time["Time"][direction['Dir']]
        time_dir0 = time["Time"][direction['Dir'] == False]

        plt.scatter(time_dir1, data_dir1)
        plt.scatter(time_dir0, data_dir0)
        plt.show()


if __name__ == '__main__':
    BSD = BSData('log.txt')
    BSD.read_data()
    BSD.scatter_single()
