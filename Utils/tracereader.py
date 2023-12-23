import warnings
import os
from Utils.filereader import fileload


class TraceReader:
    def __init__(self, log_file_path=''):
        self.log_file_path = log_file_path
        self.fileloader = fileload(self.log_file_path)
        self.trace = {}

    def append(self, data_dict):
        if not self.trace:
            self.trace = data_dict
        else:
            for key, value in data_dict.items():
                if not key in self.trace:
                    warnings.warn('The key %s to log does not exist in the trace' % key)
                self.trace[key].extend(value)

    def get_trace(self):
        for asegment in self.fileloader:
            self.append(asegment)
        return self.trace


if __name__ == "__main__":
    from pprint import pprint
    log_file_path = os.getcwd() +'\\log\\2023-12-22_15-22-26.pkl'
    aTR = TraceReader(log_file_path=log_file_path)
    trace = aTR.get_trace()
    pprint(trace)
    # for key, value in trace.items():
    #     print(key, ':')
    #     pprint(value)
