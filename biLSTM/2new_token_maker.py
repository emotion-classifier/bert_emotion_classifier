import os
import logging
from kiwipiepy import Kiwi

class ReaderExam:
    def __init__(self, filePath):
        self.file = open(filePath, encoding='utf-8')

    def read(self, id):
        if id == 0: self.file.seek(0)
        return self.file.readline()

class IOHandler:
    def __init__(self, input, output):
        self.input = open(input, encoding='utf-8')
        self.output = open(output, 'w', encoding='utf-8')

    def read(self, id):
        if id == 0:
            self.input.seek(0)
        return self.input.readline()

    def write(self, id, res):
        print('Analyzed %dth row' % id)
        for x in res[0][0]:
            self.output.write(x[0] + ' ')
        self.output.write('\n')

    def __del__(self):
        self.input.close()
        self.output.close()

train = True
if train is True:
    reader = ReaderExam('testdataset.csv')
    kiwi = Kiwi(8)
    kiwi.extractAddWords(reader.read, 10, 10, 0.25, -3)

    kiwi.prepare()
    handle = IOHandler('testdataset.csv', 'tokened_testdataset.csv')
    kiwi.analyze(handle.read, handle.write)
