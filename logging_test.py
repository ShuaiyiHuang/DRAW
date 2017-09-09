import logging
import os
logPath='./log'
fileName='log1'
if os.path.exists(logPath)==False:
    os.mkdir(logPath)
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logFormatter = logging.Formatter("%(asctime)s  %(message)s")
rootLogger = logging.getLogger()
#if you forget to set level,you will print nothing
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("{0}/{1}.txt".format(logPath, fileName))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

a=[9,9,0]
logging.info('Test log:{}'.format(len(a)))
logging.debug('wrong')