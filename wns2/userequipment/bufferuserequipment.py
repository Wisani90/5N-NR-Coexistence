import random
import math
import logging
import numpy.random
import userequipment

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class BufferUserEquipment(userequipment.UserEquipment):
    def __init__(self, bufferSize):

        self.bufferSize = bufferSize

    def computeDataRate(self):
        self.bufferSize

        return

