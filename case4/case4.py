import threading
from pathlib import Path
import time

totalthreadNum = 15
totalIterNum = 100000

class ThreadCounter:
    def __init__(self):
        self.countNum = 0
        self.logFilePath = Path(r'caseLog.txt')
        self.logFilePath.parent.mkdir(parents=True, exist_ok=True)
        if self.logFilePath.exists():
            self.logFilePath.unlink()

    def count(self, threadNum):
        print(f'Thread {threadNum} started\n')
        time.sleep(3)
        while True:
            if self.countNum >= totalIterNum:
                break
            with open(str(self.logFilePath), 'a+', encoding='utf-8') as logFile:
                self.countNum += 1
                # print(f'Increased Count to {self.countNum} from thread {threadNum}\n')
                logFile.write(f'Increased Count to {self.countNum} from thread {threadNum}\n')
                # print(f'Timestamp: {time.time()}\n')
                logFile.write(f'Timestamp: {time.time()}\n')
                # print(f'After some time... Now the count is {self.countNum} from thread {threadNum}\n\n')
                logFile.write(f'After some time... Now the count is {self.countNum} from thread {threadNum}\n\n')

tC = ThreadCounter()

startTime = time.time()
threads = []
for i in range(totalthreadNum):
    t = threading.Thread(target=tC.count, args=(i,))
    threads.append(t)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

endTime = time.time()
print(f'Time taken: {endTime - startTime} seconds')