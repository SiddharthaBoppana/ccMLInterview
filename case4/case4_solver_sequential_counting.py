'''
Problem Statement: The count needs to incremented sequentially, until it reaches 100_000. 

Explanation: In this solution, the principle of mutual exclusion is utilised to ensure synchronisation. 
Solution is suggested here: 
https://dev.to/lucaslealllc/solving-race-conditions-in-python-a-mutex-approach-for-efficiency-5ddg

Rest is similar to prior solution. 

'''

import threading
from pathlib import Path
import time

totalthreadNum = 15
totalIterNum = 100000

class ThreadCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_count = 0
        self.logFilePath = Path(r'caseLog.txt')
        self.logFilePath.parent.mkdir(parents=True, exist_ok=True)
        if self.logFilePath.exists():
            self.logFilePath.unlink()

    def count(self, threadNum):
        print(f'Thread {threadNum} started\n')
        time.sleep(3)
        while True:
            with self.lock:
                if self.current_count >= totalIterNum: # ensures count terminates at 100_000
                    break
                self.current_count += 1 
                # Mutex -> Ensures that other threads wait to access the count variable. thus count is increase by only 1
                count_num = self.current_count
                with open(str(self.logFilePath), 'a+', encoding='utf-8') as logFile:
                    logFile.write(f'Increased Count to {count_num} from thread {threadNum}\n')
                    logFile.write(f'Timestamp: {time.time()}\n')
                    logFile.write(f'After some time... Now the count is {count_num} from thread {threadNum}\n\n')

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