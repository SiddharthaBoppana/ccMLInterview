# General

- We would like to see your approaches for all of the cases.
- The cases do not have a single solution.
- Please don't forget to add comments for your approaches.
- You need to work in the relevant case directory in order to run the case scripts. For example:

```bash
cd case1
python case1.py
```

-----

# case1:
You are free to use all the optimization techniques. Prioritize what you consider important. We want you to optimize the code, in terms of using system resources, reducing the request-response time, etc.

**How to run current scripts?**

When you run case1.py, the Flask server will start running.
While the Flask server is running, you can send requests to the Flask server by running the DataLoader&ServerRequests.ipynb script.
By changing the "case_or_solution" variable in DataLoader&ServerRequests.ipynb, you can get different output results for the solution or case.
For example: when you run the notebook for case_or_solution = 'case', the file "./case1/case1_request_times.png" will be created. You can compare this file with "solution1_request_times.png".

Please don't forget to add comments for your approaches.

-----

# case2:
You are free to use all the optimization techniques. Prioritize what you consider important. We want you to optimize the code, in terms of using system resources, and reducing the total running time of the script.

Run case2.py and monitor the system resource utilization and the total running time of the script.

Please don't forget to add comments for your approaches.

----

# case3:
Optimize case3.ipynb (accelerate inference speed, reduce ram usage, etc.)
Define the comparison metrics and build/implement a strategy and tooling for comparing, profiling and monitoring these metrics

Please don't forget to add comments for your approaches.

----

# case4:
When you run case4.py, logs will be generated in the file './case4/caseLog.txt'.

As you can see in the "Example log1" below, writing to the file does not stop even though self.countNum >= totalIterNum.

Sometimes this can also be observed ("Example log2") like the numbers in the first line and the 3rd line are not equal. We ask you to debug this and write a proper multithreading solution.

Example log1:
```
'''
Increased Count to 100012 from thread 1
Timestamp: 1710174085.612473
After some time... Now the count is 100012 from thread 1
'''
```

Example log2:
```
'''
Increased Count to 591 from thread 1
Timestamp: 1710174085.612473
After some time... Now the count is 623 from thread 1
'''
```
Please don't forget to add comments for your approaches.