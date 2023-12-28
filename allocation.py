import pulp as pl
import numpy as np
import sys

args = sys.argv
unitCount = int(args[1]) # 总计算单元数
taskFile = args[2]
costFile = args[3]
duration = int(args[4])
task = []

def generate_t():
    # 读取time cost文件
    cost = []

    # 打开文件并逐行读取内容
    with open(costFile, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()

        # 将每一行内容拆分为元素，并添加到矩阵中
        for line in lines:
            elements = line.strip().split()
            elements = list(map(float, elements))
            cost.append(elements)

    # 生成 task * unitCount 矩阵
    t = []
    for i in range(len(task)):
        count = task[i]
        if count > 0:
            temp = []
            temp.append(cost[i][0])
            for j in range(unitCount-1):
                temp.append(cost[i][1])
            for j in range(count):
                t.append(temp)
    return t

def generate_task():

    # 打开task number文件并逐行读取内容
    with open(taskFile, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()

        # 将每一行内容拆分为元素，并添加到矩阵中
        for line in lines:
            elements = line.strip().split()
            for ele in elements:
               task.append(int(ele))

def GPU_First(n, unitCount):
    # Define the model
    model = pl.LpProblem(name="Model-GPU", sense=pl.LpMinimize)
    
    # Define the decision variables
    x = {(i, j): pl.LpVariable(name=f"x({i},{j})", lowBound=0, upBound=1, cat=pl.LpInteger)
         for i in range(0, n) for j in range(0, unitCount)}
    
    c = 0
    # Add constraints
    # 1个矩阵最多被1个计算单元处理
    for i in range(0, n):
        model += (pl.lpSum(x[(i, j)] for j in range(0, unitCount)) == 1, "constrain_" + str(c))
        c += 1
        
    # 每个计算单元处理 >= 1个矩阵？
    
    # GPU主计算
    for j in range(1, unitCount):
        model += (pl.lpSum(t[i][j] * x[(i,j)] for i in range(0, n)) <= pl.lpSum(t[i][0] * x[(i,0)] for i in range(0, n)), "constrain_" + str(c))
        c += 1
    
    # Set the objective
    model += pl.lpSum(t[i][0] * x[(i,0)] for i in range(0, n))
    
    # Solve the optimization problem
    solver = pl.PULP_CBC_CMD(timeLimit=duration)
    status = model.solve(solver)
    
    # Get the results
    print("====GPU====")
    print(f"status: {model.status}, {pl.LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")

    # 获取所有变量的名称
    results = np.zeros((n, unitCount))
    for var in model.variables():
        temp = var.name.split(",")
        x = int(temp[0][2:])
        y = int(temp[1][:len(temp[1]) - 1])
        results[x, y] = var.value()
    #print(results)

    prefix = 0
    allocations = np.array([])
    for i, item in enumerate(task):
        column_sums = np.sum(results[prefix:prefix + item, :], axis=0)
        prefix = prefix + item
        allocations = np.append(allocations, column_sums, axis=0)
    allocations = allocations.reshape(-1, unitCount)
    print(allocations)
        
    '''
    for name, constraint in model.constraints.items():
        print(f"{name}: {constraint.value()} {constraint.toDict()}")
    '''

def CPU_First(n, unitCount):
    # Define the model
    model = pl.LpProblem(name="Model-CPU", sense=pl.LpMinimize)
    
    # Define the decision variables
    x = {(i, j): pl.LpVariable(name=f"x({i},{j})", lowBound=0, upBound=1, cat=pl.LpInteger)
         for i in range(0, n) for j in range(0, unitCount)}
    
    c = 0
    # Add constraints
    # 1个矩阵最多被1个计算单元处理
    for i in range(0, n):
        model += (pl.lpSum(x[(i, j)] for j in range(0, unitCount)) == 1, "constrain_" + str(c))
        c += 1
    
    # 每个计算单元处理 >= 1个矩阵？
    
    # CPU主计算
    for j in range(0, unitCount - 1):
        model += (pl.lpSum(t[i][j] * x[(i,j)] for i in range(0, n)) <= pl.lpSum(t[i][unitCount - 1] * x[(i, unitCount - 1)] for i in range(0, n)), "constrain_" + str(c))
        c += 1
    
    # Set the objective
    model += pl.lpSum(t[i][unitCount-1] * x[(i,unitCount-1)] for i in range(0, n))
    
    # Solve the optimization problem
    solver = pl.PULP_CBC_CMD(timeLimit=duration)
    status = model.solve(solver)
    
    # Get the results
    print("====CPU====")
    print(f"status: {model.status}, {pl.LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")
     
    results = np.zeros((n, unitCount))
    for var in model.variables():
        temp = var.name.split(",")
        x = int(temp[0][2:])
        y = int(temp[1][:len(temp[1]) - 1])
        results[x, y] = var.value()
    
    prefix = 0
    allocations = np.array([])
    for i, item in enumerate(task):
        column_sums = np.sum(results[prefix:prefix + item, :], axis=0)
        prefix = prefix + item
        allocations = np.append(allocations, column_sums, axis=0)
    allocations = allocations.reshape(-1, unitCount)
    print(allocations)
     
    '''
    for name, constraint in model.constraints.items():
        print(f"{name}: {constraint.value()}")
    '''

generate_task()
t = generate_t()
n = len(t) # 任务数量

print("task num:", n, "calc units:", unitCount)
GPU_First(n, unitCount)
CPU_First(n, unitCount)
