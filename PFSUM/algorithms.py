import math

#5 algorithms: (1)SUM, (2)FSUM, (3)PFSUM, (4)PDLA, (5)Offline Optimal Dynamic Programming

#SUM purchases a bahncard at regular request whenever its regular T-recent-cost at time t is at least gamma.
def SUM(instance, T, C, beta):
    length = len(instance)
    if (length == 0):
        return (0, [])

    regular_cost = [0] * length
    gamma = C / (1 - beta)
    cost = 0
    solution = []
    T_recent_regular_cost = 0
    last_buy_time = -T
    
    for i in range(0, length):
        if (i - T >= 0):
            T_recent_regular_cost -= regular_cost[i - T]

        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
            if (T_recent_regular_cost + instance[i] >= gamma):
                #buy a bahncard
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
                T_recent_regular_cost = 0
            else:
                cost += instance[i]
                T_recent_regular_cost += instance[i]
                regular_cost[i] = instance[i]

    return (cost, solution)


#FSUM purchases a bahncard at a regular request whenever the predicted T-future-cost at time t is at least gamma.
def FSUM(instance, T, C, beta, prediction):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    cost = 0
    solution = []
    last_buy_time = -T

    for i in range(0, length):
        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
            if (prediction[i] >= gamma):
                #buy a bahncard
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
            else:
                cost += instance[i]

    return (cost, solution)


#PFSUM purchases a bahncard at a regular request whenever (i) T-recent-cost at t is at least gamma, and (ii) the predicted T-future-cost at t is also at least gamma.
def PFSUM(instance, T, C, beta, prediction):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    cost = 0
    solution = []
    last_buy_time = -T
    T_recent_cost = 0

    for i in range(0, length):
        T_recent_cost += instance[i]
        if (i - T >= 0):
            T_recent_cost -= instance[i - T]

        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
            if (T_recent_cost >= gamma and prediction[i] >= gamma):
                #buy a bahncard
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
            else:
                cost += instance[i]
            
    return (cost, solution)


#Online primal-dual learning augmented algorithm for the bahncard problem, which follows Algorithm 8 of Bamas's paper published at NeurIPS 2020.
def PDLA_FOR_BAHNCARD(instance, T, C, beta, Lambda, predicted_solution):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    c_lambda = (1 + 1 / gamma)**(Lambda*gamma)
    c_1_by_lambda = (1 + 1 / gamma)**(gamma/Lambda)

    pre_x_sum = [0] * length
    d = []
    f = []
    cost = 0
    solution = []
    latest_predicted_idx = -1
    
    for i in range(0, length):
        while (latest_predicted_idx + 1 < len(predicted_solution)
            and predicted_solution[latest_predicted_idx + 1] <= i):
            latest_predicted_idx += 1

        if (i > 0):
            pre_x_sum[i] = pre_x_sum[i - 1]

        T_recent_x_sum = pre_x_sum[i]
        if (i - T >= 0):
            T_recent_x_sum -= pre_x_sum[i - T]

        if (instance[i] == 0):
            continue

        #a request arrived at time i
        if (T_recent_x_sum >= 1):
            #for a minimal update, the primal cost is beta * price
            cost += beta * instance[i]
        else:
            if (latest_predicted_idx >= 0 and predicted_solution[latest_predicted_idx] + T - 1 >= i):
                #big update
                x_increment = instance[i] * (T_recent_x_sum + 1 / (c_lambda - 1)) / gamma
                pre_x_sum[i] += x_increment

                #primal cost brought by the increase in x
                cost += instance[i] * (1 - beta) / (c_lambda - 1) + instance[i]
            else:
                #small update
                x_increment = instance[i] * (T_recent_x_sum + 1 / (c_1_by_lambda - 1)) / gamma
                pre_x_sum[i] += x_increment

                #primal cost brought by the increase in x
                cost += instance[i] * (1 - beta) / (c_1_by_lambda - 1) + instance[i]

            solution.append((i, x_increment))

    return (cost, solution)


# Offline optimal algorithm return the optimal cost and the corresponding time list of buying bahncard. The algorithm is based on dynamic programming.
def OFFLINE_OPTIMAL(instance, T, C, beta):
    length = len(instance)
    if (length == 0):
        return (0, [])

    #dp[i][1] indicates the optimal cost for all requests arrived in time interval [0, i], and a bahncard expires at time i. Besides, there is no bahncard expire at time i in the case of dp[i][0].
    dp = []
    #dp_pre[i][0/1] records the suboptimal structure of dp[i][0/1]
    dp_pre = []

    pre_sum = [instance[0]]
    for i in range(1, length):
        pre_sum.append(pre_sum[i - 1] + instance[i])

    dp.append([instance[0], C + beta * instance[0]])
    dp_pre.append([[-1, -1], [-1, -1]])

    for i in range(1, length):
        dp.append([math.inf, math.inf])
        dp_pre.append([[-1, -1], [-1, -1]])

        if (i - T < 0):
            dp[i][1] = C + beta * pre_sum[i]
        else:
            if (dp[i - T][0] < dp[i - T][1]):
                dp[i][1] = dp[i - T][0] + C + beta * (pre_sum[i] - pre_sum[i - T])
                dp_pre[i][1] = [i - T, 0]
            else:
                dp[i][1] = dp[i - T][1] + C + beta * (pre_sum[i] - pre_sum[i - T])
                dp_pre[i][1] = [i - T, 1]

        if (dp[i - 1][0] < dp[i - 1][1]):
            dp[i][0] = dp[i - 1][0] + instance[i]
            dp_pre[i][0] = [i - 1, 0]
        else:
            dp[i][0] = dp[i - 1][1] + instance[i]
            dp_pre[i][0] = [i - 1, 1]

    p = length - 1
    idx = 0
    optimal_cost = dp[length - 1][0]
    if (dp[length - 1][1] < dp[length - 1][0]):
        idx = 1
        optimal_cost = dp[length - 1][1]

    optimal_solution = []

    #restore the optimal solution
    while p != -1:
        if (idx == 1):
            optimal_solution.append(max(0, p - T + 1))

        new_p = dp_pre[p][idx][0]
        new_idx = dp_pre[p][idx][1]
        p = new_p
        idx = new_idx

    optimal_solution.reverse()

    return (optimal_cost, optimal_solution)

