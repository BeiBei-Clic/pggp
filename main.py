import time
from src.nesymres.architectures.model import Model
from data_conversion import *
import numpy as np
import argparse
import operator
import math
import os
import random
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import json, re
import omegaconf
import sympy
from src.nesymres.dclasses import FitParams, BFGSParams
from backpropagation import *
import warnings
import logging
import sys
from contextlib import contextmanager

warnings.filterwarnings("ignore")
# 禁用PyTorch Lightning的日志输出
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# 禁用特定的升级警告
warnings.filterwarnings("ignore", message="Lightning automatically upgraded your loaded checkpoint")

@contextmanager
def suppress_stdout():
    """临时禁用标准输出"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



def transformer_init():
    with open('jupyter/100M/eq_setting.json', 'r') as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load("100M/config.yaml")
    return eq_setting, cfg


def get_res_transformer(X, y, BFGS, first_call=False):
    input_X = np.array(X)
    input_Y = np.array(y)
    X = torch.from_numpy(input_X)
    y = torch.from_numpy(input_Y)

    eq_setting, cfg = transformer_init()

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(word2id=eq_setting["word2id"],
                           id2word={int(k): v for k, v in eq_setting["id2word"].items()},
                           una_ops=eq_setting["una_ops"],
                           bin_ops=eq_setting["bin_ops"],
                           total_variables=list(eq_setting["total_variables"]),
                           total_coefficients=list(eq_setting["total_coefficients"]),
                           rewrite_functions=list(eq_setting["rewrite_functions"]),
                           bfgs=bfgs,
                           beam_size=cfg.inference.beam_size
                           # This parameter is a tradeoff between accuracy and fitting time
                           )
    weights_path = "weights/100M.ckpt"
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    if first_call:
        with suppress_stdout():
            _ = fitfunc(X, y, BFGS=True)

        final_equation = model.get_equation()

        with suppress_stdout():
            prefix_symbol_list = fitfunc(X, y, BFGS=False)

        return prefix_symbol_list, final_equation


    if BFGS:
        try:
            with suppress_stdout():
                prefix_symbol_list = fitfunc(X, y, BFGS)
        except (ValueError, RuntimeError):
            return [None, None]

        final_equation = model.get_equation()

        return prefix_symbol_list, final_equation, model.total_c, model.total_bfgs_time
    else:
        with suppress_stdout():
            prefix_symbol_list = fitfunc(X, y, BFGS)
        return prefix_symbol_list



def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

def protectedDiv(left, right):
    if right == 0:
        return left
    res = left / right
    if res > 1e7:
        return 1e7
    if res < -1e7:
        return -1e7
    return res
def protectedExp(arg):
    if is_complex(arg):
        return 99999
    if arg > 10:
        arg = 10
    return math.exp(arg)

def protectedLog(arg):
    if abs(arg) < 1e-5:
        arg = 1e-5
    return math.log(abs(arg))


def protectedAsin(x):
    if x < -1.0 or x > 1.0:
        return 99999
    else:
        return math.asin(x)

def protectedSqrt(x):
    if x < 0:
        return 99999
    else:
        return math.sqrt(x)

def protectedAcos(x):
    if x < -1.0 or x > 1.0:
        return 99999
    else:
        return math.asin(x)

def protectedAtan(x):
    try:
        return math.atan(x)
    except Exception:
        return 99999


def convert_to_list(trimmed_eq, accurate_constant=False):

    token_list= []
    for i in range(n_variables):
        token_list.append('x_'+str(i+1))

    for i, x in enumerate(trimmed_eq):
        if x == 'constant' :
            if not accurate_constant:
                trimmed_eq[i]='rand505'

        elif  x.startswith('x_'):
            index=int(x[-1])
            trimmed_eq[i]='x_'+str(index-1)
        else:

            trimmed_eq[i] = x

    return trimmed_eq



def get_creator():
    pset = gp.PrimitiveSet("MAIN", n_variables)
    rename_kwargs = {"ARG{}".format(i): 'x_'+str(i) for i in range(n_variables)}
    for k, v in rename_kwargs.items():
        pset.mapping[k].name = v
    pset.renameArguments(**rename_kwargs)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(protectedMul, 2, name='mul')
    pset.addPrimitive(protectedDiv, 2, name='div')
    pset.addPrimitive(protectedExp, 1, name="exp")
    pset.addPrimitive(protectedLog, 1, name="ln")
    pset.addPrimitive(protectedSqrt, 1, name="sqrt")
    pset.addPrimitive(operator.pow, 2, name="pow")
    pset.addPrimitive(operator.abs, 1, name="abs")
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.tan, 1)
    pset.addPrimitive(protectedAsin, 1,name='asin')
    pset.addPrimitive(protectedAcos, 1,name='acos')
    pset.addPrimitive(protectedAtan, 1, name='atan')
    pset.addEphemeralConstant("rand505", partial(random.uniform, -5, 5))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    return pset, creator

def init_individual(pset, creator, trimmed_eq):

    plist=[]

    for t in trimmed_eq:
        if t in pset.mapping:
            plist.append(pset.mapping[t])

        elif t in [ '-3','-2','-1','0', '1', '2', '3', '4', '5']:

            if t not in pset.terminals[pset.ret]:
                pset.addTerminal(float(t), name=t)
                term = pset.terminals[pset.ret][-1]

            else:
                for i, term in enumerate(pset.terminals[pset.ret]):
                    if term.name == t:
                        break
                term = pset.terminals[pset.ret][i]()

            plist.append(term)


        elif t=='rand505':

            index_rand505=n_variables
            term = pset.terminals[pset.ret][index_rand505]()

            plist.append(term)

        else:
            value = float(t)

            pset.addTerminal(value, name=t)

            plist.append(pset.terminals[pset.ret][-1])

    individual = creator.Individual(plist)

    return individual

def is_complex(number):
    """
    Determine whether the given number is a complex number.

    :param number: The number to be checked.
    :return: True if the number is complex, False otherwise.
    """
    # A number is complex if it has a non-zero imaginary part
    return isinstance(number, complex) and number.imag != 0


def evalSymbReg(individual, pset, toolbox):
    wrong_mark=999999999
    
    # 处理表达式编译错误，包括语法错误和过于复杂的嵌套
    try:
        func = toolbox.compile(expr=individual)
    except (SyntaxError, RecursionError, MemoryError):
        return wrong_mark,
    
    sqerrors=[]
    for i, x in enumerate(input_X):
        try:
            try:
                try:
                    result=func(*x)
                except (AttributeError, ValueError, ZeroDivisionError):
                    return wrong_mark,

                if is_complex(result):
                    return wrong_mark,
                else:
                    tmp = (func(*x) - input_Y[i]) ** 2
                    sqerrors.append(tmp)
            except TypeError:
                return wrong_mark,

        except OverflowError:
            return wrong_mark,
    try:
        res = math.sqrt(math.fsum(sqerrors) / len(input_X))
    except TypeError:
        return  wrong_mark,

    return res,

def mutReplace(individual, pset, toolbox, creator):

    global  input_X
    node_index = random.randrange(len(individual))
    if len(individual) == 1:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    while individual[node_index].arity == 0:
        node_index = random.randrange(len(individual))
    try:
        semantic = backpropogation(individual, pset, (input_X, input_Y), node_index)
    except OverflowError:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    if semantic == 'nan':
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    try:
        symbol_list, prediceted_equation,total_c, bfgs_time= get_res_transformer(input_X, semantic, BFGS=True)

    except (ValueError, RuntimeError):
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    if symbol_list is None:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)



    symbol_list=convert_to_list(symbol_list,accurate_constant=True)


    new_subtree=init_individual(pset, creator, symbol_list)

    CT_slice = individual.searchSubtree(node_index)

    individual[CT_slice] = new_subtree

    return individual,

def mutate(individual, pset, creator, toolbox, p_subtree=0.05):

    if random.random() < p_subtree:

        return mutReplace(individual, pset=pset, toolbox=toolbox, creator=creator)
    else:

        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)


def main(filename=None, seed=8346, data_count=1000, generations=200, population_size=300):
    # 默认使用指定的数据集
    if filename is None:
        file_path = "/home/xyh/pggp/dataset/Feynman_with_units/I.6.2"
    else:
        dataset_path = "benchmark_dataset/"
        file_path = os.path.join(dataset_path, filename + '.txt')
    
    global n_variables
    global input_X
    global input_Y
    file_contents = {}

    all_X = []
    all_Y = []

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            count = 0
            for line in file:
                if count >= data_count:  # 读取指定条数的数据
                    break
                
                data = line.strip().split()
                x = list(map(float, data[:-1]))
                y = float(data[-1])
                all_X.append(x)
                all_Y.append(y)
                count += 1

    # 按80%训练集和20%测试集分割数据
    total_samples = len(all_X)
    train_size = int(total_samples * 0.8)
    
    train_X = np.array(all_X[:train_size])
    train_Y = np.array(all_Y[:train_size])
    test_X = np.array(all_X[train_size:])
    test_Y = np.array(all_Y[train_size:])

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    # 对输入特征进行标准化
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)
    
    # 对目标变量进行标准化
    train_Y_scaled = scaler_Y.fit_transform(train_Y.reshape(-1, 1)).flatten()
    test_Y_scaled = scaler_Y.transform(test_Y.reshape(-1, 1)).flatten()
    
    # 转换为列表格式供后续使用
    input_X = train_X_scaled.tolist()
    input_Y = train_Y_scaled.tolist()

    n_variables = len(input_X[0])


    pset, creator = get_creator()

    # fixme

    symbol_list, equation, =  get_res_transformer(input_X, input_Y, BFGS=False, first_call=True)

    total_variables = ["x_1", "x_2", "x_3"][:n_variables]

    # 使用标准化的测试数据进行预测
    X_dict = {x: test_X_scaled[:, idx] for idx, x in enumerate(total_variables)}
    y_pred_scaled = np.array(sympy.lambdify(",".join(total_variables), equation)(**X_dict))
    
    # 使用标准化后的数据计算MSE
    transformer_rmse = math.sqrt(mean_squared_error(test_Y_scaled.ravel(), y_pred_scaled.ravel()))

    # early stop
    if transformer_rmse < 1e-10:
        print('early stop')
        return

    # fixme
    trimmed_eq = convert_to_list(symbol_list, accurate_constant=False)


    # set toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    # toolbox.register("population", init_population, list, creator.Individual, eq=trimmed_eq, primitive_set=pset, num_individuals=20)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg, pset=pset, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate, pset=pset, creator=creator, toolbox=toolbox, p_subtree=0.025)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

    np.random.seed(seed)
    random.seed(seed)

    pop = toolbox.population(n=population_size)


    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)


    # 手动实现进化算法来跟踪适应度变化
    fitness_trend = []
    
    start_time = time.time()
    
    # 评估初始种群
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    
    # 记录初始代的最佳适应度
    current_best = min([ind.fitness.values[0] for ind in pop])
    fitness_trend.append(current_best)
    hof.update(pop)
    
    # 进化过程
    for gen in range(generations):
        # 选择
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # 交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # 交叉概率
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:  # 变异概率
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 评估需要评估的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # 替换种群
        pop[:] = offspring
        
        # 更新名人堂
        hof.update(pop)
        
        # 记录当前代的最佳适应度
        current_best = min([ind.fitness.values[0] for ind in pop])
        fitness_trend.append(current_best)
        
        # 每50代输出一次进度
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"Generation {gen+1}: Best fitness = {current_best}")
    
    end_time = time.time()
    training_time = end_time - start_time

    last_generation_fitness = min([ind.fitness.values[0] for ind in pop])

    # 编译最佳个体，添加异常处理
    try:
        func = toolbox.compile(expr=hof[0])
    except (SyntaxError, ValueError, TypeError) as e:
        print(f"编译错误: {e}，使用简单线性函数")
        func = lambda *args: args[0] if args else 0
    
    # 计算测试集RMSE - 使用标准化后的数据，添加异常值处理
    test_predictions_scaled = []
    problematic_predictions_count = 0
    for row in test_X_scaled:
        try:
            pred = func(*row)
            # 检查异常值
            if np.isnan(pred) or np.isinf(pred) or np.iscomplex(pred):
                pred = 999999999
                problematic_predictions_count += 1
            test_predictions_scaled.append(pred)
        except Exception as e:
            pred = 999999999
            problematic_predictions_count += 1
            test_predictions_scaled.append(pred)
    
    if problematic_predictions_count > 0:
        print(f"警告: 在测试集预测中，有 {problematic_predictions_count} 个预测值被替换为 999999999。")
    
    test_rmse = np.sqrt(mean_squared_error(test_Y_scaled, test_predictions_scaled))
    print(f"测试集RMSE: {test_rmse:.6f}")
    
    return {
        'test_rmse': test_rmse,
        'fitness_trend': fitness_trend,
        'training_time': training_time,
        'best_individual': str(hof[0])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer-GP')
    parser.add_argument('--dataset_name', type=str, default=None, help='数据集名称，如果不指定则使用默认的Feynman数据集')
    parser.add_argument('--data_count', type=int, default=10, help='使用的数据条数 (默认: 10)')
    parser.add_argument('--generations', type=int, default=200, help='遗传算法迭代次数 (默认: 200)')
    parser.add_argument('--population_size', type=int, default=300, help='种群大小 (默认: 300)')
    parser.add_argument('--seed', type=int, default=8346, help='随机种子 (默认: 8346)')
    args = parser.parse_args()
    main(args.dataset_name, args.seed, args.data_count, args.generations, args.population_size)
