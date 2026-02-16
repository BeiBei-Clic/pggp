# pggp_sr.py - PGGP标准化符号回归实验

import argparse
import json
import os
import random
import math
import copy
import operator
import warnings
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import sympy
import omegaconf

# 屏蔽已知且不影响流程的噪声告警
warnings.filterwarnings(
    "ignore",
    message=r".*multiple `ModelCheckpoint` callback states.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Lightning automatically upgraded your loaded checkpoint.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*To copy construct from a tensor.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*overflow encountered in scalar power.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*invalid value encountered in scalar power.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*divide by zero encountered in scalar power.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*invalid value encountered in scalar multiply.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*overflow encountered in scalar multiply.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Casting complex values to real discards the imaginary part.*",
    category=np.exceptions.ComplexWarning,
)
logging.getLogger("pytorch_lightning.utilities.migration.migration").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.ERROR)

# 设置PyTorch safe globals以兼容旧版本模型
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
torch.serialization.add_safe_globals([ModelCheckpoint])

from deap import base, creator, tools, gp
from src.nesymres.architectures.model import Model
from src.nesymres.dclasses import FitParams, BFGSParams
from backpropagation import backpropogation


def convert_to_list(trimmed_eq, accurate_constant=False, n_variables=None):
    """转换表达式列表

    Args:
        trimmed_eq: 表达式token列表
        accurate_constant: 是否使用精确常量
        n_variables: 变量数量

    Returns:
        转换后的表达式列表
    """
    result = list(trimmed_eq)

    for i, x in enumerate(result):
        if x == 'constant':
            if not accurate_constant:
                result[i] = 'rand505'
        elif x.startswith('x_'):
            index = int(x.split('_')[1])
            result[i] = 'x_' + str(index - 1)

    return result


def load_feynman_csv(csv_path, max_input_points=None):
    """读取标准CSV格式的Feynman数据集

    Args:
        csv_path: CSV文件路径
        max_input_points: 最大数据点数量，None表示全部

    Returns:
        (ground_truth, X, y) - 表达式、特征矩阵、目标值
    """
    with open(csv_path, 'r') as f:
        ground_truth = f.readline().strip()

    data = np.loadtxt(csv_path, skiprows=1, max_rows=max_input_points)
    X, y = data[:, :-1], data[:, -1]

    return ground_truth, X, y


def protectedMul(left, right):
    try:
        return left * right
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
        return math.acos(x)


def protectedAtan(x):
    try:
        return math.atan(x)
    except Exception:
        return None


def is_complex(number):
    """判断是否为复数"""
    return isinstance(number, complex) and number.imag != 0


class PGGPWrapper:
    """PGGP算法封装类"""

    def __init__(self, seed=0, device="cuda:0"):
        self.seed = seed
        self.device = device
        np.random.seed(seed)
        random.seed(seed)

        # 数据相关
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.n_variables = None

        # GP组件
        self.pset = None
        self.creator = None
        self.toolbox = None
        self.pop = None

        # Transformer配置
        self.eq_setting = None
        self.cfg = None
        self.model = None

        # 获取脚本所在目录
        self.script_dir = Path(__file__).parent

    def _transformer_init(self):
        """初始化Transformer配置和模型"""
        config_path = self.script_dir / 'jupyter' / '100M' / 'eq_setting.json'
        yaml_path = self.script_dir / '100M' / 'config.yaml'

        with open(config_path, 'r') as json_file:
            self.eq_setting = json.load(json_file)
        self.cfg = omegaconf.OmegaConf.load(yaml_path)

    def get_res_transformer(self, X, y, BFGS=False, first_call=False):
        """调用Transformer获取方程表达式"""
        input_X = np.array(X)
        input_Y = np.array(y)
        X = torch.from_numpy(input_X)
        y = torch.from_numpy(input_Y)

        if self.eq_setting is None:
            self._transformer_init()

        bfgs = BFGSParams(
            activated=self.cfg.inference.bfgs.activated,
            n_restarts=self.cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=self.cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=self.cfg.inference.bfgs.normalization_o,
            idx_remove=self.cfg.inference.bfgs.idx_remove,
            normalization_type=self.cfg.inference.bfgs.normalization_type,
            stop_time=self.cfg.inference.bfgs.stop_time,
        )

        params_fit = FitParams(
            word2id=self.eq_setting["word2id"],
            id2word={int(k): v for k, v in self.eq_setting["id2word"].items()},
            una_ops=self.eq_setting["una_ops"],
            bin_ops=self.eq_setting["bin_ops"],
            total_variables=list(self.eq_setting["total_variables"]),
            total_coefficients=list(self.eq_setting["total_coefficients"]),
            rewrite_functions=list(self.eq_setting["rewrite_functions"]),
            bfgs=bfgs,
            beam_size=self.cfg.inference.beam_size
        )

        weights_path = self.script_dir / "weights" / "100M.ckpt"
        if self.model is None:
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到可用CUDA，请在GPU环境运行该脚本。")
            self.model = Model.load_from_checkpoint(
                str(weights_path),
                cfg=self.cfg.architecture,
                map_location=self.device,
            )
            self.model.to(self.device)
            self.model.eval()

        fitfunc = partial(self.model.fitfunc, cfg_params=params_fit)

        if first_call:
            _ = fitfunc(X, y, BFGS=True)
            final_equation = self.model.get_equation()
            prefix_symbol_list = fitfunc(X, y, BFGS=False)
            return prefix_symbol_list, final_equation

        if BFGS:
            try:
                prefix_symbol_list = fitfunc(X, y, BFGS)
            except ValueError:
                return [None, None]
            final_equation = self.model.get_equation()
            return prefix_symbol_list, final_equation, self.model.total_c, self.model.total_bfgs_time
        else:
            prefix_symbol_list = fitfunc(X, y, BFGS)
            return prefix_symbol_list

    def get_creator(self):
        """创建GP原始集和creator"""
        pset = gp.PrimitiveSet("MAIN", self.n_variables)
        rename_kwargs = {"ARG{}".format(i): 'x_' + str(i) for i in range(self.n_variables)}
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
        pset.addPrimitive(protectedAsin, 1, name='asin')
        pset.addPrimitive(protectedAcos, 1, name='acos')
        pset.addPrimitive(protectedAtan, 1, name='atan')
        pset.addEphemeralConstant("rand505", partial(random.uniform, -5, 5))

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        return pset, creator

    def init_individual(self, trimmed_eq):
        """初始化个体"""
        plist = []

        for t in trimmed_eq:
            if t in self.pset.mapping:
                node = self.pset.mapping[t]
                if isinstance(node, gp.MetaEphemeral):
                    node = node()
                plist.append(node)
            elif t in ['-3', '-2', '-1', '0', '1', '2', '3', '4', '5']:
                if t not in [term.name for term in self.pset.terminals[self.pset.ret]]:
                    self.pset.addTerminal(float(t), name=t)
                    term = self.pset.terminals[self.pset.ret][-1]
                else:
                    for i, term in enumerate(self.pset.terminals[self.pset.ret]):
                        if term.name == t:
                            break
                    term = self.pset.terminals[self.pset.ret][i]
                plist.append(term)
            else:
                value = float(t)
                self.pset.addTerminal(value, name=t)
                plist.append(self.pset.terminals[self.pset.ret][-1])

        individual = creator.Individual(gp.PrimitiveTree(plist))
        return individual

    def evalSymbReg(self, individual):
        """评估个体适应度"""
        func = self.toolbox.compile(expr=individual)
        wrong_mark = 999999999
        sum_sq = 0.0
        max_sqerror = 1e300

        for i, x in enumerate(self.X_train):
            try:
                result = func(*x)
                if is_complex(result) or not math.isfinite(result):
                    return wrong_mark,
                diff = result - self.y_train[i]
                if not math.isfinite(diff):
                    return wrong_mark,
                sq = diff * diff
                if is_complex(sq) or not math.isfinite(sq) or sq > max_sqerror:
                    return wrong_mark,
                sum_sq += float(sq)
                if not math.isfinite(sum_sq) or sum_sq > max_sqerror * len(self.X_train):
                    return wrong_mark,
            except (AttributeError, ValueError, ZeroDivisionError, TypeError, OverflowError):
                return wrong_mark,

        res = math.sqrt(sum_sq / len(self.X_train))
        return res,

    def mutReplace(self, individual):
        """Transformer引导的变异"""
        node_index = random.randrange(len(individual))
        if len(individual) == 1:
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)

        while individual[node_index].arity == 0:
            node_index = random.randrange(len(individual))

        try:
            semantic = backpropogation(individual, self.pset, (self.X_train, self.y_train), node_index)
        except (OverflowError, RuntimeError):
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)

        if isinstance(semantic, str) and semantic == 'nan':
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)

        try:
            symbol_list, prediceted_equation, total_c, bfgs_time = self.get_res_transformer(
                self.X_train, semantic, BFGS=True
            )
        except (ValueError, RuntimeError):
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)

        if symbol_list is None:
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)

        symbol_list = convert_to_list(symbol_list, accurate_constant=True, n_variables=self.n_variables)
        new_subtree = self.init_individual(symbol_list)

        CT_slice = individual.searchSubtree(node_index)
        individual[CT_slice] = new_subtree

        return individual,

    def mutate(self, individual, p_subtree=0.025):
        """变异操作"""
        if random.random() < p_subtree:
            return self.mutReplace(individual)
        else:
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)

    def setup_data(self, X_train, y_train, X_test, y_test, max_tree_height=17, max_tree_size=80):
        """设置数据"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_variables = X_train.shape[1]

        # 创建GP组件
        self.pset, creator = self.get_creator()
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=6)
        self.toolbox.register("random_individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("random_population", tools.initRepeat, list, self.toolbox.random_individual)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.evalSymbReg)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", self.mutate, p_subtree=0.025)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))
        self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_tree_size))
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_tree_size))

        # 初始化Transformer并获取初始表达式
        symbol_list, equation = self.get_res_transformer(X_train, y_train, BFGS=False, first_call=True)
        trimmed_eq = convert_to_list(symbol_list, accurate_constant=False, n_variables=self.n_variables)

        # 注册individual（需要在获取trimmed_eq之后）
        self.toolbox.register("individual", self.init_individual, trimmed_eq=trimmed_eq)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 创建初始种群：优先加入Transformer初始化个体，再补充满足复杂度约束的随机个体
        target_pop_size = 200
        max_init_attempts = target_pop_size * 200
        pop = []

        transformer_init_ind = self.toolbox.individual()
        if transformer_init_ind.height <= max_tree_height and len(transformer_init_ind) <= max_tree_size:
            for _ in range(20):
                pop.append(copy.deepcopy(transformer_init_ind))

        attempts = 0
        while len(pop) < target_pop_size and attempts < max_init_attempts:
            candidate = self.toolbox.random_individual()
            if candidate.height <= max_tree_height and len(candidate) <= max_tree_size:
                pop.append(candidate)
            attempts += 1

        if len(pop) < target_pop_size:
            raise ValueError(
                f"无法在限制条件下构造初始种群: size={len(pop)}/{target_pop_size}, "
                f"max_tree_height={max_tree_height}, max_tree_size={max_tree_size}"
            )

        self.pop = pop

    def run_evolution(self, ngen=300, cxpb=0.5, mutpb=0.2, p_subtree=0.025):
        """运行进化算法

        Returns:
            (best_individual, evolution_curve) - 最佳个体和进化曲线
        """
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fit.register("min", np.min)

        evolution_curve = []

        # 初始评估
        for ind in self.pop:
            if not ind.fitness.valid:
                ind.fitness.values = self.evalSymbReg(ind)

        # 记录初始最佳适应度
        hof = tools.HallOfFame(1)
        hof.update(self.pop)
        evolution_curve.append(hof[0].fitness.values[0])

        # 进化循环
        for gen in range(ngen):
            # 选择
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    try:
                        offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                        del offspring[i - 1].fitness.values, offspring[i].fitness.values
                    except (TypeError, AttributeError):
                        pass  # 交叉失败，保持原样

            # 变异
            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = self.toolbox.mutate(offspring[i], p_subtree=p_subtree)
                    del offspring[i].fitness.values

            # 评估新个体
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.evalSymbReg(ind)

            # 替换种群
            self.pop[:] = offspring

            # 记录最佳适应度
            hof.update(self.pop)
            evolution_curve.append(hof[0].fitness.values[0])

        return hof[0], evolution_curve


def compute_metrics(X, y, individual, toolbox):
    """计算RMSE和R²指标

    Returns:
        (rmse, r2)
    """
    func = toolbox.compile(expr=individual)
    predictions = []

    for x in X:
        try:
            pred = func(*x)
            if not is_complex(pred):
                predictions.append(pred)
            else:
                predictions.append(np.nan)
        except:
            predictions.append(np.nan)

    predictions = np.array(predictions)

    # 过滤无效值（NaN/inf）
    valid_mask = np.isfinite(predictions)
    if valid_mask.sum() < len(y) * 0.5:
        return float('inf'), -float('inf')

    y_valid = y[valid_mask]
    pred_valid = predictions[valid_mask]

    rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
    r2 = r2_score(y_valid, pred_valid)

    return rmse, r2


def run_single_experiment(
    csv_path,
    seed,
    max_input_points=100,
    max_tree_height=17,
    max_tree_size=80,
    device="cuda:0",
):
    """运行单次实验

    Args:
        csv_path: 数据集路径
        seed: 随机种子
        max_input_points: 训练数据点数量
        max_tree_height: 表达式树最大高度
        max_tree_size: 表达式树最大节点数
        device: 使用的CUDA设备（如cuda:0）

    Returns:
        包含实验结果的字典
    """
    # 1. 读取数据
    ground_truth, X, y = load_feynman_csv(csv_path)

    # 2. 数据采样和划分
    if len(X) > max_input_points:
        indices = np.random.choice(len(X), max_input_points, replace=False)
        X, y = X[indices], y[indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # 3. 运行PGGP
    pggp = PGGPWrapper(seed=seed, device=device)
    pggp.setup_data(
        X_train,
        y_train,
        X_test,
        y_test,
        max_tree_height=max_tree_height,
        max_tree_size=max_tree_size,
    )

    best_individual, evolution_curve = pggp.run_evolution()

    # 4. 计算指标
    train_rmse, train_r2 = compute_metrics(
        X_train, y_train, best_individual, pggp.toolbox
    )
    test_rmse, test_r2 = compute_metrics(
        X_test, y_test, best_individual, pggp.toolbox
    )

    # 5. 用SymPy化简表达式并保存原始表达式
    expression_raw = str(best_individual)
    sympy_locals = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: a ** b,
        "ln": sympy.log,
        "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "exp": sympy.exp,
    }
    expression_simplified = sympy.simplify(sympy.sympify(expression_raw, locals=sympy_locals))
    expression_str = sympy.sstr(expression_simplified)

    return {
        "seed": seed,
        "final_expression_raw": expression_raw,
        "final_expression": expression_str,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "evolution_curve": evolution_curve
    }


def parse_gpu_ids(gpus_arg):
    gpu_ids = [int(x.strip()) for x in gpus_arg.split(",") if x.strip()]
    if len(gpu_ids) == 0:
        raise ValueError("--gpus 不能为空，例如: --gpus 0,1")
    return gpu_ids


def gpu_worker(task_queue, result_queue, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_idx, csv_path, seed, max_input_points, max_tree_height, max_tree_size = task
        result = run_single_experiment(
            Path(csv_path),
            seed,
            max_input_points=max_input_points,
            max_tree_height=max_tree_height,
            max_tree_size=max_tree_size,
            device=device,
        )
        result_queue.put((task_idx, result))


def main():
    parser = argparse.ArgumentParser(description='PGGP标准化符号回归实验')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集路径（文件夹或单个CSV文件）')
    parser.add_argument('--gpus', type=str, required=True,
                        help='使用的GPU编号列表，例如: 0,1,3')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='实验重复次数（默认10）')
    parser.add_argument('--max_input_points', type=int, default=100,
                        help='训练数据点数量（默认100）')
    parser.add_argument('--max_tree_height', type=int, default=17,
                        help='表达式树最大高度（默认17）')
    parser.add_argument('--max_tree_size', type=int, default=80,
                        help='表达式树最大节点数（默认80）')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用CUDA，请在GPU环境运行该脚本。")

    gpu_ids = parse_gpu_ids(args.gpus)
    total_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= total_gpus:
            raise ValueError(f"GPU编号越界: {gpu_id}, 当前可用GPU数量: {total_gpus}")

    # 读取模型可支持的最大变量数（100M模型）
    script_dir = Path(__file__).parent
    with open(script_dir / 'jupyter' / '100M' / 'eq_setting.json', 'r') as f:
        eq_setting = json.load(f)
    cfg = omegaconf.OmegaConf.load(script_dir / '100M' / 'config.yaml')
    max_supported_vars = min(len(eq_setting["total_variables"]), int(cfg.architecture.dim_input) - 1)

    # 确定数据集列表
    dataset_path = Path(args.dataset)

    if dataset_path.is_file():
        csv_files = [dataset_path]
    elif dataset_path.is_dir():
        csv_files = sorted(dataset_path.glob('*.csv'))
    else:
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"模型最大支持输入变量数: {max_supported_vars}")
    print(f"使用GPU: {gpu_ids}（每块GPU固定1个进程）")

    # 对每个数据集运行实验
    for csv_file in csv_files:
        print(f"\n正在处理: {csv_file.name}")

        # 读取ground truth
        ground_truth, X_full, _ = load_feynman_csv(csv_file)

        if X_full.shape[1] > max_supported_vars:
            print(f"  跳过: 输入维度 {X_full.shape[1]} 超过模型上限 {max_supported_vars}")
            output = {
                "dataset": csv_file.stem,
                "ground_truth": ground_truth,
                "skipped": True,
                "skip_reason": f"input_dim_exceeds_model_capacity ({X_full.shape[1]} > {max_supported_vars})",
                "runs": []
            }
            output_path = results_dir / f"{csv_file.stem}_pggp.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"  跳过结果已保存到: {output_path}")
            continue

        # 多GPU并行运行：每块GPU一个进程
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        workers = []
        for gpu_id in gpu_ids:
            proc = ctx.Process(target=gpu_worker, args=(task_queue, result_queue, gpu_id))
            proc.start()
            workers.append(proc)

        for seed in range(args.num_seeds):
            print(f"  提交 Seed {seed}/{args.num_seeds - 1}...")
            task_queue.put((
                seed,
                str(csv_file),
                seed,
                args.max_input_points,
                args.max_tree_height,
                args.max_tree_size,
            ))

        for _ in workers:
            task_queue.put(None)

        results = [None] * args.num_seeds
        for _ in range(args.num_seeds):
            idx, result = result_queue.get()
            results[idx] = result

        for proc in workers:
            proc.join()

        # 保存结果
        output = {
            "dataset": csv_file.stem,
            "ground_truth": ground_truth,
            "runs": results
        }

        output_path = results_dir / f"{csv_file.stem}_pggp.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
