from typing import List, Tuple, Union
from itertools import zip_longest
from rdkit import Chem
import torch
import numpy as np
from FLAME.dataprocess.flsf.scaffold import make_tag

atom_dict = {'B': 0, 'Br': 1, 'C': 2, 'Cl': 3, 'F': 4, 'H': 5, 'I': 6, 'N': 7, 'O': 8, 'P': 9, 'S': 10, 'Si': 11, 'Se': 12}

# Atom feature sizes (atom_fdim: 127)
MAX_ATOMIC_NUM = 100  # 最大原子序数
# MAX_ATOMIC_NUM = len(atom_dict)
ATOM_FEATURES = {
    # type of atom (ex. C,N,O), by atomic number, size = 100
    # 原子类型（如C,N,O），按原子序数，size = 100
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    # number of bonds the atom is involved in, size = 6
    # 原子参与的键的数量，size = 6
    'degree': [0, 1, 2, 3, 4, 5],
    # integer electronic charge assigned to atom, size = 5
    # 原子的形式电荷，size = 5
    'formal_charge': [-1, -2, 1, 2, 0],
    # chirality: unspecified, tetrahedral CW/CCW, or other, size = 4
    # 手性：未指定、四面体CW/CCW或其他，size = 4
    'chiral_tag': [0, 1, 2, 3],
    # 连接的氢原子数量，size = 5
    'num_Hs': [0, 1, 2, 3, 4],  # number of bonded hydrogen atoms, size = 5
    # 杂化类型，size = 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
# 路径距离（图距离）的区间
PATH_DISTANCE_BINS = list(range(10))     # [0,1,2,3,4,5,6,7,8,9]
# 3D距离的最大值和步长
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(
    range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))
# 生成 [0,1,2,...,20]

# 计算原子特征的总维度
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
EXTRA_ATOM_FDIM = 0
# 键特征的维度
BOND_FDIM = 14
# 额外键特征的维度
EXTRA_BOND_FDIM = 0

def get_atom_label(symbol):
    if symbol in atom_dict:
        return atom_dict[symbol] # 返回原子索引
    else:
        return -1 # 未知原子返回-1

def get_atom_fdim(overwrite_default_atom: bool = False) -> int:
    """
    获取原子特征向量的维度

    :param overwrite_default_atom: 是否覆盖默认原子描述符
    :return: 原子特征向量的维度
    """
    return (not overwrite_default_atom) * ATOM_FDIM + EXTRA_ATOM_FDIM


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    global EXTRA_ATOM_FDIM
    EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False) -> int:
    """
    获取键特征向量的维度

    :param atom_messages: 是否使用原子消息传递
    :param overwrite_default_bond: 是否覆盖默认键描述符
    :param overwrite_default_atom: 是否覆盖默认原子描述符
    :return: 键特征向量的维度
    """

    return (not overwrite_default_bond) * BOND_FDIM + EXTRA_BOND_FDIM + \
           (not atom_messages) * \
        get_atom_fdim(overwrite_default_atom=overwrite_default_atom)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    global EXTRA_BOND_FDIM
    EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    创建one-hot编码，为不常见值增加一个额外类别

    :param value: 需要编码的值
    :param choices: 可能值的列表
    :return: one-hot编码列表，长度为 len(choices) + 1
             如果value不在choices中，则最后一个元素为1
    """
    encoding = [0] * (len(choices) + 1) # 创建全零列表
    index = choices.index(value) if value in choices else -1 # 找到值的索引，不在则用-1
    encoding[index] = 1# 对应位置设为1


    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
     为原子构建特征向量

     :param atom: RDKit原子对象
     :param functional_groups: 原子所属的官能团的k-hot向量
     :return: 原子特征列表
     """
    # feature vector for each atom
#     features = onek_encoding_unk(get_atom_label(atom.GetSymbol()), ATOM_FEATURES['atomic_num']) + \
    # 原子序数（-1是因为RDKit的原子序数从1开始）,度数,形式电荷,手性,氢原子数,杂化类型,是否芳香（二值）
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]  # 缩放至与其他特征大致相同的范围

    if functional_groups is not None:
        features += functional_groups  # 添加官能团信息
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    为化学键构建特征向量

    :param bond: RDKit键对象
    :return: 键特征列表
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType() # 空键（用于虚拟连接）
        fbond = [
            0, # 第一个位置表示不是空键
            bt == Chem.rdchem.BondType.SINGLE,                   # 单键
            bt == Chem.rdchem.BondType.DOUBLE,                   # 双键
            bt == Chem.rdchem.BondType.TRIPLE,                   # 三键
            bt == Chem.rdchem.BondType.AROMATIC,                 # 芳香键
            (bond.GetIsConjugated() if bt is not None else 0),   # 是否共轭
            (bond.IsInRing() if bt is not None else 0)           # 是否在环中
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6))) # 立体化学
    return fbond


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`f_adj`: The adjacency matrix of the molecule.
    * :code:`f_dist`: The distance matrix of the molecule.
    * :code:`f_clb`: The coulomb matrix of the molecule.
    """

    def __init__(self, mol: Union[str, Chem.Mol],
                 adj: np.ndarray = None,
                 dist: np.ndarray = None,
                 clb: np.ndarray = None):
        """
        :param mol: 一个 SMILES 表示或一个 RDKit 分子
        :param f_adj: 该分子的邻接矩阵
        :param f_dist: 该分子的距离矩阵
        :param f_clb: 该分子的库仑矩阵
        """

        self.smiles = mol
        # 如果是SMILES字符串，转换为RDKit分子
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)
        
        self.n_atoms = 0  # 原子数量
        self.n_bonds = 0  # 键的数量
        self.f_atoms = []   # 原子特征 [原子索引] -> 特征向量
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []   # 键特征 [键索引] -> 特征向量
        self.a2b = []  # 原子到键的映射 [原子索引] -> 入边键索引列表
        self.b2a = []  # 键到原子的映射 [键索引] -> 源原子索引
        self.b2revb = []  # 键到反向键的映射 [键索引] -> 反向键索引

        # Get atom features
        # atom feature (size = 133)
        # 1. 获取所有原子的特征
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
        self.f_adj = adj
        self.f_dist = dist
        self.f_clb = clb

        self.n_atoms = len(self.f_atoms)  # Initialize number of atoms

        # 2. 初始化原子到键的映射
        for _ in range(self.n_atoms):
            self.a2b.append([])# 每个原子一个空列表

        # Get bond features
        # iterate throgh all the bonds (based on every two adjacent atoms)
        # 3. 遍历所有原子对，构建边（化学键）
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                # if there's no bond between them, then continue
                if bond is None:
                    continue  # 没有化学键，跳过

                # bond feature (the bond between a1 and a2)
                # 获取键特征
                f_bond = bond_features(bond)

                # 添加双向边（有向图）
                # 边 a1 -> a2: 包含a1的原子特征 + 键特征
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                # a2 atom with adjacent bond (size = 147)
                # 边 a2 -> a1: 包含a2的原子特征 + 键特征
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                # 更新索引映射
                b1 = self.n_bonds           # 边a1->a2的索引
                b2 = b1 + 1                 # 边a2->a1的索引
                # b1 = a1 --> a2   the atom (a2) has the incoming bond (b1)
                self.a2b[a2].append(b1)     # 原子a2有入边b1
                # the bond (b1) is originated from atom (a1)
                self.b2a.append(a1)         # 边b1从a1出发
                # b2 = a2 --> a1   the atom (a1) has the incoming bond (b2)
                self.a2b[a1].append(b2)     # 原子a1有入边b2
                # the bond (b2) is originated from atom (a2)
                self.b2a.append(a2)         # 边b2从a2出发
                self.b2revb.append(b2)  # first append b2   b1的反向边是b2
                self.b2revb.append(b1)  # then append b1    b2的反向边是b1
                self.n_bonds += 2           # 增加了两条边

#将多个独立的分子图合并成一个大的批次图，并建立索引映射
class BatchMolGraph:

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.tags = list(map(make_tag,self.smiles_batch))#
        
        # Start n_atoms and n_bonds at 1 b/c zero padding
        # 原子计数从1开始（0用于填充）
        self.n_atoms = 1
        # 键计数从1开始（0用于填充）
        self.n_bonds = 1
        # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.a_scope = []    # 记录每个分子的原子范围
        # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.b_scope = []     # 记录每个分子的键范围
        
        # 所有数据都以0填充开始，这样索引0返回的都是0
        f_atoms = [[0] * self.atom_fdim]  # 原子特征，第一个是填充
        f_bonds = [[0] * self.bond_fdim]  # 键特征，第一个是填充
        f_adj = [[]]
        f_dist = [[]]
        f_clb = [[]]
        a2b = [[]]  # 原子到键的映射，第一个是填充
        # mapping from bond index to the index of the atom the bond is coming from
        b2a = [0]     # 键到原子的映射，第一个是填充
        b2revb = [0]  # 键到反向键的映射，第一个是填充
        for mol_graph in mol_graphs:  # for each molecule graph
            f_atoms.extend(mol_graph.f_atoms)  # n_atoms * 133
            f_bonds.extend(mol_graph.f_bonds)  # n_bonds * 147
            f_adj.append(mol_graph.f_adj)      # n_atoms * n_atoms
            f_dist.append(mol_graph.f_dist)
            f_clb.append(mol_graph.f_clb)

            for a in range(mol_graph.n_atoms):
                # 关键：将原子a的入边索引加上当前批次的键总数
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])
                # f_adj.append(mol_graph.f_adj) # n_atoms * n_atoms

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))  # 记录原子范围
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))  # 记录键范围
            self.n_atoms += mol_graph.n_atoms  # 更新原子总数
            self.n_bonds += mol_graph.n_bonds  # 更新键总数


        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.f_adj = f_adj  # batch_n_atoms * n_atoms
        self.f_clb = f_clb
        self.f_dist = f_dist
        self.a2b = torch.LongTensor(
            [a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        if atom_messages:
            # 如果使用原子消息，只取键特征的最后部分
            f_bonds = self.f_bonds[:, -
                                   get_bond_fdim(atom_messages=atom_messages):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    # try to avoid computing b2b b/c O(n_atoms^3)
    def get_b2b(self) -> torch.LongTensor:
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(
                1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    # only needed if using atom messages
    def get_a2a(self) -> torch.LongTensor:
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds
        return self.a2a

    def get_adjacency(self):
        return self.f_adj

    def get_distance(self):
        return self.f_dist

    def get_coulomb(self):
        return self.f_clb


def mol2graph(mols: Union[List[str], List[Chem.Mol]],
              mol_adj_batch: List[np.array] = None,
              mol_dist_batch: List[np.array] = None,
              mol_clb_batch: List[np.array] = None,
              ) -> BatchMolGraph:
    """
     将SMILES列表或RDKit分子列表转换为包含批处理分子图的:class:`BatchMolGraph`。

     :param mols: SMILES列表或RDKit分子列表。
     :param mol_adj_batch: 包含额外原子特征的2D numpy数组列表
     :param mol_dist_batch: 包含距离矩阵的2D numpy数组列表
     :param mol_clb_batch: 包含库仑矩阵的2D numpy数组列表
     :return: 包含合并后分子图的:class:`BatchMolGraph`
     """
    return BatchMolGraph([MolGraph(mol, adj, dist, clb)
                          for mol, adj, dist, clb
                          in zip_longest(mols, mol_adj_batch, mol_dist_batch, mol_clb_batch)])
