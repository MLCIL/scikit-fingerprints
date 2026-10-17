from collections import Counter

import numpy as np
from rdkit.Chem import BondType, GetDistanceMatrix, Mol
from rdkit.Chem.GraphDescriptors import BalabanJ, HallKierAlpha, Kappa1, Kappa2, Kappa3

from skfp.utils.validators import require_atoms


@require_atoms(min_atoms=2)
def average_wiener_index(mol: Mol, distance_matrix: np.ndarray | None = None) -> float:
    """
    Average Wiener Index (AW).

    Calculates the Average Wiener Index [1]_, defined as the Wiener index divided
    by the total number of atom pairs in the molecule. This makes it independent of
    the molecule size.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Average Wiener index is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Andrey A. Dobrynin, Ivan Gutman,
        "The average Wiener index of hexagonal chains",
        Computers & Chemistry, Volume 23, Issue 6, 1999, Pages 571-576, ISSN 0097-8485,
        <https://www.sciencedirect.com/science/article/pii/S0097848599000352>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import average_wiener_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> average_wiener_index(mol)
    1.8
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)
    wiener_idx = wiener_index(mol, distance_matrix)
    num_atoms = mol.GetNumAtoms()
    return (2 * wiener_idx) / (num_atoms * (num_atoms - 1))


def balaban_j_index(mol: Mol, distance_matrix: np.ndarray | None = None) -> float:
    r"""
    Balaban's J Index.

    Calculates the Balaban’s J index [1]_, defined as a measure of molecular
    connectivity with an emphasis on cyclic structures. The formula for calculating
    Balaban's J Index is given as:

    .. math::

        J = \frac{M}{\mu + 1} \cdot \frac{\sum_{ij} (d_{ij})^{-1}}{n},

    where:

    - :math:`M` is the number of bonds
    - :math:`\mu` is the cyclomatic number (number of independent cycles)
    - :math:`d_{ij}` is the distance between atoms :math:`i` and :math:`j`
    - :math:`n` is the number of atoms

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Balaban's J index is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated by RDKit.

    References
    ----------
    .. [1] `Balaban, Alexandru T.
        "Highly discriminating distance-based topological index."
        Chemical Physics Letters 89.5 (1982): 399-404.
        <https://doi.org/10.1016/0009-2614(82)80009-2>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import balaban_j_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> balaban_j_index(mol)  # doctest: +SKIP
    3.000000000000001
    """
    return BalabanJ(mol=mol, dMat=distance_matrix)


def burden_matrix(mol: Mol, descriptors: np.ndarray | None = None) -> np.ndarray:
    """
    Burden matrix.

    Burden matrix [1]_ [2]_ is a modified connectivity matrix, aimed to combine topological
    structure with atomic properties. Diagonal elements are atom descriptors, e.g.
    atomic number, charge, polarizability. Off-diagonal elements for bonded atoms are
    1/sqrt(bond order), with minimum of 0.001 in case of no bond between given pair of
    atoms.

    Burden proposed to use vector of smallest eigenvalues of this matrix as molecule
    descriptors. They reflect the overall topology of the molecule, while also
    incorporating the functional information via atom properties. Largest eigenvalues
    can also be used [2]_.

    If ``descriptors`` are None, default value 0.001 for non-connected atoms is used
    on the diagonal.

    We use bond orders as in RDKit, i.e. 1, 2, 3, and 1.5 for aromatic. See
    RDKit code for details:
    https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/BCUT.cpp

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Balaban's J index is to be calculated.

    descriptors : np.ndarray, default=None
        Vector of atomic descriptors, with the same length as number of atoms in the
        input molecule.

    References
    ----------
    .. [1] `Frank R. Burden
        "Molecular identification number for substructure searches"
        J. Chem. Inf. Comput. Sci. 1989, 29, 3, 225–227
        <https://doi.org/10.1021/ci00063a011>`_

    .. [2] `R. Todeschini, V. Consonni
        "Molecular Descriptors for Chemoinformatics"
        Wiley‐VCH Verlag GmbH & Co. KGaA
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import burden_matrix
    >>> mol = MolFromSmiles("C=1=C=C=C1")  # cyclobutadiyne
    >>> burden_matrix(mol)
    array([[0.001     , 0.70710678, 0.001     , 0.70710678],
           [0.70710678, 0.001     , 0.70710678, 0.001     ],
           [0.001     , 0.70710678, 0.001     , 0.70710678],
           [0.70710678, 0.001     , 0.70710678, 0.001     ]])
    """
    num_atoms = mol.GetNumAtoms()

    if descriptors is not None and len(descriptors) != num_atoms:
        raise ValueError(
            f"Number of descriptors {len(descriptors)} "
            f"does not match number of atoms {num_atoms}"
        )

    matrix = np.empty((num_atoms, num_atoms), dtype=float)
    matrix.fill(0.001)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        if bond.GetBondType() == BondType.SINGLE:
            value = 1.0  #  1/sqrt(1.0)
        elif bond.GetBondType() == BondType.DOUBLE:
            value = 0.7071067811865475  # 1/sqrt(2.0)
        elif bond.GetBondType() == BondType.TRIPLE:
            value = 0.5773502691896258  # 1/sqrt(3.0)
        elif bond.GetBondType() == BondType.AROMATIC:
            value = 0.8164965809277261  # 1/sqrt(1.5)
        else:
            raise ValueError(
                "Bond order for Burden matrix must be single, double, triple, or aromatic"
            )

        matrix[i, j] = matrix[j, i] = value

    return matrix


def diameter(mol: Mol, distance_matrix: np.ndarray | None = None) -> int:
    """
    Diameter.

    Calculates the diameter [1]_, defined as the maximum length of the shortest path.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the diameter is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated.

    References
    ----------
    .. [1] `Petitjean, Michel
        "Applications of the radius-diameter diagram to the classification
        of topological and geometrical shapes of chemical compounds."
        Journal of Chemical Information and Computer Sciences 32.4 (1992): 331-337.
        <https://doi.org/10.1021/ci00008a012>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import diameter
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> diameter(mol)
    3
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)

    eccentricities = np.max(distance_matrix, axis=1)

    return int(np.max(eccentricities))


def graph_distance_index(mol: Mol, distance_matrix: np.ndarray | None = None) -> int:
    r"""
    Graph Distance Index (GDI).

    Calculates the Graph Distance Index [1]_, defined as the squared sum of all
    graph distance counts in the molecular graph's distance matrix. The formula for
    calculating GDI is given as:

    .. math::

        GDI = \sum_{k=1}^{D} \left(k \cdot f_k\right)^2

    where:

    - :math:`D` is the topological diameter of the graph (the largest graph distance)
    - :math:`f_k` is the total number of distances in the graph equal to :math:`k`

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Graph Distance Index is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Konstantinova, Elena V.
        "The Discrimination Ability of Some Topological and Information Distance
        Indices for Graphs of Unbranched Hexagonal Systems"
        Journal of Chemical Information and Computer Sciences 36.1 (1996): 54-57.
        <https://pubs.acs.org/doi/10.1021/ci9502461>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import graph_distance_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> graph_distance_index(mol)
    261
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    distance_counts = Counter(distances)
    return int(sum((k * f) ** 2 for k, f in distance_counts.items()))


def hall_kier_alpha(mol: Mol) -> float:
    r"""
    Hall-Kier alpha index.

    Computes the Hall-Kier alpha index [1]_, which is a measure of molecular flexibility.
    It is calculated by summing atomic contributions alpha:

    .. math::
        \alpha = \frac{r}{r(Csp^3)} - 1

    where:

    - r is the covalent radius of the atom
    - r(Csp3) is the covalent radius of a sp3 hybridized carbon

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Hall-Kier alpha index is computed.

    References
    ----------
    .. [1] `Lowell H. Hall, Lemont B. Kier
        "The Molecular Connectivity Chi Indexes and Kappa Shape Indexes in Structure-Property Modeling"
        Reviews in Computational Chemistry vol 2 (1991): 367-422.
        <https://doi.org/10.1002/9780470125793.ch9>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import hall_kier_alpha
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> hall_kier_alpha(mol)
    -0.78
    """
    return HallKierAlpha(mol)


def petitjean_index(mol: Mol, distance_matrix: np.ndarray | None = None) -> float:
    r"""
    Petitjean Index.

    Calculates the Petitjean Index [1]_, defined as a measure of molecular shape based on graph
    topology. It is derived from two fundamental properties of molecular graphs: radius (R) and
    diameter (D). The formula for calculating the Petitjean Index is given as:

    .. math::
        I_2 = \frac{D - R}{R}

    where:

    - :math:`D` is the graph diameter
    - :math:`R` is the graph radius

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Petitjean index is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated.

    References
    ----------
    .. [1] `Petitjean, Michel
        "Applications of the radius-diameter diagram to the classification
        of topological and geometrical shapes of chemical compounds."
        Journal of Chemical Information and Computer Sciences 32.4 (1992): 331-337.
        <https://doi.org/10.1021/ci00008a012>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import petitjean_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> petitjean_index(mol)
    0.0
    """
    D = diameter(mol, distance_matrix)
    R = radius(mol, distance_matrix)

    return (D - R) / R if R != 0 else 0.0


def polarity_number(
    mol: Mol, distance_matrix: np.ndarray | None = None, carbon_only: bool = False
) -> int:
    """
    Polarity Number.

    Calculates the Polarity Number [1]_ [2]_, defined as the total number of
    unordered pairs of vertices (atoms) in a molecular graph that are separated by
    a graph distance of exactly 3. It provides information about the structural
    connectivity of a molecule.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Polarity Number is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    carbon_only : bool, default=False
        Whether to consider only carbon-carbon distances. If True, the distance
        matrix will be filtered to include only rows and columns corresponding to
        carbon atoms.

    References
    ----------
    .. [1] `Wiener, Harry.
        "Structural Determination of Paraffin Boiling Points"
        Journal of the American Chemical Society 69.1 (1947): 17-20.
        <https://pubs.acs.org/doi/10.1021/ja01193a005>`_

    .. [2] `Liu, Muhuo, and Bolian Liu.
        "On the Wiener Polarity Index"
        MATCH Commun. Math. Comput. Chem 66.1 (2011): 293-304.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match66/n1/match66n1_293-304.pdf>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import polarity_number
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> polarity_number(mol)
    3
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)

    if carbon_only:
        atom_indices = [
            i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == "C"
        ]

        if not atom_indices:
            raise ValueError(
                "The molecule contains no carbon atoms, so carbon-only filtering is not possible."
            )

        distance_matrix = distance_matrix[np.ix_(atom_indices, atom_indices)]

    return int((distance_matrix == 3).sum() // 2)


def kappa1_index(mol: Mol) -> float:
    r"""
    First Kappa shape index (K1).

    Computes the first kappa shape index [1]_, which measures molecular shape based on
    single bonds. It is given by the equation:

    .. math::
        K_1 = \frac{(A + \alpha) (A + \alpha - 1)^2}{P_1^2}

    where:

    - A is the number of heavy atoms
    - α is the Hall-Kier alpha index
    - P1 is the number of single bonds

    This index provides insight into the molecular shape and branching properties.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the first Kappa shape index is calculated.

    References
    ----------
    .. [1] `Lowell H. Hall, Lemont B. Kier
        "The Molecular Connectivity Chi Indexes and Kappa Shape Indexes in Structure-Property Modeling"
        Reviews in Computational Chemistry vol 2 (1991): 367-422.
        <https://doi.org/10.1002/9780470125793.ch9>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import kappa1_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> kappa1_index(mol)
    3.4115708812260532
    """
    return Kappa1(mol)


def kappa2_index(mol: Mol) -> float:
    r"""
    Second Kappa shape index (K2).

    Computes the second kappa shape index [1]_, which measures molecular shape based on
    paths of length 2. It is given by the equation:

    .. math::
        K_2 = \frac{(A + \alpha - 1) (A + \alpha - 2)^2}{P_2^2}

    where:

    - A is the number of heavy atoms
    - α is the Hall-Kier alpha index
    - P2 is the number of paths of length 2

    This index captures molecular branching and shape characteristics.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the second Kappa shape index is calculated.

    References
    ----------
    .. [1] `Lowell H. Hall, Lemont B. Kier
        "The Molecular Connectivity Chi Indexes and Kappa Shape Indexes in Structure-Property Modeling"
        Reviews in Computational Chemistry vol 2 (1991): 367-422.
        <https://doi.org/10.1002/9780470125793.ch9>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import kappa2_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> kappa2_index(mol)
    1.6057694396735218
    """
    return Kappa2(mol)


def kappa3_index(mol: Mol) -> float:
    r"""
    Third Kappa shape index (K3).

    Computes the third kappa shape index [1]_, which measures molecular shape based on
    paths of length 3. It is given by the equation:

    .. math::
        K_3 = \frac{(A + \alpha - 1) (A + \alpha - 3)^2}{P_3^2}

    where:

    - A is the number of heavy atoms,
    - α is the Hall-Kier alpha index,
    - P3 is the number of paths of length 3.

    This index helps characterize the overall shape and structural complexity of molecules.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the third Kappa shape index is calculated.

    References
    ----------
    .. [1] `Lowell H. Hall, Lemont B. Kier
        "The Molecular Connectivity Chi Indexes and Kappa Shape Indexes in Structure-Property Modeling"
        Reviews in Computational Chemistry vol 2 (1991): 367-422.
        <https://doi.org/10.1002/9780470125793.ch9>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import kappa3_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> kappa3_index(mol)
    0.5823992601400448
    """
    return Kappa3(mol)


def radius(mol: Mol, distance_matrix: np.ndarray | None = None) -> int:
    """
    Radius.

    Calculates the radius [1]_, defined as the minimal length of the longest path between atoms.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the radius is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated.

    References
    ----------
    .. [1] `Petitjean, Michel
        "Applications of the radius-diameter diagram to the classification
        of topological and geometrical shapes of chemical compounds."
        Journal of Chemical Information and Computer Sciences 32.4 (1992): 331-337.
        <https://doi.org/10.1021/ci00008a012>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import radius
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> radius(mol)
    3
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)

    eccentricities = np.max(distance_matrix, axis=1)

    return int(np.min(eccentricities))


def wiener_index(mol: Mol, distance_matrix: np.ndarray | None = None) -> int:
    """
    Wiener Index (W).

    Calculates the Wiener Index [1]_ [2]_, defined as the sum of all pairwise
    distances in the molecular graph distance matrix.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the Wiener index is to be calculated.

    distance_matrix : np.ndarray, default=None
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Wiener, Harry.
        "Structural Determination of Paraffin Boiling Points"
        Journal of the American Chemical Society 69.1 (1947): 17-20.
        <https://pubs.acs.org/doi/10.1021/ja01193a005>`_

    .. [2] `Rouvray, Dennis H.
        "Chapter 2 - The Rich Legacy of Half a Century of the Wiener Index"
        Topology in Chemistry: 16-37.
        <https://www.sciencedirect.com/science/article/abs/pii/B9781898563761500068>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import wiener_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> wiener_index(mol)
    27
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)
    return int(np.sum(distance_matrix) // 2)


def zagreb_index_m1(mol: Mol) -> int:
    """
    First Zagreb Index.

    Calculates the first Zagreb index [1]_, defined as the sum of the squares of the
    degrees of all atoms in the molecule. Also known as simply the Zagreb index. It is
    a measure of molecular branching.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the first Zagreb index is to be calculated.

    References
    ----------
    .. [1] `Gutman, Ivan.
        "Degree-Based Topological Indices"
        Croatica Chemica Acta 86.4 (2013): 352.
        <http://dx.doi.org/10.5562/cca2294>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import zagreb_index_m1
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> zagreb_index_m1(mol)
    24
    """
    return int(sum(atom.GetDegree() ** 2 for atom in mol.GetAtoms()))


def zagreb_index_m2(mol: Mol) -> int:
    r"""
    Second Zagreb Index.

    Calculates the second Zagreb index [1]_, defined as the sum of the product of
    degrees of all pairs of adjacent atoms in the molecule. It provides a measure
    related to molecular branching and the connectivity of bonds. The formula for
    calculating the second Zagreb index is given as:

    .. math::

        M_2 = \sum_{(u,v) \in E} d_u \cdot d_v

    where:

    - :math:`E` is the set of bonds (edges) in the molecular graph
    - :math:`d_u` and :math:`d_v` are the degrees of the atoms :math:`u` and :math:`v`
      connected by a bond

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the second Zagreb index is to be calculated.

    References
    ----------
    .. [1] `Gutman, Ivan.
        "Degree-Based Topological Indices"
        Croatica Chemica Acta 86.4 (2013): 352.
        <http://dx.doi.org/10.5562/cca2294>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import zagreb_index_m2
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> zagreb_index_m2(mol)
    24
    """
    return int(
        sum(
            mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetDegree()
            * mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetDegree()
            for bond in mol.GetBonds()
        )
    )
