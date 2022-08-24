import os
import sys
import numpy as np
from typing import List, Tuple, Union
from util import beta_from_V, alpha_beta
from scipy.interpolate import PPoly
from dataclasses import dataclass
import msprime as msp
import functools
import stdpopsim
import _lsss
import pickle
import random
import subprocess
import tempfile
import shutil
from cyvcf2 import VCF, Writer


@dataclass
class LiStephensSurface:
    """Compute the solution surface for the Li-Stephens model.
    Args:
        gh: Either a 1-dimensional integer array of length L, or a 2-dimensional integer array of shape [L, 2],
            representing focal haplotypes or genotypes, respectively.
        H: A two-dimensional integer array of shape [L, H] representing panel haplotypes.
    Notes:
        The integers in gh and H can represent any encoding of genotype information; it is only necessary to test for a
        allele equality in the L-S model. For example, in a biallelic model, the entries of gh and H are binary, while
        in a general tetra-allelic model, they can be in {0, 1, 2, 3}.
    """

    gh: np.ndarray
    H: np.ndarray
    num_threads: int = os.cpu_count()

    def __post_init__(self):
        gh = self.gh = np.ascontiguousarray(self.gh, dtype=np.uint8)
        H = self.H = np.ascontiguousarray(self.H, dtype=np.uint8)
        if gh.shape[0] != H.shape[0]:
            raise ValueError("shape mismatch between gh and H")
        if gh.ndim == 2 and gh.shape[1] != 2:
            raise ValueError("genotypes shape should be Lx2")
        if 0 in H.shape:
            raise ValueError("empty genotype/haplotype array")
        self.V = _lsss.partition_ls(self.gh, self.H, self.num_threads)

    @property
    def L(self):
        """Number of loci"""
        return self.H.shape[0]

    @property
    def N(self):
        """Number of haplotypes in panel"""
        return self.H.shape[1]

    @property
    def diploid(self) -> bool:
        """True if this is a diploid model"""
        return self.gh.ndim == 2

    @property
    def s_beta(self) -> PPoly:
        r""":math:`s(\beta)` with the property that
        .. math:: s(\beta) = (m(\pi^*),r(\pi^*)) \text{ where } \pi^* = \arg\min_\pi m(\pi) + \beta k(\pi).
        (See manuscript for notation.)
        """
        c = np.array(self.V[:-1][::-1])[None]
        return PPoly(x=self.C_beta.x, c=c)

    @property
    def C_beta(self) -> PPoly:
        r""":math:`C(\beta)` with the property that
        .. math:: C(\beta) = \min_\pi m(\pi) + \beta k(\pi).
        (See manuscript for notation.)
        """
        return beta_from_V(self.V)

    def __call__(self, theta, rho):
        # probability of mutation to any bp
        alpha, beta = alpha_beta(theta, rho, self.N)
        assert np.all([alpha > 0, beta > 0]), (alpha, beta)
        return self.s_beta(beta / alpha)

    def draw(self):
        r"""Plot a phase diagram of solution space.
        Args:
            ax: A matplotlib axis on which to draw the plot, or matplotlib.pyplot.gca() if None.
        Notes:
            Assumes a tetraallelic model where the probability of a mutation between any two nucleotides
            :math:`X,Y \in \{A,C,G,T\}, X\neq Y` is
            .. math:: p_\theta = \frac{1 - e^{-\theta}}{3},
            where the population-scaled mutation rate is :math:`\theta`. Similarly, the probability of recombination
            onto another haplotype
            .. math:: \frac{1 - e^{-\rho}}{N},
            where :math:`N` is the size of the panel.
            FIXME
            See Figure XX in paper for an example of this plot.
        """
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar")
        except ImportError as e:
            raise ImportError("Plotting requires matplotlib") from e

        theta = np.arctan(self.s_beta.x)[None, :].T
        r = np.r_[np.zeros_like(theta), np.ones_like(theta)]
        ax.plot(theta, r)
        return fig, ax

    @classmethod
    def from_ts(
        cls,
        ts: "tskit.TreeSequence",
        focal: Union[int, Tuple[int, int]],
        panel: List[int],
    ) -> "LiStephensSurface":
        G = ts.genotype_matrix()
        gh = G[:, focal].astype(np.int32)
        H = G[:, panel].astype(np.int32)
        return cls(gh, H)


def _backtrace(c_star):
    ell = len(c_star) - 1
    path = []
    while ell >= 0:
        path.append(c_star[ell])
        ell -= path[-1][-1]
    path = np.array(path)[::-1]
    return path


def ls_hap(ts_or_G, focal, panel, beta, alpha=1.0):
    try:
        G = ts_or_G.genotype_matrix()
    except AttributeError:
        G = ts_or_G
    D = (G[:, [focal]] != G[:, panel]).astype(int)
    L, N = D.shape
    cost = np.zeros(N)
    m, r = np.zeros([2, N], int)
    ibd = np.zeros(N, int)
    c_star = np.zeros([L, 2], int)  # the lowest cost haplotype and its ibd tract length
    n_star = 0
    for ell in range(L):
        F_t = cost.min() + beta
        recomb = F_t < cost
        ibd += 1
        ibd[recomb] = 1
        r[recomb] = 1 + r[n_star]
        m = D[ell] + np.where(recomb, m[n_star], m)
        cost = alpha * D[ell] + np.minimum(cost, F_t)
        n_star = cost.argmin()
        c_star[ell] = [n_star, ibd[n_star]]
    np.testing.assert_allclose(cost[n_star], alpha * m[n_star] + beta * r[n_star])
    # backtrack to find the optimal copying path
    path = _backtrace(c_star)
    assert len(path) - 1 == r[n_star]
    assert path[:, 1].sum() == L
    path = np.repeat(path[:, 0], path[:, 1])
    return {"c": cost[n_star], "r": r[n_star], "m": m[n_star], "path": path}

def vcf_gmatrix(fname, path):
    vcf = VCF("%s/%s.vcf" %(path, fname))
    matrix = []
    for v in vcf:
        word = v.genotypes
        row = [w[0:1] for w in word]
        row = list(np.concatenate(row).flat)
        matrix.append(row)
    matrix = np.array(matrix, dtype=int)
    return matrix

def get_id(true, filtered, path, seed):
    p = subprocess.Popen('vcftools --vcf %s.vcf --diff %s.vcf.recode.vcf --diff-site --out in1_v_in2%s' %(true, filtered, seed), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path)
    p.communicate()
    retain_id = []
    i = 0
    with open('%s/in1_v_in2%s.diff.sites_in_files' %(path, seed), 'rb') as f:
                for row in f:
                    row = row.strip().decode("utf-8").split('\t')
                    if row[-1]=='1':
                        retain_id.append(i-1)
                    i+=1
    return retain_id


def imputation_accuracy(ts, focal: int, panel: List[int], k: float, seed: int):
    '''
    compute imputation accuracy for focal on panel across all settings of the li-stephens model, 
    when only total k sites are retained. Impute a missing SNP by pasting the copying path from 
    the nearest retained SNP behind it
    '''
    G = ts.genotype_matrix()
    L, N = G.shape
    dirpath = tempfile.mkdtemp()
    with open("%s/truth%s.vcf" % (dirpath, seed), "w") as vcf_file:
        ts.write_vcf(vcf_file)
    p = subprocess.Popen('vcftools --vcf truth%s.vcf --remove-indels --maf %s --recode --out SNP%s.vcf' %(seed, k, seed), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dirpath)
    p.communicate()
    retain_id = get_id('truth%s' %seed, 'SNP%s' %seed, dirpath, seed)
    num_retain_id = np.size(retain_id)
    Gk = G[retain_id]
    ell = np.arange(L)
    ls = LiStephensSurface(gh=Gk[:, focal], H=Gk[:, panel])
    p = beta_from_V(ls.V)
    ret = []
    for beta, c_star in zip(p.x[:-1], p.c[-1]):
        d = ls_hap(Gk, focal, panel, beta)
       # assert np.allclose(d['c'], c_star)
        # compute imputation score
        I = np.searchsorted(retain_id, ell)
        I = np.where(I==num_retain_id, num_retain_id-1, I)
        imp = d['path'][I]
        err = (G[:, focal] != G[:, panel][ell, imp]).sum()
        ret.append((beta, err))
    shutil.rmtree(dirpath)
    return np.array(ret)


def get_imputation_error(focal: int, panel: List[int], sample_size: int, length: float, k: float, seed: int):
    ts = msp.simulate(
        sample_size=sample_size,
        length=length,
        recombination_rate=1e-4,
        mutation_rate=1e-4,
        random_seed=seed+1
    )
    beta, y = imputation_accuracy(ts, focal, panel, k, seed).T
    beta_id = 0
    try:
        opt_imputed_beta = [beta[y.argmin()], beta[y.argmin()+1]]
    except IndexError:
        opt_imputed_beta = [beta[y.argmin()-1], beta[y.argmin()]]
        beta_id = 1
    return [opt_imputed_beta, beta_id, y.argmin()]


def get_imputation_error_std(focal: int, panel: List[int], sample_size: int, len_multiplier: float, k: int, seed: int):
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig('chr22', length_multiplier=len_multiplier)
    model = species.get_demographic_model('Africa_1T12')
    samples = model.get_samples(sample_size)
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples, seed=seed+1)
    G = ts.genotype_matrix()
    L, N = np.shape(G)
    k = int(L*k)
    beta, y = imputation_accuracy(ts, focal, panel, k).T
    opt_imputed_beta = [beta[y.argmin()], beta[y.argmin()+1]]
    return opt_imputed_beta, y.min()


def main():
    focal = 0
    panel = list(range(1, int(1e3)+1))
    sample_size = int(1e3+1)
    length = 1e8
    len_multiplier = 1.970
    k = 0.05
    N = 2
    array = int(sys.argv[1])
#    results = [get_imputation_error_std(focal, panel, sample_size, len_multiplier, k, i) for i in np.arange(array*N, (array+1)*N)]
    results = [get_imputation_error(focal, panel, sample_size, length, k, i) for i in np.arange(array*N, (array+1)*N)]
    with open('results_alg1/opt_imputed_beta%s.pkl' %array, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()


