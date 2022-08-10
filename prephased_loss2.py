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

def ls_dip(ts_or_G, focal, panel, beta, alpha=1.0):
    try:
        G = ts_or_G.genotype_matrix()
    except AttributeError:
        G = ts_or_G
    g = G[:, focal].sum(1)
    H = G[:, panel]
    D = abs(H[:, :, None] + H[:, None, :] - g[:, None, None])
    L, N, _ = D.shape
    NN = np.arange(N)
    cost = np.zeros([N, N])
    m, r = np.zeros([2, N, N], int)
    m_star, switch = _switch(beta, cost)
    # quantities needed for traceback
    ibd = np.zeros([N, N], int)
    c_star = np.zeros([L, 3], int)  # the lowest cost haplotypes
    n_star = [0, 0]
    for ell in range(L):
        # cost'[h1, h2] = D[ell, h1, h2] + min(cost[h1, h2],  # no recombination
        #                                      beta + cost[h1, :].min(),  # single recombination
        #                                      beta + cost[:, h2].min(),  # single recombination
        #                                      2 * beta + cost.min())  # double recombination
        # update ibd tract lengths
        ibd += 1
        ibd[~switch[3]] = 1
        r = np.select(
            switch,
            [
                1 + r[cost.argmin(axis=0), NN][None],
                1 + r[NN, cost.argmin(axis=1)][:, None],
                2 + r.flat[cost.argmin()][None, None],
                r,
            ],
        )
        m = D[ell] + np.select(
            switch,
            [
                m[cost.argmin(axis=0), NN][None],
                m[NN, cost.argmin(axis=1)][:, None],
                m.flat[cost.argmin()][None, None],
                m,
            ],
        )
        cost = alpha * D[ell] + m_star
        n_star = np.unravel_index(cost.argmin(), cost.shape)
        c_star[ell] = np.r_[n_star, ibd[n_star]]
        m_star, switch = _switch(beta, cost)
    # np.testing.assert_allclose(cost[n_star], alpha * m[n_star] + beta * r[n_star])
    path = _backtrace(c_star)
    return {
        "c": cost.min(),
        "r": r[n_star],
        "m": m[n_star],
        "path": path,
        "g": g,
        "G": H,
        "alpha": alpha,
        "beta": beta,
    }

def _switch(beta, cost):
    c1 = beta + cost.min(axis=0, keepdims=True)
    c2 = beta + cost.min(axis=1, keepdims=True)
    c3 = 2 * beta + cost.min(keepdims=True)
    m_star = functools.reduce(np.minimum, [c1, c2, c3, cost])
    switch = [m_star == c1, m_star == c2, m_star == c3, m_star == cost]
    return m_star, switch

def vcf_genotype(fname, vcf, g):
    """"
    Write genotyoe to a vcf template
    Args:
    fname: the output file name,
    vcf: the template name,
    g: the genotype sequence
    """
    i = 0
    vcf = VCF("%s.vcf" %vcf)
    fname = "%s.vcf" %fname
    w = Writer(fname, vcf)
    L, N = np.shape(g)

    for v, a in zip(vcf, g):
        v.genotypes[0] = [a[0]]+[a[1]]+[True]
        v.genotypes = v.genotypes
        w.write_record(v)
        i+=1
        if i == L:
            break
    w.close(); vcf.close()

def vcf_gmatrix(fname, path):
    """
    Write a single vcf dipoid record to a np array matrix
    Args:
    fname: string vcf file name
    path: string vcf file address
    """
    vcf = VCF("%s/%s.vcf" %(path, fname))
    matrix = []
    for v in vcf:
        word = v.genotypes
        row = [w[:-1] for w in word]
        row = list(np.concatenate(row).flat)
        matrix.append(row)
    matrix = np.array(matrix, dtype=int)
    return matrix

def phase_impute(focal, panel, sample_size, length, len_multiplier, k, seed, flag):

    def get_sample():
        if flag == 'msprime':
            ts = msp.simulate(
            sample_size=sample_size,
            length=length,
            recombination_rate=1e-4,
            mutation_rate=1e-4,
            random_seed=seed+1
        )
            G = ts.genotype_matrix()
            g = G[:, focal]
        else:
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig('chr2', length_multiplier=len_multiplier)
            model = species.get_demographic_model('Africa_1T12')
            samples = model.get_samples(sample_size)
            engine = stdpopsim.get_engine("msprime")
            ts = engine.simulate(model, contig, samples, seed=seed+1)
            G = ts.genotype_matrix()
            g = G[:, focal]
        return ts, G, g

    def genotype_phasing(G):
        L, N = G.shape
        g = G[:, focal].sum(1)
        ell = np.arange(L)
        ls = LiStephensSurface(gh=G[:, focal], H=G[:, panel])
        b = beta_from_V(ls.V)
        phased_genotype = np.zeros([1, 2, L], int)
        for beta, c_star in zip(b.x[:-1], b.c[-1]):
            d = ls_dip(G, focal, panel, beta)
            #assert np.allclose(d['c'], c_star), (d['c'], c_star)
            path = d['path']
            p1 = np.repeat(path[:, 0], path[:, -1])
            p2 = np.repeat(path[:, 1], path[:, -1])
            new_genotype = np.vstack((G[:, panel][ell, p1], G[:, panel][ell, p2])).reshape(1, 2, L)
            phased_genotype = np.vstack((phased_genotype, new_genotype))
        return b.x[:-1], phased_genotype[1:]

    def pre_phase(beta, d, path):
        num_beta = np.size(beta)
        switch_error = []
        for i in np.arange(num_beta):
            phased_genotype = np.column_stack((d[i][0], d[i][1])).reshape(-1, 2)
            vcf_genotype("%s/in%s%s" % (path, seed, i), "template.vcf.recode", phased_genotype)
            p = subprocess.Popen('vcftools --vcf truth%s.vcf --diff in%s%s.vcf --diff-switch-error --out true_v_in%s%s' %(seed, seed, i, seed, i), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path)
            p.communicate()
            with open('%s/true_v_in%s%s.diff.indv.switch' %(path, seed, i), 'r') as f:
                var = f.read()
            s = var.strip().split('\t')[-1]
            switch_error.append(float(s))
        beta_id = 0
        try:
            opt_switch_beta = [beta[np.argmin(switch_error)], beta[np.argmin(switch_error)+1]]
        except IndexError:
            opt_switch_beta = [beta[np.argmin(switch_error)-1], beta[np.argmin(switch_error)]]
            beta_id = 1
        return "in%s%s" % (seed, np.argmin(switch_error))

    def get_maf(ftruth, path, k):
        """
        Compute the allele frequency and retained id
        Args:
        ftruth: string file name contained the information of focal and panel genotype
        path: string address of the file
        k: float keep af >= k
        """
        p = subprocess.Popen('vcftools --vcf %s.vcf --freq --out %s%s' %(ftruth, ftruth, seed), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path)
        p.communicate()
        i = 0
        maf = []; retain_id = []
        with open('%s/%s%s.frq' %(path, ftruth, seed), 'rb') as f:
                    for row in f:
                        if i == 0:
                            i += 1
                            continue
                        row = row.strip().decode("utf-8").split('\t')[4:]
                        af = [float(w[2:]) for w in row]
                        if np.min(af) > k:
                            retain_id.append(i-1)
                        maf.append(1-np.max(af)); i += 1
        return np.array(maf), retain_id

    def imputation_accuracy(G, path, fpre_phase):
        """
        Compute imputation accuracy for focal on panel across all settings of the li-stephens model, when only total k sites are retained. Impute a missing SNP by pasting the copying path from the nearest retained SNP behind it
        """
        L, N = G.shape
        num_retain_id = np.size(retain_id)
        gh = vcf_gmatrix(fpre_phase, path)
        Gk = G[retain_id]
        ell = np.arange(L)
        ls = LiStephensSurface(gh=gh, H=Gk[:, panel])
        p = beta_from_V(ls.V)
        ret = []
        for beta, c_star in zip(p.x[:-1], p.c[-1]):
            I = np.searchsorted(retain_id, ell)
            I = np.where(I==num_retain_id, num_retain_id-1, I)
            d = ls_dip(Gk, focal, panel, beta)
            path = d['path']
            p1 = np.repeat(path[:, 0], path[:, -1])
            p2 = np.repeat(path[:, 1], path[:, -1])
            imp1 = p1[I]
            imp2 = p2[I]
            Y = np.array(G[:, panel][ell, imp1] + G[:, panel][ell, imp2])
            # err = (np.abs(Y - np.sum(G[:, focal], axis=1)) / (2*maf*(1-maf))).sum()
            err = np.abs(Y - np.sum(G[:, focal], axis=1)).sum()
            ret.append((beta, err))
        return np.array(ret)

    def get_imputation_error(G, path, fpre_phase):
        beta, y = imputation_accuracy(G, dirpath, fpre_phase).T
        beta_id = 0
        try:
            opt_imputed_beta = [beta[y.argmin()], beta[y.argmin()+1]]
        except IndexError:
            opt_imputed_beta = [beta[y.argmin()-1], beta[y.argmin()]]
            beta_id = 1
        return [opt_imputed_beta, beta_id, y.argmin()]

    ts, G, g = get_sample()
    dirpath = tempfile.mkdtemp()
    with open("%s/allofG%s.vcf" % (dirpath, seed), "w") as vcf_file:
        ts.write_vcf(vcf_file)
    maf, retain_id = get_maf('allofG%s' % seed, dirpath, k)
    vcf_genotype("%s/truth%s" %(dirpath, seed), "template.vcf.recode", g[retain_id])
    beta, d = genotype_phasing(G[retain_id])
    fpre_phase = pre_phase(beta, d, dirpath)
    ret = get_imputation_error(G, dirpath, fpre_phase)
    shutil.rmtree(dirpath)
    return ret

def main():
    focal = [0, 1]
    panel = list(range(2, int(1e2)+2))
    sample_size = int(1e2+2)
    length = 1e7
    len_multiplier = 4.13e-2
    k = 0.05
    flag = 'stdpopsim'
    N = 1
    array = int(sys.argv[1])
    results = [phase_impute(focal, panel, sample_size, length, len_multiplier, k, array, flag) for i in np.arange(array*N, (array+1)*N)]
    if flag == 'msprime':
        with open('prephase_loss1/opt_imputed_beta%s.pkl' %array, 'wb') as f:
            pickle.dump(results, f)
    else:
        with open('prephase_loss1_std/opt_imputed_beta%s.pkl' %array, 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()


