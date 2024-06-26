from scipy.special import binom
from scipy.stats import chi2_contingency, mannwhitneyu, norm, t, ttest_ind
import numpy as np
import pandas as pd

class ABTesting:
    def __init__(self, discrete:bool, sample_size_large:bool, variances_known:bool, normally_distributed:bool):
        self.discrete=discrete
        self.sample_size_large=sample_size_large
        self.variances_known=variances_known
        self.normally_distributed=normally_distributed

    #Fisher's exact test
    def fisher_prb(self, matrix_vals):
        def hypergeom(k, K, n, N):
            """PMF of hypergeometric dist."""
            return binom(K,k)*binom(N-K, n-k)/binom(N,n)
        [[x1, y1], [x2, y2]]=matrix_vals
        K=x2+y2 #all clicks
        k=x2 #clicks in X-variant
        n=x1+x2  #entire pool of X-variant users
        N=x1+y1+x2+y2
        return hypergeom(k, K, n, N)
    
    def fisher_prb_histogram(self, matrix_vals):
        """PMF hist. computation using Fisher's exact test"""
        self.matrix_vals=matrix_vals #store original matrix values
        m=self.matrix_vals
        neg_val= -min(m[0,0], m[1,1])
        pos_val= min(m[1,0], m[0,1])
        probs=[]
        for k in range(neg_val, pos_val+1):
            m1=m+np.array([[1,-1], [-1,1]])*k
            probs.append(self.fisher_prb(m1))
        return probs
    
    def fisher_pval(self, matrix_vals):
        if not self.discrete or self.sample_size_large:
            return 'invalid usage'
        hist_probs=self.fisher_prb_histogram(matrix_vals)
        bars_h=np.array(hist_probs)
        fisher_prb=self.fisher_prb(self.matrix_vals)
        idxs=bars_h<=fisher_prb
        p_val = bars_h[idxs].sum()
        return fisher_prb, p_val

    def pearson_chisquare_t_test(self, matrix_vals, correction=False):
        if not self.discrete or not self.sample_size_large:
            return 'invalid usage'
        chi2_val, p_val=chi2_contingency(matrix_vals, correction)[:2]
        return chi2_val, p_val
    
    def two_sample_ztest(self, x: list, y: list, s_x: float, s_y: float, n_x: int, n_y: int):
        """x and y are the revenue generated from two different variants such as two product layouts"""
        """s_x and s_y are the standard deviations of both groups"""
        """n_x and n_y are the sample sizes of each variant"""
        if self.discrete or not self.variances_known or not self.normally_distributed:
            return 'invalid usage'
        z_score=np.mean(x)-np.mean(y) / np.sqrt((s_x**2)/n_x + (s_y**2)/n_y)

        stat_distrib=norm(loc=0, scale=1)

        #assuming we are doing a two tailed z-test for the alternative hypothesis that the results from the 2 variants  are unequal
        p_val=stat_distrib.cdf(z_score) * 2

        return p_val
    
    def students_t_test(self, x: list, y: list, n_x: int, n_y: int):
        if self.discrete or self.variances_known or not self.normally_distributed:
            return 'invalid usage'
        #ddof=1 for sample variance
        s_x, s_y=np.sqrt(np.var(x, ddof=1)), np.sqrt(np.var(y, ddof=1))

        #calculate pooled variance, assuming variances are the same
        s_pooled=np.sqrt((
            (n_x-1)*(s_x**2) + 
            (n_y-1)*(s_y**2))/ (n_x+n_y-2)
        )

        #test stat dist under null hypothesis
        degrees_of_freedom=n_x+n_y-2
        stat_distrib=t(df=degrees_of_freedom, loc=0, scale=1)

        t_val=(np.mean(x)-np.mean(y))/(s_pooled*np.sqrt(1/n_x + 1/n_y))

        p_val=stat_distrib.cdf(t_val)*2

        return p_val
    
    def welchs_t_test(self, x: list, y: list, n_x: int, n_y: int):
        if self.discrete or self.variances_known or not self.normally_distributed:
            return 'invalid usage'
        s_x, s_y=np.sqrt(np.var(x, ddof=1)), np.sqrt(np.var(y, ddof=1))

        #calculate NON-pooled variance
        s_d=np.sqrt((s_x**2/n_x + s_y**2/n_y))

        #test stat dist under null hypothesis
        degrees_of_freedom=s_d**4/((s_x**2/n_x)**2/(n_x-1)+
                                    (s_y**2/n_y)**2/(n_y-1))
        stat_distrib=t(df=degrees_of_freedom, loc=0, scale=1)

        t_val=(np.mean(x)-np.mean(y))/s_d

        p_val=stat_distrib.cdf(t_val)*2

        return p_val

    def mannwhitney_u_test(self, x:list, y:list, use_continuity=False, alternative="two-sided"):
        if self.discrete or self.sample_size_large or self.normally_distributed:
            return 'invalid usage'
        mwu=mannwhitneyu(x, y, use_continuity, alternative)
        m_stat=mwu.statistic
        m_pval=mwu.pvalue
        return m_stat, m_pval

# def main():
#     data=[[7,15],[8,4]]
#     result_matrix=pd.DataFrame(data=data,
#                              index=["no click", "click"],
#                              columns=["X_variant","Y_variant"])
#     matrix_vals=result_matrix.values
#     ABT=ABTesting(discrete=True, sample_size_large=True, variances_known=False, normally_distributed=False) 
#     try:
#         fisher_prb, p_val=ABT.fisher_pval(matrix_vals)
#     except ValueError:
#         print("invalid initialization")
#     chi2_val, p_val=ABT.pearson_chisquare_t_test(matrix_vals)
#     return chi2_val, p_val

# if __name__=="__main__":
#     chi2_val, p_val=main()
#     print(chi2_val)
#     print(p_val)
