from firedrake import *
from utility import Workspace
from scipy import stats
import matplotlib.pyplot as plt

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":
    nx =512
    alpha = 400 
    procno = ensemble_comm.rank 
    fname = "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha if alpha != None else 'none')
    workspace = Workspace(fname)
    # gridfname = workspace.output_name("streamfunction_increment_error_grid_value_time_series", "visuals")
    gridfname_reg = workspace.output_name("__streamfunction_increment_regularised_error_grid_value_time_series", "visuals")

    grid_pos = {0:"left",1:"middle",2:"right"}
    if procno == 0:
        time_series = np.load(gridfname_reg + ".npy") 
        PETSc.Sys.Print(gridfname_reg, flush=True)
        plotfname = workspace.output_name("_psi_inc_stats_res_{}.png".format(nx))
        list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']
        
        for grid_pt_id in [0,1,2]:
            pltdata = time_series[:,grid_pt_id]
            actual_series = np.cumsum(pltdata)

            results = []
            for i in list_of_dists:
                dist = getattr(stats, i)
                param = dist.fit(pltdata)
                a = stats.kstest(pltdata, i, args=param)
                results.append((i,a[0],a[1]))

            results.sort(key=lambda x:float(x[2]), reverse=True)
            with open(workspace.output_name("__psi_inc_dist_tests_grid_pos_{}.txt".format(grid_pos[grid_pt_id])), "w") as text_file:
                for j in results:
                    print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]), file=text_file)
