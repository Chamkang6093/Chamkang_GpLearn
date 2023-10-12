from Control.Control import multi_worldquant_brain, aggregate_process
from Chamkang_GpLearn.genetic import SymbolicTransformer

if __name__ =="__main__":

    path = './'
    pro_name = '1'
    engines = multi_worldquant_brain(12, headless = True, log_info = False)

    for i in range(0, 2):
        gp_transformer = SymbolicTransformer(population_size=50,
                                        tournament_size=5,
                                        generations=7,
                                        low_memory=True,
                                        p_crossover=0.9,
                                        p_subtree_mutation=0.04,
                                        p_hoist_mutation=0.03,
                                        p_point_mutation=0.03,
                                        settings=("USA", 3000, 1, "Sector", 10, 0.05),
                                        engines=engines,
                                        log_dir=path,
                                        project_name=pro_name,
                                        output_dir=str(i))
        gp_transformer.fit()

    aggregate_process(path + "project_" + pro_name + "/")