from genetic_algoritm import GeneticAlgorithmRunner

if __name__ == "__main__":
    runner = GeneticAlgorithmRunner()
    best_ind, best_score = runner.main()
    print(f"[MAIN] Done. best_ind => {best_ind}, best_score={best_score:.4f}")
