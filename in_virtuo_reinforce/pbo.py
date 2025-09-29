import json
import os
import random
import lightning
import matplotlib.pyplot as plt
import numpy as np
import tdc
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from datetime import datetime
from tdc.generation import MolGen
import wandb

import os
import pandas as pd
from datetime import datetime
class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / len(buffer)


class Oracle:
    def __init__(self, max_oracle_calls=10000, freq_log=100, output_dir="results", mol_buffer=None, device=None):
        self.name = None
        self.evaluator = None
        self.task_label = None
        self.device = device
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log
        self.output_dir = output_dir

        self.mol_buffer = {} if mol_buffer is None else mol_buffer
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")
        self.last_log = 0
        self.avg_top10 = []
        self.num_calls = []
        self.predicted_auc = 0
        self.same_auc = 0


    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator
        self.name = evaluator.name
        self.name = self.name.replace("_current", "").lower()
        print("Oracle: " + self.name)

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.output_dir, "results.yaml")
        else:
            output_file_path = os.path.join(self.output_dir, "results_" + suffix + ".yaml")

        self.sort_buffer()
        # with open(output_file_path, "w") as f:
        #     yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[: self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)

        print(f"{n_calls}/{self.max_oracle_calls} | ")
        current_auc = top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls) * n_calls
        remaining_frac = 10000 - n_calls
        best_reachable_auc = (current_auc + remaining_frac) / 10000
        predicted_auc = (current_auc + remaining_frac*avg_top10)/10000
        if torch.isclose(torch.tensor(self.predicted_auc).float(), torch.tensor(predicted_auc).float(), atol=1e-3):
            self.same_auc += 1
        else:
            self.same_auc = 0
        self.predicted_auc = predicted_auc
        print(
            {
                "avg_top1": np.round(avg_top1,3),
                "avg_top10": np.round(avg_top10,3),
                "avg_top100": np.round(avg_top100,3),
                "auc_top10": np.round(top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),3),
                "best_reachable_auc": np.round(best_reachable_auc,3),
                "predicted_auc": np.round(predicted_auc,3),
                "n_oracle": np.round(n_calls,3),
            }
        )
        result_dict = {
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
        }
        wandb.log({"best_reachable_auc": best_reachable_auc, "predicted_auc": predicted_auc, "avg_top10": avg_top10})
        self.avg_top10.append(avg_top10)
        self.num_calls.append(n_calls)
        plt.plot(self.num_calls, self.avg_top10)
        plt.ylim(0, 1)
        plt.xlabel("Number of oracle calls")
        plt.ylabel("Average top-10 score")
        plt.title("Average top-10 score vs number of oracle calls, AUC: {:.2f}".format(current_auc / n_calls))
        plt.xlim(0, self.max_oracle_calls)
        print(self.name)
        os.makedirs(f"plots/tdc/{self.name}", exist_ok=True)
        plt.savefig(f"plots/tdc/{self.name}/tdc_avg_top10{self.device}.png", format="png")
        plt.close()

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls if self.mol_buffer else False


class TempOracle:
    def __init__(self, func=None) -> None:
        self.func = func
        self.name = func.__name__.split("_")[0]

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)


class BaseOptimizer:

    def __init__(self, smi_file=None, n_jobs=-1, max_oracle_calls=10000, freq_log=100, output_dir="results", log_results=True, device=None):
        self.model_name = "Default"
        self.n_jobs = n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = smi_file
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.log_results = log_results
        self.oracle = Oracle(max_oracle_calls=max_oracle_calls, freq_log=freq_log, output_dir=output_dir, device=device)
        # if self.smi_file is not None:
        #     self.all_smiles = self.load_smiles_from_file(self.smi_file)
        # else:
        #     data = MolGen(name = 'ZINC')
        #     self.all_smiles = data.get_data()['smiles'].tolist()

        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters=["PAINS", "SureChEMBL", "Glaxo"], property_filters_flag=False)
        self.qed = []
        self.sa = []
        self.scores = []
        self.ga_scores = []

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print("bad smiles")
        return new_mol_list

    def sort_buffer(self):
        self.oracle.sort_buffer()

    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)

    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > self.max_oracle_calls:
            results = results[:self.max_oracle_calls]

        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))


    def save_result_tot(self, suffix=None):
        """
        Append summary metrics (best score, top-10 AUC, top-100 AUC) as a new row
        in <output_dir>/<self.name>/results.csv (or results_<suffix>.csv).
        """
        # 1) Compute metrics
        self.sort_buffer()
        best_score = max(val[0] for val in self.mol_buffer.values())
        auc10  = top_auc(self.mol_buffer, 10, False, self.freq_log, self.max_oracle_calls)
        auc100 = top_auc(self.mol_buffer, 100, False, self.freq_log, self.max_oracle_calls)
        predicted_auc = self.oracle.predicted_auc
        # 2) Prepare output path
        out_dir = os.path.join(self.output_dir, self.oracle.name)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"results_{suffix}.csv" if suffix else "results.csv"
        out_path = os.path.join(out_dir, fname)

        # 3) Load existing CSV or create new DF
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
        else:
            df = pd.DataFrame(columns=["timestamp", "best_score", "auc_top10", "auc_top100", "predicted_auc"])

        # 4) Append new row
        new_row = {
            "timestamp":   datetime.utcnow().isoformat(),
            "best_score":  best_score,
            "auc_top10":   auc10,
            "auc_top100":  auc100,
            "predicted_auc": predicted_auc,
            "seed":        self.seed
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # 5) Save back to CSV
        df.to_csv(out_path, index=False)
        print(f"Appended results to {out_path}")

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores), np.mean(scores[:10]), np.max(scores), self.diversity_evaluator(smis), np.mean(self.sa_scorer(smis)), float(len(smis_pass) / 100), top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(
            max_oracle_calls=self.max_oracle_calls,
            freq_log=self.freq_log,
            output_dir=self.output_dir,
        )

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish

    def _optimize(self, oracle, config):
        raise NotImplementedError

    def optimize(self, oracle, config=None, patience=5, seed=0, project="test", **kwargs):
        if type(oracle) == str:
            oracle = tdc.Oracle(oracle)
        elif callable(oracle):
            oracle = TempOracle(oracle)
        # assert type(oracle) == tdc.Oracle

        assert callable(oracle)


        if config is not None:
            config = yaml.safe_load(open(config))
        else:
            config = dict()
        for key, value in kwargs.items():
            config[key] = value

        self.patience = patience
        run_name = self.model_name + "_" + oracle.name
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        self.oracle.task_label = self.model_name + "_" + oracle.name + "_" + str(seed)
        self._optimize(oracle, config)
        if self.log_results:
            self.log_result()
        # self.reset()

    def production(self, oracle, config, num_runs=5, project="production"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if num_runs > len(seeds):
            raise ValueError(f"Current implementation only allows at most {len(seeds)} runs.")
        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, seed, project)
            # self.reset()

    def save_qed_sa(self,oracle):
        df = pd.DataFrame(columns=['qed', 'sa','score' ])
        df_ga = pd.DataFrame(columns=['qed', 'sa','score' ])
        df['qed']= self.qed
        df['sa'] = self.sa
        df_ga['qed'] = self.qed_ga
        df_ga['sa'] = self.sa_ga
        df['score'] = self.scores
        df_ga['score'] = self.ga_scores
        out_dir = os.path.join(self.output_dir, self.oracle.name)
        os.makedirs(out_dir, exist_ok=True)

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, 'qed_sa.csv')
        df.to_csv(out_path, index=False)
        out_path = os.path.join(out_dir, 'qed_sa_ga.csv')
        df_ga.to_csv(out_path, index=False)