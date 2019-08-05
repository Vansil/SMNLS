"""display our Word in Context (WiC) results as an HTML table"""

import visual
results_files_dict = {
    # # old
    # 'elmo0': "output/baseline_elmo0/evaluation/results.pt",
    # 'elmo1': "output/baseline_elmo1/evaluation/results.pt",
    # 'elmo2': "output/baseline_elmo2/evaluation/results.pt",
    # 'elmo012': "output/baseline_elmo012/evaluation/results.pt",
    # 'random': "output/empty_jmt/evaluation/results.pt",
    # 'pos': "output/pos/evaluation/results.pt",
    # 'pos-snli': "output/pos-snli/evaluation/results.pt",
    # 'pos-vua-snli': "output/pos-vua-snli/evaluation/results.pt",
    # 'snli': "output/snli/evaluation/results.pt",
    # 'vua': "output/vua/evaluation/results.pt",
    # 'vua-pos': "output/vua-pos/evaluation/results.pt",
    # 'vua-snli': "output/vua-snli/evaluation/results.pt",
    # 'vpos': "output/vpos/evaluation/results.pt",
    # 'vpos-snli': "output/vpos-snli/evaluation/results.pt",
    # 'vpos-vua-snli': "output/vpos-vua-snli/evaluation/results.pt",
    # 'vua-vpos': "output/vua-vpos/evaluation/results.pt",

    # elmo-2
    'elmo-2_snli':      "output/elmo-2/snli/evaluation/results.pt",
    'elmo-2_vpos':      "output/elmo-2/vpos/evaluation/results.pt",
    'elmo-2_vpos-snli': "output/elmo-2/vpos-snli/evaluation/results.pt",
    'elmo-2_vpos-vua-snli': "output/elmo-2/vpos-vua-snli/evaluation/results.pt",
    'elmo-2_vua':      "output/elmo-2/vua/evaluation/results.pt",
    'elmo-2_vua-snli': "output/elmo-2/vua-snli/evaluation/results.pt",
    'elmo-2_vua-vpos': "output/elmo-2/vua-vpos/evaluation/results.pt",
    # elmo-3
    'elmo-3_snli':      "output/elmo-3/snli/evaluation/results.pt",
    'elmo-3_vpos':      "output/elmo-3/vpos/evaluation/results.pt",
    'elmo-3_vpos-snli': "output/elmo-3/vpos-snli/evaluation/results.pt",
    'elmo-3_vpos-vua-snli': "output/elmo-3/vpos-vua-snli/evaluation/results.pt",
    'elmo-3_vua':      "output/elmo-3/vua/evaluation/results.pt",
    'elmo-3_vua-snli': "output/elmo-3/vua-snli/evaluation/results.pt",
    'elmo-3_vua-vpos': "output/elmo-3/vua-vpos/evaluation/results.pt",
}
visual.wic_table(results_files_dict, 'results/WicTableVuapos.html', True, True)
