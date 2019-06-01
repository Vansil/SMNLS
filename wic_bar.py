import visual
results_files_dict = {
    'elmo0': "output/baseline_elmo0/evaluation/results.pt",
    'elmo1': "output/baseline_elmo1/evaluation/results.pt",
    'elmo2': "output/baseline_elmo2/evaluation/results.pt",
    'elmo012': "output/baseline_elmo012/evaluation/results.pt",
    'random': "output/empty_jmt/evaluation/results.pt",
    'pos': "output/pos/evaluation/results.pt",
    'pos-snli': "output/pos-snli/evaluation/results.pt",
    'pos-vua-snli': "output/pos-vua-snli/evaluation/results.pt",
    'snli': "output/snli/evaluation/results.pt",
    'vua': "output/vua/evaluation/results.pt",
    'vua-pos': "output/vua-pos/evaluation/results.pt",
    'vua-snli': "output/vua-snli/evaluation/results.pt",
    'vpos': "output/vpos/evaluation/results.pt",
    'vpos-snli': "output/vpos-snli/evaluation/results.pt",
    'vpos-vua-snli': "output/vpos-vua-snli/evaluation/results.pt",
    'vua-vpos': "output/vua-vpos/evaluation/results.pt",
}
visual.wic_barplot(results_files_dict, 'results/WicBarplotVuapos.png')
